"""Stage 1 -- subprocess wrapper around wagon_count/run_global_count.py.

The wagon_count package owns:
    * gap detection per camera
    * cross-camera gap fusion
    * RIGHT_UP master classification
    * deterministic GW_n id assignment

We invoke it as a subprocess with `--no-frames` so we get
`global_train_state.json` + `per_camera_tracking.json` plus the per-camera
tracking-overlay mp4s under `<output_dir>/processed_videos/` (kept as debug
artifacts; the rich feature-overlay videos are produced separately by
`rendering.feature_overlay_renderer`).  The new materializer/ owns frame
extraction so the wagon_count step does not duplicate it.

Returns the parsed GlobalTrainState (lightweight dataclass from
core.global_state_loader) or raises on failure.  Caller is responsible
for marking the batch as `failed_no_global_state` when this raises.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from core import constants as C
from core.global_state_loader import (
    GlobalTrainState, load_global_train_state, load_per_camera_fps,
)
from core.logging_setup import get_logger

log = get_logger("reconstruction")


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------

@dataclass
class ReconstructionResult:
    """Outcome of one Stage-1 invocation."""
    state: GlobalTrainState
    per_camera_fps: Dict[str, float]
    state_json_path: str
    per_camera_tracking_path: str
    output_dir: str
    elapsed_seconds: float
    # Master-first reconstruction provenance (mirrored from the sealed state)
    master_camera: str = C.MASTER_CAMERA
    reconstruction_mode: str = ""
    participating_cameras: Optional[list] = None
    missing_at_reconstruction: Optional[list] = None
    support_cameras_present: Optional[list] = None
    support_fusion_used: bool = False
    support_gap_recoveries: int = 0
    reconstruction_confidence: float = 1.0
    fallback_master_used: bool = False
    sealing_reason: str = ""


class ReconstructionError(RuntimeError):
    pass


# -----------------------------------------------------------------------------
# Subprocess driver
# -----------------------------------------------------------------------------

def _find_wagon_count_dir(repo_root: str) -> str:
    """Locate the wagon_count subpackage shipped next to this file."""
    candidate = os.path.join(repo_root, "wagon_count")
    if os.path.isfile(os.path.join(candidate, "run_global_count.py")):
        return candidate
    raise ReconstructionError(
        f"wagon_count/ not found under {repo_root}. "
        f"Expected {candidate}/run_global_count.py."
    )


_CAM_FLAG = {
    C.CAMERA_RIGHT_UP:     "--right_up",
    C.CAMERA_LEFT_UP:      "--left_up",
    C.CAMERA_RIGHT_UP_TOP: "--right_up_top",
    C.CAMERA_LEFT_UP_TOP:  "--left_up_top",
}


def _write_trace(output_dir: str, cmd: list, returncode: int,
                 elapsed: float, lines: List[str]) -> None:
    """Persist the full (stdout+stderr merged) subprocess trace per batch."""
    try:
        trace_path = os.path.join(output_dir, "stage1_wagon_count.log")
        with open(trace_path, "w", encoding="utf-8") as fh:
            fh.write(f"# cmd: {' '.join(cmd)}\n# exit={returncode} "
                     f"elapsed={elapsed:.1f}s\n\n--- OUTPUT (stdout+stderr) ---\n")
            fh.write("\n".join(lines))
            fh.write("\n")
    except Exception as e:  # never let logging bookkeeping fail the stage
        log.warning("[STAGE1] could not write stage1 trace file: %s", e)


def run_camera_gaps(
    *,
    camera_id: str,
    video_path: str,
    reconstruction_models_dir: str,
    gap_cache_dir: str,
    repo_root: str,
    source_identity: Optional[Dict[str, object]] = None,
    master_camera: str = C.MASTER_CAMERA,
    python_executable: Optional[str] = None,
    timeout_seconds: int = 3600,
    verbose: bool = True,
) -> int:
    """Gap-extract ONE camera into `gap_cache_dir`.  Returns the exit code.

    Invokes the same `run_global_count.py` in `--camera-only` mode, so gap
    detection, tracking, NMS/merge and the per-camera classification are the
    existing implementations -- there is no second detector.  Produces NO
    GlobalTrainState: assembly stays in `run()` below.

    Never raises for a processing failure; the caller isolates failures per
    camera and retries on a later tick.
    """
    if camera_id not in _CAM_FLAG:
        log.error("[GAP] unknown camera %s", camera_id)
        return 4
    if not os.path.exists(video_path):
        log.error("[GAP] %s video does not exist: %s", camera_id, video_path)
        return 4

    wagon_count_dir = _find_wagon_count_dir(repo_root)
    os.makedirs(gap_cache_dir, exist_ok=True)

    cmd = [python_executable or sys.executable,
           os.path.join(wagon_count_dir, "run_global_count.py"),
           _CAM_FLAG[camera_id], video_path,
           "--camera-only", camera_id,
           "--gap-cache", gap_cache_dir,
           "--models-dir", reconstruction_models_dir,
           # --master-camera tells the child whether THIS camera owns the
           # pre-fusion classification pass; it selects no master here.
           "--master-camera", master_camera]
    if source_identity is not None:
        cmd += ["--source-identity", json.dumps(source_identity)]

    log.info("[AUTO/GAP] Starting %s gap extraction", camera_id)
    t0 = time.time()
    rc, captured = _stream_subprocess(cmd, wagon_count_dir, timeout_seconds,
                                      verbose, tag=f"GAP:{camera_id}")
    elapsed = time.time() - t0
    if rc != 0:
        if captured:
            log.error("[AUTO/GAP] %s failed (exit=%s) --- output tail ---\n%s",
                      camera_id, rc, "\n".join(captured[-30:]))
        else:
            log.error("[AUTO/GAP] %s failed (exit=%s)", camera_id, rc)
        return rc if isinstance(rc, int) else 3
    log.info("[AUTO/GAP] %s gap extraction subprocess OK (%.1fs)",
             camera_id, elapsed)
    return 0


def _stream_subprocess(cmd: list, cwd: str, timeout_seconds: int,
                       verbose: bool, *, tag: str):
    """Run `cmd`, relaying each child line to the log as it is printed.

    Returns `(returncode_or_None, captured_lines)`; None means it timed out and
    was killed.  Shared by the per-camera and full Stage-1 invocations so both
    stream identically.
    """
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    captured: List[str] = []
    proc = subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )

    def _pump() -> None:
        try:
            for raw in proc.stdout:
                line = raw.rstrip("\n")
                captured.append(line)
                if verbose and line:
                    log.info("[%s] %s", tag, line)
        except Exception:                               # pragma: no cover
            pass

    reader = threading.Thread(target=_pump, name=f"{tag}-log-pump", daemon=True)
    reader.start()
    try:
        proc.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        proc.kill()
        reader.join(timeout=5)
        log.error("[%s] timed out after %ds", tag, timeout_seconds)
        return None, captured
    reader.join(timeout=10)
    return proc.returncode, captured


def run(
    *,
    video_paths: Dict[str, str],
    reconstruction_models_dir: str,
    output_dir: str,
    repo_root: str,
    master_camera: str = C.MASTER_CAMERA,
    allow_fallback_master: bool = False,
    python_executable: Optional[str] = None,
    timeout_seconds: int = 7200,
    verbose: bool = True,
    gap_cache_dir: Optional[str] = None,
) -> ReconstructionResult:
    """Run Stage 1 over the PRESENT cameras (master-first, subset-capable).

    Args:
        video_paths: {camera_id -> local path} for the cameras present NOW.
            Must include `master_camera`.  Absent cameras are simply not
            reconstructed -- their features attach later without a reseal.
        master_camera: which present camera drives the master timeline
            (default RIGHT_UP).  A non-RIGHT_UP master requires
            allow_fallback_master.
        allow_fallback_master: opt-in for a non-RIGHT_UP (LEFT_UP) master.
        reconstruction_models_dir: path to models/reconstruction/.
        output_dir: where wagon_count writes its outputs.
        repo_root: parent that contains the wagon_count/ subpackage.

    Raises:
        ReconstructionError on any failure (master absent, subprocess exit
        != 0, no JSON produced, zero wagons).
    """
    if master_camera not in video_paths:
        raise ReconstructionError(
            f"Stage 1 master camera {master_camera} is not present "
            f"(present: {sorted(video_paths)})")
    if master_camera != C.MASTER_CAMERA and not allow_fallback_master:
        raise ReconstructionError(
            f"master {master_camera} != {C.MASTER_CAMERA} requires "
            f"allow_fallback_master=True")
    for cam, p in video_paths.items():
        if not os.path.exists(p):
            raise ReconstructionError(f"Video for {cam} does not exist: {p}")

    if not os.path.isdir(reconstruction_models_dir):
        raise ReconstructionError(
            f"reconstruction_models_dir does not exist: "
            f"{reconstruction_models_dir}")

    wagon_count_dir = _find_wagon_count_dir(repo_root)
    script = os.path.join(wagon_count_dir, "run_global_count.py")
    os.makedirs(output_dir, exist_ok=True)

    cmd = [python_executable or sys.executable, script]
    for cam in C.ALL_CAMERAS:            # deterministic flag order
        if cam in video_paths:
            cmd += [_CAM_FLAG[cam], video_paths[cam]]
    cmd += ["--master-camera", master_camera]
    if allow_fallback_master:
        cmd += ["--allow-fallback-master"]
    cmd += [
        "--models-dir", reconstruction_models_dir,
        "--output",     output_dir,
        "--no-frames",      # materializer owns frame extraction
        # wagon_count's tracking overlay videos are kept (no --no-videos).
    ]
    # AUTO only: consume per-camera gap results already extracted incrementally.
    # Omitted by LOCAL mode, which therefore always infers in-process as before.
    if gap_cache_dir:
        cmd += ["--gap-cache", gap_cache_dir]

    if verbose:
        log.info("[STAGE1] launching wagon_count: %s", " ".join(cmd))

    # Stream the subprocess output LIVE into wagon_eye.log instead of buffering
    # it until exit.  A reader thread relays each line the moment it is printed
    # (the child runs with PYTHONUNBUFFERED=1 so its own print()s are flushed
    # per line), so `tail -f logs/wagon_eye.log` shows reconstruction progress
    # in real time.  stderr is merged into stdout to keep a single ordered trace.
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"

    t0 = time.time()
    captured: List[str] = []

    proc = subprocess.Popen(
        cmd, cwd=wagon_count_dir,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )

    def _pump() -> None:
        # Relay every child line immediately; also keep it for the trace file.
        try:
            for raw in proc.stdout:                     # blocks per line
                line = raw.rstrip("\n")
                captured.append(line)
                if verbose and line:
                    log.info("[STAGE1] %s", line)
        except Exception:                               # pragma: no cover
            pass

    reader = threading.Thread(target=_pump, name="stage1-log-pump", daemon=True)
    reader.start()

    try:
        proc.wait(timeout=timeout_seconds)              # hard cap, even on hang
    except subprocess.TimeoutExpired as e:
        proc.kill()
        reader.join(timeout=5)
        elapsed = time.time() - t0
        log.error("[STAGE1] wagon_count timed out after %.0fs (limit %ds)",
                  elapsed, timeout_seconds)
        _write_trace(output_dir, cmd, -1, elapsed, captured)
        raise ReconstructionError(
            f"wagon_count subprocess timed out after {timeout_seconds}s"
        ) from e
    reader.join(timeout=10)                             # drain remaining lines
    elapsed = time.time() - t0

    # Persist the FULL subprocess trace to a per-batch file (wagon_count stays
    # standalone -- it must not import core.logging_setup -- so its complete
    # output is captured here as well as streamed above).
    _write_trace(output_dir, cmd, proc.returncode, elapsed, captured)

    if verbose:
        log.info("[STAGE1] subprocess exit=%d (%.1fs)", proc.returncode, elapsed)

    if proc.returncode != 0:
        if captured:
            log.error("[STAGE1] --- output tail ---\n%s\n[STAGE1] ----------------------",
                      "\n".join(captured[-40:]))
        raise ReconstructionError(
            f"wagon_count subprocess exited {proc.returncode}"
        )

    state_path = os.path.join(output_dir, "global_train_state.json")
    if not os.path.isfile(state_path):
        raise ReconstructionError(
            f"wagon_count did not produce {state_path}"
        )

    state = load_global_train_state(state_path)
    if state.total_wagons <= 0:
        raise ReconstructionError(
            f"wagon_count returned total_wagons={state.total_wagons}; "
            f"aborting batch"
        )

    pcf_path = os.path.join(output_dir, "per_camera_tracking.json")
    per_camera_fps = load_per_camera_fps(pcf_path) if os.path.exists(pcf_path) else {}

    if verbose:
        log.info("[STAGE1] OK  total_wagons=%d  (E:%d  W:%d  B:%d)  master_fps=%.2f",
                 state.total_wagons, state.engine_count,
                 state.regular_wagon_count, state.brake_van_count,
                 state.master_fps)

    return ReconstructionResult(
        state=state,
        per_camera_fps=per_camera_fps,
        state_json_path=state_path,
        per_camera_tracking_path=pcf_path,
        output_dir=output_dir,
        elapsed_seconds=elapsed,
        master_camera=getattr(state, "master_camera", master_camera),
        reconstruction_mode=getattr(state, "reconstruction_mode", ""),
        participating_cameras=list(getattr(state, "participating_cameras", []) or []),
        missing_at_reconstruction=list(getattr(state, "missing_at_reconstruction", []) or []),
        support_cameras_present=list(getattr(state, "support_cameras_present", []) or []),
        support_fusion_used=bool(getattr(state, "support_fusion_used", False)),
        support_gap_recoveries=int(getattr(state, "support_gap_recoveries", 0) or 0),
        reconstruction_confidence=float(getattr(state, "reconstruction_confidence", 1.0) or 1.0),
        fallback_master_used=bool(getattr(state, "fallback_master_used", False)),
        sealing_reason=getattr(state, "sealing_reason", "") or "",
    )
