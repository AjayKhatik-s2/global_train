"""Incremental per-camera gap extraction for the AUTO/S3 pipeline.

WHAT THIS CHANGES
-----------------
Stage 1's per-camera work (gap detection + tracking, plus that camera's own
classification pass) used to happen entirely inside the single blocking
subprocess launched at seal time.  The first camera to land therefore sat idle
until the seal trigger fired, and every camera's inference was paid for in one
serial burst.

This module turns S3 into an incremental input source: the moment a camera's
object is proven stable it is downloaded and gap-extracted on its own, and the
complete result is persisted.  When the reconstruction eventually runs it loads
those results instead of re-running inference.

WHAT THIS DOES NOT CHANGE
-------------------------
* Nothing about the reconstruction mathematics -- fusion, trust weights, the
  RIGHT_UP canonical rule, ownership transitions, minimum gap distance,
  numbering and semantic classification all still happen in
  `global_alignment.assemble_global_train_state`, unchanged, from the same
  inputs.
* The seal TRIGGER.  Deciding when there are enough cameras to build a Global
  Train remains `lifecycle_runner.advance`'s existing policy.  "Enough cameras
  to reconstruct" and "enough cameras to start gap extraction" are now two
  separate conditions -- that separation is the whole point.
* LOCAL mode, which never calls into this module.

SAFETY PROPERTIES
-----------------
STABILITY   An object is only claimed once the same identity (ETag, size,
            last-modified) has been observed on two separate polls at least
            `WAGONEYE_GAP_STABILITY_SECONDS` apart.  A still-uploading file
            changes between polls and is left alone.
VALIDATION  After download the local file is checked for a plausible size and
            a readable moov/frame count before inference starts.
IDEMPOTENCY A completed (camera, object-identity) pair is never re-extracted.
            A replaced video has a different ETag and so is a NEW input.
LOCKING     A lock file carrying pid + hostname stops two AUTO workers from
            claiming the same camera. Stale locks expire.
ISOLATION   One camera's failure is recorded against that camera only; the
            other three continue to be discovered and processed.
"""

from __future__ import annotations

import errno
import json
import os
import socket
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from core import config as CFG
from core import constants as C
from core.logging_setup import get_logger

log = get_logger("gap_extraction")

#: Cache lives beside the batch's other Stage-1 artifacts.
DIR_GAP_CACHE = "gap_cache"


# -----------------------------------------------------------------------------
# Tunables
# -----------------------------------------------------------------------------

def stability_seconds() -> float:
    """Minimum age of an unchanged object before it may be claimed.

    Guards against reading a multipart upload mid-flight: S3 makes the object
    visible only on completion for a normal PUT, but a re-uploaded or
    still-being-written key can change under us, and the trimmed-clip producer
    writes `_train_incomplete.mp4` before its final form.
    """
    try:
        return max(0.0, float(os.environ.get(
            "WAGONEYE_GAP_STABILITY_SECONDS", "30")))
    except ValueError:
        return 30.0


def lock_stale_seconds() -> float:
    """After this long an unrefreshed lock is assumed dead (worker crashed)."""
    try:
        return max(60.0, float(os.environ.get(
            "WAGONEYE_GAP_LOCK_STALE_SECONDS", "5400")))
    except ValueError:
        return 5400.0


def enabled() -> bool:
    """Incremental extraction on?  Default ON; set 0 to fall back to seal-time.

    The fallback is a genuine escape hatch: with it off, `stage_seal` runs the
    unchanged full-inference path, so a problem here cannot strand production.
    """
    raw = os.environ.get("WAGONEYE_INCREMENTAL_GAPS", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def min_video_bytes() -> int:
    try:
        return max(0, int(os.environ.get("WAGONEYE_GAP_MIN_VIDEO_BYTES", "65536")))
    except ValueError:
        return 65536


# -----------------------------------------------------------------------------
# Cache access (state file only -- the heavy result file belongs to wagon_count)
# -----------------------------------------------------------------------------

def cache_dir(batch_root: str) -> str:
    return os.path.join(batch_root, DIR_GAP_CACHE)


def _state_file(cdir: str, camera: str) -> str:
    return os.path.join(cdir, f"{camera}.state.json")


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    os.replace(tmp, path)


def read_gap_state(cdir: str, camera: str) -> Optional[Dict[str, Any]]:
    return _read_json(_state_file(cdir, camera))


def write_gap_state(cdir: str, camera: str, state: str,
                    identity: Optional[Dict[str, Any]] = None, *,
                    error: Optional[str] = None,
                    gap_count: Optional[int] = None) -> None:
    _write_json(_state_file(cdir, camera), {
        "camera_id": camera,
        "state": state,
        "identity": identity,
        "error": error,
        "gap_count": gap_count,
        "owner": _owner_token(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })


# -----------------------------------------------------------------------------
# Object identity
# -----------------------------------------------------------------------------

def slot_identity(slot) -> Dict[str, Any]:
    """Identity of the S3 object currently in a CameraSlot.

    Mirrors `wagon_count.gap_cache.make_identity` -- kept as a small duplicate
    rather than an import because the orchestrator and the standalone
    wagon_count package do not share a sys.path.
    """
    return {
        "bucket": getattr(slot, "bucket", "") or "",
        "s3_key": getattr(slot, "s3_key", "") or "",
        "etag": (getattr(slot, "etag", None) or "").strip('"') or None,
        "file_size": int(getattr(slot, "file_size", 0) or 0),
        "last_modified": getattr(slot, "last_modified", None) or None,
    }


def identity_matches(a: Optional[Dict[str, Any]],
                     b: Optional[Dict[str, Any]]) -> bool:
    if not a or not b:
        return False
    if a.get("s3_key") != b.get("s3_key") or a.get("bucket") != b.get("bucket"):
        return False
    ea, eb = a.get("etag"), b.get("etag")
    if ea and eb:
        return ea == eb
    return (a.get("file_size") == b.get("file_size")
            and a.get("last_modified") == b.get("last_modified"))


def is_completed(cdir: str, camera: str, identity: Dict[str, Any]) -> bool:
    """This exact object already gap-extracted?  Requires BOTH files.

    A state record alone is not enough: a run killed between writing the state
    and the result would otherwise look complete and the reconstruction would
    find no gaps.
    """
    st = read_gap_state(cdir, camera)
    if not st or st.get("state") != "completed":
        return False
    if not identity_matches(st.get("identity"), identity):
        return False
    return os.path.isfile(os.path.join(cdir, f"{camera}.result.json"))


# -----------------------------------------------------------------------------
# Stability
# -----------------------------------------------------------------------------

def _seen_file(cdir: str, camera: str) -> str:
    return os.path.join(cdir, f"{camera}.seen.json")


def check_stable(cdir: str, camera: str, identity: Dict[str, Any]) -> bool:
    """True when this identity has been unchanged for `stability_seconds()`.

    First sighting records the identity and returns False -- one poll can never
    prove stability.  A changed identity restarts the clock, which is exactly
    the "if the object changes between checks, treat it as still uploading"
    rule.
    """
    path = _seen_file(cdir, camera)
    prev = _read_json(path)
    now = time.time()
    if prev and identity_matches(prev.get("identity"), identity):
        first = float(prev.get("first_seen_epoch") or now)
        if now - first >= stability_seconds():
            return True
        log.info("[AUTO/S3] %s object seen %.0fs ago, needs %.0fs to be "
                 "considered stable", camera, now - first, stability_seconds())
        return False
    if prev:
        log.info("[AUTO/S3] %s object CHANGED while waiting (still uploading?) "
                 "-- stability clock restarted", camera)
    _write_json(path, {"identity": identity, "first_seen_epoch": now})
    log.info("[AUTO/S3] Available: %s (awaiting stability)", camera)
    return False


# -----------------------------------------------------------------------------
# Locking
# -----------------------------------------------------------------------------

def _owner_token() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


def _lock_file(cdir: str, camera: str) -> str:
    return os.path.join(cdir, f"{camera}.lock")


def acquire(cdir: str, camera: str) -> bool:
    """Exclusively claim one camera.  O_CREAT|O_EXCL is the atomic primitive.

    A lock older than `lock_stale_seconds()` is treated as abandoned and broken,
    so a killed worker cannot block a camera forever.
    """
    os.makedirs(cdir, exist_ok=True)
    path = _lock_file(cdir, camera)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except OSError as e:
        if e.errno != errno.EEXIST:
            log.warning("[AUTO/GAP] %s lock error: %s", camera, e)
            return False
        try:
            age = time.time() - os.path.getmtime(path)
        except OSError:
            return False
        if age < lock_stale_seconds():
            held = (_read_json(path) or {}).get("owner", "?")
            log.info("[AUTO/GAP] %s already claimed by %s (%.0fs) -- skipping",
                     camera, held, age)
            return False
        log.warning("[AUTO/GAP] %s lock is stale (%.0fs) -- breaking it",
                    camera, age)
        try:
            os.unlink(path)
        except OSError:
            return False
        return acquire(cdir, camera)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump({"owner": _owner_token(),
                   "acquired_at": datetime.now(timezone.utc).isoformat()}, fh)
    return True


def release(cdir: str, camera: str) -> None:
    try:
        os.unlink(_lock_file(cdir, camera))
    except OSError:
        pass


# -----------------------------------------------------------------------------
# Download + validation
# -----------------------------------------------------------------------------

def _download(slot, cdir: str, dl_root: str, ctx) -> Optional[str]:
    """Fetch the camera video, returning its local path or None on failure.

    Downloads to a `.part` file and renames on success so an interrupted
    transfer can never be mistaken for a complete video on the next tick.
    """
    if slot.bucket == "__local__":
        return slot.s3_key if os.path.exists(slot.s3_key) else None
    os.makedirs(dl_root, exist_ok=True)
    name = f"{slot.camera_id}_{slot.filename or os.path.basename(slot.s3_key)}"
    final = os.path.join(dl_root, name)
    if os.path.exists(final) and _validate(final):
        return final
    part = final + ".part"
    try:
        ctx.s3_client.download_file(slot.bucket, slot.s3_key, part)
        os.replace(part, final)
    except Exception as e:
        log.error("[AUTO/GAP] %s download failed: %s", slot.camera_id, e)
        try:
            os.unlink(part)
        except OSError:
            pass
        return None
    return final if _validate(final) else None


def _validate(path: str) -> bool:
    """Cheap sanity check that a downloaded file is a usable video.

    Size first (a truncated transfer is usually tiny), then ask OpenCV to open
    it and report a frame count -- which fails fast on a partial container
    whose moov atom never arrived.  Passing here does not guarantee decodable
    content; it guarantees we do not launch a subprocess against rubbish.
    """
    try:
        if os.path.getsize(path) < min_video_bytes():
            log.error("[AUTO/GAP] %s is only %d bytes -- treating as incomplete",
                      path, os.path.getsize(path))
            return False
    except OSError as e:
        log.error("[AUTO/GAP] cannot stat %s: %s", path, e)
        return False
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        try:
            if not cap.isOpened():
                log.error("[AUTO/GAP] %s could not be opened as video", path)
                return False
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            cap.release()
        if frames <= 0:
            log.error("[AUTO/GAP] %s reports %d frames -- incomplete download",
                      path, frames)
            return False
    except ImportError:
        return True          # no cv2 here; size check is the best we can do
    except Exception as e:
        log.error("[AUTO/GAP] %s validation error: %s", path, e)
        return False
    return True


# -----------------------------------------------------------------------------
# Main entry: extract every camera that is ready RIGHT NOW
# -----------------------------------------------------------------------------

def readiness(manifest, cdir: str) -> Dict[str, str]:
    """`{camera: READY|PENDING|MISSING|FAILED}` for observability."""
    out: Dict[str, str] = {}
    for cam in C.ALL_CAMERAS:
        slot = manifest.cameras.get(cam)
        if slot is None or not getattr(slot, "s3_key", ""):
            out[cam] = "MISSING"
            continue
        ident = slot_identity(slot)
        if is_completed(cdir, cam, ident):
            out[cam] = "READY"
            continue
        st = read_gap_state(cdir, cam) or {}
        out[cam] = "FAILED" if st.get("state") == "failed" else "PENDING"
    return out


def extract_ready_cameras(manifest, ctx, batch_root: str) -> Dict[str, str]:
    """Gap-extract every present, stable, not-yet-done camera -- one at a time.

    Returns the readiness map after the pass.  Called on every AUTO tick while
    the batch is still collecting cameras, so a camera that arrives later is
    picked up on the next tick without waiting for the others.

    Deliberately sequential: gap extraction is GPU/CPU-bound and this box runs
    four cameras' worth of models, so parallel extraction would thrash rather
    than help.  Each camera returns to the poll loop as soon as it finishes.
    """
    from reconstruction import runner as reconstruction_runner

    cdir = cache_dir(batch_root)
    os.makedirs(cdir, exist_ok=True)
    dl_root = os.path.join(batch_root, CFG.DIR_DOWNLOADS)

    log.info("[AUTO/S3] Checking for available camera videos")
    for cam in C.ALL_CAMERAS:
        slot = manifest.cameras.get(cam)
        if slot is None or not getattr(slot, "s3_key", ""):
            continue
        ident = slot_identity(slot)

        if is_completed(cdir, cam, ident):
            continue
        # A different ETag for a camera we already did = a replacement video.
        st = read_gap_state(cdir, cam)
        if st and st.get("state") == "completed" and not identity_matches(
                st.get("identity"), ident):
            log.info("[AUTO/GAP] %s source object changed since its last gap "
                     "extraction -- re-extracting the new object", cam)

        if not check_stable(cdir, cam, ident):
            continue
        log.info("[AUTO/S3] %s object stable", cam)

        if not acquire(cdir, cam):
            continue
        try:
            write_gap_state(cdir, cam, "downloading", ident)
            local = _download(slot, cdir, dl_root, ctx)
            if not local:
                write_gap_state(cdir, cam, "failed", ident,
                                error="download_or_validation_failed")
                continue
            slot.local_path = local
            write_gap_state(cdir, cam, "downloaded", ident)

            rc = reconstruction_runner.run_camera_gaps(
                camera_id=cam,
                video_path=local,
                reconstruction_models_dir=ctx.recon_models_dir,
                gap_cache_dir=cdir,
                repo_root=ctx.repo_root,
                source_identity=ident,
                master_camera=C.MASTER_CAMERA,
                verbose=ctx.verbose,
            )
            # The subprocess owns the completed/failed state record (it knows
            # the gap count); only correct it if it died without writing one.
            if rc != 0:
                cur = read_gap_state(cdir, cam) or {}
                if cur.get("state") != "failed":
                    write_gap_state(cdir, cam, "failed", ident,
                                    error=f"subprocess_exit_{rc}")
                log.error("[AUTO/GAP] %s gap extraction FAILED -- other cameras "
                          "are unaffected and this one retries next tick", cam)
            else:
                done = read_gap_state(cdir, cam) or {}
                log.info("[AUTO/GAP] %s gap extraction complete: gaps=%s",
                         cam, done.get("gap_count"))
                log.info("[AUTO/GAP] Saved %s gap result", cam)
        except Exception as e:                       # never kill the tick
            log.error("[AUTO/GAP] %s unexpected error: %s", cam, e, exc_info=True)
            write_gap_state(cdir, cam, "failed", ident,
                            error=f"{type(e).__name__}: {e}")
        finally:
            release(cdir, cam)
        log.info("[AUTO/S3] Checking again for newly available cameras")

    ready = readiness(manifest, cdir)
    missing = [c for c, v in ready.items() if v != "READY"]
    log.info("[AUTO/GLOBAL] Gap results available: %s",
             " ".join(f"{c}={ready[c]}" for c in C.ALL_CAMERAS))
    if missing:
        log.info("[AUTO/GLOBAL] Still missing gap results for: %s",
                 ", ".join(missing))
    return ready
