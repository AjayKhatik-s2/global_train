"""
tracker_engine.py
=================

Phase-1 per-camera tracking engine.

Responsibilities (per camera):
  1. Open the input video.
  2. Run the appropriate gap-detection YOLO model on every frame.
  3. Track gap detections temporally with a constant-velocity Kalman
     filter on the bounding-box horizontal center, plus a hit/miss
     persistence rule.
  4. Emit one GapEvent per stable track.
  5. (RIGHT_UP only) Classify each pre-fusion master segment using
     side_classification.pt by majority voting across sampled frames.

This module deliberately reuses YOLO via `ultralytics` -- the same
pattern as RIGHT_UP/gap_validation.py and RIGHT_UP/wagon_classifier.py.

Determinism notes:
  * YOLO inference is deterministic given the same weights, device, and
    pre-processing.  We never use stochastic augmentation.
  * Track id assignment is monotonic in detection arrival order, which
    is itself deterministic (frame index then ymin then xmin).
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

import numpy as np

# OpenCV used only for reading frames + measuring video metadata.
# All model inference goes through ultralytics.YOLO.
import cv2

from global_train_state import (
    GapEvent,
    LocalCameraTracks,
    SegmentClass,
    _MasterClassification,
    CAMERA_RIGHT_UP,
    SIDE_CAMERAS,
    TOP_CAMERAS,
)


# =============================================================================
# DEVICE RESOLUTION
# =============================================================================
#
# wagon_count/ is a self-contained package (it must zip and run on its own,
# see its README), so it does NOT import core.config.  This mirrors the same
# logic locally: honour WAGONEYE_DEVICE, else auto-detect CUDA, else CPU.
# The Stage-1 subprocess inherits WAGONEYE_DEVICE from the orchestrator's
# environment, so the whole pipeline agrees on the device without any extra
# flag plumbing.  Passing the resolved device explicitly into inference keeps
# GPU behaviour identical (cuda == ultralytics' own default there) while
# giving a clean CPU fallback on a CPU-only host.

# =============================================================================
# STAGE-1 DETECTION BATCHING
# =============================================================================
#
# Stage 1 runs the gap detector over EVERY frame of all four full videos -- the
# single largest CPU cost in a batch, and until now strictly one frame per model
# call.  Batching the detector amortizes the per-call Python/pre/post overhead
# the same way `features/_common.iter_wagon_detections` does for Stage 3.
#
# DEFAULTS TO 1 (the original, unbatched path) ON PURPOSE.  Stage 1 produces the
# SEALED CANONICAL wagon count, GW ids and boundaries: everything downstream is
# keyed off them, and a resealed batch is never renumbered.  CPU batching can
# shift box coordinates by <=1e-3 px (BLAS reduction order) -- far below any
# threshold here, but "far below" is a judgement that deserves evidence from real
# video rather than assumption.  So the fast path is available and off by default;
# turn it on, compare the wagon count and GW boundaries against a batch=1 run on
# the same clips, and only then adopt it.
#
#   WAGONEYE_STAGE1_INFER_BATCH=1   (default) exact original behaviour
#   WAGONEYE_STAGE1_INFER_BATCH=16  batched detector, tracking order unchanged
#
# Only the raw detector call is batched.  Frames are consumed in strict order, and
# the class/confidence/height filters, same-frame NMS and sequential Kalman
# tracking all run per frame exactly as before.

def _stage1_infer_batch() -> int:
    raw = os.environ.get("WAGONEYE_STAGE1_INFER_BATCH")
    if not raw:
        return 1
    try:
        return max(1, int(raw))
    except ValueError:
        return 1


STAGE1_INFER_BATCH = _stage1_infer_batch()


# -----------------------------------------------------------------------------
# Last-segment head sampling
# -----------------------------------------------------------------------------
#
# The FINAL segment has no closing gap -- it runs to the end of the video.  The
# trimmed clip deliberately keeps a few seconds after the rake has passed (the
# extractor's END_EXTRA_BUFFER, needed so a real brake van is never cut off), so
# that segment is "vehicle, then empty track".
#
# Classifying it by sampling evenly across the whole span therefore votes mostly
# on grass.  Observed on batch 20260808_125052: GW_59 spanned 182 frames of
# which ~45 held a wagon and ~137 were empty track; all three cameras returned
# BRAKE_VAN at >=0.99 confidence for what was actually a wagon.
#
# The vehicle always sits at the START of that segment (gap -> vehicle -> empty
# track), so classify it from the head of the span.  A rake that genuinely ends
# with a brake van still reads BRAKE_VAN -- its head shows a brake van.
#
# ONLY the last segment, and NEVER when the last is also the first: a phantom
# leading segment can be DROPPED when it classifies UNKNOWN
# (global_alignment.build_global_wagons), and this must not be able to influence
# that decision.  Segment count, boundaries and numbering are untouched -- this
# changes a LABEL, never how many vehicles exist.
#
# WAGONEYE_STAGE1_LAST_SEGMENT_HEAD: fraction of the last segment to classify
# from (default 0.35).  0 or >=1 disables it and restores the previous
# whole-span sampling exactly.

def _last_segment_head_fraction() -> float:
    raw = os.environ.get("WAGONEYE_STAGE1_LAST_SEGMENT_HEAD")
    if raw is None or raw.strip() == "":
        return 0.35
    try:
        v = float(raw)
    except ValueError:
        return 0.35
    return v if 0.0 < v < 1.0 else 0.0      # out of range == disabled


def _resolve_device(force: Optional[str] = None) -> str:
    choice = (force or os.environ.get("WAGONEYE_DEVICE") or "auto").strip().lower()
    if choice in ("cuda", "gpu"):
        return "cuda"
    if choice == "cpu":
        return "cpu"
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# =============================================================================
# CONSTANT-VELOCITY KALMAN FILTER (1-D on bbox center_x)
# =============================================================================
#
# State vector  : [x, vx]
# Measurement   : [x]
# Transition    : x_{t+1} = x_t + vx_t,   vx_{t+1} = vx_t
# We keep this hand-rolled rather than pulling in filterpy so the module
# has no extra runtime dependency beyond numpy/cv2/ultralytics.

class _KF1D:
    __slots__ = ("x", "P", "F", "H", "Q", "R")

    def __init__(self, init_x: float, process_var: float = 4.0, meas_var: float = 9.0):
        self.x = np.array([[init_x], [0.0]], dtype=np.float64)
        self.P = np.eye(2, dtype=np.float64) * 100.0
        self.F = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
        self.H = np.array([[1.0, 0.0]], dtype=np.float64)
        self.Q = np.eye(2, dtype=np.float64) * process_var
        self.R = np.array([[meas_var]], dtype=np.float64)

    def predict(self) -> float:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0, 0])

    def update(self, z: float) -> None:
        y = np.array([[z]]) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P

    @property
    def cx(self) -> float:
        return float(self.x[0, 0])


# =============================================================================
# INTERNAL TRACK STATE
# =============================================================================

@dataclass
class _Track:
    track_id: int
    first_frame: int
    last_seen_frame: int
    confidences: List[float] = field(default_factory=list)
    centers: List[float] = field(default_factory=list)
    hit_frames: List[int] = field(default_factory=list)
    bboxes: List[List[float]] = field(default_factory=list)
    hit_count: int = 0
    miss_count: int = 0
    kf: Optional[_KF1D] = None
    confirmed: bool = False

    def predicted_center(self) -> float:
        if self.kf is None:
            return self.centers[-1] if self.centers else 0.0
        return self.kf.predict()

    def update(self, frame_idx: int, center_x: float, conf: float,
               bbox: Optional[List[float]] = None) -> None:
        if self.kf is None:
            self.kf = _KF1D(center_x)
        else:
            self.kf.update(center_x)
        self.centers.append(self.kf.cx)
        self.confidences.append(conf)
        self.hit_frames.append(frame_idx)
        if bbox is not None:
            self.bboxes.append([float(v) for v in bbox])
        self.hit_count += 1
        self.miss_count = 0
        self.last_seen_frame = frame_idx

    def mark_miss(self) -> None:
        self.miss_count += 1


# =============================================================================
# GAP TRACKER
# =============================================================================

class GapTracker:
    """Per-camera gap detection + tracking + GapEvent emission.

    Parameters
    ----------
    camera_id        : 'RIGHT_UP' | 'LEFT_UP' | 'RIGHT_UP_TOP' | 'LEFT_UP_TOP'
    model_path       : path to the YOLO weights for this camera --
                       right_up_wagon_gap.pt for RIGHT_UP,
                       left_up_wagon_gap.pt  for LEFT_UP,
                       top_gap.pt            for either top camera.
    confidence       : YOLO confidence threshold
    min_height_ratio : min bbox height as a fraction of frame height
                       (rejects floor-strip and tiny detections)
    match_distance_px: max horizontal distance to associate a detection with
                       an existing track
    min_hits         : a track must accumulate at least this many hits to be
                       emitted as a confirmed GapEvent
    max_miss         : a track is closed after this many consecutive misses
    """

    def __init__(
        self,
        camera_id: str,
        model_path: str,
        confidence: float = 0.4,
        min_height_ratio: float = 0.35,
        match_distance_px: float = 80.0,
        min_hits: int = 3,
        max_miss: int = 30,
        device: Optional[str] = None,
        verbose: bool = True,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Gap model not found for {camera_id}: {model_path}")
        # Defer ultralytics import so the file can be parsed without it.
        # Mirror the torch.load monkey-patch used by RIGHT_UP/wagon_classifier.py
        # so .pt weights load on torch >= 2.6.
        import torch
        _orig_load = torch.load
        def _patched(*a, **kw):
            kw.setdefault("weights_only", False)
            return _orig_load(*a, **kw)
        torch.load = _patched
        from ultralytics import YOLO

        self.camera_id = camera_id
        self.model_path = model_path
        self.confidence = float(confidence)
        self.min_height_ratio = float(min_height_ratio)
        self.match_distance_px = float(match_distance_px)
        self.min_hits = int(min_hits)
        self.max_miss = int(max_miss)
        # Was previously stored but never used (ultralytics guessed the device).
        # Resolve it now and actually pass it into inference below.
        self.device = device or _resolve_device()
        self.verbose = verbose

        if verbose:
            print(f"[GapTracker/{camera_id}] Loading {model_path}")
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        # Single-class models (very common for gap detectors) get a permissive
        # class filter -- whatever the one class is named, we accept it.
        # Multi-class models still require the class name to contain "gap".
        self._is_single_class_model = (len(self.class_names) == 1)
        if verbose:
            print(f"[GapTracker/{camera_id}] Classes: {self.class_names}  "
                  f"(single_class={self._is_single_class_model})")
            print(f"[GapTracker/{camera_id}] Filters: conf>={self.confidence}  "
                  f"min_height_ratio={self.min_height_ratio}")

        # Per-process diagnostic counters, reset at the start of each
        # process_video() call.  Help diagnose "no bbox shown" cases by
        # revealing whether YOLO never returned boxes vs. the filters
        # ate them.
        self._diag_total_yolo_boxes = 0
        self._diag_after_class = 0
        self._diag_after_conf = 0
        self._diag_kept = 0

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def process_video(
        self,
        video_path: str,
        frame_limit: int = 0,
        keep_raw_detections: bool = True,
    ) -> LocalCameraTracks:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        total_frames_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        if fps <= 0:
            cap.release()
            raise RuntimeError(f"Video reports non-positive fps: {video_path}")

        if self.verbose:
            print(f"\n[GapTracker/{self.camera_id}] {os.path.basename(video_path)}")
            print(f"  fps={fps:.3f}  frames={total_frames_meta}  size={width}x{height}")

        # ---- Stage-1-only frame trimming --------------------------------------
        # Ignore the first/last N% of frames DURING RECONSTRUCTION ONLY, to avoid
        # false wagons from partially-visible wagons at the video edges.  Frame
        # numbering is preserved: `frame_idx` still counts from 0 over the ORIGINAL
        # video, so gap events (and therefore Stage 2 materialization, Stage 3
        # inference, processed videos, reports, JSON) reference the same frame
        # indices as before.  0% reproduces the prior behaviour exactly.
        try:
            trim_pct = float(os.getenv("WAGONEYE_STAGE1_FRAME_TRIM_PERCENT", "4") or 4)
        except ValueError:
            trim_pct = 4.0
        trim_pct = max(0.0, min(49.0, trim_pct))
        trim_start = int(total_frames_meta * trim_pct / 100.0)
        trim_end = int(total_frames_meta * trim_pct / 100.0)
        process_lo = trim_start                       # first analyzed frame_idx
        process_hi = total_frames_meta - trim_end     # exclusive upper bound
        if self.verbose:
            print(f"[STAGE1] {self.camera_id}: total_frames={total_frames_meta}, "
                  f"trimmed_start={trim_start}, trimmed_end={trim_end}, "
                  f"processing_frames={max(0, process_hi - process_lo)}")

        active_tracks: List[_Track] = []
        completed_tracks: List[_Track] = []
        next_track_id = 1
        raw_detections: Dict[int, List[Dict[str, Any]]] = {}

        # Reset diagnostic counters for this video -- these drive the Stage-1
        # gap-lifecycle audit (raw -> filtered -> deduped -> tracked -> merged
        # -> final).  Exposed via self.stats after process_video().
        self._diag_total_yolo_boxes = 0   # every raw YOLO box
        self._diag_after_class = 0        # survived class filter
        self._diag_after_conf = 0         # survived confidence floor
        self._diag_kept = 0               # survived height filter (candidates)
        self._diag_after_nms = 0          # candidates surviving same-frame NMS
        self._diag_tracks_created = 0     # _Track objects ever created
        self._diag_tracks_confirmed = 0   # tracks that reached min_hits
        self._diag_tracks_merged = 0      # duplicate tracks folded away
        self.stats: Dict[str, int] = {}

        frame_idx = 0
        frame_idx_read = 0
        t0 = time.time()
        # Live progress cadence (frames).  Env-overridable so operators can make
        # it chattier/quieter without a code change; default 100.
        try:
            progress_interval = max(1, int(
                os.getenv("WAGONEYE_PROGRESS_LOG_INTERVAL", "100") or 100))
        except ValueError:
            progress_interval = 100

        # Detection prefetch buffer: (frame, detections) pairs already inferred,
        # consumed in strict frame order.  Empty and unused when
        # STAGE1_INFER_BATCH == 1 (the default), in which case the loop below runs
        # the original one-frame-at-a-time path unchanged.
        _pending: List[Any] = []

        def _fill_batch() -> None:
            """Read up to STAGE1_INFER_BATCH analyzable frames and infer as one."""
            window: List[np.ndarray] = []
            nonlocal frame_idx_read  # read cursor, ahead of frame_idx when batching
            while len(window) < STAGE1_INFER_BATCH:
                if frame_limit and frame_idx_read >= frame_limit:
                    break
                if frame_idx_read >= process_hi:
                    break
                ok, f = cap.read()
                if not ok:
                    break
                idx = frame_idx_read
                frame_idx_read += 1
                if idx < process_lo:
                    continue            # trimmed head: consumed, not analyzed
                window.append(f)
            if window:
                for f, d in zip(window, self._detect_gaps_batched(window, height)):
                    _pending.append((f, d))

        while True:
            if STAGE1_INFER_BATCH > 1:
                # ---- batched path (opt-in) ----
                if not _pending:
                    _fill_batch()
                    if not _pending:
                        break
                frame, detections = _pending.pop(0)
                # `frame_idx` still advances one analyzed frame at a time, so gap
                # events keep their ORIGINAL frame numbers.
                if frame_idx < process_lo:
                    frame_idx = process_lo
            else:
                # ---- original single-frame path (default, unchanged) ----
                if frame_limit and frame_idx >= frame_limit:
                    break
                ret, frame = cap.read()
                if not ret:
                    break

                # Stage-1 trim: analyze only [process_lo, process_hi); skip the
                # edge frames but keep frame_idx advancing so gap events keep
                # ORIGINAL frame numbers.  Stop once past the trimmed tail
                # (nothing left to analyze).  With trim=0 this is a no-op
                # (identical prior behaviour).
                if frame_idx >= process_hi:
                    break
                if frame_idx < process_lo:
                    frame_idx += 1
                    continue

                detections = self._detect_gaps(frame, height)

            if keep_raw_detections and detections:
                # Lightweight payload for the overlay renderer (bbox + conf)
                raw_detections[frame_idx] = [
                    {
                        "bbox": [float(x) for x in d["bbox"]],
                        "confidence": d["confidence"],
                        "center_x": d["center_x"],
                    }
                    for d in detections
                ]

            # Predict step for all active tracks
            for tr in active_tracks:
                tr.predicted_center()

            # Greedy nearest-neighbor association on x distance.
            # Sort detections by center_x so association is deterministic.
            detections_sorted = sorted(detections, key=lambda d: d["center_x"])
            used_track_idx: set = set()

            for det in detections_sorted:
                best_i, best_d = -1, float("inf")
                cx = det["center_x"]
                for i, tr in enumerate(active_tracks):
                    if i in used_track_idx:
                        continue
                    d = abs(cx - tr.kf.cx) if tr.kf is not None else abs(cx - tr.centers[-1])
                    if d < best_d and d <= self.match_distance_px:
                        best_d = d
                        best_i = i
                if best_i >= 0:
                    used_track_idx.add(best_i)
                    active_tracks[best_i].update(frame_idx, cx, det["confidence"], det["bbox"])
                    if active_tracks[best_i].hit_count >= self.min_hits:
                        active_tracks[best_i].confirmed = True
                else:
                    tr = _Track(
                        track_id=next_track_id,
                        first_frame=frame_idx,
                        last_seen_frame=frame_idx,
                    )
                    tr.update(frame_idx, cx, det["confidence"], det["bbox"])
                    active_tracks.append(tr)
                    next_track_id += 1
                    self._diag_tracks_created += 1

            # Increment miss for tracks not matched this frame
            for i, tr in enumerate(active_tracks):
                if i not in used_track_idx:
                    tr.mark_miss()

            # Close tracks that exceeded max_miss
            still_active: List[_Track] = []
            for tr in active_tracks:
                if tr.miss_count >= self.max_miss:
                    if tr.confirmed:
                        completed_tracks.append(tr)
                else:
                    still_active.append(tr)
            active_tracks = still_active

            frame_idx += 1
            # ---- live progress (every WAGONEYE_PROGRESS_LOG_INTERVAL frames) ----
            # Streamed straight into logs/wagon_eye.log by the Stage-1 reader
            # thread (PYTHONUNBUFFERED=1), so `tail -f` shows it in real time.
            if self.verbose and frame_idx % progress_interval == 0:
                elapsed = time.time() - t0
                proc_fps = frame_idx / elapsed if elapsed > 0 else 0.0
                if total_frames_meta > 0:
                    pct = 100.0 * frame_idx / total_frames_meta
                    remaining = max(0, total_frames_meta - frame_idx)
                    eta = remaining / proc_fps if proc_fps > 0 else 0.0
                    frac = f"{frame_idx}/{total_frames_meta} ({pct:.1f}%)"
                    eta_s = f"{eta:.0f}s"
                else:
                    frac = f"{frame_idx}/?"
                    eta_s = "?"
                print(f"[GapTracker/{self.camera_id}] frame={frac} "
                      f"fps={proc_fps:.1f} elapsed={elapsed:.0f}s eta={eta_s} "
                      f"active_tracks={len(active_tracks)} "
                      f"completed_gaps={len(completed_tracks)}")

        cap.release()
        # Flush surviving confirmed tracks
        for tr in active_tracks:
            if tr.confirmed:
                completed_tracks.append(tr)

        self._diag_tracks_confirmed = len(completed_tracks)

        # DUPLICATE REMOVAL -- fold together tracks that represent the SAME
        # physical gap (two ids alive over the same frames at the same x).
        # Distinct gaps are separated either in TIME (sequential as the train
        # passes) or in SPACE (two boundaries in view sit at different x), so
        # this can never merge two real gaps -- it only heals duplicates the
        # per-frame NMS did not already prevent.
        before = len(completed_tracks)
        completed_tracks = self._merge_duplicate_tracks(completed_tracks)
        self._diag_tracks_merged = before - len(completed_tracks)

        # Sort by first_frame so GapEvents are temporally ordered, then
        # rewrite track_ids 1..N for determinism
        completed_tracks.sort(key=lambda t: (t.first_frame, t.last_seen_frame))

        events: List[GapEvent] = []
        for new_id, tr in enumerate(completed_tracks, start=1):
            mean_conf = float(np.mean(tr.confidences)) if tr.confidences else 0.0
            span = max(1, tr.last_seen_frame - tr.first_frame + 1)
            tcs = float(min(1.0, tr.hit_count / span))
            events.append(GapEvent(
                track_id=new_id,
                camera_id=self.camera_id,
                start_frame=tr.first_frame,
                end_frame=tr.last_seen_frame,
                confidence=mean_conf,
                hit_count=tr.hit_count,
                center_x_trajectory=list(tr.centers),
                fps=fps,
                temporal_consistency_score=tcs,
                hit_frames=list(tr.hit_frames),
                bbox_history=[list(b) for b in tr.bboxes],
            ))

        # Effective frame count = whatever we actually iterated through
        effective_frames = frame_idx
        # Some containers misreport CAP_PROP_FRAME_COUNT; trust what we read.
        total_frames = max(effective_frames, total_frames_meta if total_frames_meta > 0 else 0)

        # Persist the gap-lifecycle audit for this camera (used by the Stage-1
        # validation report + the [STAGE1] tracker-audit log line).
        self.stats = {
            "raw_yolo_boxes":   self._diag_total_yolo_boxes,
            "after_class":      self._diag_after_class,
            "after_confidence": self._diag_after_conf,
            "candidates":       self._diag_kept,        # survived all filters
            "after_nms":        self._diag_after_nms,   # per-frame deduped
            "tracks_created":   self._diag_tracks_created,
            "tracks_confirmed": self._diag_tracks_confirmed,
            "tracks_rejected":  self._diag_tracks_created - self._diag_tracks_confirmed,
            "tracks_merged":    self._diag_tracks_merged,
            "final_gaps":       len(events),
        }

        elapsed = time.time() - t0
        if self.verbose:
            print(f"[GapTracker/{self.camera_id}] done in {elapsed:.1f}s  "
                  f"emitted {len(events)} confirmed gaps  "
                  f"({frame_idx} frames processed)")
            # Full gap-lifecycle audit: raw -> filtered -> deduped -> tracked ->
            # confirmed -> merged -> final.  final == GapEvents == what the video
            # annotates == boundaries fed to reconstruction.
            s = self.stats
            print(f"[STAGE1] {self.camera_id} gap lifecycle: "
                  f"raw={s['raw_yolo_boxes']} -> class={s['after_class']} -> "
                  f"conf={s['after_confidence']} -> candidates={s['candidates']} -> "
                  f"nms={s['after_nms']} | tracks_created={s['tracks_created']} "
                  f"confirmed={s['tracks_confirmed']} "
                  f"rejected={s['tracks_rejected']} merged={s['tracks_merged']} "
                  f"-> FINAL={s['final_gaps']}")
            # Filter-stage diagnostics -- helps spot "no bbox shown" cases.
            print(f"  YOLO boxes: total={self._diag_total_yolo_boxes}  "
                  f"after_class={self._diag_after_class}  "
                  f"after_conf={self._diag_after_conf}  "
                  f"kept={self._diag_kept}")
            if self._diag_total_yolo_boxes > 0 and self._diag_kept == 0:
                print(f"  ⚠ All {self._diag_total_yolo_boxes} YOLO boxes were "
                      f"rejected by filters.  Lower --side/top-confidence or "
                      f"--side/top-min-height-ratio for this camera.")

        return LocalCameraTracks(
            camera_id=self.camera_id,
            video_path=video_path,
            fps=fps,
            total_frames=total_frames,
            width=width,
            height=height,
            gaps=events,
            raw_frame_detections=raw_detections if keep_raw_detections else {},
        )

    # ------------------------------------------------------------------
    # Per-frame YOLO inference
    # ------------------------------------------------------------------
    def _detect_gaps(self, frame: np.ndarray, frame_h: int) -> List[Dict[str, Any]]:
        """Single-frame detection -- the exact original call path."""
        results = self.model(frame, verbose=False, device=self.device)[0]
        return self._postprocess_gaps(results, frame_h)

    def _detect_gaps_batched(self, frames: List[np.ndarray],
                             frame_h: int) -> List[List[Dict[str, Any]]]:
        """Detect on N frames in ONE model call; return per-frame detection lists.

        Only the raw detector call is batched -- every downstream step (class
        filter, confidence floor, height filter, same-frame NMS, and the caller's
        sequential Kalman tracking) is the SAME code applied in the SAME frame
        order.  Opt-in via WAGONEYE_STAGE1_INFER_BATCH; see the constant's note on
        why it defaults to 1.
        """
        if not frames:
            return []
        if len(frames) == 1:
            return [self._detect_gaps(frames[0], frame_h)]
        results = self.model(frames, verbose=False, device=self.device)
        return [self._postprocess_gaps(r, frame_h) for r in results]

    def _postprocess_gaps(self, results, frame_h: int) -> List[Dict[str, Any]]:
        """Filter one frame's raw YOLO output into gap detections.

        Unchanged from the original inline body -- factored out so the
        single-frame and batched paths cannot diverge.
        """
        dets: List[Dict[str, Any]] = []
        if results.boxes is None or len(results.boxes) == 0:
            return dets

        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        clss = results.boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls_id in zip(boxes, confs, clss):
            self._diag_total_yolo_boxes += 1

            name = self.class_names.get(int(cls_id), "unknown").lower()
            # Permissive class filter:
            #   - single-class models: accept any (model is gap-only by design)
            #   - multi-class models: require "gap" as a substring of the name
            if not self._is_single_class_model and "gap" not in name:
                continue
            self._diag_after_class += 1

            if float(conf) < self.confidence:
                continue
            self._diag_after_conf += 1

            # Height-ratio filter rejects tiny noise detections.  For top
            # cameras the gap is a thin horizontal strip and the caller
            # should configure min_height_ratio to a low value (~0.05).
            h = float(box[3] - box[1])
            if h < frame_h * self.min_height_ratio:
                continue
            self._diag_kept += 1

            cx = float((box[0] + box[2]) / 2.0)
            dets.append({
                "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                "confidence": float(conf),
                "center_x": cx,
                "height": h,
            })

        # Same-frame NMS: one physical gap must yield ONE detection per frame.
        # Without this, two overlapping YOLO boxes on the same gap would each
        # spawn/feed a separate track -> duplicate gaps.  Distinct gaps sit at
        # different x with low IoU, so they are never suppressed.
        dets = self._nms_same_frame(dets)
        self._diag_after_nms += len(dets)
        return dets

    @staticmethod
    def _bbox_iou(a: List[float], b: List[float]) -> float:
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    def _nms_same_frame(self, dets: List[Dict[str, Any]],
                        iou_thresh: float = 0.5) -> List[Dict[str, Any]]:
        """Greedy IoU NMS, keeping the highest-confidence box in each cluster."""
        kept: List[Dict[str, Any]] = []
        for d in sorted(dets, key=lambda x: x["confidence"], reverse=True):
            if all(self._bbox_iou(d["bbox"], k["bbox"]) < iou_thresh for k in kept):
                kept.append(d)
        return kept

    def _merge_duplicate_tracks(self, tracks: List["_Track"]) -> List["_Track"]:
        """Fold tracks that are the SAME physical gap into one stable track.

        A candidate is absorbed into an existing kept track when their frame
        spans OVERLAP (same gap alive under two ids) AND their mean x-centres are
        within ``match_distance_px`` (same location).  Both conditions are
        required, so two genuinely different gaps -- separated in time OR in x --
        are never merged.  Absorbing preserves every hit (union of hit_frames /
        bboxes / confidences) so no evidence is suppressed.
        """
        kept: List[_Track] = []
        for tr in sorted(tracks, key=lambda t: (t.first_frame, t.last_seen_frame)):
            tr_cx = float(np.mean(tr.centers)) if tr.centers else 0.0
            target = None
            for m in kept:
                overlap = (min(m.last_seen_frame, tr.last_seen_frame)
                           - max(m.first_frame, tr.first_frame))
                if overlap <= 0:
                    continue
                m_cx = float(np.mean(m.centers)) if m.centers else 0.0
                if abs(m_cx - tr_cx) <= self.match_distance_px:
                    target = m
                    break
            if target is None:
                kept.append(tr)
                continue
            # merge tr INTO target, de-duplicating shared hit frames
            by_frame = dict(zip(target.hit_frames, zip(target.centers,
                                                       target.confidences, target.bboxes)))
            for f, cx, cf, bb in zip(tr.hit_frames, tr.centers, tr.confidences, tr.bboxes):
                if f not in by_frame:
                    by_frame[f] = (cx, cf, bb)
            order = sorted(by_frame)
            target.hit_frames = order
            target.centers = [by_frame[f][0] for f in order]
            target.confidences = [by_frame[f][1] for f in order]
            target.bboxes = [by_frame[f][2] for f in order]
            target.hit_count = len(order)
            target.first_frame = min(target.first_frame, tr.first_frame)
            target.last_seen_frame = max(target.last_seen_frame, tr.last_seen_frame)
        return kept


# =============================================================================
# MASTER CLASSIFIER (RIGHT_UP only)
# =============================================================================

# Reuse the label mapping from the existing wagon_classifier so behavior
# is consistent with the legacy pipeline.
_ENGINE_LABELS = {"engine", "loco", "engine_head", "locono", "locomotive"}
_BRAKEVAN_LABELS = {"tail", "brake_van", "brakevan", "guard_van", "wagon_tail"}
_TRACK_LABELS = {"track", "background", "empty_track", "rail", "tracks"}

# Minimum mean vote confidence required before a segment may be called ENGINE.
# Below this, an 'engine'/'loco' vote is treated as UNCERTAIN (UNKNOWN), not
# promoted to ENGINE.  This is the single guard that stops a low-confidence
# loco-front / track-strip read at train entry from creating a phantom leading
# engine.  NEVER assume "first detected wagon == ENGINE".
ENGINE_MIN_CONFIDENCE = 0.55


class MasterClassifier:
    """Classify video segments into ENGINE / WAGON / BRAKE_VAN / UNKNOWN.

    A *segment* is the span between two consecutive gaps of the driving camera
    (or between video start and the first gap / between the last gap and video
    end).  For each segment we sample N frames evenly, run the classifier, and
    majority-vote.

    This class is model-agnostic: RIGHT_UP drives it with side_classification.pt
    (``tag="MASTER"``); the TOP cameras drive the SAME class with
    top_classification.pt (``tag="TOP"``) to provide the extra semantic evidence
    fused in ``assemble_global_train_state``.  The label mapping
    (``_label_to_class``) is name-based, so it handles either model's class list.
    """

    def __init__(
        self,
        model_path: str,
        num_samples: int = 5,
        verbose: bool = True,
        device: Optional[str] = None,
        tag: str = "MASTER",
    ):
        self.tag = tag
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Classification model not found: {model_path}")
        import torch
        _orig_load = torch.load
        def _patched(*a, **kw):
            kw.setdefault("weights_only", False)
            return _orig_load(*a, **kw)
        torch.load = _patched
        from ultralytics import YOLO

        self.model_path = model_path
        self.num_samples = int(num_samples)
        self.verbose = verbose
        self.device = device or _resolve_device()
        if verbose:
            print(f"[MasterClassifier] Loading {model_path}")
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        if verbose:
            print(f"[MasterClassifier] Classes: {self.class_names}")

    def classify_frame(self, frame: np.ndarray) -> Tuple[str, float]:
        results = self.model(frame, verbose=False, device=self.device)[0]
        if getattr(results, "probs", None) is not None:
            top1 = int(results.probs.top1)
            conf = float(results.probs.top1conf)
            return self.class_names.get(top1, "unknown").lower(), conf
        if results.boxes is not None and len(results.boxes) > 0:
            confs = results.boxes.conf.cpu().numpy()
            cls = results.boxes.cls.cpu().numpy().astype(int)
            best = int(np.argmax(confs))
            return self.class_names.get(int(cls[best]), "unknown").lower(), float(confs[best])
        return "wagon", 0.0

    def classify_segments(
        self,
        video_path: str,
        segments: List[Tuple[int, int]],
        last_segment_head_fraction: Optional[float] = None,
    ) -> List[_MasterClassification]:
        """Classify each (start_frame, end_frame) segment of `video_path`.

        The LAST segment (when it is not also the first) is classified from the
        head of its span -- see `_last_segment_head_fraction` for why.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video for classification: {video_path}")

        head = (_last_segment_head_fraction()
                if last_segment_head_fraction is None
                else float(last_segment_head_fraction))
        n = len(segments)

        out: List[_MasterClassification] = []
        try:
            for idx, (sf, ef) in enumerate(segments):
                # last, and never the first -- the leading segment feeds the
                # phantom-drop guard and must be classified exactly as before.
                is_last_only = (idx == n - 1 and idx != 0)
                frac = head if (is_last_only and 0.0 < head < 1.0) else None
                if frac is not None and self.verbose:
                    kept = max(1, int(round((ef - sf + 1) * frac)))
                    print(f"[Classify/{self.tag}] last segment {sf}-{ef}: "
                          f"classifying from the first {kept} frame(s) "
                          f"({frac:.0%}) -- the tail is post-rake empty track")
                label, conf = self._classify_one(cap, sf, ef, head_fraction=frac)
                seg_class = self._label_to_class(label, conf)
                out.append(_MasterClassification(
                    segment_index=idx,
                    start_frame=sf,
                    end_frame=ef,
                    label=seg_class,
                    confidence=conf,
                ))
                if self.verbose:
                    print(f"[Classify/{self.tag}] segment {idx + 1}/"
                          f"{len(segments)} frames {sf}-{ef} -> {seg_class} "
                          f"(raw='{label}', conf={conf:.2f})")
        finally:
            cap.release()
        return out

    def _classify_one(
        self,
        cap: cv2.VideoCapture,
        start_frame: int,
        end_frame: int,
        *,
        head_fraction: Optional[float] = None,
    ) -> Tuple[str, float]:
        # Narrow to the head of the span BEFORE the usual 10% margin, so the
        # margin still trims the edges of the window we actually sample.
        if head_fraction is not None and 0.0 < head_fraction < 1.0:
            full = max(1, end_frame - start_frame + 1)
            end_frame = start_frame + max(0, int(round(full * head_fraction)) - 1)
        span = max(1, end_frame - start_frame + 1)
        margin = max(1, int(span * 0.1))
        safe_s = start_frame + margin
        safe_e = end_frame - margin
        if safe_e <= safe_s:
            sample_idxs = [start_frame + span // 2]
        elif self.num_samples == 1:
            sample_idxs = [safe_s + (safe_e - safe_s) // 2]
        else:
            step = (safe_e - safe_s) / max(self.num_samples - 1, 1)
            sample_idxs = [int(round(safe_s + i * step)) for i in range(self.num_samples)]

        labels: List[str] = []
        confs: List[float] = []
        for fi in sample_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                continue
            lbl, c = self.classify_frame(frame)
            labels.append(lbl)
            confs.append(c)

        if not labels:
            return "wagon", 0.0

        # Majority vote, deterministic tiebreak by alphabetical order
        from collections import Counter
        counts = Counter(labels)
        top = max(counts.items(), key=lambda kv: (kv[1], -ord(kv[0][0])))[0]
        kept_confs = [c for l, c in zip(labels, confs) if l == top]
        return top, float(np.mean(kept_confs)) if kept_confs else 0.0

    @staticmethod
    def _label_to_class(label: str, confidence: float = 1.0) -> str:
        lbl = (label or "").lower()
        if lbl in _ENGINE_LABELS:
            # NEVER assume the first wagon == ENGINE.  Promote to ENGINE only
            # when the model is confident; otherwise leave it UNCERTAIN
            # (UNKNOWN) so a phantom leading segment can be dropped rather than
            # emitted as a false engine.
            if confidence >= ENGINE_MIN_CONFIDENCE:
                return SegmentClass.ENGINE
            return SegmentClass.UNKNOWN
        if lbl in _BRAKEVAN_LABELS:
            return SegmentClass.BRAKE_VAN
        if lbl in _TRACK_LABELS:
            return SegmentClass.UNKNOWN
        return SegmentClass.WAGON


# =============================================================================
# CONVENIENCE: split a camera's frame range into segments using its gaps
# =============================================================================

def segments_from_gaps(
    gaps: List[GapEvent],
    total_frames: int,
) -> List[Tuple[int, int]]:
    """Convert a list of GapEvents into (start_frame, end_frame) segments.

    Gaps must come from one camera.  The function uses each gap's
    midpoint frame as the inter-wagon boundary, in temporal order.

    Returns a list of [start, end] inclusive segment ranges covering
    [0, total_frames - 1].
    """
    if total_frames <= 0:
        return []
    if not gaps:
        return [(0, total_frames - 1)]

    boundaries: List[int] = []
    for g in sorted(gaps, key=lambda x: x.center_frame):
        b = int(round(g.center_frame))
        b = max(0, min(total_frames - 1, b))
        boundaries.append(b)

    segments: List[Tuple[int, int]] = []
    prev = 0
    for b in boundaries:
        if b <= prev:
            continue
        segments.append((prev, b - 1))
        prev = b
    if prev <= total_frames - 1:
        segments.append((prev, total_frames - 1))
    return segments
