"""Shared low-level primitives for every feature processor.

Each processor (door, load, damage, ocr) is a small ~150-line module
that uses only these helpers + its own aggregation rule:

    1. `load_yolo(path)` -- lazy + cached YOLO loader (one load per
       process per .pt file).
    2. `iter_wagon_frames(...)` -- iterate frame_NNNNNN.jpg files in a
       wagon-cache folder.
    3. `run_detection(...)` / `run_classification(...)` -- one-line
       calls to ultralytics that return clean dicts.
    4. `majority_vote(...)` / `confidence_mean(...)` -- deterministic
       aggregation helpers.
    5. `write_per_wagon_json(...)` -- consistent JSON shape with status
       sentinels (`OK` / `NO_FRAMES` / `FAILED`).
"""

from __future__ import annotations

import glob
import json
import os
import tempfile
import threading
import time
from collections import Counter
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

from core import constants as C
from core import config as CFG


# -----------------------------------------------------------------------------
# Inference device -- resolved once for the whole process (CUDA if available,
# else CPU; WAGONEYE_DEVICE overrides).  Before the EC2 migration nothing
# branched on device: ultralytics guessed and half-precision was hardcoded ON,
# which is a footgun on a CPU-only host.  DEVICE + HALF make the choice explicit
# while preserving the exact GPU behaviour (DEVICE='cuda' => HALF=True).
# -----------------------------------------------------------------------------

DEVICE = CFG.resolve_device()
HALF = CFG.use_half_precision(DEVICE)


# -----------------------------------------------------------------------------
# FP16 precision selection -- migrated OFF the deprecated ultralytics `half=`
# predict argument.  Newer ultralytics deprecates `half` in favour of `quantize`
# ("'half' is deprecated ... Use 'quantize' instead").
#
# Root-cause fix (no warning suppression):
#   * CPU  -> pass NOTHING (FP32 is the default; `half` was always False on CPU,
#            so omitting it is behaviourally identical AND never triggers the
#            deprecation warning).
#   * CUDA -> use the CURRENT supported mechanism: `quantize` when the installed
#            ultralytics exposes it, else the legacy `half=True` (older builds
#            where `half` is still the supported key, e.g. 8.4.53).
# `fp16=False` forces FP32 on every device (used by damage, which always ran
# FP32 -- it never passed `half`).
# -----------------------------------------------------------------------------

_CUDA_FP16_KWARG: Optional[Dict[str, Any]] = None


def _cuda_fp16_kwarg() -> Dict[str, Any]:
    """The FP16 predict kwarg for CUDA on the installed ultralytics (cached)."""
    global _CUDA_FP16_KWARG
    if _CUDA_FP16_KWARG is None:
        kw: Dict[str, Any] = {"half": True}          # legacy supported key
        try:
            from ultralytics.cfg import DEFAULT_CFG_DICT
            if "quantize" in DEFAULT_CFG_DICT:        # current supported key
                kw = {"quantize": "fp16"}
        except Exception:
            pass
        _CUDA_FP16_KWARG = kw
    return dict(_CUDA_FP16_KWARG)


def precision_kwargs(device: Optional[str] = None, fp16: bool = True) -> Dict[str, Any]:
    """Return the precision kwargs to splat into a YOLO predict call.

    CPU (or fp16=False) -> {} (FP32, no deprecated arg).  CUDA + fp16 -> the
    installed version's supported FP16 mechanism.
    """
    dev = device if device is not None else DEVICE
    if not fp16 or dev != "cuda":
        return {}
    return _cuda_fp16_kwarg()


# -----------------------------------------------------------------------------
# CPU throughput knobs (all env-overridable; safe -- they change SPEED, not the
# business logic).  On CPU, batching YOLO across frames amortizes per-call
# Python/pre/post-processing overhead; using all cores speeds each inference.
#
#   WAGONEYE_INFER_BATCH   frames per YOLO batch (default 24 -- benchmarked
#                          fastest on CPU for door + damage; 32 was SLOWER for
#                          both).  Set 1 for the exact pre-batch, bit-identical
#                          single-frame path.
#   WAGONEYE_INFER_BATCH_<CAMERA>
#                          per-camera override, e.g.
#                          WAGONEYE_INFER_BATCH_RIGHT_UP_TOP=16.  Restores the
#                          per-camera tuning V4 has as `detection_batch_size` in
#                          configs/cameras/*.yaml.  Falls back to
#                          WAGONEYE_INFER_BATCH when unset.
#   WAGONEYE_TORCH_THREADS intra-op CPU threads (default: all cores).
#   WAGONEYE_RAW_DETECTIONS bypass post-inference FILTERS (benchmark-only; see
#                          each processor).  Default False = production behaviour.
# -----------------------------------------------------------------------------

def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


INFER_BATCH = max(1, _env_int("WAGONEYE_INFER_BATCH", 24))


def infer_batch_for(camera_id: Optional[str] = None) -> int:
    """Frames per YOLO batch for `camera_id`.

    ``WAGONEYE_INFER_BATCH_<CAMERA>`` overrides the global
    ``WAGONEYE_INFER_BATCH`` for one camera -- the equivalent of V4's per-camera
    ``detection_batch_size`` (configs/cameras/*.yaml).  Read at CALL time, not
    import time, so it is settable per run without reimporting.

    Batch size is a pure throughput knob: outputs are unchanged except for
    <=1e-3 px box-coordinate jitter from BLAS reduction order (class + confidence
    are identical), and ``1`` takes the exact single-frame path.
    """
    if camera_id:
        raw = os.getenv(f"WAGONEYE_INFER_BATCH_{camera_id}")
        if raw:
            try:
                return max(1, int(raw))
            except ValueError:
                pass
    return INFER_BATCH
RAW_DETECTIONS = os.getenv("WAGONEYE_RAW_DETECTIONS", "").strip().lower() in (
    "1", "true", "yes", "on")

# Use all CPU cores by default (pure speed knob; deterministic outputs).
try:
    import torch as _torch
    _threads = _env_int("WAGONEYE_TORCH_THREADS", os.cpu_count() or 0)
    if _threads > 0:
        _torch.set_num_threads(_threads)
except Exception:
    pass


# -----------------------------------------------------------------------------
# YOLO loader cache
# -----------------------------------------------------------------------------

_MODEL_CACHE: Dict[str, Any] = {}
_MODEL_LOCK = threading.Lock()


def load_yolo(model_path: str):
    """Cached YOLO loader.  Returns None if the file is missing or unusable --
    the caller is expected to short-circuit to NO_DATA in that case.

    Patches `torch.load` once on first call so .pt weights load on
    torch >= 2.6 (the same monkey-patch used by wagon_count).
    """
    if not model_path or not os.path.isfile(model_path):
        return None
    # A clone made without git-lfs on PATH leaves a ~130-byte TEXT pointer where
    # the weights should be.  It passes isfile(), so without this check torch
    # would fail deep in deserialization with an unhelpful error.  Startup
    # model_sync is the primary gate; this is the second line of defence.
    try:
        from core.model_sync import is_lfs_pointer, lfs_pointer_size
        if is_lfs_pointer(model_path):
            size = lfs_pointer_size(model_path)
            print(f"[MODEL] {model_path} is an UNPULLED GIT-LFS POINTER"
                  + (f" (expects {size/1e6:.0f} MB)" if size else "")
                  + " -- run `git lfs pull` (and check git-lfs is on PATH for "
                    "this user, not just an interactive shell)")
            return None
    except Exception:
        pass

    abspath = os.path.abspath(model_path)
    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(abspath)
        if cached is not None:
            return cached

        # torch.load shim for newer torch versions
        import torch
        _orig_load = torch.load
        def _patched(*a, **kw):
            kw.setdefault("weights_only", False)
            return _orig_load(*a, **kw)
        torch.load = _patched

        from ultralytics import YOLO
        model = YOLO(abspath)
        _MODEL_CACHE[abspath] = model
        return model


def model_class_names(model) -> Dict[int, str]:
    """Return YOLO's class-id -> name mapping for the loaded model."""
    if model is None:
        return {}
    return dict(getattr(model, "names", {}) or {})


# -----------------------------------------------------------------------------
# Frame iteration
# -----------------------------------------------------------------------------

def wagon_camera_dir(cache_root: str, gw_id: str, camera_id: str) -> str:
    """Path to wagon_cache/<gw>/<camera_folder>/."""
    return os.path.join(cache_root, gw_id, C.CAMERA_FOLDER[camera_id])


# -----------------------------------------------------------------------------
# Stable-interior trimming
# -----------------------------------------------------------------------------
# A wagon's first/last frames are the noisiest part of its pass (entering /
# leaving view, motion blur, partial occlusion at the gap).  For FEATURE
# INFERENCE we drop a margin from each end and use only the stable interior.
# The full span is still used for processed-video rendering and report
# continuity (those read the raw video / evidence, not this iterator).
#
# The trim is adaptive: 5% of the span, clamped to [3, 12] frames per side,
# so it scales with variable wagon durations (speed / visibility / geometry).

_STABLE_TRIM_FRACTION = 0.05
_STABLE_TRIM_MIN = 3
_STABLE_TRIM_MAX = 12


def stable_trim_count(span_length: int) -> int:
    """Frames to trim from EACH end of a wagon span for stable inference.

        trim_k = int(span_length * 0.05)   clamped to [3, 12]

    Returns 0 (no trim) when the span is too short to leave a usable interior
    (2*trim_k would consume the whole span) so short wagons still get inference
    instead of dropping to NO_DATA.
    """
    if span_length <= 0:
        return 0
    trim_k = int(span_length * _STABLE_TRIM_FRACTION)
    trim_k = max(_STABLE_TRIM_MIN, min(_STABLE_TRIM_MAX, trim_k))
    if span_length <= 2 * trim_k:
        return 0
    return trim_k


def stable_interior(paths: List[str]) -> List[str]:
    """Symmetric stable interior of a sorted frame-path list (see
    stable_trim_count).  Returns the list unchanged when the span is too short
    to trim."""
    k = stable_trim_count(len(paths))
    if k <= 0:
        return paths
    return paths[k:len(paths) - k]


def list_wagon_frames(
    cache_root: str, gw_id: str, camera_id: str,
    *, trim_stable: bool = False,
) -> List[str]:
    """Return sorted JPEG paths for one (gw, camera) pair.

    When ``trim_stable`` is True, returns only the adaptive stable interior
    (5% per side, clamped [3, 12]) used for feature inference.
    """
    d = wagon_camera_dir(cache_root, gw_id, camera_id)
    if not os.path.isdir(d):
        return []
    paths = glob.glob(os.path.join(d, "frame_*.jpg"))
    paths.sort()
    if trim_stable:
        paths = stable_interior(paths)
    return paths


def iter_wagon_frames(
    cache_root: str, gw_id: str, camera_id: str,
    *, every_nth: int = 1, max_frames: Optional[int] = None,
    trim_stable: bool = False,
) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield (frame_idx, BGR ndarray) in monotonic order.

    Args:
        every_nth: yield 1 of every N frames (default 1 = all).
        max_frames: hard cap; useful when a wagon has hundreds of
            frames and we only need a sample.
        trim_stable: iterate only the adaptive stable interior (drop the
            noisy first/last 5% per side, clamped [3, 12]) -- used for
            feature inference, NOT for rendering.
    """
    paths = list_wagon_frames(cache_root, gw_id, camera_id, trim_stable=trim_stable)
    if every_nth > 1:
        paths = paths[::every_nth]
    if max_frames is not None and len(paths) > max_frames:
        # evenly-spaced subsample so first / last / middle are covered
        idx = np.linspace(0, len(paths) - 1, max_frames).round().astype(int)
        paths = [paths[i] for i in idx]
    for p in paths:
        frame = cv2.imread(p)
        if frame is None:
            continue
        # parse frame_NNNNNN.jpg
        try:
            fi = int(os.path.basename(p).split("_")[1].split(".")[0])
        except (IndexError, ValueError):
            fi = -1
        yield fi, frame


# -----------------------------------------------------------------------------
# YOLO calls
# -----------------------------------------------------------------------------

def run_detection(
    model, frame: np.ndarray,
    *, confidence: float = 0.4, fp16: Optional[bool] = None,
    device: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run a YOLO detection model on a frame; return clean dicts.

    Each dict: {class_id, class_name, confidence, bbox: [x1, y1, x2, y2]}.

    device/fp16 default to the process-resolved DEVICE/HALF (CUDA => FP16,
    CPU => FP32) -- callers may override but the defaults preserve the
    pre-migration GPU behaviour exactly.
    """
    if model is None:
        return []
    dev = device if device is not None else DEVICE
    _fp16 = HALF if fp16 is None else bool(fp16)
    res = model(frame, verbose=False, device=dev, **precision_kwargs(dev, _fp16))[0]
    if res.boxes is None or len(res.boxes) == 0:
        return []

    boxes = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    clss  = res.boxes.cls.cpu().numpy().astype(int)
    names = getattr(model, "names", {}) or {}

    out: List[Dict[str, Any]] = []
    for bbox, conf, cls_id in zip(boxes, confs, clss):
        if float(conf) < confidence:
            continue
        out.append({
            "class_id": int(cls_id),
            "class_name": str(names.get(int(cls_id), "unknown")).lower(),
            "confidence": float(conf),
            "bbox": [float(bbox[0]), float(bbox[1]),
                     float(bbox[2]), float(bbox[3])],
        })
    return out


def iter_wagon_detections(
    model, cache_root: str, gw_id: str, camera_id: str,
    *, batch: Optional[int] = None, trim_stable: bool = True,
    fp16: Optional[bool] = None, device: Optional[str] = None,
) -> Iterator[Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield (frame_idx, BGR frame, boxes_xyxy, confs, class_ids) in monotonic
    frame order, running YOLO in BATCHES to amortize CPU per-call overhead.

    This is a drop-in replacement for the per-frame
    ``for fi,frame in iter_wagon_frames(...): res = model(frame)...`` pattern:
    the caller still sees one (frame, detections) tuple per frame IN ORDER, so
    any stateful/sequential downstream (illumination, geometric prior, Kalman
    tracker, evidence bucketing) is unchanged.  Only the raw detector call is
    batched.

    batch=1 uses the exact single-frame call ``model(frame)[0]`` -- bit-identical
    to the pre-batch pipeline (the CPU-batched path can differ from single-frame
    by <=1e-3 px in box coords due to BLAS reduction order; conf/class match).

    Empty arrays are yielded for frames with no detections so the caller's
    frame loop still runs (tracker predict-only step, event frames, etc.).
    """
    if model is None:
        return
    # Per-camera override wins, then the explicit arg, then the global default.
    b = infer_batch_for(camera_id) if batch is None else max(1, int(batch))
    dev = device if device is not None else DEVICE
    _fp16 = HALF if fp16 is None else bool(fp16)
    prec = precision_kwargs(dev, _fp16)

    paths = list_wagon_frames(cache_root, gw_id, camera_id, trim_stable=trim_stable)
    if not paths:
        return

    def _fi(p: str) -> int:
        try:
            return int(os.path.basename(p).split("_")[1].split(".")[0])
        except (IndexError, ValueError):
            return -1

    def _extract(res):
        if res.boxes is None or len(res.boxes) == 0:
            z = np.empty((0, 4), dtype=np.float32)
            return z, np.empty((0,), np.float32), np.empty((0,), np.int64)
        return (res.boxes.xyxy.cpu().numpy(),
                res.boxes.conf.cpu().numpy(),
                res.boxes.cls.cpu().numpy().astype(int))

    for start in range(0, len(paths), b):
        chunk_paths = paths[start:start + b]
        frames = [cv2.imread(p) for p in chunk_paths]
        keep = [(p, f) for p, f in zip(chunk_paths, frames) if f is not None]
        if not keep:
            continue
        kp, kf = [k[0] for k in keep], [k[1] for k in keep]
        if len(kf) == 1:
            results = [model(kf[0], verbose=False, device=dev, **prec)[0]]
        else:
            results = model(kf, verbose=False, device=dev, **prec)
        for p, f, res in zip(kp, kf, results):
            boxes, confs, clss = _extract(res)
            yield _fi(p), f, boxes, confs, clss


def run_classification(model, frame: np.ndarray,
                       *, device: Optional[str] = None) -> Tuple[str, float]:
    """Run a YOLO classification model. Returns (top1_class_name, conf).

    device defaults to the process-resolved DEVICE (CUDA if available, else
    CPU) so behaviour no longer depends on ultralytics' internal guess.
    """
    if model is None:
        return "", 0.0
    dev = device if device is not None else DEVICE
    res = model(frame, verbose=False, device=dev)[0]
    if getattr(res, "probs", None) is None:
        # Some "classification" models still emit boxes; pick the
        # highest-conf detection's class as a fallback.
        if res.boxes is not None and len(res.boxes) > 0:
            confs = res.boxes.conf.cpu().numpy()
            clss  = res.boxes.cls.cpu().numpy().astype(int)
            i = int(np.argmax(confs))
            names = getattr(model, "names", {}) or {}
            return str(names.get(int(clss[i]), "unknown")).lower(), float(confs[i])
        return "", 0.0
    top1 = int(res.probs.top1)
    conf = float(res.probs.top1conf)
    names = getattr(model, "names", {}) or {}
    return str(names.get(top1, "unknown")).lower(), conf


def crop_bbox(frame: np.ndarray, bbox: List[float], pad: int = 0) -> Optional[np.ndarray]:
    """Crop a frame to a bbox with optional pixel padding."""
    if frame is None or bbox is None or len(bbox) != 4:
        return None
    h, w = frame.shape[:2]
    x1 = max(0, int(bbox[0]) - pad)
    y1 = max(0, int(bbox[1]) - pad)
    x2 = min(w, int(bbox[2]) + pad)
    y2 = min(h, int(bbox[3]) + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


# -----------------------------------------------------------------------------
# Aggregation helpers
# -----------------------------------------------------------------------------

def majority_vote(
    items: List[Dict[str, Any]],
    *, key: str = "class_name", conf_key: str = "confidence",
) -> Tuple[Optional[str], float, int]:
    """Confidence-weighted majority vote.

    Returns (winning_value, mean_conf_of_winner, vote_count).
    Ties broken by:  more votes  >  higher mean conf  >  alphabetical.
    """
    if not items:
        return None, 0.0, 0
    votes: Counter = Counter()
    confs: Dict[str, List[float]] = {}
    for it in items:
        v = it.get(key)
        if v is None:
            continue
        v = str(v).lower()
        votes[v] += 1
        confs.setdefault(v, []).append(float(it.get(conf_key, 0.0) or 0.0))
    if not votes:
        return None, 0.0, 0

    def sort_key(item):
        cls, n = item
        mean_c = sum(confs[cls]) / len(confs[cls]) if confs[cls] else 0.0
        return (-n, -mean_c, cls)

    best, n = sorted(votes.items(), key=sort_key)[0]
    mean_c = sum(confs[best]) / len(confs[best]) if confs[best] else 0.0
    return best, mean_c, n


def fraction_with(items: List[Dict[str, Any]], predicate) -> float:
    if not items:
        return 0.0
    return sum(1 for it in items if predicate(it)) / float(len(items))


# -----------------------------------------------------------------------------
# Per-wagon JSON I/O
# -----------------------------------------------------------------------------

def write_per_wagon_json(
    output_dir: str, gw_id: str, payload: Dict[str, Any],
) -> str:
    """Atomically write <output_dir>/<gw_id>.json (temp file + os.replace)."""
    os.makedirs(output_dir, exist_ok=True)
    p = os.path.join(output_dir, f"{gw_id}.json")
    fd, tmp = tempfile.mkstemp(dir=output_dir, prefix=f".{gw_id}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
    return p


def feature_camera_dir(output_dir: str, feature: str, camera_id: str) -> str:
    """Per-camera feature output dir: <output_dir>/<feature>/<CAMERA>/.

    Each camera writes ONLY inside its own namespace so a late camera can never
    overwrite another camera's per-wagon feature files."""
    d = os.path.join(output_dir, feature, camera_id)
    os.makedirs(d, exist_ok=True)
    return d


def empty_payload(gw_id: str, feature: str, status: str, **extra) -> Dict[str, Any]:
    payload = {
        "global_id": gw_id,
        "feature": feature,
        "status": status,
        "frame_count": 0,
    }
    payload.update(extra)
    return payload


# -----------------------------------------------------------------------------
# Stage-3 per-feature timing + progress
# -----------------------------------------------------------------------------

class FeatureTimer:
    """Timing + progress instrumentation for one feature processor.

    Captures three things a Stage-3 operator wants to see:

      * ``model_load``  -- the one-time YOLO/OCR model-load wall-clock, recorded
                           via :meth:`set_model_load`.
      * named ``phases``-- cumulative time spent in each stage across all wagons,
                           recorded via the :meth:`phase` context manager (the
                           processors use ``"inference"`` and ``"evidence"``).
      * ``per_wagon``   -- per-wagon total wall time, recorded via the
                           :meth:`wagon` context manager, used both for the
                           progress line and the slowest-wagon summary.

    A progress line is emitted every ``log_every`` wagons (and on the last one),
    and a single summary line at the end -- both through ``logger`` (a
    ``logging.Logger``).  With no logger the timer still accumulates numbers but
    stays silent, so it is safe in tests / library use.
    """

    def __init__(self, name: str, *, logger=None,
                 total_units: int = 0, log_every: int = 25):
        self.name = name
        self.log = logger
        self.start = time.time()
        self.total_units = int(total_units or 0)
        self.log_every = max(1, int(log_every))
        self.model_load_s = 0.0
        self.phase_totals: Dict[str, float] = {}
        self.per_wagon: Dict[str, float] = {}
        self._done = 0

    def set_model_load(self, seconds: float) -> None:
        self.model_load_s = round(float(seconds), 3)

    @contextmanager
    def phase(self, name: str):
        """Accumulate wall-clock into the named phase bucket."""
        t0 = time.time()
        try:
            yield
        finally:
            self.phase_totals[name] = self.phase_totals.get(name, 0.0) + (time.time() - t0)

    def _record(self, gw_id: str, dt: float, camera_id: Optional[str] = None) -> None:
        """Record one finished wagon and emit a periodic progress line."""
        self.per_wagon[gw_id] = self.per_wagon.get(gw_id, 0.0) + dt
        self._done += 1
        if self.log is not None and (
            self._done % self.log_every == 0 or self._done == self.total_units):
            tot = f"/{self.total_units}" if self.total_units else ""
            cam = f" {camera_id}" if camera_id else ""
            self.log.info("[FEAT/%s] progress %d%s%s  last=%s %.2fs  elapsed=%.1fs",
                          self.name, self._done, tot, cam, gw_id, dt, self.total())

    @contextmanager
    def wagon(self, gw_id: str, camera_id: Optional[str] = None):
        """Time one wagon's processing and emit a periodic progress line."""
        t0 = time.time()
        try:
            yield
        finally:
            self._record(gw_id, time.time() - t0, camera_id)

    # -- back-compat call site: timer.stamp(gw_id, t0[, camera_id]) in a finally --
    def stamp(self, gw_id: str, t0: float, camera_id: Optional[str] = None) -> float:
        dt = time.time() - t0
        self._record(gw_id, dt, camera_id)
        return dt

    def total(self) -> float:
        return time.time() - self.start

    def log_summary(self, *, ok: Optional[int] = None,
                    total: Optional[int] = None) -> None:
        """Emit one structured DONE line with model-load, per-phase totals, and
        the three slowest wagons."""
        if self.log is None:
            return
        phases = "  ".join(f"{k}={v:.1f}s"
                           for k, v in sorted(self.phase_totals.items())) or "-"
        slow = sorted(self.per_wagon.items(), key=lambda x: x[1], reverse=True)[:3]
        slow_s = ", ".join(f"{g}:{s:.2f}s" for g, s in slow) or "-"
        okpart = f"  ok={ok}/{total}" if ok is not None else ""
        self.log.info(
            "[FEAT/%s] DONE total=%.1fs  model_load=%.2fs  %s%s  slowest=[%s]",
            self.name, self.total(), self.model_load_s, phases, okpart, slow_s)


def phase(timer: Optional[FeatureTimer], name: str):
    """Return ``timer.phase(name)`` when a timer is present, else a no-op context.

    Lets a per-wagon helper accept an optional timer and write
    ``with phase(timer, "inference"):`` without a None-check at every call site.
    """
    return timer.phase(name) if timer is not None else nullcontext()
