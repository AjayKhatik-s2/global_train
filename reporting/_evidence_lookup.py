"""Resolve evidence-snapshot + cache-frame paths for the reporting layer.

The legacy door report rendered a 4-frame quartile wagon overview
(12.5 / 37.5 / 62.5 / 87.5%); the damage report rendered a single
midpoint snapshot for loaded / no-damage / non-wagon pages.  Both
sourced frames from the per-camera raw videos via cv2.VideoCapture.

In v4 every wagon's per-camera frames are already on disk under
    wagon_cache/<gw_id>/<camera_folder_lower>/frame_NNNNNN.jpg
because the materializer extracts them in a single pass during Stage 2.
This module computes those paths so the report builders can read them
directly without touching any video file.

It also resolves evidence snapshot paths by feature (e.g.
    evidence/<gw_id>/door/left_best.jpg
) so the combined "Damaged Wagon Report" and the camera-wise reports all
share one helper.

Pure path resolution + JSON read.  No model loads, no decoder calls.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core import constants as C


# -----------------------------------------------------------------------------
# Per-camera local frame range for a given GlobalWagon
# -----------------------------------------------------------------------------

def wagon_local_frames(
    wagon_start_time: float, wagon_end_time: float,
    local_fps: float, local_total_frames: int,
) -> Tuple[int, int]:
    """Same arithmetic as wagon_count/video_segmenter.py:70.

    Returns (start_frame, end_frame) inclusive, clipped into the camera.
    """
    if local_fps <= 0 or local_total_frames <= 0:
        return (0, -1)
    sf = int(round(wagon_start_time * local_fps))
    ef = int(round(wagon_end_time * local_fps)) - 1
    sf = max(0, min(local_total_frames - 1, sf))
    ef = max(0, min(local_total_frames - 1, ef))
    if ef < sf:
        ef = sf
    return (sf, ef)


# -----------------------------------------------------------------------------
# Per-camera tracking JSON read (fps + total_frames per camera)
# -----------------------------------------------------------------------------

def load_per_camera_meta(
    per_camera_tracking_path: Optional[str],
) -> Dict[str, Dict[str, Any]]:
    """Return {camera_id -> {fps, total_frames, width, height}}.  Empty if
    the file is missing / unreadable.
    """
    if not per_camera_tracking_path or not os.path.isfile(per_camera_tracking_path):
        return {}
    try:
        with open(per_camera_tracking_path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except Exception:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for cam, meta in doc.items():
        if isinstance(meta, dict):
            out[cam] = {
                "fps":          float(meta.get("fps") or 0.0),
                "total_frames": int(meta.get("total_frames") or 0),
                "width":        int(meta.get("width") or 0),
                "height":       int(meta.get("height") or 0),
                "gaps":         list(meta.get("gaps") or []),
            }
    return out


# -----------------------------------------------------------------------------
# Cache frame paths
# -----------------------------------------------------------------------------

def _cache_frame_path(
    cache_root: str, gw_id: str, camera_id: str, frame_idx: int,
) -> str:
    folder = C.CAMERA_FOLDER.get(camera_id, camera_id.lower())
    return os.path.join(
        cache_root, gw_id, folder, f"frame_{int(frame_idx):06d}.jpg",
    )


def quartile_cache_paths(
    *,
    cache_root: Optional[str],
    gw_id: str,
    camera_id: str,
    wagon_start_time: float,
    wagon_end_time: float,
    local_fps: float,
    local_total_frames: int,
) -> List[Optional[str]]:
    """Return four paths (12.5/37.5/62.5/87.5%) into the wagon_cache for
    one (wagon, camera) pair.  Entries that don't exist on disk are
    returned as None so the caller can render placeholders.
    """
    if not cache_root:
        return [None, None, None, None]
    sf, ef = wagon_local_frames(
        wagon_start_time, wagon_end_time, local_fps, local_total_frames,
    )
    if ef <= sf:
        return [None, None, None, None]
    span = ef - sf
    fractions = (0.125, 0.375, 0.625, 0.875)
    paths: List[Optional[str]] = []
    for frac in fractions:
        idx = sf + int(round(frac * span))
        idx = max(sf, min(ef, idx))
        p = _cache_frame_path(cache_root, gw_id, camera_id, idx)
        paths.append(p if os.path.isfile(p) else None)
    return paths


def midpoint_cache_path(
    *,
    cache_root: Optional[str],
    gw_id: str,
    camera_id: str,
    wagon_start_time: float,
    wagon_end_time: float,
    local_fps: float,
    local_total_frames: int,
) -> Optional[str]:
    """Return the single mid-wagon cache frame path.  Mirrors the legacy
    damage report's `_extract_wagon_snapshot` (legacy :952-1004) which
    used `(start + end) // 2`."""
    if not cache_root:
        return None
    sf, ef = wagon_local_frames(
        wagon_start_time, wagon_end_time, local_fps, local_total_frames,
    )
    if ef <= sf:
        return None
    mid = (sf + ef) // 2
    p = _cache_frame_path(cache_root, gw_id, camera_id, mid)
    return p if os.path.isfile(p) else None


# -----------------------------------------------------------------------------
# Wagon-CENTRE overview frame  (combined report's 4-camera wagon pages)
# -----------------------------------------------------------------------------
#
# This is the GENERAL WAGON OVERVIEW selector -- deliberately distinct from the
# feature-specific "best snapshot" selectors (`evidence_snapshot`,
# `damage_track_snapshots`, `core.frame_quality.snapshot_score`).  Those pick the
# frame that best shows a DETECTED DEFECT; this one picks the frame that best
# shows the MIDDLE OF THE WAGON, with no reference to any model output.  Both
# stay available side by side; neither replaces the other.
#
# Source of truth for the mapping is Stage 2's own materialization: every frame
# the materializer wrote lives at
#     wagon_cache/<gw_id>/<camera_folder>/frame_<ORIGINAL_LOCAL_FRAME>.jpg
# so the filename IS the original source-video frame number for that camera and
# the directory listing IS that camera's mapped interval for that Global Wagon.
# `wagon_local_frames()` above (identical arithmetic to
# materializer/wagon_cache_builder._wagon_local_range and
# wagon_count/video_segmenter) supplies the authoritative interval whenever
# per-camera fps/total_frames are known, and is used to place the centre target.

# A truncated/0-byte JPEG is smaller than this; used as a cheap first gate
# before the (more expensive) header decode.
_MIN_FRAME_BYTES = 512

# Hard cap on decode probes per (wagon, camera) so a wholly corrupt cache
# directory can never make report generation quadratic.
_MAX_DECODE_PROBES = 32

# Selection outcome sentinels (carried into the report + the combined JSON).
OVERVIEW_OK               = "OK"
OVERVIEW_NO_CACHE_ROOT    = "NO_CACHE_ROOT"
OVERVIEW_NO_FRAMES        = "NO_FRAMES"
OVERVIEW_NO_READABLE      = "NO_READABLE_FRAME"
OVERVIEW_OUTSIDE_VIDEO    = "OUTSIDE_VIDEO_RANGE"


def list_cache_frame_indices(
    cache_root: Optional[str], gw_id: str, camera_id: str,
) -> List[int]:
    """Sorted ORIGINAL local frame numbers materialized for one (wagon, camera).

    Parsed straight out of the `frame_NNNNNN.jpg` names Stage 2 wrote, so the
    numbering stays tied to the source video and every returned index is
    traceable back to it.  Empty list when the camera contributed nothing to
    this wagon (missing feed, empty/invalid mapped interval, or a cache that
    was pruned).
    """
    if not cache_root:
        return []
    folder = C.CAMERA_FOLDER.get(camera_id, camera_id.lower())
    d = os.path.join(cache_root, gw_id, folder)
    try:
        names = os.listdir(d)
    except OSError:
        return []
    out: List[int] = []
    for name in names:
        if not (name.startswith("frame_") and name.endswith(".jpg")):
            continue
        try:
            out.append(int(name[len("frame_"):-len(".jpg")]))
        except ValueError:
            continue
    out.sort()
    return out


def _frame_readable(path: str) -> bool:
    """True when the JPEG can actually be embedded in the PDF.

    Uses reportlab's own `ImageReader` (header-only size probe) so the check
    agrees with what the report will do at build time.  Falls back to a size
    check if reportlab/PIL cannot be imported, so this module stays usable in a
    pure path-resolution context.
    """
    try:
        if os.path.getsize(path) < _MIN_FRAME_BYTES:
            return False
    except OSError:
        return False
    try:
        from reportlab.lib.utils import ImageReader
    except Exception:
        return True
    try:
        w, h = ImageReader(path).getSize()
    except Exception:
        return False
    return bool(w and h)


def _in_any_gap(frame_idx: int, gaps: Sequence[Dict[str, Any]]) -> bool:
    """True when a local frame falls inside a Stage-1 detected gap for this
    camera.  Read-only use of the existing per_camera_tracking.json gap list --
    no gap detection is performed or altered here."""
    for g in gaps or ():
        if not isinstance(g, dict):
            continue
        try:
            gs = int(g["start_frame"])
            ge = int(g["end_frame"])
        except (KeyError, TypeError, ValueError):
            continue
        if gs <= frame_idx <= ge:
            return True
    return False


def center_cache_frame(
    *,
    cache_root: Optional[str],
    gw_id: str,
    camera_id: str,
    wagon_start_time: float,
    wagon_end_time: float,
    local_fps: float = 0.0,
    local_total_frames: int = 0,
    gaps: Optional[Sequence[Dict[str, Any]]] = None,
    avoid_boundary: bool = True,
    max_probes: int = _MAX_DECODE_PROBES,
) -> Dict[str, Any]:
    """Pick ONE wagon-centre overview frame for a (Global Wagon, camera) pair.

    Deterministic, model-free, and scoped strictly to this wagon's own mapped
    interval in THIS camera's local frame space -- a frame belonging to GW_n+1
    can never be returned for GW_n because it is not in GW_n's cache directory
    and, when fps is known, not inside GW_n's `wagon_local_frames()` interval.

    Selection order:
      1. Candidate pool = frames Stage 2 materialized for (gw, camera).
      2. Intersect with the mapped interval from `wagon_local_frames()` when
         fps/total_frames are known (skipped only if that would empty the pool,
         which would mean cache and mapping disagree -- reported via `clipped`).
      3. Drop frames inside a Stage-1 gap for this camera (skipped if it would
         empty the pool).
      4. Target = temporal centre of the mapped interval, `(start + end) // 2`
         -- the same midpoint convention as `midpoint_cache_path`.
      5. Prefer the interior (drop the first/last frame of the pool) when the
         pool is long enough, so a boundary frame is never chosen while an
         interior one exists.
      6. Walk candidates by (|frame - target|, frame) and return the first that
         decodes.  Ties break to the LOWER frame number, so repeated runs on the
         same inputs always select the same frame.

    Returns a dict -- never raises, never returns another wagon's frame:
        {status, path, frame, start_frame, end_frame, target_frame,
         candidates, clipped, gap_filtered}
    `status != OK` means the caller must render a NO FRAME AVAILABLE placeholder
    for this camera and carry on with the other three.
    """
    res: Dict[str, Any] = {
        "status": OVERVIEW_NO_CACHE_ROOT,
        "path": None,
        "frame": None,
        "start_frame": None,
        "end_frame": None,
        "target_frame": None,
        "candidates": 0,
        "clipped": False,
        "gap_filtered": False,
    }
    if not cache_root:
        return res

    # --- mapped interval (authoritative when per-camera meta is available) ---
    sf: Optional[int] = None
    ef: Optional[int] = None
    if local_fps > 0 and local_total_frames > 0:
        # UNCLAMPED first.  A camera whose clip was cut short never saw the tail
        # of the rake at all.  `wagon_local_frames` (and Stage 2, which shares
        # the arithmetic) CLAMPS such a wagon to the final frame, and the
        # materializer's last-write-wins then hands that one real frame to
        # whichever past-the-end wagon is numbered last -- so GW_49 would be
        # illustrated with a frame showing some earlier wagon.  Refuse it: a
        # wagon that begins at or after this camera's last frame has no frame
        # here, and the page must say so rather than borrow another wagon's.
        raw_sf = int(round(wagon_start_time * local_fps))
        raw_ef = int(round(wagon_end_time * local_fps)) - 1
        if raw_sf >= local_total_frames or raw_ef < 0:
            res["status"] = OVERVIEW_OUTSIDE_VIDEO
            res["start_frame"], res["end_frame"] = raw_sf, raw_ef
            return res
        _sf, _ef = wagon_local_frames(
            wagon_start_time, wagon_end_time, local_fps, local_total_frames,
        )
        if _ef >= _sf:
            sf, ef = _sf, _ef
    res["start_frame"], res["end_frame"] = sf, ef

    indices = list_cache_frame_indices(cache_root, gw_id, camera_id)
    if not indices:
        res["status"] = OVERVIEW_NO_FRAMES
        return res

    pool = indices
    if sf is not None:
        inside = [i for i in pool if sf <= i <= ef]
        if inside:
            pool = inside
            res["clipped"] = len(inside) != len(indices)
        # else: cache and mapping disagree -- keep the materialized frames (they
        # are this wagon's own, by construction) and fall back to their span.
    if sf is None or not (sf <= pool[0] and pool[-1] <= ef):
        sf, ef = pool[0], pool[-1]
        res["start_frame"], res["end_frame"] = sf, ef

    if gaps:
        ungapped = [i for i in pool if not _in_any_gap(i, gaps)]
        if ungapped and len(ungapped) != len(pool):
            pool = ungapped
            res["gap_filtered"] = True

    target = (sf + ef) // 2
    res["target_frame"] = target
    res["candidates"] = len(pool)

    # Interior-first: never hand back an entry/exit boundary frame while an
    # interior frame is usable.  (Feature inference's stable-interior trim is a
    # separate, inference-only concept and is intentionally not reused here --
    # reports use the full wagon span.)
    interior = pool[1:-1] if (avoid_boundary and len(pool) >= 5) else []

    seen = set()
    for cand in (interior, pool):
        if not cand:
            continue
        for idx in sorted(cand, key=lambda i: (abs(i - target), i))[:max_probes]:
            if idx in seen:
                continue
            seen.add(idx)
            p = _cache_frame_path(cache_root, gw_id, camera_id, idx)
            if os.path.isfile(p) and _frame_readable(p):
                res["status"] = OVERVIEW_OK
                res["path"] = p
                res["frame"] = idx
                return res

    res["status"] = OVERVIEW_NO_READABLE
    return res


# -----------------------------------------------------------------------------
# Evidence snapshot path resolution
# -----------------------------------------------------------------------------

def evidence_snapshot(
    evidence_root: Optional[str], gw_id: str, feature: str, slot: str,
    camera_id: Optional[str] = None,
) -> Optional[str]:
    """Resolve a single evidence file path; returns None if it doesn't exist.

    With `camera_id`, resolves the camera-scoped path
    `evidence/<gw>/<feature>/<CAMERA>/<slot>.jpg` first (so a per-camera report
    only ever shows THAT camera's evidence), then falls back to the legacy flat
    `evidence/<gw>/<feature>/<slot>.jpg` for old batches.

    `slot` examples:
        door:   left_best | left_crop | right_best | right_crop
        damage: track_1 | track_1_crop | ...
        ocr:    best_frame | number_crop
        load:   best_frame
    """
    if not evidence_root:
        return None
    if camera_id:
        cp = os.path.join(evidence_root, gw_id, feature, camera_id, f"{slot}.jpg")
        if os.path.isfile(cp):
            return cp
    p = os.path.join(evidence_root, gw_id, feature, f"{slot}.jpg")
    return p if os.path.isfile(p) else None


def evidence_metadata(
    evidence_root: Optional[str], gw_id: str, feature: str,
    camera_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Read metadata.json for a feature; camera-scoped first, then legacy flat."""
    if not evidence_root:
        return {}
    candidates = []
    if camera_id:
        candidates.append(os.path.join(evidence_root, gw_id, feature, camera_id,
                                       "metadata.json"))
    candidates.append(os.path.join(evidence_root, gw_id, feature, "metadata.json"))
    for p in candidates:
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f) or {}
            except Exception:
                return {}
    return {}


def damage_track_snapshots(
    evidence_root: Optional[str], gw_id: str, max_tracks: int = 3,
    camera_id: Optional[str] = None,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Resolve up to `max_tracks` damage track snapshots for one wagon.

    With `camera_id`, only that camera's damage evidence is used, so one top
    camera never shows the other's snapshots.  Sorted by `best_confidence` desc.
    """
    meta = evidence_metadata(evidence_root, gw_id, "damage", camera_id=camera_id)
    tracks = meta.get("tracks") or []
    out: List[Tuple[str, Dict[str, Any]]] = []
    for tr in tracks:
        if not isinstance(tr, dict):
            continue
        idx = tr.get("track_idx")
        if not idx:
            continue
        p = evidence_snapshot(evidence_root, gw_id, "damage", f"track_{int(idx)}",
                              camera_id=camera_id)
        if not p:
            continue
        out.append((p, tr))
    out.sort(key=lambda x: float(x[1].get("best_confidence") or 0.0), reverse=True)
    return out[:max_tracks]


# -----------------------------------------------------------------------------
# Per-wagon raw feature JSON read (for confidences not folded into UWS)
# -----------------------------------------------------------------------------

def read_wagon_feature_json(
    wagon_states_root: Optional[str], feature: str, gw_id: str,
    camera_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Read a per-wagon feature JSON.  With `camera_id`, prefers the camera-
    scoped `wagon_states/<feature>/<CAMERA>/<gw>.json`, then falls back to the
    legacy flat `wagon_states/<feature>/<gw>.json` for old batches."""
    if not wagon_states_root:
        return {}
    candidates = []
    if camera_id:
        candidates.append(os.path.join(wagon_states_root, feature, camera_id,
                                       f"{gw_id}.json"))
    candidates.append(os.path.join(wagon_states_root, feature, f"{gw_id}.json"))
    for p in candidates:
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f) or {}
            except Exception:
                return {}
    return {}
