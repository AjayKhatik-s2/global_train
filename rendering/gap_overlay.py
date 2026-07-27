"""Stage-1 gap overlay for the processed videos (visualization only).

The final `<CAMERA>_processed.mp4` is produced by
``rendering/feature_overlay_renderer.py`` from the raw input video, and until now
carried only the feature annotations (door / damage / load / OCR).  This module
adds the **Stage-1 reconstruction gaps** on top of those, so one video shows gap
detection + wagon segmentation + wagon IDs + every feature at once.

It does NOT re-run or duplicate any reconstruction logic.  It replays what
Stage 1 already computed and persisted:

  * per-camera tracked gaps   -> ``per_camera_tracking.json`` ``[camera]["gaps"]``
      (each gap carries ``bbox_history`` + ``hit_frames`` -- the exact per-hit
      image-plane box trajectory the tracker recorded; see
      ``wagon_count/global_train_state.GapEvent``)
  * fused wagon boundaries    -> ``global_train_state.json`` wagon time windows,
      mapped to this camera's local frames (same arithmetic Stage 1 uses)

Only the FINAL accepted gaps are drawn: ``[camera]["gaps"]`` are the tracks that
survived Stage-1 temporal filtering (the rejected raw single-frame candidates in
``LocalCameraTracks.raw_frame_detections`` are never serialized, so they can
never be drawn here).  The interpolation + colours mirror
``wagon_count/video_segmenter.render_processed_video`` so the appearance matches
Stage 1's own debug video exactly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import cv2

# BGR colours -- chosen to stay DISTINCT from every feature colour so gaps are
# visually separable from door/damage/load/OCR in the same video.  The feature
# palette already uses yellow (0,255,255) for partial-closed doors and
# outer-wall damage, so the tracked-gap box uses CYAN instead (Stage 1's own
# debug video uses yellow, but there gaps are the only overlay).  Magenta is not
# used by any feature, so the fused-boundary flash keeps Stage 1's magenta.
GAP_COLOR = (255, 255, 0)        # cyan    -- tracked gap (interpolated bbox)
BOUNDARY_COLOR = (255, 0, 255)   # magenta -- fused wagon boundary flash
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def interp_gap_bbox(gap: Dict[str, Any], frame_idx: int) -> Optional[List[float]]:
    """Interpolate a tracked gap's bbox at ``frame_idx``.

    Faithful port of ``wagon_count/video_segmenter._interp_gap_bbox`` operating on
    the serialized gap dict (``bbox_history`` + ``hit_frames``): clamp before the
    first / after the last hit, linear interpolation between bracketing hits.
    Returns ``None`` outside the gap span or when no bbox history was recorded.
    """
    bh = gap.get("bbox_history")
    hf = gap.get("hit_frames")
    if not bh or not hf:
        return None
    sf, ef = gap.get("start_frame"), gap.get("end_frame")
    if sf is None or ef is None or frame_idx < sf or frame_idx > ef:
        return None
    if frame_idx <= hf[0]:
        return list(bh[0])
    if frame_idx >= hf[-1]:
        return list(bh[-1])
    for i in range(len(hf) - 1):
        f0, f1 = hf[i], hf[i + 1]
        if f0 <= frame_idx <= f1:
            if f1 == f0:
                return list(bh[i])
            t = (frame_idx - f0) / (f1 - f0)
            b0, b1 = bh[i], bh[i + 1]
            return [b0[j] + t * (b1[j] - b0[j]) for j in range(4)]
    return list(bh[-1])


def build_gap_frame_index(gaps: Sequence[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """frame_idx -> the tracked gap active at that frame (mirror of Stage 1's
    ``frame_to_active_gap``)."""
    idx: Dict[int, Dict[str, Any]] = {}
    for g in gaps or []:
        sf, ef = g.get("start_frame"), g.get("end_frame")
        if sf is None or ef is None:
            continue
        for f in range(int(sf), int(ef) + 1):
            idx[f] = g
    return idx


def draw_gap_overlays(
    frame,
    frame_idx: int,
    gap_by_frame: Dict[int, Dict[str, Any]],
    boundary_frames: Sequence[int],
) -> None:
    """Overlay the Stage-1 gaps for ONE frame, IN PLACE, on top of whatever
    feature annotations are already drawn.  Two elements, matching Stage 1:

      * magenta top/bottom flash + ``GW_BOUNDARY`` banner within +/-3 frames of a
        fused wagon boundary;
      * yellow interpolated bbox for the active tracked gap, labelled with its id,
        confidence and frame number.
    """
    h, w = frame.shape[:2]

    # magenta fused wagon-boundary flash (drawn for every camera)
    for b in boundary_frames:
        if abs(int(b) - frame_idx) <= 3:
            cv2.line(frame, (0, 0), (w, 0), BOUNDARY_COLOR, 4)
            cv2.line(frame, (0, h - 1), (w, h - 1), BOUNDARY_COLOR, 4)
            label = "GW_BOUNDARY"
            (tw, th), _ = cv2.getTextSize(label, _FONT, 0.8, 2)
            tx = max(0, (w - tw) // 2)
            ty = th + 16
            cv2.rectangle(frame, (tx - 8, ty - th - 8), (tx + tw + 8, ty + 8),
                          BOUNDARY_COLOR, -1)
            cv2.putText(frame, label, (tx, ty), _FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            break

    # yellow tracked-gap bbox, interpolated on every frame inside the gap span
    g = gap_by_frame.get(frame_idx)
    if g is None:
        return
    bbox = interp_gap_bbox(g, frame_idx)
    if bbox is None:
        return
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), GAP_COLOR, 2)
    conf = float(g.get("confidence") or 0.0)
    cv2.putText(frame, f"TRACKED_GAP #{g.get('track_id')}", (x1, max(0, y1 - 24)),
                _FONT, 0.55, GAP_COLOR, 2, cv2.LINE_AA)
    cv2.putText(frame, f"conf={conf:.2f} f={frame_idx}", (x1, max(0, y1 - 6)),
                _FONT, 0.55, GAP_COLOR, 2, cv2.LINE_AA)
