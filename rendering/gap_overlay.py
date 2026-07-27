"""Stage-1 gap overlay for the processed videos (visualization only).

Draws the FINAL accepted Stage-1 gaps on top of the feature annotations, so one
video shows gap detection + wagon segmentation + wagon IDs + every feature.  It
replays what Stage 1 already computed and persisted (no reconstruction re-run):

  * per-camera final gaps  -> ``per_camera_tracking.json[camera]["gaps"]``
      each with ``bbox_history`` + ``hit_frames`` (per-hit image-plane trajectory)
  * fused wagon boundaries -> ``global_train_state.json`` wagon time windows

Every drawn gap is the SAME object fed to reconstruction: gap #N (running order),
its stable track id, and its confidence.  A live "Detected Gaps: k/N" counter
increments as the train passes; at the end k == N == final gap count, which is
mathematically tied to the wagon count.  The rejected raw single-frame
candidates are never serialized here, so they can never be drawn in production.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2

# BGR colours -- distinct from every feature colour (door/damage use yellow
# (0,255,255)); the tracked-gap box is CYAN, the fused boundary MAGENTA.
GAP_COLOR = (255, 255, 0)        # cyan    -- tracked gap (interpolated bbox)
BOUNDARY_COLOR = (255, 0, 255)   # magenta -- fused wagon boundary flash
_HUD_BG = (0, 0, 0)
_HUD_FG = (255, 255, 0)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def number_gaps(gaps: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return gaps sorted by start_frame with a 1-based running ``_num`` stamped
    on each (the 'Gap #N' shown in the video)."""
    ordered = sorted((g for g in (gaps or []) if g.get("start_frame") is not None),
                     key=lambda g: g["start_frame"])
    for i, g in enumerate(ordered, start=1):
        g["_num"] = i
    return ordered


def interp_gap_bbox(gap: Dict[str, Any], frame_idx: int) -> Optional[List[float]]:
    """Interpolate a tracked gap's bbox at ``frame_idx`` (faithful port of
    ``wagon_count/video_segmenter._interp_gap_bbox`` on the serialized dict)."""
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
    """frame_idx -> the tracked gap active at that frame (gaps must be numbered)."""
    idx: Dict[int, Dict[str, Any]] = {}
    for g in gaps or []:
        sf, ef = g.get("start_frame"), g.get("end_frame")
        if sf is None or ef is None:
            continue
        for f in range(int(sf), int(ef) + 1):
            idx[f] = g
    return idx


def running_gap_count(gaps: Sequence[Dict[str, Any]], frame_idx: int) -> int:
    """Accepted gaps whose boundary has been reached by ``frame_idx`` (the live
    counter -- reaches the final total by end of clip)."""
    return sum(1 for g in gaps or [] if g.get("start_frame") is not None
               and int(g["start_frame"]) <= frame_idx)


def _hud(frame, text: str) -> None:
    """Top-right translucent 'Detected Gaps' counter (kept clear of the top-left
    feature info panel)."""
    h, w = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, _FONT, 0.7, 2)
    x0, y0 = w - tw - 24, 12
    ov = frame.copy()
    cv2.rectangle(ov, (x0 - 10, y0), (w - 8, y0 + th + 16), _HUD_BG, -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, text, (x0, y0 + th + 6), _FONT, 0.7, _HUD_FG, 2, cv2.LINE_AA)


def draw_gap_overlays(
    frame,
    frame_idx: int,
    gap_by_frame: Dict[int, Dict[str, Any]],
    boundary_frames: Sequence[int],
    *,
    all_gaps: Optional[Sequence[Dict[str, Any]]] = None,
    show_counter: bool = True,
) -> None:
    """Overlay the Stage-1 gaps for ONE frame, IN PLACE, on top of the feature
    annotations:

      * magenta boundary flash + ``GW_BOUNDARY`` within +/-3 frames of a fused
        wagon boundary;
      * cyan interpolated bbox for the active tracked gap, labelled
        ``Gap #<n> | Track <id> | Conf: <c>``;
      * a live ``Detected Gaps: k/N`` counter (top-right).
    """
    h, w = frame.shape[:2]

    for b in boundary_frames:
        if abs(int(b) - frame_idx) <= 3:
            cv2.line(frame, (0, 0), (w, 0), BOUNDARY_COLOR, 4)
            cv2.line(frame, (0, h - 1), (w, h - 1), BOUNDARY_COLOR, 4)
            label = "GW_BOUNDARY"
            (tw, th), _ = cv2.getTextSize(label, _FONT, 0.8, 2)
            tx, ty = max(0, (w - tw) // 2), th + 16
            cv2.rectangle(frame, (tx - 8, ty - th - 8), (tx + tw + 8, ty + 8),
                          BOUNDARY_COLOR, -1)
            cv2.putText(frame, label, (tx, ty), _FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            break

    g = gap_by_frame.get(frame_idx)
    if g is not None:
        bbox = interp_gap_bbox(g, frame_idx)
        if bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), GAP_COLOR, 2)
            conf = float(g.get("confidence") or 0.0)
            lines = [f"Gap #{g.get('_num', '?')}",
                     f"Track {g.get('track_id')}",
                     f"Conf: {conf:.2f}"]
            ly = max(0, y1 - 6 - 18 * (len(lines) - 1))
            for ln in lines:
                cv2.putText(frame, ln, (x1, ly), _FONT, 0.55, GAP_COLOR, 2, cv2.LINE_AA)
                ly += 18

    if show_counter and all_gaps is not None:
        _hud(frame, f"Detected Gaps: {running_gap_count(all_gaps, frame_idx)}/{len(all_gaps)}")
