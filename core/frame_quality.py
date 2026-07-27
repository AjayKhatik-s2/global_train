"""Evidence-frame quality scoring.

Faithful port of the legacy WagonEye snapshot heuristics that lived in
``old_system/RIGHT_UP/door_processor.py``:

    _compute_detection_quality   -> detection_quality()
    _is_edge_detection           -> is_edge_detection()
    _score_detection             -> snapshot_score()

These are the heuristics the legacy side/door pipeline actually used to pick
the report snapshot for each track (the multi-metric ``SnapshotSelector`` was
dead code in both the old and the new system and is deliberately NOT revived).

The scorer blends bbox area, horizontal centre proximity, model confidence and
a crop-quality term (brightness + Laplacian texture), then applies a hard
multiplicative penalty for boxes hugging a frame edge.  There is NO hard
blur/quality REJECTION gate -- legacy never had one; quality is a soft
down-weight only.

Imported directly by the feature processors (``from core.frame_quality import
...``); intentionally NOT re-exported from ``core/__init__`` so the
reportlab-only reporting layer never transitively imports cv2.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np


# --- legacy constants (verbatim from door_processor.py) ----------------------

_EDGE_MARGIN_RATIO  = 0.08    # _is_edge_detection: within 8% of any edge
_AREA_OPTIMAL_RATIO = 0.15    # EXACT old production value: 15% of frame = perfect
_EDGE_PENALTY       = 0.3     # 70% score reduction for edge-hugging boxes

# _score_detection term weights -- EXACT old production values (door_processor.py
# _score_detection): area 2.0, centre 2.5, confidence 1.0, quality 0.5.  A prior
# revision biased this toward the largest door (area peak 0.28, weight 3.5, plus
# a raw-area tie-break); that changed WHICH frame was selected vs old production,
# so it has been reverted to reproduce the identical snapshot choice.
_W_AREA          = 2.0
_W_CENTER        = 2.5
_W_CONF          = 1.0
_W_QUALITY       = 0.5

Bbox = Sequence[float]


def detection_quality(frame: np.ndarray, bbox: Bbox, *, pad: int = 5) -> float:
    """Crop brightness + Laplacian-texture quality in ``[0.1, 1.0]``.

    Faithful port of ``_compute_detection_quality`` (the quality scalar only;
    the legacy glare/reason flags are not needed downstream).  Brighter-than-200
    crops and low-texture (blurry / featureless) crops are penalised; the result
    is clamped to ``[0.1, 1.0]`` so a poor frame is down-weighted, never excluded.
    """
    if frame is None or bbox is None or len(bbox) != 4:
        return 1.0
    h, w = frame.shape[:2]
    x1 = max(0, int(bbox[0]) - pad)
    y1 = max(0, int(bbox[1]) - pad)
    x2 = min(w, int(bbox[2]) + pad)
    y2 = min(h, int(bbox[3]) + pad)
    if x2 <= x1 or y2 <= y1:
        return 1.0
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return 1.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    brightness = float(np.mean(gray))
    texture = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    brightness_penalty = max(0.0, (brightness - 200.0) / 55.0)
    texture_penalty = max(0.0, (100.0 - texture) / 100.0)
    quality = 1.0 - 0.5 * brightness_penalty - 0.5 * texture_penalty
    return max(0.1, min(1.0, quality))


def is_edge_detection(
    bbox: Bbox, frame_w: int, frame_h: int,
    *, margin_ratio: float = _EDGE_MARGIN_RATIO,
) -> bool:
    """True if the bbox hugs any frame edge (door entering / leaving view)."""
    x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]),
                      float(bbox[2]), float(bbox[3]))
    mx = frame_w * margin_ratio
    my = frame_h * margin_ratio
    return (x1 < mx or x2 > frame_w - mx or y1 < my or y2 > frame_h - my)


def snapshot_score(
    bbox: Bbox, confidence: float, quality: float,
    frame_w: int, frame_h: int,
) -> float:
    """Snapshot scorer -- EXACT port of old production ``_score_detection``:

    ``(area*2.0 + center_h*2.5 + conf*1.0 + quality*0.5) * edge_penalty``

    with the area term peaking at 15% of the frame and a 0.3 edge-hugging
    penalty.  Centre proximity is horizontal-only (the door crosses the frame
    horizontally as the train passes).  No hard rejection gate (quality is a
    soft down-weight only), matching production.
    """
    x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]),
                      float(bbox[2]), float(bbox[3]))
    frame_area = max(1.0, float(frame_w) * float(frame_h))
    bbox_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_score = min(1.0, bbox_area / (frame_area * _AREA_OPTIMAL_RATIO))

    frame_cx = max(1.0, frame_w / 2.0)
    cx = (x1 + x2) / 2.0
    center_score = 1.0 - abs(cx - frame_cx) / frame_cx   # legacy: unclamped

    edge_penalty = (_EDGE_PENALTY
                    if is_edge_detection(bbox, frame_w, frame_h) else 1.0)

    score = (
        area_score * _W_AREA
        + center_score * _W_CENTER
        + float(confidence) * _W_CONF
        + float(quality) * _W_QUALITY
    ) * edge_penalty
    return float(score)


# NOTE: an earlier revision expanded the chosen door box by 15% ("ITEM 4") for
# both the processed-video overlay and the evidence crop.  Old production drew
# and stored the RAW box, so that expansion was removed for exact annotation
# parity; the helper is intentionally gone (no expansion anywhere).
