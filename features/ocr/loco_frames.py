"""Loco-band candidate extraction -- the input `LocoNumberOCR` consumes.

v4-native equivalent of the V4 Train-Inspection-Engine
`inspection/segments.extract_loco_frames`.  Same outputs, same `role`
semantics, same loco triplet; two things necessarily differ:

  * **Band source.**  Upstream derives loco bands from a `GapLocoDetector`
    sweep over the video.  Here the sealed GlobalTrainState already says which
    segment is an ENGINE, so the engine wagon IS the loco band -- no detector
    pass and no re-segmentation (Stage 1 owns that).
  * **Bbox source.**  Upstream reads a loco-detection CSV.  This pipeline ships
    no loco-plate detector, so number-plate boxes come from the same
    `wagon_id_counting.pt` the wagon path uses.  When a frame has no box the
    full frame is used instead -- exactly the degradation upstream documents.

Frames are read from the materialized wagon cache
(`wagon_cache/<GW_n>/right_up/frame_%06d.jpg`); no video is opened, which keeps
the Stage-3 "no cv2.VideoCapture for inference" constraint intact.

Returns `(summary_rows, ocr_rows)` as plain lists of dicts:

    summary_rows  one row per (band, representative position) -- the report /
                  dashboard loco gallery.
                  {loco_id, position, frame_num, frame_path, timestamp_sec}

    ocr_rows      at most TWO candidates per band, both already written to disk:
                  role="sheet"  a vertical sheet of the Middle-2 / Middle /
                                Middle+2 crops (primary, one Rekognition call)
                  role="best"   the highest-confidence single frame (fallback)
                  {loco_id, middle_frame, frame_path, timestamp_sec, role,
                   confidence}
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from core import constants as C
from core.logging_setup import get_logger

from features._common import list_wagon_frames
from features.inference_lib.ocr import build_vertical_sheet, select_frame_positions

log = get_logger("features.ocr.loco")

# Same padding fraction the wagon-number crop uses.
LOCO_CROP_PADDING = 0.25

# Representative positions for the report gallery -- the fractional offsets the
# wagon frames use, so a loco surfaces the same shape of evidence as a wagon.
REPRESENTATIVE_POSITIONS = (0.25, 0.55, 0.75, 0.80)
REPRESENTATIVE_POSITION_NAMES = ("start", "mid1", "mid2", "end")


def _frame_index(path: str) -> int:
    try:
        return int(os.path.basename(path).split("_")[1].split(".")[0])
    except (IndexError, ValueError):
        return -1


def _crop_to_bbox(frame: np.ndarray, bbox: Sequence[float]) -> Optional[np.ndarray]:
    """Padded crop of `bbox` from `frame` (None if degenerate)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    pad_x = (x2 - x1) * LOCO_CROP_PADDING
    pad_y = (y2 - y1) * LOCO_CROP_PADDING
    cx1, cy1 = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
    cx2, cy2 = min(w, int(x2 + pad_x)), min(h, int(y2 + pad_y))
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return frame[cy1:cy2, cx1:cx2].copy()


def _plate_bbox_by_frame(
    yolo_model, frame_paths: Sequence[str], det_confidence: float,
) -> Dict[int, Tuple[float, float, float, float, float]]:
    """Map frame index -> highest-confidence plate box (x1,y1,x2,y2,conf).

    Local YOLO only; no OCR and no API calls happen here."""
    out: Dict[int, Tuple[float, float, float, float, float]] = {}
    if yolo_model is None:
        return out
    for path in frame_paths:
        fnum = _frame_index(path)
        if fnum < 0:
            continue
        frame = cv2.imread(path)
        if frame is None:
            continue
        try:
            results = yolo_model.predict(source=frame, verbose=False,
                                         conf=det_confidence)
        except Exception:
            continue
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                prev = out.get(fnum)
                if prev is None or conf > prev[4]:
                    out[fnum] = (float(x1), float(y1), float(x2), float(y2), conf)
    return out


def extract_loco_candidates(
    *,
    yolo_model,
    cache_root: str,
    gw_id: str,
    loco_id: int,
    out_dir: str,
    det_confidence: float = C.CONF_OCR_BOX,
    fps: float = 0.0,
    camera_id: str = C.CAMERA_RIGHT_UP,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build this engine segment's loco gallery + OCR candidates.

    `out_dir` receives the written JPEGs (typically the evidence temp dir).
    Returns `(summary_rows, ocr_rows)`; both empty when the segment has no
    cached frames."""
    frame_paths = list_wagon_frames(cache_root, gw_id, camera_id, trim_stable=True)
    if not frame_paths:
        return [], []

    frames = sorted(f for f in (_frame_index(p) for p in frame_paths) if f >= 0)
    if not frames:
        return [], []
    by_index = {_frame_index(p): p for p in frame_paths}
    total = len(frames)
    middle_frame = frames[total // 2]

    bbox_by_frame = _plate_bbox_by_frame(yolo_model, frame_paths, det_confidence)
    os.makedirs(out_dir, exist_ok=True)

    def _read_loco_crop(fnum: int) -> Tuple[Optional[np.ndarray], bool]:
        """(image, was_cropped).  Full frame when there is no usable box."""
        path = by_index.get(fnum)
        if not path:
            return None, False
        frame = cv2.imread(path)
        if frame is None:
            return None, False
        bbox = bbox_by_frame.get(fnum)
        img = _crop_to_bbox(frame, bbox) if bbox else None
        if img is None or img.size == 0:
            return frame, False
        return img, True

    def _conf_of(fnum: int) -> float:
        box = bbox_by_frame.get(fnum)
        return float(box[4]) if box else 0.0

    # ---- report gallery: one frame per representative position ----
    summary: List[Dict[str, Any]] = []
    for pos_name, pos_fraction in zip(REPRESENTATIVE_POSITION_NAMES,
                                      REPRESENTATIVE_POSITIONS):
        idx = min(int(total * pos_fraction), total - 1)
        frame_num = frames[idx]
        path = by_index.get(frame_num)
        img = cv2.imread(path) if path else None
        if img is None:
            continue
        rep_path = os.path.join(out_dir, f"loco_{loco_id:03d}_{pos_name}.jpg")
        cv2.imwrite(rep_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        summary.append({
            "loco_id": loco_id,
            "position": pos_name,
            "frame_num": frame_num,
            "frame_path": rep_path,
            "timestamp_sec": (frame_num / fps) if fps else 0.0,
        })

    # ---- OCR candidates ----
    ocr_rows: List[Dict[str, Any]] = []
    n_cropped = 0

    # 1. primary: ONE sheet of the Middle-2 / Middle / Middle+2 crops.
    seen: set = set()
    sheet_frame_nums = [
        frames[p] for p in select_frame_positions(total, "loco")
        if not (frames[p] in seen or seen.add(frames[p]))
    ]
    raw_crops: List[np.ndarray] = []
    used_frame_nums: List[int] = []
    for fnum in sheet_frame_nums:
        crop, was_cropped = _read_loco_crop(fnum)
        if crop is None:
            continue
        n_cropped += int(was_cropped)
        raw_crops.append(crop)
        used_frame_nums.append(fnum)
    if raw_crops:
        sheet = build_vertical_sheet(raw_crops)
        sheet_path = os.path.join(out_dir, f"loco_{loco_id:03d}_sheet.jpg")
        cv2.imwrite(sheet_path, sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        ocr_rows.append({
            "loco_id": loco_id,
            "middle_frame": middle_frame,
            "frame_path": sheet_path,
            "timestamp_sec": (middle_frame / fps) if fps else 0.0,
            "role": "sheet",
            "confidence": _conf_of(middle_frame),
            # Audit trail: the cache frames stacked onto this sheet, top to
            # bottom, so the reading can be checked against its own input.
            "sheet_frames": list(used_frame_nums),
        })

    # 2. fallback: the band's highest-confidence single frame.
    frames_with_conf = [(f, bbox_by_frame[f][4]) for f in frames
                        if f in bbox_by_frame]
    best_frame = (max(frames_with_conf, key=lambda i: i[1])[0]
                  if frames_with_conf else middle_frame)
    crop, was_cropped = _read_loco_crop(best_frame)
    if crop is not None:
        n_cropped += int(was_cropped)
        crop_path = os.path.join(
            out_dir, f"loco_{loco_id:03d}_frame_{best_frame:06d}.jpg")
        cv2.imwrite(crop_path, crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        ocr_rows.append({
            "loco_id": loco_id,
            "middle_frame": best_frame,
            "frame_path": crop_path,
            "timestamp_sec": (best_frame / fps) if fps else 0.0,
            "role": "best",
            "confidence": _conf_of(best_frame),
        })

    log.debug("[OCR/loco %s] band=%d frames=%d gallery=%d candidates=%d "
              "(%d cropped to a plate box)",
              gw_id, loco_id, total, len(summary), len(ocr_rows), n_cropped)
    return summary, ocr_rows
