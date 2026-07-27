"""Wagon-number detection + OCR.

Mirrors `inspection/ocr/wagon_number.py` (V4).

`detect_on_segment` scans a directory of `frame_%06d.jpg` files — in this
pipeline that is a wagon's cache directory,
``wagon_cache/<GW_n>/right_up/``, which uses exactly that naming.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from core.logging_setup import get_logger
from core.rekognition import RekognitionClient

from .preprocessor import Preprocessor
from .rekognition_reader import read_digits
from .three_frame_sheet import build_vertical_sheet, select_frame_positions


# ---------------------------------------------------------------------------
# Class-name aliases
# ---------------------------------------------------------------------------

# Different wagon-number models emit different class labels for the same
# concept.  Normalise them all to the canonical name so detect_on_segment
# filters correctly regardless of which camera/model produced the output.
WAGON_NUMBER_CLASS_ALIASES: Dict[str, str] = {
    "wagonno": "wagon_id",   # right_top model → canonical right_up name
}
_WAGON_NUMBER_CLASS = "wagon_id"


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def is_valid_wagon_number(number: str) -> bool:
    """Wagon numbers are 11 digits."""
    return bool(number) and number.isdigit() and len(number) == 11


# ---------------------------------------------------------------------------
# YOLO bbox detector — produces bands of wagon-number detections per segment
# ---------------------------------------------------------------------------


class WagonNumberDetector:
    """Returns wagon-number bbox detection bands per segment.

    Works with any YOLO version (v8-v11) — detection is driven purely by the
    generic ultralytics ``Results.boxes`` API, no version-specific parsing.
    """

    def __init__(
        self,
        model,
        confidence_threshold: float = 0.50,
        gap_tolerance: int = 8,
        logger: Optional[logging.Logger] = None,
    ):
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.gap_tolerance = gap_tolerance
        self.logger = logger or get_logger("features.ocr.detector")
        self._unknown_cls_warned: set = set()

    def detect_on_segment(
        self, segment_dir: str, start_frame: int, end_frame: int
    ) -> List[Dict[str, Any]]:
        raw: List[tuple] = []
        for fn in range(start_frame, end_frame + 1):
            path = os.path.join(segment_dir, f"frame_{fn:06d}.jpg")
            if not os.path.exists(path):
                continue
            frame = cv2.imread(path)
            if frame is None:
                continue
            results = self.model.predict(
                source=frame, verbose=False, conf=self.confidence_threshold
            )
            for r in results:
                for box in r.boxes:
                    cls_idx = int(box.cls[0])
                    cls_name = r.names.get(cls_idx)
                    if cls_name is None:
                        # Class index not present in model metadata — accept the
                        # box anyway.  This model is specialized for wagon-number
                        # plates so any detection is a plate candidate.  Log once
                        # per unseen index so the mismatch is visible in logs.
                        if cls_idx not in self._unknown_cls_warned:
                            self.logger.warning(
                                "WagonNumberDetector: class index %d not in model "
                                "names %s — accepted as wagon-number detection. "
                                "Model metadata may be incomplete.",
                                cls_idx, r.names,
                            )
                            self._unknown_cls_warned.add(cls_idx)
                    else:
                        cls_name = WAGON_NUMBER_CLASS_ALIASES.get(cls_name, cls_name)
                        if cls_name != _WAGON_NUMBER_CLASS:
                            continue
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    raw.append((fn, conf, float(x1), float(y1), float(x2), float(y2)))

        if not raw:
            return []
        raw.sort(key=lambda d: d[0])

        bands: List[dict] = []
        cur = self._new_band(raw[0], 1)
        for det in raw[1:]:
            if det[0] - cur["end_frame"] <= self.gap_tolerance + 1:
                cur["end_frame"] = det[0]
                if det[0] not in cur["frames"]:
                    cur["frames"].append(det[0])
                cur["confidences"].append(det[1])
                cur["detections"].append(det)
            else:
                bands.append(cur)
                cur = self._new_band(det, len(bands) + 1)
        bands.append(cur)

        result = []
        for band in bands:
            sorted_dets = sorted(band["detections"], key=lambda d: d[1], reverse=True)
            seen, all_frames = set(), []
            for det in sorted_dets:
                fn = det[0]
                if fn in seen:
                    continue
                seen.add(fn)
                all_frames.append({
                    "frame": fn,
                    "confidence": det[1],
                    "bbox": [det[2], det[3], det[4], det[5]],
                })
            best = all_frames[0]
            result.append({
                "band_id": band["band_id"],
                "best_frame": best["frame"],
                "best_confidence": best["confidence"],
                "best_bbox": best["bbox"],
                "top_frames": all_frames,
                # Chronological view (frame + bbox + confidence), ascending by
                # frame number — used to pick "the Nth frame of the band" /
                # "the Nth-from-last frame", as opposed to top_frames above
                # which is sorted by confidence.
                "frames_by_time": sorted(all_frames, key=lambda f: f["frame"]),
                "frame_count": len(set(band["frames"])),
                "avg_confidence": float(np.mean(band["confidences"])),
            })
        return result

    @staticmethod
    def _new_band(det: tuple, band_id: int) -> dict:
        return {
            "band_id": band_id,
            "start_frame": det[0],
            "end_frame": det[0],
            "frames": [det[0]],
            "confidences": [det[1]],
            "detections": [det],
        }


# ---------------------------------------------------------------------------
# Crop + enhance helper
# ---------------------------------------------------------------------------


class WagonImageEnhancer:
    PADDING_FRACTION = 0.25

    def crop(self, frame_path: str, bbox: List[float]) -> Optional[np.ndarray]:
        frame = cv2.imread(frame_path)
        if frame is None:
            return None
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in bbox]
        pad_x = (x2 - x1) * self.PADDING_FRACTION
        pad_y = (y2 - y1) * self.PADDING_FRACTION
        cx1 = max(0, int(x1 - pad_x))
        cy1 = max(0, int(y1 - pad_y))
        cx2 = min(w, int(x2 + pad_x))
        cy2 = min(h, int(y2 + pad_y))
        if cx2 <= cx1 or cy2 <= cy1:
            return None
        return frame[cy1:cy2, cx1:cx2].copy()

    @staticmethod
    def apply_primary(crop: np.ndarray) -> np.ndarray:
        return Preprocessor.primary(crop)


# ---------------------------------------------------------------------------
# OCR orchestrator — AWS Rekognition
# ---------------------------------------------------------------------------


class WagonNumberOCR:
    """Reads the 11-digit wagon number via AWS Rekognition DetectText.

    Frame selection: for each band, build ONE vertical sheet of THREE crops
    instead of reading a single frame, then send that sheet through the existing
    Preprocessor + Rekognition. The crops are stacked vertically so each reads
    as its own LINE; the reader then picks the single best valid 11-digit
    reading rather than concatenating the three. Two sheets are tried per band,
    in an order that depends on rake load state (cargo can occlude the plate
    from one end of the band depending on travel direction):

      * loaded rake → primary End-7/-5/-3, fallback Start+3/+5/+7
      * empty rake  → primary Start+3/+5/+7, fallback End-7/-5/-3

    Offsets index the band's detected frames ordered by time (see
    :mod:`three_frame_sheet`). Bands with too few detected frames fall back to a
    single-crop sheet of the highest-confidence frame. The first sheet that
    reads a valid 11-digit number wins; otherwise the highest-confidence attempt
    is still reported (raw digits + representative frame), just flagged invalid.

    Crops are placed on the sheet exactly as extracted — the Preprocessor runs
    on the assembled sheet, never on the individual crops.
    """

    MIN_FRAMES_FOR_SHEET_SELECTION = 3

    def __init__(
        self,
        region: str,
        logger: Optional[logging.Logger] = None,
        aws_access_key: Optional[str] = None,
        aws_secret_key: Optional[str] = None,
        client: Optional[RekognitionClient] = None,
    ):
        self.logger = logger or get_logger("features.ocr.wagon")
        # `client` injection is an additive convenience for tests / reuse of a
        # process-wide client; upstream always constructs its own.
        self.client = client or RekognitionClient(
            region, aws_access_key, aws_secret_key, logger=self.logger,
        )

    def extract_number_from_segment(
        self,
        bands: List[dict],
        segment_dir: str,
        enhancer: WagonImageEnhancer,
        is_loaded: bool,
        save_dir: Optional[str] = None,
        segment_id: int = 0,
    ) -> dict:
        best_raw, best_conf, best_img, best_frame_info = "", 0.0, None, None
        best_sheet_frames: List[int] = []
        primary_band = max(bands, key=lambda b: b["best_confidence"]) if bands else {}

        attempt_index = 0
        for band in bands:
            for triplet in self._select_candidate_triplets(band, is_loaded):
                sheet, rep_info, used = self._build_sheet(
                    triplet, segment_dir, enhancer)
                if sheet is None:
                    attempt_index += 1
                    continue
                # Preprocessor runs on the assembled sheet, not per crop.
                img = enhancer.apply_primary(sheet)
                # Vertical sheet → the 3 crops read as separate LINEs; pick the
                # single best valid 11-digit reading instead of concatenating.
                num, conf = read_digits(
                    self.client, img, validator=is_valid_wagon_number,
                )
                if num and conf >= best_conf:
                    best_raw, best_conf, best_img, best_frame_info = num, conf, img, rep_info
                    best_sheet_frames = used
                if is_valid_wagon_number(num):
                    return self._build_result(
                        num, conf, img, rep_info, band["band_id"],
                        is_valid=True, fallback_triggered=attempt_index > 0,
                        save_dir=save_dir, segment_id=segment_id,
                        sheet_frames=used,
                    )
                attempt_index += 1

        if best_frame_info is None:
            best_frame_info = {
                "frame": primary_band.get("best_frame"),
                "bbox": primary_band.get("best_bbox"),
            }
        return self._build_result(
            best_raw, best_conf, best_img, best_frame_info,
            primary_band.get("band_id", 0), is_valid=False, fallback_triggered=True,
            save_dir=save_dir, segment_id=segment_id,
            sheet_frames=best_sheet_frames,
        )

    # ------------------------------------------------------------------

    def _select_candidate_triplets(
        self, band: dict, is_loaded: bool
    ) -> List[List[dict]]:
        """Return the ordered list of three-frame triplets to sheet for a band.

        Each triplet is a list of ``frame_info`` dicts (``frame``/``bbox``/
        ``confidence``). Normally two triplets are returned — the primary set
        then the fallback set (see the class docstring) — deduped so a short
        band that clamps offsets onto the same frame doesn't repeat crops. Bands
        with too few detected frames yield a single one-crop triplet of the
        highest-confidence frame.
        """
        frames = band.get("frames_by_time") or []
        n = len(frames)
        if n == 0:
            return []
        if n < self.MIN_FRAMES_FOR_SHEET_SELECTION:
            return [[{
                "frame": band["best_frame"],
                "confidence": band["best_confidence"],
                "bbox": band["best_bbox"],
            }]]
        category = "loaded" if is_loaded else "empty"
        triplets: List[List[dict]] = []
        for use_fallback in (False, True):
            positions = select_frame_positions(n, category, use_fallback)
            seen: set = set()
            triplet = [frames[p] for p in positions
                       if not (p in seen or seen.add(p))]
            if triplet:
                triplets.append(triplet)
        return triplets

    def _build_sheet(
        self, triplet: List[dict], segment_dir: str, enhancer: WagonImageEnhancer
    ):
        """Crop each frame in ``triplet`` and lay the raw crops on one sheet.

        Returns ``(sheet, representative_frame_info, used_frame_numbers)`` where
        the representative is the highest-confidence frame in the triplet (used
        for reporting the winning frame/bbox) and ``used_frame_numbers`` names
        the frames actually stacked, top to bottom -- the audit trail for the
        image Rekognition is asked to read.  Returns ``(None, None, [])`` when
        no crop could be extracted.
        """
        raw_crops: List[np.ndarray] = []
        used: List[dict] = []
        for fi in triplet:
            frame_path = os.path.join(segment_dir, f"frame_{fi['frame']:06d}.jpg")
            crop = enhancer.crop(frame_path, fi["bbox"])
            if crop is None or crop.size == 0:
                continue
            raw_crops.append(crop)
            used.append(fi)
        if not raw_crops:
            return None, None, []
        sheet = build_vertical_sheet(raw_crops)
        rep = max(used, key=lambda f: f.get("confidence", 0.0))
        return sheet, rep, [int(f["frame"]) for f in used]

    def _build_result(
        self, number, conf, img, frame_info, band_id,
        is_valid, fallback_triggered, save_dir, segment_id,
        sheet_frames=None,
    ) -> dict:
        return {
            "raw_number": number,
            "display_number": number if is_valid else "-",
            "is_valid_11_digit": is_valid,
            "confidence": conf,
            "ocr_confidence": conf,
            "band_id": band_id,
            "best_frame": frame_info.get("frame"),
            "best_bbox": frame_info.get("bbox"),
            "enhanced_image_path": self._save_debug_image(img, save_dir, segment_id),
            "fallback_triggered": fallback_triggered,
            # Additive: the assembled+preprocessed sheet, so the caller can
            # persist it as evidence without re-reading frames from disk --
            # this IS the image posted to Rekognition, and `sheet_frames` names
            # the cache frames it was stacked from, so a reviewer can check the
            # reading against the source.
            "_sheet_image": img,
            "sheet_frames": list(sheet_frames or []),
        }

    @staticmethod
    def _save_debug_image(img, save_dir, segment_id) -> Optional[str]:
        if img is None or not save_dir or not segment_id:
            return None
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"segment_{segment_id:03d}_wagon_number.jpg")
        cv2.imwrite(path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return path
