"""Loco-number OCR — 5-digit numbers on the locomotive face, via AWS Rekognition.

Mirrors `inspection/ocr/loco_number.py` (V4).

Takes the pre-extracted loco-frame candidate rows, which carry at most two
candidates per band — a three-frame Middle-2/Middle/Middle+2 sheet
(``role="sheet"``) and the band's highest-confidence single frame
(``role="best"``) — rather than every frame of the band, since each candidate is
a single deterministic Rekognition call.

DEVIATION from upstream: upstream types the candidate rows as a pandas
DataFrame (``ocr_df`` from ``extract_loco_frames``).  This pipeline does not
depend on pandas, so the two entry points accept EITHER a DataFrame or a plain
list of row dicts with the same columns
(``loco_id`` / ``role`` / ``frame_path`` / ``middle_frame``).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import cv2

from core.logging_setup import get_logger
from core.rekognition import RekognitionClient

from .preprocessor import Preprocessor
from .rekognition_reader import read_digits


def is_valid_loco_number(number: str) -> bool:
    return bool(number) and number.isdigit() and len(number) == 5


def _rows(candidates) -> List[Dict[str, Any]]:
    """Normalise a DataFrame OR a list of dicts into a list of row dicts."""
    if candidates is None:
        return []
    if hasattr(candidates, "to_dict"):          # pandas DataFrame
        if getattr(candidates, "empty", False):
            return []
        return candidates.to_dict("records")
    return list(candidates)


# ---------------------------------------------------------------------------
# OCR orchestrator — AWS Rekognition
# ---------------------------------------------------------------------------


class LocoNumberOCR:
    """Reads the 5-digit loco number via AWS Rekognition DetectText.

    Frame selection: try the band's three-frame Middle-2/Middle/Middle+2
    **sheet** first (role ``"sheet"``, a single white sheet read in one
    Rekognition call); if that doesn't read a valid 5-digit number, fall back to
    the band's highest-confidence single frame (role ``"best"``).
    """

    def __init__(
        self,
        region: str,
        logger: Optional[logging.Logger] = None,
        aws_access_key: Optional[str] = None,
        aws_secret_key: Optional[str] = None,
        client: Optional[RekognitionClient] = None,
    ):
        self.logger = logger or get_logger("features.ocr.loco")
        self.client = client or RekognitionClient(
            region, aws_access_key, aws_secret_key, logger=self.logger,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_per_band(self, loco_ocr_df, save_dir: Optional[str] = None) -> dict:
        """Read one loco number **per loco band**.

        Candidate rows carry a ``loco_id`` = the band id.  Each band is OCR'd
        independently so a rake with multiple locos surfaces every loco number,
        keyed by ``loco_id`` (the dashboard keys ``loco_number_results`` by
        ``str(loco_id)``).

        Returns ``{loco_id (int): result_dict}`` — empty when there are no
        loco frames.
        """
        rows = _rows(loco_ocr_df)
        if not rows:
            return {}

        grouped: Dict[int, List[Dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(int(row.get("loco_id", 0)), []).append(row)

        results: Dict[int, dict] = {}
        for loco_id in sorted(grouped):
            results[loco_id] = self.extract_from_frames(
                grouped[loco_id], save_dir=save_dir,
                debug_name=f"loco_number_band_{loco_id}.jpg",
            )
        valid = sum(1 for r in results.values() if r.get("is_valid_5_digit"))
        self.logger.info(
            "Loco-number OCR: %d band(s), %d valid 5-digit number(s)",
            len(results), valid,
        )
        return results

    def extract_from_frames(
        self,
        loco_summary_df,
        save_dir: Optional[str] = None,
        debug_name: str = "loco_number_best.jpg",
    ) -> dict:
        """Read the loco number from a single band's candidate image(s).

        Tries ``role="sheet"`` (the three-frame sheet) before ``role="best"``
        (single-frame fallback). The first candidate to read a valid 5-digit
        number wins; otherwise the highest-confidence attempt is still reported
        (raw digits + frame used), just flagged invalid.
        """
        rows = _rows(loco_summary_df)
        if not rows:
            return self._build_result(
                "", 0.0, None, None, is_valid=False, fallback_triggered=False,
                save_dir=save_dir, debug_name=debug_name,
            )

        rows.sort(key=lambda r: 0 if r.get("role") == "sheet" else 1)

        best_raw, best_conf, best_img, best_frame = "", 0.0, None, None
        best_row: Optional[Dict[str, Any]] = None
        for idx, row in enumerate(rows):
            frame_path = row.get("frame_path")
            if not frame_path or not os.path.exists(frame_path):
                continue
            img = cv2.imread(frame_path)
            if img is None:
                continue
            frame_num = int(row.get("middle_frame", 0))

            processed = Preprocessor.primary(img)
            # The "sheet" candidate stacks 3 crops → each reads as a separate
            # LINE; pick the single best valid 5-digit reading rather than
            # concatenating them.
            num, conf = read_digits(
                self.client, processed, validator=is_valid_loco_number,
            )
            if num and conf >= best_conf:
                best_raw, best_conf, best_img, best_frame = num, conf, processed, frame_num
                best_row = row
            if is_valid_loco_number(num):
                return self._build_result(
                    num, conf, processed, frame_num, is_valid=True,
                    fallback_triggered=idx > 0, save_dir=save_dir,
                    debug_name=debug_name, row=row,
                )

        return self._build_result(
            best_raw, best_conf, best_img, best_frame, is_valid=False,
            fallback_triggered=True, save_dir=save_dir, debug_name=debug_name,
            row=best_row or rows[0],
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _build_result(
        number, conf, img, frame_num, is_valid, fallback_triggered,
        save_dir, debug_name, row=None,
    ) -> dict:
        enhanced_path = None
        if save_dir and img is not None:
            os.makedirs(save_dir, exist_ok=True)
            enhanced_path = os.path.join(save_dir, debug_name)
            cv2.imwrite(enhanced_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return {
            "raw_number": number,
            "display_number": number if is_valid else "-",
            "is_valid_5_digit": is_valid,
            "confidence": conf,
            "ocr_confidence": conf,
            "best_frame": frame_num,
            "enhanced_image_path": enhanced_path,
            "fallback_triggered": fallback_triggered,
            # Additive: which candidate image Rekognition actually read the
            # winning number from -- "sheet" (the three-frame sheet) or "best"
            # (the single-frame fallback).  The caller surfaces this file so a
            # reviewer verifies the number against the exact OCR input.
            "candidate_role": (row or {}).get("role"),
            "candidate_path": (row or {}).get("frame_path"),
        }
