"""Image preprocessing applied before sending OCR crops to AWS Rekognition.

Mirrors `inspection/ocr/preprocessor.py` (V4).

Deliberately NOT a denoise / CLAHE / unsharp chain: Rekognition is trained on
natural imagery and reads raw upscaled crops better than aggressively filtered
ones (V4 commit e9bbd44 removed that preprocessing).
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


class Preprocessor:
    """Preprocessing applied to every OCR image. Takes BGR, returns BGR."""

    OCR_TARGET_WIDTH = 800
    RESIZE_FACTOR = 3

    @staticmethod
    def _upscale(img: np.ndarray, factor: float = 3.0) -> np.ndarray:
        h, w = img.shape[:2]
        return cv2.resize(
            img, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_CUBIC
        )

    @classmethod
    def _ensure_min_width(cls, img: np.ndarray,
                          min_w: Optional[int] = None) -> np.ndarray:
        min_w = min_w or cls.OCR_TARGET_WIDTH
        h, w = img.shape[:2]
        if w < min_w:
            scale = min_w / w
            img = cv2.resize(
                img, (min_w, int(h * scale)), interpolation=cv2.INTER_CUBIC
            )
        return img

    @classmethod
    def primary(cls, crop: np.ndarray) -> np.ndarray:
        """Enlarge the image and ensure it meets the minimum width for OCR."""
        img = cls._upscale(crop, cls.RESIZE_FACTOR)
        return cls._ensure_min_width(img)
