"""AWS Rekognition OCR package.

Mirrors `train_inspection_engine/inspection/ocr/` (V4): the same modules, the
same public names, the same behaviour.  The Rekognition client itself lives at
`core.rekognition`, matching upstream's `core/rekognition.py`.
"""

from .preprocessor import Preprocessor
from .wagon_number import (
    WagonNumberDetector,
    WagonImageEnhancer,
    WagonNumberOCR,
    is_valid_wagon_number,
)
from .loco_number import LocoNumberOCR, is_valid_loco_number
from .rekognition_reader import digits_only, read_digits
from .three_frame_sheet import (
    build_horizontal_sheet,
    build_sheet,
    build_sheet_from_frames,
    build_vertical_sheet,
    ocr_sheet,
    preprocess_sheet,
    select_frame_positions,
    select_frames,
)

__all__ = [
    "Preprocessor",
    "WagonNumberDetector",
    "WagonImageEnhancer",
    "WagonNumberOCR",
    "LocoNumberOCR",
    "is_valid_wagon_number",
    "is_valid_loco_number",
    "digits_only",
    "read_digits",
    "build_horizontal_sheet",
    "build_sheet",
    "build_sheet_from_frames",
    "build_vertical_sheet",
    "ocr_sheet",
    "preprocess_sheet",
    "select_frame_positions",
    "select_frames",
]
