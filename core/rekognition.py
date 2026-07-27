"""AWS Rekognition client wrapper used for wagon/loco number-plate OCR.

Mirrors `Train-Inspection-Engine/src/train_inspection_engine/core/rekognition.py`
(V4).  Credential conventions match the rest of the pipeline: explicit keys when
supplied, otherwise boto3's default chain (an EC2 IAM instance role).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from core.logging_setup import get_logger


class RekognitionClient:
    """Thin wrapper over ``boto3.client('rekognition')``."""

    def __init__(
        self,
        region: str,
        aws_access_key: Optional[str] = None,
        aws_secret_key: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        import boto3

        self.region = region
        self.logger = logger or get_logger("core.rekognition")

        if aws_access_key and aws_secret_key:
            self.client = boto3.client(
                "rekognition",
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=region,
            )
            self.logger.info(
                "Rekognition client initialised with provided credentials "
                "(region=%s).", region)
        else:
            self.client = boto3.client("rekognition", region_name=region)
            self.logger.info(
                "Rekognition client initialised with IAM role credentials "
                "(region=%s).", region)

    def detect_text(self, image_bytes: bytes) -> List[Dict[str, Any]]:
        response = self.client.detect_text(Image={"Bytes": image_bytes})
        return response.get("TextDetections", [])
