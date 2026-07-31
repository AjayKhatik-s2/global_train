"""Stage 6 -- upload `batch_outputs/<key>/` to S3.

Strategy:
    * PDF goes to the report microservice first; falls back to S3.
    * JSON goes directly to S3 with application/json content-type.
    * Everything else (wagon_cache + wagon_states + global_state) is
      recursively uploaded under
        s3://<bucket>/<train_batch_prefix>/<batch_key>/...
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional

from core import constants as C
from core.logging_setup import get_logger

log = get_logger("delivery.s3")


# -----------------------------------------------------------------------------
# Content-type per extension (very small mapping)
# -----------------------------------------------------------------------------

_CONTENT_TYPES = {
    ".pdf":  "application/pdf",
    ".json": "application/json",
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".mp4":  "video/mp4",
    ".txt":  "text/plain",
    ".md":   "text/markdown",
}


def _content_type_for(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return _CONTENT_TYPES.get(ext, "application/octet-stream")


# -----------------------------------------------------------------------------
# Microservice PDF upload (proven helper preserved from the legacy
# master_runner; same API and product name).
# -----------------------------------------------------------------------------

def _upload_pdf_microservice(pdf_path: str) -> Optional[str]:
    import requests
    ist = timezone(timedelta(hours=5, minutes=30))
    today = datetime.now(ist).strftime("%d-%m-%Y")
    for attempt in range(1, 4):
        try:
            with open(pdf_path, "rb") as f:
                files = {"file": (os.path.basename(pdf_path), f, "application/pdf")}
                data  = {"product_name": C.PRODUCT_NAME, "folder_name": today}
                resp = requests.post(C.UPLOAD_API_URL, data=data, files=files,
                                     timeout=120)
            if resp.status_code == 200:
                url = resp.json().get("url")
                if url:
                    log.info("[DELIVERY] PDF microservice URL: %s", url)
                    return url
        except Exception as e:
            log.warning("[DELIVERY] PDF microservice attempt %d/3 failed: %s",
                        attempt, e)
        time.sleep(10)
    return None


# -----------------------------------------------------------------------------
# Public entry
# -----------------------------------------------------------------------------

def upload_pdf(s3_client, pdf_path: str, batch_key: str) -> Optional[str]:
    """Microservice first; S3 direct fallback."""
    if not os.path.exists(pdf_path):
        return None
    url = _upload_pdf_microservice(pdf_path)
    if url:
        return url
    bucket = C.S3_OUTPUT_BUCKET
    key = f"{C.S3_TRAIN_BATCH_PREFIX}/{batch_key}/reports/combined_train_report.pdf"
    try:
        s3_client.upload_file(
            pdf_path, bucket, key,
            ExtraArgs={"ContentType": "application/pdf"},
        )
        return f"https://{bucket}.s3.{C.S3_REGION}.amazonaws.com/{key}"
    except Exception as e:
        log.error("[DELIVERY] S3 PDF fallback failed: %s", e)
        return None


def upload_json(s3_client, json_path: str, batch_key: str) -> Optional[str]:
    if not os.path.exists(json_path):
        return None
    bucket = C.S3_OUTPUT_BUCKET
    key = f"{C.S3_TRAIN_BATCH_PREFIX}/{batch_key}/reports/combined_train_report.json"
    try:
        s3_client.upload_file(
            json_path, bucket, key,
            ExtraArgs={"ContentType": "application/json"},
        )
        url = f"https://{bucket}.s3.{C.S3_REGION}.amazonaws.com/{key}"
        log.info("[DELIVERY] JSON URL: %s", url)
        return url
    except Exception as e:
        log.error("[DELIVERY] JSON upload failed: %s", e)
        return None


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _compressed_or_original(local: str, cam: str) -> str:
    """H.264-compress `local` next to itself; return the original on any failure.

    Gated by ``WAGONEYE_COMPRESS_DETECTED_VIDEO`` (default on).  Size cap and
    quality come from ``WAGONEYE_DETECTED_VIDEO_MAX_MB`` (default 50, matching V4)
    and ``WAGONEYE_DETECTED_VIDEO_CRF`` (default 26).  Never raises: if ffmpeg is
    absent or the encode fails, the un-recompressed file is published rather than
    nothing.
    """
    if not _env_bool("WAGONEYE_COMPRESS_DETECTED_VIDEO", True):
        return local
    out = os.path.join(os.path.dirname(local), f"{cam}_detected_h264.mp4")
    try:
        import cv2
        from train_extraction.video_io import compress_video
        duration = None
        cap = cv2.VideoCapture(local)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
            if fps > 0 and frames > 0:
                duration = frames / fps
        cap.release()
        compress_video(
            local, out, log, duration_sec=duration,
            max_size_mb=_env_float("WAGONEYE_DETECTED_VIDEO_MAX_MB", 50.0),
            crf=int(_env_float("WAGONEYE_DETECTED_VIDEO_CRF", 26)),
        )
        if os.path.isfile(out) and os.path.getsize(out) > 0:
            before = os.path.getsize(local) / 1e6
            after = os.path.getsize(out) / 1e6
            log.info("[DELIVERY] %s overlay compressed %.1f MB -> %.1f MB",
                     cam, before, after)
            return out
    except FileNotFoundError:
        log.warning("[DELIVERY] ffmpeg not found -- publishing %s un-recompressed",
                    cam)
    except Exception as e:
        log.warning("[DELIVERY] %s compression failed (%s) -- publishing "
                    "un-recompressed", cam, e)
    return local


def upload_detected_videos(s3_client, processed_dir: str,
                           batch_key: str) -> Dict[str, str]:
    """Mirror the overlay videos into V4's per-camera `detected_video_bucket`.

    V4 publishes each camera's annotated video to
    ``s3://biro-wagon-processed-video-copy/<camera_folder>/`` and links THAT url
    from the report.  global_train also archives a copy under
    ``train_batch/<key>/processed_videos/`` -- this mirror is what makes the
    per-camera location match V4.

    Returns ``{camera_id -> url}`` for whatever was uploaded.  Never raises: a
    failed mirror is logged and the archive copy still stands.  Set
    ``WAGONEYE_S3_DETECTED_VIDEO_BUCKET=`` (empty) to disable.
    """
    bucket = C.S3_DETECTED_VIDEO_BUCKET
    urls: Dict[str, str] = {}
    if not bucket or s3_client is None or not os.path.isdir(processed_dir):
        return urls
    for cam in C.ALL_CAMERAS:
        local = os.path.join(processed_dir, f"{cam}_processed.mp4")
        if not os.path.isfile(local):
            continue
        # The overlay renderer writes an mp4v-codec file, which is bulky and not
        # broadly browser-playable.  Re-encode to H.264 under a size cap before
        # publishing (V4 does the same via core.video_io.compress_video).  A
        # failure here is non-fatal -- the original is uploaded instead.
        upload_path = _compressed_or_original(local, cam)
        folder = C.CAMERA_S3_FOLDER.get(cam, cam)
        key = f"{folder}/{batch_key}_{cam.lower()}_detected.mp4"
        try:
            s3_client.upload_file(upload_path, bucket, key,
                                  ExtraArgs={"ContentType": "video/mp4"})
            urls[cam] = f"https://{bucket}.s3.{C.S3_REGION}.amazonaws.com/{key}"
        except Exception as e:
            log.warning("[DELIVERY] detected-video mirror failed %s -> s3://%s/%s: %s",
                        upload_path, bucket, key, e)
    if urls:
        log.info("[DELIVERY] mirrored %d detected video(s) to s3://%s/",
                 len(urls), bucket)
    return urls


def upload_tree(
    s3_client, local_dir: str, batch_key: str,
    *, sub_prefix: str = "",
    skip_extensions: Optional[set] = None,
) -> int:
    """Recursively upload everything under `local_dir` to
    s3://<bucket>/<train_batch_prefix>/<batch_key>/<sub_prefix>/...

    Returns the number of files uploaded.
    """
    if not os.path.isdir(local_dir):
        return 0
    bucket = C.S3_OUTPUT_BUCKET
    base   = f"{C.S3_TRAIN_BATCH_PREFIX}/{batch_key}"
    if sub_prefix:
        base = f"{base}/{sub_prefix.strip('/')}"
    skip = skip_extensions or set()

    count = 0
    for root, _, files in os.walk(local_dir):
        for fn in files:
            if any(fn.lower().endswith(ext) for ext in skip):
                continue
            local = os.path.join(root, fn)
            rel   = os.path.relpath(local, local_dir).replace(os.sep, "/")
            key   = f"{base}/{rel}"
            try:
                s3_client.upload_file(
                    local, bucket, key,
                    ExtraArgs={"ContentType": _content_type_for(fn)},
                )
                count += 1
            except Exception as e:
                log.warning("[DELIVERY] upload failed %s -> s3://%s/%s: %s",
                            local, bucket, key, e)
    return count
