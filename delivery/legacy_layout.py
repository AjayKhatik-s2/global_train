"""Stage 6 -- reproduce the OLD PRODUCTION S3 output layout + filenames.

The old 4-instance pipeline wrote, at the OUTPUT BUCKET ROOT:

    camera_reports/<CAMERA>/<DD-MM-YYYY>_<NNN>.pdf
    camera_reports/<CAMERA>/<DD-MM-YYYY>_<NNN>_data.json
    camera_reports/<CAMERA>/<DD-MM-YYYY>_<NNN>_meta.json
    camera_reports/<CAMERA>/<DD-MM-YYYY>_<NNN>_snapshot_<i>.jpg       (side open/damage)
    camera_reports/<CAMERA>/<DD-MM-YYYY>_<NNN>_dmg_snapshot_<i>.jpg   (top damage)
    camera_reports/<CAMERA>/<DD-MM-YYYY>_<NNN>_tracked.mp4            (processed video)
    combined_reports/<DD-MM-YYYY>_<NNN>_combined.pdf
    camera_reports/<CAMERA>/.counter_<DD-MM-YYYY>.json               (daily counter)

This module builds that exact tree LOCALLY (so `--local-only` runs are directly
comparable to old production output) and uploads it to S3 under the same
bucket-root keys.  The per-camera `_data.json` is the same serialized `data`
dict the old pipeline produced (the `_legacy_data_adapter` payload, JSON-safed);
`_meta.json` carries the same marker fields (single-process, so `merged=true`).

Only the S3 KEY SCHEME + filenames are the old ones; the file CONTENTS are the
byte-identical old-generator outputs produced upstream in Stage 5.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from core import constants as C
from core.logging_setup import get_logger

log = get_logger("delivery.legacy_layout")

IST = timezone(timedelta(hours=5, minutes=30))

CAMERA_REPORT_PREFIX = "camera_reports"
COMBINED_REPORT_PREFIX = "combined_reports"

# Per-camera processed-video + report basenames produced upstream (Stage 4b/5).
_CAMERA_FOLDER_LOWER = {
    C.CAMERA_RIGHT_UP:     "right_up",
    C.CAMERA_LEFT_UP:      "left_up",
    C.CAMERA_RIGHT_UP_TOP: "right_up_top",
    C.CAMERA_LEFT_UP_TOP:  "left_up_top",
}


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------

def date_str_from_batch_key(batch_key: str) -> str:
    """`20260724_204400` -> `24-07-2026` (old DD-MM-YYYY train-date folder)."""
    m = re.match(r"(\d{4})(\d{2})(\d{2})", batch_key or "")
    if m:
        y, mo, d = m.groups()
        return f"{d}-{mo}-{y}"
    return datetime.now(IST).strftime("%d-%m-%Y")


def report_name(date_str: str, counter: int) -> str:
    return f"{date_str}_{counter:03d}"


def next_counter(s3_client, date_str: str, local_root: str) -> int:
    """Increment the daily counter, mirroring old `get_next_report_counter`.

    Prefers the shared S3 counter (so numbering is continuous across restarts);
    falls back to a local counter file for offline / --local-only runs.
    """
    counter_key = f"{CAMERA_REPORT_PREFIX}/.counter_{date_str}.json"
    if s3_client is not None and C.S3_OUTPUT_BUCKET:
        try:
            resp = s3_client.get_object(Bucket=C.S3_OUTPUT_BUCKET, Key=counter_key)
            data = json.loads(resp["Body"].read().decode("utf-8"))
            counter = int(data.get("counter", 0)) + 1
        except Exception:
            counter = 1
        try:
            s3_client.put_object(
                Bucket=C.S3_OUTPUT_BUCKET, Key=counter_key,
                Body=json.dumps({"counter": counter, "date": date_str}).encode(),
                ContentType="application/json")
        except Exception as e:
            log.warning("[LEGACY] counter put failed: %s", e)
        return counter
    # local fallback
    local_counter = os.path.join(local_root, f".counter_{date_str}.json")
    counter = 1
    try:
        with open(local_counter, "r", encoding="utf-8") as f:
            counter = int(json.load(f).get("counter", 0)) + 1
    except Exception:
        counter = 1
    try:
        with open(local_counter, "w", encoding="utf-8") as f:
            json.dump({"counter": counter, "date": date_str}, f)
    except OSError:
        pass
    return counter


# ---------------------------------------------------------------------------
# JSON-safe per-camera data.json (matches old `_serialize_wagon_data` shape)
# ---------------------------------------------------------------------------

def _json_safe_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize an adapter payload to the old `_data.json` shape.

    - `wagon_summary[].ocr_wagon_number`: object -> full_number string | None
    - door/damage entries keep their fields; `local_snapshot_path` is replaced
      by `snapshot_<i>` bookkeeping by the caller after images are staged.
    """
    out: Dict[str, Any] = {}
    summary = []
    for w in payload.get("wagon_summary", []):
        sw = dict(w)
        ocr = sw.get("ocr_wagon_number")
        if ocr is not None and hasattr(ocr, "full_number"):
            sw["ocr_wagon_number"] = ocr.full_number if getattr(ocr, "is_valid", False) else ocr.full_number
        elif ocr is not None and not isinstance(ocr, (str, type(None))):
            sw["ocr_wagon_number"] = str(ocr)
        summary.append(sw)
    out["wagon_summary"] = summary

    key = "damages" if "damages" in payload else "doors"
    entries = []
    for d in payload.get(key, []):
        sd = {k: v for k, v in d.items() if k not in ("snapshot",)}
        entries.append(sd)
    out[key] = entries
    out["state_counts"] = payload.get("state_counts", {})
    out["source_video_url"] = payload.get("source_video_url")
    out["main_report_url"] = payload.get("main_report_url")
    out["tracked_video_url"] = payload.get("tracked_video_url")
    out["session_id"] = payload.get("session_id")
    return out


def _copy(src: Optional[str], dst: str) -> bool:
    if src and os.path.isfile(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        return True
    return False


# ---------------------------------------------------------------------------
# Build the old-structure tree locally
# ---------------------------------------------------------------------------

def build_local_layout(
    *,
    out_root: str,
    batch_key: str,
    date_str: str,
    counter: int,
    payloads: Dict[str, Dict[str, Any]],
    camera_pdf_paths: Dict[str, str],
    processed_root: str,
    combined_pdf_path: Optional[str],
    session_id: str,
    missing_cameras: List[str],
) -> Dict[str, str]:
    """Write the old bucket-root layout under `out_root`.  Returns
    {relative_s3_key -> local_path} for every file (used to upload)."""
    rname = report_name(date_str, counter)
    files: Dict[str, str] = {}

    for cam in C.ALL_CAMERAS:
        folder = _CAMERA_FOLDER_LOWER[cam]
        cam_dir = os.path.join(out_root, CAMERA_REPORT_PREFIX, cam)
        os.makedirs(cam_dir, exist_ok=True)
        payload = payloads.get(cam, {}) or {}

        # PDF
        pdf_local = os.path.join(cam_dir, f"{rname}.pdf")
        if _copy(camera_pdf_paths.get(cam), pdf_local):
            files[f"{CAMERA_REPORT_PREFIX}/{cam}/{rname}.pdf"] = pdf_local

        # snapshots (open-door for side; damage for top) + rewrite data.json refs
        data = _json_safe_payload(payload)
        snap_prefix = "dmg_snapshot" if cam in C.TOP_CAMERAS else "snapshot"
        entries = data.get("damages" if cam in C.TOP_CAMERAS else "doors", [])
        snap_idx = 0
        for d in entries:
            src = d.get("local_snapshot_path")
            state = str(d.get("state", "")).lower()
            keep = (cam in C.TOP_CAMERAS) or ("open" in state and "partial" not in state) \
                or ("damage" in state)
            if keep and src and os.path.isfile(src):
                snap_name = f"{rname}_{snap_prefix}_{snap_idx}.jpg"
                snap_local = os.path.join(cam_dir, snap_name)
                _copy(src, snap_local)
                d["snapshot_s3_key"] = f"{CAMERA_REPORT_PREFIX}/{cam}/{snap_name}"
                files[d["snapshot_s3_key"]] = snap_local
                snap_idx += 1
            d.pop("local_snapshot_path", None)

        # data.json
        data_local = os.path.join(cam_dir, f"{rname}_data.json")
        with open(data_local, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        files[f"{CAMERA_REPORT_PREFIX}/{cam}/{rname}_data.json"] = data_local

        # meta.json (single-process: always merged=true, no partner polling)
        meta = {
            "camera_id": cam, "report_name": rname, "session_id": session_id,
            "batch_key": batch_key, "date": date_str, "counter": counter,
            "merged": True, "merged_at": datetime.now(IST).isoformat(),
            "present": cam not in missing_cameras,
        }
        meta_local = os.path.join(cam_dir, f"{rname}_meta.json")
        with open(meta_local, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        files[f"{CAMERA_REPORT_PREFIX}/{cam}/{rname}_meta.json"] = meta_local

        # processed/tracked video: <CAMERA>_processed.mp4 -> {rname}_tracked.mp4
        mp4_src = os.path.join(processed_root, f"{cam}_processed.mp4")
        mp4_local = os.path.join(cam_dir, f"{rname}_tracked.mp4")
        if _copy(mp4_src, mp4_local):
            files[f"{CAMERA_REPORT_PREFIX}/{cam}/{rname}_tracked.mp4"] = mp4_local

    # combined report
    if combined_pdf_path and os.path.isfile(combined_pdf_path):
        comb_dir = os.path.join(out_root, COMBINED_REPORT_PREFIX)
        os.makedirs(comb_dir, exist_ok=True)
        comb_local = os.path.join(comb_dir, f"{rname}_combined.pdf")
        _copy(combined_pdf_path, comb_local)
        files[f"{COMBINED_REPORT_PREFIX}/{rname}_combined.pdf"] = comb_local

    return files


# ---------------------------------------------------------------------------
# Upload the built tree to S3 under old bucket-root keys
# ---------------------------------------------------------------------------

_CT = {".pdf": "application/pdf", ".json": "application/json",
       ".jpg": "image/jpeg", ".mp4": "video/mp4"}


def upload_layout(s3_client, files: Dict[str, str]) -> Dict[str, str]:
    """Upload {key -> local_path} to the output bucket root.  Returns
    {key -> https url} for successful uploads."""
    urls: Dict[str, str] = {}
    if s3_client is None or not C.S3_OUTPUT_BUCKET:
        return urls
    for key, local in files.items():
        if not os.path.isfile(local):
            continue
        ct = _CT.get(os.path.splitext(local)[1].lower(), "application/octet-stream")
        try:
            s3_client.upload_file(local, C.S3_OUTPUT_BUCKET, key,
                                  ExtraArgs={"ContentType": ct})
            urls[key] = f"https://{C.S3_OUTPUT_BUCKET}.s3.{C.S3_REGION}.amazonaws.com/{key}"
        except Exception as e:
            log.warning("[LEGACY] upload failed %s: %s", key, e)
    return urls


# ---------------------------------------------------------------------------
# One-call orchestration (build tree -> upload) used by stage_finalize
# ---------------------------------------------------------------------------

def deliver(
    *,
    s3_client,
    batch_root: str,
    batch_key: str,
    state,
    unified: Dict[str, Any],
    evidence_root: Optional[str],
    wagon_states_root: Optional[str],
    per_camera_tracking_path: Optional[str],
    processed_root: str,
    camera_pdf_paths: Dict[str, str],
    combined_pdf_path: Optional[str],
    source_video_urls: Optional[Dict[str, str]] = None,
    missing_cameras: Optional[List[str]] = None,
    counter: Optional[int] = None,
    skip_upload: bool = False,
) -> Dict[str, Any]:
    """Build the old-structure output tree locally and (unless skip_upload)
    upload it to S3 under the old bucket-root keys.

    `counter` may be supplied to reuse a previously assigned daily counter
    (idempotent re-entry); when None a fresh counter is claimed.
    Returns metadata incl. report_name, counter, combined_url, and local_root.
    """
    from reporting import _legacy_data_adapter as LDA  # local import: avoid cycle

    date_str = date_str_from_batch_key(batch_key)
    if counter is None:
        counter = next_counter(None if skip_upload else s3_client, date_str, batch_root)

    payloads = LDA.build_camera_payloads(
        state=state, unified=unified,
        evidence_root=evidence_root, wagon_states_root=wagon_states_root,
        per_camera_tracking_path=per_camera_tracking_path,
        source_video_urls=source_video_urls or {}, session_id=batch_key,
    )

    out_root = os.path.join(batch_root, "legacy_output")
    files = build_local_layout(
        out_root=out_root, batch_key=batch_key, date_str=date_str, counter=counter,
        payloads=payloads, camera_pdf_paths=camera_pdf_paths,
        processed_root=processed_root, combined_pdf_path=combined_pdf_path,
        session_id=batch_key, missing_cameras=list(missing_cameras or []),
    )
    urls = {} if skip_upload else upload_layout(s3_client, files)
    rname = report_name(date_str, counter)
    combined_key = f"{COMBINED_REPORT_PREFIX}/{rname}_combined.pdf"
    log.info("[LEGACY] delivered %s (%d files, %d uploaded) -> %s",
             rname, len(files), len(urls), out_root)
    return {
        "report_name": rname, "counter": counter, "date_str": date_str,
        "files": files, "urls": urls, "combined_url": urls.get(combined_key),
        "local_root": out_root,
    }
