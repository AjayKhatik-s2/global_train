"""delivery/dashboard_ingest.py -- Stage-6-only, read-only legacy dashboard adapter.

Purpose
-------
The pre-migration ("old") pipeline fed a per-camera dashboard by POSTing one
``*_inspection.json`` (schema ``{camera_id, version, inspection_data}``) per
camera angle to an S3 bucket and then calling a ``cctv-receiver/inspections/ingest``
API.  The train-state-native v4 pipeline does NOT produce that per-camera feed --
it emits one combined ``combined_train_report.json`` per train.

This module RE-DERIVES the legacy per-camera dashboard payload from finalized v4
artifacts so the existing dashboard keeps working, WITHOUT changing anything
about how the new system computes results.

Hard guarantees (by construction)
---------------------------------
* **Read-only w.r.t. the pipeline.**  It reads finalized artifacts only:
    <batch_root>/reports/combined_train_report.json
    <batch_root>/evidence/<GW>/<feature>/<CAMERA>/{metadata.json,*.jpg}
    <batch_root>/delivery/finalization.json
  It NEVER imports or mutates GlobalTrainState, feature processors, fusion, or
  the report builders, and it NEVER loads a model or opens a video.
* **Writes only under <batch_root>/delivery/.**  Generated JSON goes to
  ``delivery/dashboard/<CAMERA>_inspection.json``; ingest status is merged into
  ``delivery/finalization.json``.  Nothing else on disk is touched.
* **Enabled by default.**  ``WAGONEYE_DASHBOARD_INGEST_ENABLED`` defaults to
  ``true`` -- every finalized batch posts to the live ingest API (version v1).
  Set it to ``false`` to make ``run()`` a no-op (staging / shadow runs).
* **Failure-isolating.**  ``run()`` never raises; any error is logged and
  recorded.  It cannot corrupt the final report or the sealed batch state.
* **Idempotent across restarts.**  Per-camera ingest status (keyed by the
  generated JSON's sha256 + report revision) is persisted; an already-ingested
  camera is skipped on re-entry -- no duplicate uploads, no duplicate ingest.

Degraded fields (documented, never invented)
--------------------------------------------
* ``direction``            -> "unknown" (optical-flow direction is not in any
                              finalized artifact; recompute is out of scope for a
                              read-only adapter).
* ``rake_status``          -> derived from FUSED load results (Loaded/Empty),
                              a measured proxy for the old direction heuristic.
* ``loco_frames`` /
  ``loco_number_results`` /
  ``total_loco_frames``    -> empty (v4 has no loco-specific frame/OCR feed).
* ``wagon_frames`` gallery -> synthesized from whatever per-camera evidence JPEGs
                              exist; only files that are actually present are
                              referenced (never fabricated).

The degraded set for each payload is echoed under ``inspection_data._adapter`` so
a consumer can see exactly what was and was not faithfully reproduced.

NOTE: this posts to the LIVE dashboard on every run.  Confirm with the dashboard
team that (a) reused ``train_batch/.../evidence/...`` HTTPS URLs are accepted and
(b) the degraded loco/direction/gallery fields are acceptable.  Set
``WAGONEYE_DASHBOARD_INGEST_ENABLED=false`` to disable without a code change.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from core import constants as C
from core.logging_setup import get_logger
from delivery import finalization as FIN

log = get_logger("delivery.dashboard")

_IST = timezone(timedelta(hours=5, minutes=30))
_TS_RE = re.compile(r"(\d{8})_(\d{6})")
_DATE_RE = re.compile(r"(\d{8})")

# Local (delivery/) scratch subdir for generated per-camera JSON.
_LOCAL_SUBDIR = os.path.join("delivery", "dashboard")


# -----------------------------------------------------------------------------
# Configuration (all self-contained here; nothing shared is modified).
# Every value defaults to the pre-migration production value and is
# env-overridable so a staging deployment needs no source edit.
# -----------------------------------------------------------------------------

def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_json_map(name: str, default: Dict[str, str]) -> Dict[str, str]:
    """Merge a JSON-object env override over `default` (override wins)."""
    raw = os.getenv(name)
    if not raw:
        return dict(default)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            merged = dict(default)
            merged.update({str(k): str(v) for k, v in parsed.items()})
            return merged
    except (ValueError, TypeError):
        log.warning("[DASHBOARD] ignoring malformed %s (not a JSON object)", name)
    return dict(default)


# Full CCTV camera ids (dashboard primary key).  Defaults taken from the old
# per-camera run_service.py INPUT_BUCKET suffixes.
_DEFAULT_FULL_IDS = {
    C.CAMERA_RIGHT_UP:     "camera_CCTV_HZBN_DHN_2_RIGHT_UP",
    C.CAMERA_LEFT_UP:      "camera_CCTV_HZBN_DHN_1_LEFT_UP",
    C.CAMERA_RIGHT_UP_TOP: "camera_CCTV_HZBN_DHN_5_RIGHT_TOP",
    C.CAMERA_LEFT_UP_TOP:  "camera_CCTV_HZBN_DHN_6_LEFT_TOP",
}

# Legacy dashboard S3 folder (prefix) per camera.  RIGHT_UP="Right_up" is the
# only one confirmed from the old env; the others follow the same convention and
# MUST be confirmed with the dashboard team before enabling.
_DEFAULT_FOLDERS = {
    C.CAMERA_RIGHT_UP:     "Right_up",
    C.CAMERA_LEFT_UP:      "Left_up",
    C.CAMERA_RIGHT_UP_TOP: "Right_up_top",
    C.CAMERA_LEFT_UP_TOP:  "Left_up_top",
}


def is_enabled() -> bool:
    # ON by default: every finalized batch posts the legacy per-camera feed to
    # the dashboard ingest API (version v1).  Set WAGONEYE_DASHBOARD_INGEST_ENABLED=false
    # to turn it off (e.g. staging / shadow runs).
    return _env_bool("WAGONEYE_DASHBOARD_INGEST_ENABLED", True)


def _inspection_bucket() -> str:
    return _env("WAGONEYE_INSPECTION_JSON_BUCKET", "ankit-version-1-prod")


def _ingest_api_url() -> str:
    return _env(
        "WAGONEYE_INSPECTION_INGEST_API_URL",
        "https://ms-pnr-location-notification-api.suvidhaen.com/"
        "cctv-receiver/inspections/ingest",
    )


def _version() -> str:
    # The dashboard chooses its tab from this value: version "v1" -> V1 tab.
    # Override with WAGONEYE_INSPECTION_VERSION (e.g. v2/v3/v4) if needed.
    return _env("WAGONEYE_INSPECTION_VERSION", "v1")


def _model_id() -> str:
    return _env("WAGONEYE_INSPECTION_MODEL_ID", "model-v3")


def _reuse_evidence_urls() -> bool:
    return _env_bool("WAGONEYE_DASHBOARD_REUSE_EVIDENCE_URLS", True)


def full_camera_id(camera: str) -> str:
    return _env_json_map("WAGONEYE_INSPECTION_CAMERA_FULL_IDS",
                         _DEFAULT_FULL_IDS).get(camera, camera)


def folder_for(camera: str) -> str:
    return _env_json_map("WAGONEYE_INSPECTION_FOLDERS",
                         _DEFAULT_FOLDERS).get(camera, C.CAMERA_FOLDER.get(camera, camera))


# -----------------------------------------------------------------------------
# Pure helpers (timestamp / date-folder / URLs) -- fully unit-testable
# -----------------------------------------------------------------------------

def extract_train_timestamp(*texts: Optional[str]) -> Optional[datetime]:
    """First ``YYYYMMDD_HHMMSS`` (or ``YYYYMMDD``) token across `texts`.

    Returns a naive datetime (interpreted as train local/IST wall-clock, exactly
    as the old pipeline treated the filename timestamp)."""
    for t in texts:
        if not t:
            continue
        m = _TS_RE.search(t)
        if m:
            try:
                return datetime.strptime(f"{m.group(1)}_{m.group(2)}", "%Y%m%d_%H%M%S")
            except ValueError:
                pass
        m = _DATE_RE.search(t)
        if m:
            try:
                return datetime.strptime(m.group(1), "%Y%m%d")
            except ValueError:
                pass
    return None


def date_folder(dt: Optional[datetime]) -> str:
    """Operational-day folder with the old 05:00 IST boundary: a train recorded
    before 05:00 lands in the PREVIOUS calendar day's folder."""
    if dt is None:
        dt = datetime.now(_IST)
    shifted = (dt - timedelta(days=1)) if dt.hour < 5 else dt
    return shifted.strftime("%Y-%m-%d")


def evidence_url(output_bucket: str, region: str, batch_key: str,
                 gw: str, feature: str, camera: str, filename: str) -> str:
    """Deterministic HTTPS URL for an evidence JPEG already mirrored to S3 by the
    Stage-6 tree upload (``train_batch/<key>/evidence/...``)."""
    key = (f"{C.S3_TRAIN_BATCH_PREFIX}/{batch_key}/evidence/"
           f"{gw}/{feature}/{camera}/{filename}")
    return f"https://{output_bucket}.s3.{region}.amazonaws.com/{key}"


def _seg_type(classification: Optional[str]) -> str:
    return {
        C.CLASS_ENGINE:    "engine",
        C.CLASS_WAGON:     "wagon",
        C.CLASS_BRAKE_VAN: "brake_van",
    }.get(classification or "", "wagon")


def _door_side(camera: str) -> Optional[str]:
    if camera == C.CAMERA_RIGHT_UP:
        return "right"
    if camera == C.CAMERA_LEFT_UP:
        return "left"
    return None


# -----------------------------------------------------------------------------
# Evidence reads (read-only)
# -----------------------------------------------------------------------------

def _read_meta(evidence_root: str, gw: str, feature: str, camera: str) -> Dict[str, Any]:
    p = os.path.join(evidence_root, gw, feature, camera, "metadata.json")
    if not os.path.isfile(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _evidence_file(evidence_root: str, gw: str, feature: str, camera: str,
                   filename: str) -> Optional[str]:
    p = os.path.join(evidence_root, gw, feature, camera, filename)
    return p if os.path.isfile(p) else None


def _ocr_input_url(url_maker, evidence_root: str, gw: str, camera: str,
                   meta: Dict[str, Any], default_name: str) -> Optional[str]:
    """URL of the image Rekognition was actually asked to read.

    The OCR feature records that filename as `ocr_input_image`; `default_name`
    covers evidence written before that key existed.  Falls back to the legacy
    crop/frame pair so a wagon never loses its thumbnail."""
    for name in (meta.get("ocr_input_image"), default_name,
                 "number_crop.jpg", "best_frame.jpg"):
        if not name:
            continue
        url = url_maker.url(evidence_root, gw, "ocr", camera, str(name))
        if url:
            return url
    return None


class _UrlMaker:
    """Turn a local evidence JPEG into a dashboard-usable HTTPS URL.

    Default (reuse=True): reference the already-uploaded train_batch evidence URL
    -- no extra upload.  reuse=False: copy ONLY the referenced JPEG into the
    legacy inspection bucket and return that URL."""

    def __init__(self, *, s3_client, output_bucket: str, region: str,
                 inspection_bucket: str, batch_key: str, folder: str,
                 date_folder_str: str, reuse: bool, skip_upload: bool):
        self.s3 = s3_client
        self.output_bucket = output_bucket
        self.region = region
        self.inspection_bucket = inspection_bucket
        self.batch_key = batch_key
        self.folder = folder
        self.date_folder = date_folder_str
        self.reuse = reuse
        self.skip_upload = skip_upload

    def url(self, evidence_root: str, gw: str, feature: str, camera: str,
            filename: str) -> Optional[str]:
        local = _evidence_file(evidence_root, gw, feature, camera, filename)
        if local is None:
            return None
        if self.reuse:
            return evidence_url(self.output_bucket, self.region, self.batch_key,
                                gw, feature, camera, filename)
        # copy-only mode: upload just this JPEG to the legacy bucket
        key = (f"{self.folder}/{self.date_folder}/evidence/"
               f"{gw}/{feature}/{camera}/{filename}")
        if self.skip_upload or self.s3 is None:
            return f"https://{self.inspection_bucket}.s3.{self.region}.amazonaws.com/{key}"
        try:
            self.s3.upload_file(local, self.inspection_bucket, key,
                                ExtraArgs={"ContentType": "image/jpeg"})
        except Exception as e:  # pragma: no cover - network path
            log.warning("[DASHBOARD] evidence copy failed %s: %s", key, e)
            return None
        return f"https://{self.inspection_bucket}.s3.{self.region}.amazonaws.com/{key}"


# -----------------------------------------------------------------------------
# Per-camera legacy payload builder (PURE: no I/O beyond reading evidence)
# -----------------------------------------------------------------------------

# Ordered gallery candidates per camera role.
_SIDE_GALLERY = ("door/{side}_best.jpg", "door/{side}_crop.jpg",
                 "ocr/best_frame.jpg", "ocr/number_crop.jpg")
_TOP_GALLERY = ("load/best_frame.jpg", "damage/track_1.jpg",
                "damage/track_2.jpg", "damage/track_3.jpg")
# Side galleries carry four frames, top galleries three (matches the dashboard's
# per-flavour expectation).
_POSITIONS_SIDE = ("start", "mid1", "mid2", "end")
_POSITIONS_TOP = ("start", "mid1", "end")
_POSITIONS = _POSITIONS_SIDE          # back-compat alias

# Canonical door state -> dashboard problem-frame / door_status vocabulary.
_DOOR_PROBLEM_TYPE = {
    C.DOOR_OPEN:    "open_door",
    C.DOOR_CLOSED:  "closed_door",
    C.DOOR_PARTIAL: "partially_closed",
    C.DOOR_DAMAGED: "damage",
}
_DOOR_STATUS = {
    C.DOOR_OPEN:    "open",
    C.DOOR_CLOSED:  "closed",
    C.DOOR_PARTIAL: "partially_closed",
    C.DOOR_DAMAGED: "damage",
}

# v4 damage class names -> the dashboard's short vocabulary.  `floor_dmg_probable`
# has no v4 producer (the damage tracker emits confirmed tracks only) but the key
# is still reported as 0 so the dashboard's shape is stable.
_DAMAGE_CLASS_TO_DASH = {
    "floor_damage":      "floor_dmg",
    "inner_wall_damage": "inner_wall_dmg",
}
_TOP_PROBLEM_TYPES = ("floor_dmg", "inner_wall_dmg", "floor_dmg_probable")
_SIDE_PROBLEM_TYPES = ("damage", "open_door", "closed_door", "partially_closed")


def build_inspection_json(*, camera: str, report_doc: Dict[str, Any],
                          evidence_root: str, url_maker: "_UrlMaker") -> Dict[str, Any]:
    """Build ONE legacy ``{camera_id, version, inspection_data}`` document for
    `camera` from the finalized combined report + this camera's evidence.

    Pure w.r.t. the pipeline: reads only `report_doc` (already loaded) and files
    under `evidence_root`.  Never invents image URLs or numbers."""
    wagons = report_doc.get("wagons", []) or []
    summary = report_doc.get("summary", {}) or {}
    train_meta = report_doc.get("train_metadata", {}) or {}
    report_meta = report_doc.get("report_meta", {}) or {}
    batch_key = report_doc.get("batch_key", "")
    source_urls = train_meta.get("source_video_urls", {}) or {}
    processed_urls = train_meta.get("processed_video_urls", {}) or {}

    side = _door_side(camera)
    is_top = camera in C.TOP_CAMERAS

    src_url = source_urls.get(camera, "")
    raw_video_name = os.path.basename(src_url) if src_url else \
        f"{batch_key}_{C.CAMERA_FOLDER.get(camera, camera)}.mp4"
    ts = extract_train_timestamp(raw_video_name, batch_key)
    upload_ts = (ts or datetime.now(_IST)).strftime("%Y-%m-%dT%H:%M:%S")
    upload_ts_readable = (ts or datetime.now(_IST)).strftime("%d-%m-%Y %H:%M:%S") + " IST"

    # Train-level counts.  total_wagons uses the GLOBAL fused count (authoritative
    # across cameras); door/damage counts are scoped to THIS camera's authority.
    total_wagons = int(summary.get("total_wagons", len(wagons)))
    num_engines = int(summary.get("engine_count", 0))
    loaded = int(summary.get("loaded", 0))
    empty = int(summary.get("empty", 0))
    if loaded == 0 and empty == 0:
        rake_status = "Unknown"
    else:
        rake_status = "Loaded" if loaded >= empty else "Empty"

    num_brakevans = sum(1 for w in wagons
                        if w.get("classification") == C.CLASS_BRAKE_VAN)

    doors_open = doors_closed = doors_partial = 0
    if side:
        state_key = f"{side}_door"
        doors_open = sum(1 for w in wagons if w.get(state_key) == C.DOOR_OPEN)
        doors_closed = sum(1 for w in wagons if w.get(state_key) == C.DOOR_CLOSED)
        doors_partial = sum(1 for w in wagons if w.get(state_key) == C.DOOR_PARTIAL)

    wagon_number_results: Dict[str, Any] = {}
    loco_number_results: Dict[str, Any] = {}
    loco_frames: List[Dict[str, Any]] = []
    segment_type_map: Dict[str, Any] = {}
    wagon_segments: List[Dict[str, Any]] = []
    problem_frames: List[Dict[str, Any]] = []
    # Every key in this camera's flavour is pre-seeded to 0 so the dashboard
    # always receives the same shape, whether or not anything was detected.
    pf_type_counts: Dict[str, int] = {
        t: 0 for t in (_TOP_PROBLEM_TYPES if is_top else _SIDE_PROBLEM_TYPES)}
    damaged_wagons: set = set()
    # Top-camera per-class wagon tallies (distinct wagons, not track counts).
    dmg_class_wagons: Dict[str, set] = {t: set() for t in _TOP_PROBLEM_TYPES}
    wagons_loaded = wagons_empty = 0
    # Running counters used by the top-camera segment_type_map, which numbers
    # each segment WITHIN its own type and tracks a separate wagon ordinal.
    type_ordinal: Dict[str, int] = {}
    wagon_ordinal = 0

    def _bump(t: str) -> None:
        pf_type_counts[t] = pf_type_counts.get(t, 0) + 1

    for w in wagons:
        gw = w.get("global_id", "")
        idx = w.get("wagon_index", 0)
        classification = w.get("classification")
        is_non_wagon = classification in (C.CLASS_ENGINE, C.CLASS_BRAKE_VAN)

        # ---- per-camera load verdict (THIS camera's own evidence, not fused) --
        load_meta = _read_meta(evidence_root, gw, "load", camera) if is_top else {}
        load_state = load_meta.get("load_status") or w.get("load_status")
        if is_top and not is_non_wagon:
            if load_state == C.LOAD_LOADED:
                wagons_loaded += 1
            elif load_state == C.LOAD_EMPTY:
                wagons_empty += 1

        # ---- segment type ----
        if is_top and not is_non_wagon:
            # Top cameras distinguish loaded from empty bodies.
            seg_type = ("wagon_loaded" if load_state == C.LOAD_LOADED else
                        "wagon_empty" if load_state == C.LOAD_EMPTY else "wagon")
        else:
            seg_type = _seg_type(classification)

        if not is_non_wagon:
            wagon_ordinal += 1
        type_ordinal[seg_type] = type_ordinal.get(seg_type, 0) + 1

        if is_top:
            # Top flavour numbers each segment within its own type and carries a
            # separate running wagon ordinal (null for engines / brake vans).
            segment_type_map[str(idx)] = {
                "type": seg_type,
                "number": type_ordinal[seg_type],
                "wagon_count": (None if is_non_wagon else wagon_ordinal),
            }
        else:
            segment_type_map[str(idx)] = {"type": seg_type, "number": idx}

        # ---- Loco number (ENGINE segments, RIGHT_UP authority) ----
        # Read from the engine wagon's OCR evidence, written by the loco branch
        # of features/ocr.  Keyed by loco_id, matching the dashboard contract.
        if camera == C.CAMERA_RIGHT_UP and classification == C.CLASS_ENGINE:
            lm = _read_meta(evidence_root, gw, "ocr", camera)
            if lm.get("segment_role") == "loco":
                lid = str(lm.get("loco_id") or (len(loco_number_results) + 1))
                conf = float(lm.get("ocr_confidence") or 0.0)
                loco_number_results[lid] = {
                    "is_valid_5_digit": bool(lm.get("is_valid_5_digit")),
                    "display_number": lm.get("loco_number") or "-",
                    "raw_number": lm.get("loco_raw_number") or "",
                    "confidence": conf,
                    "ocr_confidence": conf,
                    # The image Rekognition actually read -- normally the
                    # three-frame sheet, the single-frame fallback when the
                    # sheet failed -- so the number can be verified against it.
                    "ocr_frame_s3_url": _ocr_input_url(
                        url_maker, evidence_root, gw, camera, lm,
                        C.OCR_LOCO_SHEET_TEMPLATE.format(
                            loco_id=int(lm.get("loco_id") or 1))),
                }
                for fr in (lm.get("loco_frames") or []):
                    u = url_maker.url(evidence_root, gw, "ocr", camera,
                                      fr.get("filename", ""))
                    if u:
                        loco_frames.append({
                            "loco_id": int(lm.get("loco_id") or 1),
                            "position": fr.get("position"),
                            "frame_number": fr.get("frame_num"),
                            "s3_url": u,
                        })

        # ---- OCR identity (RIGHT_UP only; top cameras have no OCR authority) --
        ident = w.get("wagon_identifier") or C.NO_DATA
        digits = re.sub(r"[^0-9]", "", str(ident)) if ident != C.NO_DATA else ""
        is_valid = len(digits) == C.WAGON_NUMBER_LENGTH
        if camera == C.CAMERA_RIGHT_UP:
            ocr_meta = _read_meta(evidence_root, gw, "ocr", camera)
            wagon_number_results[str(idx)] = {
                "is_valid_11_digit": bool(is_valid),
                "display_number": digits if digits else "-",
                "is_manipulated": bool(ocr_meta.get("is_manipulated", False)),
                "original_number": str(ocr_meta.get("original_number")
                                       or (digits if is_valid else "")),
                # Points at number_sheet.jpg -- the exact three-frame sheet
                # posted to Rekognition -- so the displayed number can be
                # verified against the image the engine read.
                "ocr_frame_s3_url": _ocr_input_url(
                    evidence_root=evidence_root, url_maker=url_maker, gw=gw,
                    camera=camera, meta=ocr_meta,
                    default_name=C.OCR_SHEET_FILENAME),
            }

        # ---- wagon gallery (synthesized from EXISTING evidence only) ----
        positions = _POSITIONS_TOP if is_top else _POSITIONS_SIDE
        templates = (_TOP_GALLERY if is_top
                     else tuple(t.format(side=side) for t in _SIDE_GALLERY))
        frames: List[Dict[str, Any]] = []
        for rel in templates:
            feat, fn = rel.split("/", 1)
            u = url_maker.url(evidence_root, gw, feat, camera, fn)
            if u:
                frames.append({"position": positions[min(len(frames),
                                                         len(positions) - 1)],
                               "s3_url": u})
            if len(frames) >= len(positions):
                break

        seg: Dict[str, Any] = {
            "segment_id": idx,
            "segment_type": seg_type,
            "wagon_count": (wagon_ordinal if is_top else idx),
            "is_valid_wagon_id": bool(is_valid) if side else False,
            "damage_detected": False,
            "wagon_frames": frames,
        }

        if is_top:
            seg.update({
                "load_status": (str(load_state).lower()
                                if load_state and load_state != C.NO_DATA else None),
                # No v4 producer for a load-condition grade -- reported as null
                # rather than guessed.
                "load_condition": None,
                "probable_damage_detected": False,
                "floor_dmg_detected": False,
                "inner_wall_dmg_detected": False,
                "floor_dmg_probable_detected": False,
            })
        else:
            dstate = w.get(f"{side}_door") if side else None
            seg.update({
                "door_status": _DOOR_STATUS.get(dstate, "N/A") if side else "N/A",
                "door_close_detected": dstate == C.DOOR_CLOSED,
                "door_partial_detected": dstate == C.DOOR_PARTIAL,
            })
            if is_valid:
                seg["wagon_number"] = digits

        # ---- problem frames scoped to this camera's authority ----
        if side:
            meta = _read_meta(evidence_root, gw, "door", camera)
            side_meta = (meta.get("sides") or {}).get(side, {})
            dstate = w.get(f"{side}_door")
            ptype = _DOOR_PROBLEM_TYPE.get(dstate)
            if ptype:
                # Every observed door state is reported, not just anomalies --
                # the dashboard tallies closed/partial alongside open/damage.
                _bump(ptype)
                is_damage = dstate == C.DOOR_DAMAGED
                if is_damage:
                    damaged_wagons.add(idx)
                    seg["damage_detected"] = True
                problem_frames.append(_problem_frame(
                    idx=idx, gw=gw, camera=camera, evidence_root=evidence_root,
                    url_maker=url_maker, feature="door", img=f"{side}_best.jpg",
                    problem_type=ptype,
                    class_name=str(side_meta.get("raw_class") or ptype),
                    bbox=side_meta.get("bbox"), conf=side_meta.get("confidence"),
                    door_status=_DOOR_STATUS.get(dstate, "N/A"), damage=is_damage))

        if is_top:
            dmeta = _read_meta(evidence_root, gw, "damage", camera)
            for tr in (dmeta.get("tracks") or []):
                ti = tr.get("track_idx", 1)
                cls = str(tr.get("class_name") or "damage")
                dash = _DAMAGE_CLASS_TO_DASH.get(cls, cls)
                if dash in pf_type_counts:
                    _bump(dash)
                if dash in dmg_class_wagons:
                    dmg_class_wagons[dash].add(idx)
                damaged_wagons.add(idx)
                seg["damage_detected"] = True
                if dash == "floor_dmg":
                    seg["floor_dmg_detected"] = True
                elif dash == "inner_wall_dmg":
                    seg["inner_wall_dmg_detected"] = True
                problem_frames.append(_problem_frame(
                    idx=idx, gw=gw, camera=camera, evidence_root=evidence_root,
                    url_maker=url_maker, feature="damage", img=f"track_{ti}.jpg",
                    problem_type=dash, class_name=cls,
                    bbox=tr.get("bbox"),
                    conf=tr.get("best_confidence", tr.get("confidence")),
                    door_status="N/A", damage=True))

        # Top cameras list only wagon bodies; engines / brake vans appear in
        # segment_type_map but carry no inspectable body.
        if not (is_top and is_non_wagon):
            wagon_segments.append(seg)

    damaged_count = len(damaged_wagons) if (is_top or side) else 0

    degraded = ["direction", "raw_video_urls"]
    if camera != C.CAMERA_RIGHT_UP:
        # Loco OCR is RIGHT_UP-authority, like the wagon number.
        degraded += ["loco_frames", "loco_number_results", "total_loco_frames"]
    if is_top:
        # The v4 damage tracker emits confirmed tracks only -- there is no
        # "probable" tier to report, so those counters are structurally 0.
        degraded += ["probable_damage_wagons", "floor_dmg_probable_wagons",
                     "load_condition"]
    if not is_top and not side:
        degraded.append("doors_open/doors_partially_closed/doors_closed")

    inspection_data = {
        "raw_video_name": raw_video_name,
        "identified_by": _model_id(),
        "upload_timestamp": upload_ts,
        "upload_timestamp_readable": upload_ts_readable,
        "direction": "unknown",                       # DEGRADED (see module docstring)
        "rake_status": rake_status,                   # DEGRADED: fused load proxy
        "pdf_report_url": _pdf_url(report_meta, camera),
        "trimmed_video_url": src_url,
        "detected_video_url": processed_urls.get(camera, ""),
        # DEGRADED: v4 consumes one trimmed clip per camera and does not retain
        # the list of raw clips it was cut from.
        "raw_video_urls": [src_url] if src_url else [],
        "total_wagons": total_wagons,
        "damaged_wagons": damaged_count,
        "num_engines": num_engines,
        "total_loco_frames": len(loco_frames),
        "total_problem_frames": len(problem_frames),
        "problem_frames_by_type": pf_type_counts,
        "wagon_number_results": wagon_number_results,
        "loco_number_results": loco_number_results,
        "segment_type_map": segment_type_map,
        "wagon_segments": wagon_segments,
        "loco_frames": loco_frames,
        "problem_frames": problem_frames,
        "_adapter": {
            "generated_by": "wagon_eye_v4 delivery.dashboard_ingest",
            "source": "combined_train_report.json",
            "report_revision": report_meta.get("report_revision", 0),
            "report_status": report_meta.get("report_status", ""),
            "global_state_version":
                report_meta.get("generated_from_global_state_version", ""),
            "camera_authority": ("top:load+damage" if is_top
                                 else (f"side:{side}_door+ocr" if side else "none")),
            "degraded_fields": degraded,
            "flavour": "top" if is_top else ("side" if side else "none"),
        },
    }

    # Flavour-specific counters.  Side cameras report door tallies; top cameras
    # report load + per-damage-class tallies.  Keys are inserted next to
    # `total_wagons` so the document reads in the dashboard's field order.
    if is_top:
        _insert_after(inspection_data, "total_wagons", [
            ("wagons_loaded", wagons_loaded),
            ("wagons_empty", wagons_empty),
        ])
        _insert_after(inspection_data, "damaged_wagons", [
            ("probable_damage_wagons", 0),            # DEGRADED: no probable tier
            ("floor_dmg_wagons", len(dmg_class_wagons["floor_dmg"])),
            ("inner_wall_dmg_wagons", len(dmg_class_wagons["inner_wall_dmg"])),
            ("floor_dmg_probable_wagons", 0),         # DEGRADED
        ])
        _insert_after(inspection_data, "num_engines",
                      [("num_brakevans", num_brakevans)])
    else:
        _insert_after(inspection_data, "total_wagons", [
            ("doors_open", doors_open),
            ("doors_partially_closed", doors_partial),
            ("doors_closed", doors_closed),
        ])

    return {
        "camera_id": full_camera_id(camera),
        "version": _version(),
        "inspection_data": inspection_data,
    }


def _insert_after(d: Dict[str, Any], anchor: str,
                  pairs: List[tuple]) -> None:
    """Insert `pairs` immediately after `anchor`, preserving dict order.

    Purely cosmetic: the dashboard reads by key, but keeping the emitted field
    order stable makes the JSON diffable against the reference documents."""
    if anchor not in d:
        d.update(dict(pairs))
        return
    items = list(d.items())
    out: List[tuple] = []
    for k, v in items:
        out.append((k, v))
        if k == anchor:
            out.extend(pairs)
    d.clear()
    d.update(out)


def _problem_frame(*, idx, gw, camera, evidence_root, url_maker, feature, img,
                   problem_type, class_name, bbox, conf, door_status, damage):
    u = url_maker.url(evidence_root, gw, feature, camera, img)
    coords = list(bbox)[:4] if isinstance(bbox, (list, tuple)) and len(bbox) >= 4 \
        else [0, 0, 0, 0]
    return {
        "wagon_count": idx, "segment_type": "wagon", "segment_number": idx,
        "problem_type": problem_type, "frame_number": 0,
        "filename": f"{gw}_{camera}_{img}",
        "s3_url": u,
        "is_annotated": True,
        "annotated_image_url": u,
        "bounding_box": {
            "bounding_box_coordinates": coords,
            "confidence": round(float(conf), 3) if conf is not None else 0.0,
            "class_name": class_name,
        },
        "door_status": door_status,
        "door_close_detected": False,
        "damage_detected": bool(damage),
    }


def _pdf_url(report_meta: Dict[str, Any], camera: str) -> str:
    # Prefer a per-camera PDF url if the finalization marker carried one; the
    # caller injects finalization upload_urls into report_meta before building.
    urls = report_meta.get("_upload_urls", {}) or {}
    return urls.get(f"camera_{camera}") or urls.get("pdf") or ""


# -----------------------------------------------------------------------------
# Ingest (HTTP) with retries -- mirrors the old ingest loop
# -----------------------------------------------------------------------------

def ingest_idempotency_key(batch_key: str, camera: str, report_revision: int,
                           json_sha256: str) -> str:
    raw = f"{batch_key}|{camera}|{report_revision}|{json_sha256}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _post_ingest(*, api_url: str, payload: Dict[str, Any], idem_key: str,
                 max_retries: int = 3, base_delay: float = 15.0,
                 requests_mod=None) -> Dict[str, Any]:
    """POST once (with retries).  Returns {ok, status_code, run_id, error}.

    Retries only on >=500 (transient); 422 is treated as a permanent validation
    failure (no retry).  Never raises."""
    if requests_mod is None:  # pragma: no cover - exercised via injection in tests
        import requests as requests_mod  # type: ignore
    headers = {"Idempotency-Key": idem_key}
    body = dict(payload, idempotency_key=idem_key)
    delay = base_delay
    last: Dict[str, Any] = {"ok": False, "status_code": None, "run_id": None,
                            "error": "not_attempted"}
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests_mod.post(api_url, json=body, headers=headers, timeout=60)
            code = getattr(resp, "status_code", None)
            if code == 200:
                data = {}
                try:
                    data = resp.json()
                except Exception:
                    pass
                return {"ok": True, "status_code": 200,
                        "run_id": data.get("run_id"), "error": None}
            if code == 422:
                txt = ""
                try:
                    txt = resp.text[:300]
                except Exception:
                    pass
                return {"ok": False, "status_code": 422, "run_id": None,
                        "error": f"validation: {txt}"}
            last = {"ok": False, "status_code": code, "run_id": None,
                    "error": f"http_{code}"}
            if code is not None and code < 500:
                return last  # non-retryable client error
        except Exception as e:  # network/timeout -> retryable
            last = {"ok": False, "status_code": None, "run_id": None,
                    "error": str(e)}
        if attempt < max_retries:
            time.sleep(delay)
            delay *= 2
    return last


# -----------------------------------------------------------------------------
# finalization.json per-camera status (idempotency ledger)
# -----------------------------------------------------------------------------

_DASH_KEY = "dashboard_ingested"


def _load_status(batch_root: str) -> Dict[str, Any]:
    marker = FIN.load(batch_root) or {}
    return dict(marker.get(_DASH_KEY) or {})


def _record_status(batch_root: str, camera: str, entry: Dict[str, Any]) -> None:
    marker = FIN.load(batch_root) or {}
    block = dict(marker.get(_DASH_KEY) or {})
    block[camera] = entry
    marker[_DASH_KEY] = block
    FIN.write(batch_root, marker)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# -----------------------------------------------------------------------------
# Public entry
# -----------------------------------------------------------------------------

def run(*, batch_root: str, s3_client=None, skip_upload: bool = False,
        skip_ingest: Optional[bool] = None, requests_mod=None) -> Dict[str, Any]:
    """Generate + (optionally) deliver the legacy per-camera dashboard feed.

    NEVER raises.  Returns a summary dict.  A no-op (returns {'enabled': False})
    unless WAGONEYE_DASHBOARD_INGEST_ENABLED is truthy.

    skip_upload=True (shadow/dry-run) -> build + record locally, do NOT upload
    JSON and do NOT POST ingest.  skip_ingest defaults to skip_upload."""
    result: Dict[str, Any] = {"enabled": is_enabled(), "cameras": {}}
    if not is_enabled():
        return result
    if skip_ingest is None:
        skip_ingest = skip_upload
    try:
        return _run_inner(batch_root=batch_root, s3_client=s3_client,
                          skip_upload=skip_upload, skip_ingest=skip_ingest,
                          requests_mod=requests_mod, result=result)
    except Exception as e:  # absolute isolation: never propagate
        log.error("[DASHBOARD] ingest aborted (non-fatal): %s", e)
        result["error"] = str(e)
        return result


def _run_inner(*, batch_root, s3_client, skip_upload, skip_ingest,
               requests_mod, result) -> Dict[str, Any]:
    # reports/ is the fixed finalized-artifact location (core.config.DIR_REPORTS);
    # hardcoded here to keep the adapter decoupled from config internals.
    report_path = os.path.join(batch_root, "reports", "combined_train_report.json")
    if not os.path.isfile(report_path):
        log.warning("[DASHBOARD] no combined_train_report.json -- nothing to ingest")
        result["error"] = "no_report"
        return result
    with open(report_path, "r", encoding="utf-8") as f:
        report_doc = json.load(f)

    report_meta = report_doc.get("report_meta", {}) or {}
    # inject finalization upload_urls so per-camera pdf urls resolve
    fin_marker = FIN.load(batch_root) or {}
    report_meta = dict(report_meta)
    report_meta["_upload_urls"] = fin_marker.get("upload_urls", {}) or {}
    report_doc = dict(report_doc, report_meta=report_meta)
    report_revision = int(report_meta.get("report_revision", 0))

    present = report_meta.get("cameras_present") or [
        c for c in C.ALL_CAMERAS
        if c in {w0 for w in report_doc.get("wagons", [])
                 for w0 in (w.get("supporting_cameras") or [])}
    ]
    present = [c for c in C.ALL_CAMERAS if c in present]  # canonical order

    evidence_root = os.path.join(batch_root, "evidence")
    local_dir = os.path.join(batch_root, _LOCAL_SUBDIR)
    os.makedirs(local_dir, exist_ok=True)

    output_bucket = C.S3_OUTPUT_BUCKET
    region = C.S3_REGION
    inspection_bucket = _inspection_bucket()
    api_url = _ingest_api_url()
    reuse = _reuse_evidence_urls()

    batch_key = report_doc.get("batch_key", "")
    ts = extract_train_timestamp(batch_key)
    df = date_folder(ts)

    prior = _load_status(batch_root)

    for camera in present:
        url_maker = _UrlMaker(
            s3_client=s3_client, output_bucket=output_bucket, region=region,
            inspection_bucket=inspection_bucket, batch_key=batch_key,
            folder=folder_for(camera), date_folder_str=df,
            reuse=reuse, skip_upload=skip_upload)
        try:
            doc = build_inspection_json(camera=camera, report_doc=report_doc,
                                        evidence_root=evidence_root,
                                        url_maker=url_maker)
        except Exception as e:
            log.error("[DASHBOARD] build failed for %s: %s", camera, e)
            result["cameras"][camera] = {"status": "build_failed", "error": str(e)}
            continue

        raw_video_name = doc["inspection_data"]["raw_video_name"]
        json_name = f"{os.path.splitext(raw_video_name)[0]}_inspection.json"
        text = json.dumps(doc, indent=2, default=str)
        json_sha = _sha256_text(text)
        idem = ingest_idempotency_key(batch_key, camera, report_revision, json_sha)

        # ---- idempotency: already ingested this exact payload? ----
        pj = prior.get(camera) or {}
        if pj.get("status") == "ingested" and pj.get("json_sha256") == json_sha:
            log.info("[DASHBOARD] %s already ingested (rev=%s) -- skip",
                     camera, report_revision)
            result["cameras"][camera] = {"status": "already_ingested",
                                         "run_id": pj.get("run_id")}
            continue

        # ---- write local JSON (delivery/ only) ----
        local_json = os.path.join(local_dir, json_name)
        with open(local_json, "w", encoding="utf-8") as f:
            f.write(text)

        folder = folder_for(camera)
        s3_key = f"{folder}/{df}/{json_name}"
        s3_uri = f"s3://{inspection_bucket}/{s3_key}"

        entry = {
            "camera_id": full_camera_id(camera),
            "json_sha256": json_sha,
            "idempotency_key": idem,
            "report_revision": report_revision,
            "s3_uri": s3_uri,
            "run_id": None,
            "status": "prepared",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }

        # ---- upload JSON ----
        if skip_upload or s3_client is None:
            entry["status"] = "prepared_local_only"
        else:
            try:
                s3_client.upload_file(local_json, inspection_bucket, s3_key,
                                      ExtraArgs={"ContentType": "application/json"})
                entry["status"] = "uploaded"
            except Exception as e:
                log.error("[DASHBOARD] JSON upload failed %s: %s", s3_uri, e)
                entry["status"] = "upload_failed"
                entry["error"] = str(e)
                _record_status(batch_root, camera, entry)
                result["cameras"][camera] = {"status": entry["status"]}
                continue

        # ---- ingest POST ----
        if skip_ingest:
            entry["status"] = "prepared" if entry["status"] == "prepared_local_only" \
                else entry["status"]
            _record_status(batch_root, camera, entry)
            result["cameras"][camera] = {"status": entry["status"], "dry_run": True}
            continue

        payload = {"camera_id": full_camera_id(camera),
                   "inspection_s3_uri": s3_uri, "version": _version()}
        res = _post_ingest(api_url=api_url, payload=payload, idem_key=idem,
                           requests_mod=requests_mod)
        if res["ok"]:
            entry["status"] = "ingested"
            entry["run_id"] = res.get("run_id")
        else:
            entry["status"] = "ingest_failed"
            entry["error"] = res.get("error")
            entry["last_status_code"] = res.get("status_code")
        _record_status(batch_root, camera, entry)
        result["cameras"][camera] = {"status": entry["status"],
                                     "run_id": entry.get("run_id")}

    return result
