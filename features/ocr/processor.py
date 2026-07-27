"""OCR feature processor -- wagon numbers via AWS Rekognition.

AUTHORITY: RIGHT_UP only.  `core.camera_features` registers ocr as RIGHT_UP
"sole", the scheduler scopes it to RIGHT_UP, and Stage-4 fusion takes
`wagon_identifier` from RIGHT_UP.  A call scoped to any other camera is a no-op.

This module is a thin ADAPTER.  All OCR logic lives in
`features.inference_lib.ocr`, which mirrors the V4 Train-Inspection-Engine
`inspection/ocr/` package (detector -> bands -> three-frame sheet ->
Rekognition DetectText -> best valid LINE).  Per wagon this costs at most two
DetectText calls -- one per sheet -- not one per frame.

The adapter's own responsibilities:
  * point the upstream detector at this pipeline's wagon cache directory
    (`wagon_cache/<GW_n>/right_up/`, already `frame_%06d.jpg`);
  * supply the wagon's load state, which selects the sheet triplet;
  * apply this pipeline's wagon-type correction (C1-C2 into 10-39), which
    upstream does not have and which produces `is_manipulated` for the
    dashboard feed;
  * persist per-wagon JSON + evidence in this pipeline's layout.

Output JSON shape:
    {
        "global_id":  "GW_7",
        "feature":    "ocr",
        "status":     "OK" | "NO_FRAMES" | "FAILED" | "NO_DATA",
        "wagon_identifier":  "32145678901",
        "wagon_identifier_confidence": 0.83,
        "is_manipulated":  false,
        "original_number": "32145678901",
        "candidates":  [...],
        "supporting_cameras": ["RIGHT_UP"],
        "frame_count": ...,
    }
"""

from __future__ import annotations

import json
import os
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import cv2

from core import constants as C
from core.global_state_loader import GlobalTrainState
from core.logging_setup import get_logger
from core.rekognition import RekognitionClient

from features._common import (
    list_wagon_frames, write_per_wagon_json, empty_payload,
    FeatureTimer, feature_camera_dir, phase,
)
from features.inference_lib.ocr import (
    WagonNumberDetector, WagonImageEnhancer, WagonNumberOCR, LocoNumberOCR,
)
from features.ocr.loco_frames import extract_loco_candidates
from features._evidence import (
    atomic_camera_evidence, save_jpeg, write_metadata,
)


FEATURE_NAME = "ocr"
log = get_logger("features.ocr")

# Wagon-type correction (C1-C2 must fall in 10-39).  OCR commonly misreads the
# FIRST digit on weathered stencilled plates; each mapping pulls an
# out-of-range type back into the valid band.  Upstream has no equivalent --
# this is what makes `is_manipulated` meaningful on the dashboard feed.
_VALID_TYPE_MIN, _VALID_TYPE_MAX = 10, 39
_FIRST_DIGIT_CORRECTIONS = {
    "0": "2", "4": "1", "5": "3", "6": "1", "7": "1", "8": "3", "9": "3",
}


def correct_wagon_type(digits: str) -> Tuple[str, bool]:
    """Pull an out-of-range wagon type into 10-39.  -> (digits, manipulated)."""
    if len(digits) < 2 or not digits.isdigit():
        return digits, False
    if _VALID_TYPE_MIN <= int(digits[:2]) <= _VALID_TYPE_MAX:
        return digits, False
    repl = _FIRST_DIGIT_CORRECTIONS.get(digits[0])
    if not repl:
        return digits, False
    corrected = repl + digits[1:]
    if _VALID_TYPE_MIN <= int(corrected[:2]) <= _VALID_TYPE_MAX:
        return corrected, True
    return digits, False


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

def _rek_region() -> str:
    return os.getenv("WAGONEYE_REKOGNITION_REGION") or C.S3_REGION


def _band_gap_tolerance() -> int:
    raw = os.getenv("WAGONEYE_OCR_BAND_GAP_TOLERANCE")
    if not raw:
        return 8
    try:
        return max(0, int(raw))
    except ValueError:
        return 8


_CLIENT: Optional[RekognitionClient] = None
_CLIENT_FAILED = False


def _get_client() -> Optional[RekognitionClient]:
    """Cached Rekognition client, or None if it cannot be constructed."""
    global _CLIENT, _CLIENT_FAILED
    if _CLIENT is not None or _CLIENT_FAILED:
        return _CLIENT
    try:
        _CLIENT = RekognitionClient(
            region=_rek_region(),
            aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
    except Exception as e:
        log.error("[FEAT/ocr] Rekognition client init failed: %s", e)
        _CLIENT_FAILED = True
        _CLIENT = None
    return _CLIENT


def reset_client_cache() -> None:
    """Drop the cached client (tests / credential rotation)."""
    global _CLIENT, _CLIENT_FAILED
    _CLIENT = None
    _CLIENT_FAILED = False


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _wagon_is_loaded(output_dir: str, gw_id: str) -> bool:
    """This wagon's load verdict -- selects the sheet triplet (loaded reads the
    end of the band first, empty reads the start).

    Load is a TOP-camera feature while OCR is RIGHT_UP, so this reads across
    cameras -- safe because the wagon-wise scheduler finalizes `load` before
    `ocr` for the same wagon.  Missing load defaults to False (empty), which
    only changes which sheet is tried first, never the final result."""
    for cam in C.TOP_CAMERAS:
        p = os.path.join(output_dir, "load", cam, f"{gw_id}.json")
        if not os.path.isfile(p):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, ValueError):
            continue
        if payload.get("status") == C.STATUS_OK:
            return payload.get("load_status") == C.LOAD_LOADED
    return False


def _cache_frame_range(cache_root: str, gw_id: str) -> Tuple[str, int, int, int]:
    """(cache_dir, first_frame, last_frame, n_frames) for this wagon's RIGHT_UP
    cache, restricted to the stable interior used for all feature inference."""
    cache_dir = os.path.join(cache_root, gw_id, C.CAMERA_FOLDER[C.CAMERA_RIGHT_UP])
    paths = list_wagon_frames(cache_root, gw_id, C.CAMERA_RIGHT_UP,
                              trim_stable=True)
    if not paths:
        return cache_dir, 0, -1, 0

    def _idx(p: str) -> int:
        try:
            return int(os.path.basename(p).split("_")[1].split(".")[0])
        except (IndexError, ValueError):
            return -1

    idxs = [i for i in (_idx(p) for p in paths) if i >= 0]
    if not idxs:
        return cache_dir, 0, -1, 0
    return cache_dir, min(idxs), max(idxs), len(idxs)


def _best_frame_images(cache_dir: str, enhancer, frame_idx, bbox):
    """(plate_crop, full_frame) for the winning frame -- (None, None) if it
    cannot be recovered.

    These are the SOURCE of the sheet, kept alongside it so a reviewer can
    check the OCR input against the raw frame it was cut from."""
    if frame_idx is None or enhancer is None:
        return None, None
    path = os.path.join(cache_dir, f"frame_{int(frame_idx):06d}.jpg")
    if not os.path.isfile(path):
        return None, None
    frame = cv2.imread(path)
    if frame is None:
        return None, None
    crop = enhancer.crop(path, bbox) if bbox else None
    if crop is not None and crop.size == 0:
        crop = None
    return crop, frame


# -----------------------------------------------------------------------------
# Loco (ENGINE) branch
# -----------------------------------------------------------------------------

def _process_loco(
    *, yolo_model, loco_ocr: Optional[LocoNumberOCR], cache_root: str, gw,
    loco_id: int, feature_out: str, evidence_root: Optional[str],
    camera_id: str, det_confidence: float, master_fps: float,
    timer: Optional[FeatureTimer] = None,
) -> str:
    """Read the 5-digit loco number for one ENGINE segment.

    Candidates (a Middle-2/Middle/Middle+2 sheet, then the highest-confidence
    single frame) are written into this wagon's evidence directory, and OCR runs
    over them there -- so the images the dashboard shows are exactly the images
    Rekognition was asked to read."""
    import tempfile
    gw_id = gw.global_id

    if loco_ocr is None:
        write_per_wagon_json(feature_out, gw_id, empty_payload(
            gw_id, FEATURE_NAME, C.NO_DATA, camera_id=camera_id,
            wagon_identifier=C.NO_DATA, wagon_identifier_confidence=0.0,
            is_manipulated=False, original_number="",
            candidates=[], supporting_cameras=[],
            segment_role="loco", loco_id=loco_id,
            error="Rekognition client unavailable",
        ))
        return C.NO_DATA

    def _extract_and_read(out_dir: str):
        gallery, candidates = extract_loco_candidates(
            yolo_model=yolo_model, cache_root=cache_root, gw_id=gw_id,
            loco_id=loco_id, out_dir=out_dir, det_confidence=det_confidence,
            fps=master_fps, camera_id=camera_id,
        )
        if not candidates:
            return gallery, candidates, None
        return gallery, candidates, loco_ocr.extract_from_frames(
            candidates, debug_name=f"loco_number_band_{loco_id}.jpg")

    evidence_paths: Dict[str, str] = {}
    if evidence_root:
        final_dir = os.path.join(evidence_root, gw_id, FEATURE_NAME, camera_id)
        with phase(timer, "inference"), atomic_camera_evidence(
                evidence_root, gw_id, FEATURE_NAME, camera_id) as ev_tmp:
            gallery, candidates, result = _extract_and_read(ev_tmp)
            for row in gallery:
                evidence_paths[f"loco_{row['position']}"] = os.path.join(
                    final_dir, os.path.basename(row["frame_path"]))
            for row in candidates:
                evidence_paths[f"loco_{row['role']}"] = os.path.join(
                    final_dir, os.path.basename(row["frame_path"]))
            sheet_name = C.OCR_LOCO_SHEET_TEMPLATE.format(loco_id=loco_id)
            # The candidate Rekognition actually read the winning number from
            # -- the sheet, or the single-frame fallback when the sheet failed.
            won = (result or {}).get("candidate_path")
            write_metadata(os.path.join(ev_tmp, "metadata.json"), {
                "global_id": gw_id, "feature": FEATURE_NAME,
                "camera_id": camera_id, "engine": "rekognition",
                "segment_role": "loco", "loco_id": loco_id,
                "loco_number": (result or {}).get("display_number", "-"),
                "loco_raw_number": (result or {}).get("raw_number", ""),
                "is_valid_5_digit": bool((result or {}).get("is_valid_5_digit")),
                "ocr_confidence": float((result or {}).get("ocr_confidence", 0.0) or 0.0),
                "best_frame": (result or {}).get("best_frame"),
                "fallback_triggered": bool((result or {}).get("fallback_triggered")),
                # Verification trail, mirroring the wagon branch.
                "ocr_input_image": os.path.basename(won) if won else sheet_name,
                "ocr_input_role": (result or {}).get("candidate_role") or "sheet",
                "sheet_filename": sheet_name,
                "sheet_frames": next(
                    (list(r.get("sheet_frames") or []) for r in candidates
                     if r.get("role") == "sheet"), []),
                "loco_frames": [
                    {"position": r["position"], "frame_num": r["frame_num"],
                     "filename": os.path.basename(r["frame_path"])}
                    for r in gallery
                ],
            })
    else:
        with phase(timer, "inference"):
            tmp = tempfile.mkdtemp(prefix=f".loco_{gw_id}_")
            try:
                gallery, candidates, result = _extract_and_read(tmp)
            finally:
                import shutil
                shutil.rmtree(tmp, ignore_errors=True)

    result = result or {}
    valid = bool(result.get("is_valid_5_digit"))
    write_per_wagon_json(feature_out, gw_id, {
        "global_id":   gw_id,
        "feature":     FEATURE_NAME,
        "camera_id":   camera_id,
        "status":      C.STATUS_OK,
        "engine":      "rekognition",
        "segment_role": "loco",
        "loco_id":     loco_id,
        # An engine has no 11-digit wagon number -- these stay NO_DATA so
        # fusion and the wagon table are unaffected.
        "wagon_identifier": C.NO_DATA,
        "wagon_identifier_confidence": 0.0,
        "is_manipulated": False,
        "original_number": "",
        "loco_number":       result.get("display_number", "-"),
        "loco_raw_number":   result.get("raw_number", ""),
        "is_valid_5_digit":  valid,
        "loco_confidence":   round(float(result.get("ocr_confidence", 0.0) or 0.0), 4),
        "loco_best_frame":   result.get("best_frame"),
        "loco_frame_count":  len(gallery),
        "fallback_triggered": bool(result.get("fallback_triggered")),
        "candidates": [],
        "supporting_cameras": [camera_id],
        "frame_count": len(gallery),
        "evidence": evidence_paths,
    })
    log.info("  [ocr/%s] LOCO #%d %s (valid=%s, conf=%.2f)",
             gw_id, loco_id, result.get("display_number", "-"), valid,
             float(result.get("ocr_confidence", 0.0) or 0.0))
    return C.STATUS_OK


# -----------------------------------------------------------------------------
# Public entry
# -----------------------------------------------------------------------------

def run(
    *,
    state: GlobalTrainState,
    cache_root: str,
    feature_models_dir: str,
    output_dir: str,
    evidence_root: Optional[str] = None,
    cameras: Optional[List[str]] = None,
    wagon_ids: Optional[List[str]] = None,   # None = all wagons; subset = wagon-wise
    det_confidence: float = C.CONF_OCR_BOX,
    wagon_number_length: int = C.WAGON_NUMBER_LENGTH,
    every_nth: int = 1,
    max_frames: int = 0,
    verbose: bool = True,
) -> Dict[str, str]:
    """Read wagon numbers on RIGHT_UP, writing
    wagon_states/ocr/RIGHT_UP/<gw>.json and evidence/<gw>/ocr/RIGHT_UP/."""
    del every_nth, max_frames, wagon_number_length

    camera_id = C.CAMERA_RIGHT_UP
    if cameras is not None and camera_id not in cameras:
        return {}

    from features._common import load_yolo
    model_path = os.path.join(feature_models_dir, C.MODEL_WAGON_ID_COUNTING)
    _t_ml = time.time()
    yolo_model = load_yolo(model_path)
    client = _get_client()
    _model_load_s = time.time() - _t_ml

    wagons = (state.wagons if wagon_ids is None
              else [w for w in state.wagons if w.global_id in wagon_ids])
    if not wagons:
        return {}

    feature_out = feature_camera_dir(output_dir, FEATURE_NAME, camera_id)
    timer = FeatureTimer(FEATURE_NAME, logger=log, total_units=len(wagons))
    timer.set_model_load(_model_load_s)
    summary: Dict[str, str] = {}

    detector = ocr = enhancer = loco_ocr = None
    if yolo_model is not None and client is not None:
        detector = WagonNumberDetector(
            yolo_model, confidence_threshold=det_confidence,
            gap_tolerance=_band_gap_tolerance(), logger=log)
        ocr = WagonNumberOCR(region=_rek_region(), logger=log, client=client)
        loco_ocr = LocoNumberOCR(region=_rek_region(), logger=log, client=client)
        enhancer = WagonImageEnhancer()

    # ENGINE segments are numbered sequentially in rake order (1, 2, 3, ...),
    # mirroring wagon_count -- this is the dashboard's `loco_id`.
    loco_ordinal = 0

    if yolo_model is None:
        log.warning("[FEAT/ocr] %s missing -- NO_DATA for all wagons.", model_path)
    if client is None:
        log.warning("[FEAT/ocr] Rekognition unavailable -- NO_DATA for all wagons.")
    if verbose:
        log.info("[FEAT/ocr] start: %d wagons (RIGHT_UP only)  model_load=%.2fs  "
                 "engine=rekognition region=%s three-frame-sheet",
                 len(wagons), _model_load_s, _rek_region())

    for gw in wagons:
        gw_id = gw.global_id
        t0 = time.time()
        try:
            if detector is None:
                write_per_wagon_json(feature_out, gw_id, empty_payload(
                    gw_id, FEATURE_NAME, C.NO_DATA, camera_id=camera_id,
                    wagon_identifier=C.NO_DATA, wagon_identifier_confidence=0.0,
                    is_manipulated=False, original_number="",
                    candidates=[], supporting_cameras=[],
                    error="detector or Rekognition client unavailable",
                ))
                summary[gw_id] = C.NO_DATA
                continue

            # ENGINE -> the 5-digit LOCO number (different plate, different
            # validator, Middle-2/Middle/Middle+2 sheet).
            if gw.classification == C.CLASS_ENGINE:
                loco_ordinal += 1
                summary[gw_id] = _process_loco(
                    yolo_model=yolo_model, loco_ocr=loco_ocr,
                    cache_root=cache_root, gw=gw, loco_id=loco_ordinal,
                    feature_out=feature_out, evidence_root=evidence_root,
                    camera_id=camera_id, det_confidence=det_confidence,
                    master_fps=float(getattr(state, "master_fps", 0.0) or 0.0),
                    timer=timer,
                )
                continue

            # BRAKE_VAN carries neither an 11-digit wagon number nor a loco
            # number; running OCR would burn Rekognition calls on noise.
            if gw.classification == C.CLASS_BRAKE_VAN:
                write_per_wagon_json(feature_out, gw_id, empty_payload(
                    gw_id, FEATURE_NAME, C.STATUS_OK, camera_id=camera_id,
                    wagon_identifier=C.NO_DATA, wagon_identifier_confidence=0.0,
                    is_manipulated=False, original_number="",
                    candidates=[], supporting_cameras=[camera_id],
                    skipped_reason=f"classification={gw.classification}",
                ))
                summary[gw_id] = C.STATUS_OK
                continue

            cache_dir, first, last, n_frames = _cache_frame_range(cache_root, gw_id)
            if n_frames == 0:
                write_per_wagon_json(feature_out, gw_id, empty_payload(
                    gw_id, FEATURE_NAME, C.STATUS_NO_FRAMES, camera_id=camera_id,
                    wagon_identifier=C.NO_DATA, wagon_identifier_confidence=0.0,
                    is_manipulated=False, original_number="",
                    candidates=[], supporting_cameras=[],
                ))
                summary[gw_id] = C.STATUS_NO_FRAMES
                continue

            with phase(timer, "inference"):
                bands = detector.detect_on_segment(cache_dir, first, last)
                result = ocr.extract_number_from_segment(
                    bands=bands, segment_dir=cache_dir, enhancer=enhancer,
                    is_loaded=_wagon_is_loaded(output_dir, gw_id),
                    segment_id=int(gw.wagon_index),
                )

            raw = str(result.get("raw_number") or "")
            valid_upstream = bool(result.get("is_valid_11_digit"))
            conf = float(result.get("ocr_confidence") or 0.0)

            # This pipeline's wagon-type correction on top of upstream's
            # length-only validity.
            digits, manipulated = correct_wagon_type(raw) if valid_upstream else (raw, False)
            in_band = (len(digits) == C.WAGON_NUMBER_LENGTH
                       and digits.isdigit()
                       and _VALID_TYPE_MIN <= int(digits[:2]) <= _VALID_TYPE_MAX)
            ident = digits if (valid_upstream and in_band) else C.NO_DATA
            if ident == C.NO_DATA:
                conf = 0.0

            evidence_paths: Dict[str, str] = {}
            sheet = result.get("_sheet_image")
            if evidence_root and sheet is not None:
                final_dir = os.path.join(evidence_root, gw_id, FEATURE_NAME, camera_id)
                crop_img, frame_img = _best_frame_images(
                    cache_dir, enhancer,
                    result.get("best_frame"), result.get("best_bbox"))
                with phase(timer, "evidence"), atomic_camera_evidence(
                        evidence_root, gw_id, FEATURE_NAME, camera_id) as ev_tmp:
                    # number_sheet.jpg IS the OCR input: same array, same JPEG
                    # quality as `read_digits` encodes with, so the file is
                    # byte-identical to what Rekognition was posted and a
                    # reviewer reads the number off the very image the engine
                    # saw.  The other two are the source it was cut from.
                    save_jpeg(os.path.join(ev_tmp, C.OCR_SHEET_FILENAME), sheet,
                              quality=C.OCR_SHEET_JPEG_QUALITY)
                    save_jpeg(os.path.join(ev_tmp, "number_crop.jpg"),
                              crop_img if crop_img is not None else sheet)
                    save_jpeg(os.path.join(ev_tmp, "best_frame.jpg"),
                              frame_img if frame_img is not None else sheet)
                    write_metadata(os.path.join(ev_tmp, "metadata.json"), {
                        "global_id": gw_id, "feature": FEATURE_NAME,
                        "camera_id": camera_id, "engine": "rekognition",
                        "frame_idx": result.get("best_frame"),
                        "bbox": result.get("best_bbox"),
                        "band_id": result.get("band_id"),
                        "fallback_triggered": result.get("fallback_triggered"),
                        "full_number": digits,
                        "raw_number": raw,
                        "ocr_confidence": result.get("ocr_confidence"),
                        "is_full_length": valid_upstream,
                        "aggregated_winner": ident,
                        "aggregated_confidence": conf,
                        "is_manipulated": manipulated,
                        "original_number": raw,
                        # Verification trail: the image Rekognition read, and
                        # the cache frames it was stacked from (top to bottom).
                        "ocr_input_image": C.OCR_SHEET_FILENAME,
                        "sheet_frames": result.get("sheet_frames") or [],
                    })
                evidence_paths["number_sheet"] = os.path.join(
                    final_dir, C.OCR_SHEET_FILENAME)
                evidence_paths["best_frame"] = os.path.join(final_dir, "best_frame.jpg")
                evidence_paths["number_crop"] = os.path.join(final_dir, "number_crop.jpg")

            write_per_wagon_json(feature_out, gw_id, {
                "global_id":   gw_id,
                "feature":     FEATURE_NAME,
                "camera_id":   camera_id,
                "status":      C.STATUS_OK,
                "engine":      "rekognition",
                "wagon_identifier":            ident,
                "wagon_identifier_confidence": round(float(conf), 4),
                "is_manipulated":  manipulated,
                "original_number": raw,
                "candidates": [{
                    "full_number":     digits,
                    "raw_number":      raw,
                    "ocr_confidence":  round(float(result.get("ocr_confidence") or 0.0), 4),
                    "band_id":         result.get("band_id"),
                    "frame_idx":       result.get("best_frame"),
                    "bbox":            result.get("best_bbox"),
                    "is_manipulated":  manipulated,
                    "is_full_length":  valid_upstream,
                }] if raw else [],
                "bands":              len(bands),
                "fallback_triggered": bool(result.get("fallback_triggered")),
                "supporting_cameras": [camera_id],
                "frame_count": n_frames,
                "evidence":    evidence_paths,
            })
            summary[gw_id] = C.STATUS_OK
            if verbose:
                log.info("  [ocr/%s] %s (conf=%.2f, bands=%d, frames=%d, fallback=%s)",
                         gw_id, ident, conf, len(bands), n_frames,
                         bool(result.get("fallback_triggered")))
        except Exception as e:
            write_per_wagon_json(feature_out, gw_id, empty_payload(
                gw_id, FEATURE_NAME, C.STATUS_FAILED, camera_id=camera_id,
                wagon_identifier=C.NO_DATA,
                is_manipulated=False, original_number="",
                error=f"{type(e).__name__}: {e}",
                traceback=traceback.format_exc(limit=2),
            ))
            summary[gw_id] = C.STATUS_FAILED
            log.error("  [ocr/%s] FAILED: %s", gw_id, e)
        finally:
            timer.stamp(gw_id, t0, camera_id)

    n_ok = sum(1 for v in summary.values() if v == C.STATUS_OK)
    if verbose:
        timer.log_summary(ok=n_ok, total=len(summary))
    return summary
