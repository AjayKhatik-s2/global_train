"""OCR feature processor (v4, train-state-native).

Two interchangeable OCR engines, selected by ``WAGONEYE_OCR_ENGINE``:

``rekognition`` (DEFAULT) -- AWS Rekognition ``DetectText``
-----------------------------------------------------------
Matches the V4 Train-Inspection-Engine's OCR exactly:

    1. YOLO `wagon_id_counting.pt` detects wagon-number plate regions on
       RIGHT_UP frames (master / OCR authority).
    2. Per-frame detections are grouped into temporally contiguous **bands**
       (one band == one plate passing through view).
    3. For each band, THREE frames are chosen and their padded crops are stacked
       on ONE vertical white sheet -- each plate lands on its own horizontal band
       so Rekognition reads each as a separate ``LINE``.  Which three depends on
       rake load state, because cargo occludes the plate from one end of the band
       depending on travel direction:
           loaded rake -> primary End-7/-5/-3, fallback Start+3/+5/+7
           empty rake  -> primary Start+3/+5/+7, fallback End-7/-5/-3
    4. The sheet goes through the 3x-upscale preprocessor and ONE DetectText
       call; the reader picks the single best VALID 11-digit LINE (never
       concatenates the three copies).  First valid read wins; the call budget is
       capped per wagon.

Load state comes from this wagon's already-written `load` JSON -- the wagon-wise
scheduler runs LOAD before OCR precisely so this (and the damage floor filter)
sees a finalized result.

``easyocr`` -- the legacy local pipeline (kept as a no-network fallback)
-----------------------------------------------------------------------
    padding 10 -> 3x cubic upscale -> NLMeans denoise -> CLAHE -> unsharp ->
    easyocr (allowlist='0123456789') -> digit extraction -> wagon-type
    confusion-map correction -> WagonNumberValidator -> cross-frame
    WagonNumberAggregator (min 2 frames, min conf 0.3).

Output JSON shape (both engines):
    {
        "global_id":  "GW_7",
        "feature":    "ocr",
        "status":     "OK" | "NO_FRAMES" | "FAILED" | "NO_DATA",
        "wagon_identifier":  "32145678901",
        "wagon_identifier_confidence": 0.83,
        "candidates":  [...],
        "supporting_cameras": ["RIGHT_UP"],
        "frame_count": ...,
        "engine": "rekognition" | "easyocr",
        # rekognition only -- carried into the per-camera inspection JSON:
        "raw_number", "display_number", "is_valid_11_digit",
        "fallback_triggered", "band_id", "best_frame", "best_bbox"
    }
"""

from __future__ import annotations

import json
import os
import time
import traceback
from typing import Any, Dict, List, Optional

import numpy as np

from core import constants as C
from core.global_state_loader import GlobalTrainState
from core.logging_setup import get_logger

from features._common import (
    load_yolo, run_detection, iter_wagon_frames, crop_bbox, list_wagon_frames,
    iter_wagon_detections, model_class_names,
    write_per_wagon_json, empty_payload, FeatureTimer, feature_camera_dir, phase,
    DEVICE, HALF, precision_kwargs,
)

# Mature intelligence ported from legacy (easyocr engine)
from features.inference_lib.wagon_number_ocr import WagonNumberOCR, WagonNumber
from features.inference_lib.wagon_number_aggregator import (
    WagonNumberAggregator, AggregatorConfig,
)
# Rekognition engine (V4 parity)
from features.inference_lib.rekognition_wagon_number import (
    RekognitionWagonNumberOCR, group_detections_into_bands, crop_plate,
    is_valid_wagon_number,
)
from features._evidence import (
    BestFrameTracker, atomic_camera_evidence,
    save_jpeg, safe_crop, write_metadata, draw_annotated_bbox,
)


FEATURE_NAME = "ocr"
log = get_logger("features.ocr")


# -----------------------------------------------------------------------------
# Engine selection
# -----------------------------------------------------------------------------

ENGINE_REKOGNITION = "rekognition"
ENGINE_EASYOCR = "easyocr"


def resolve_engine() -> str:
    """Which OCR engine this process uses.

    ``WAGONEYE_OCR_ENGINE=rekognition`` (default) | ``easyocr``.  An unknown
    value falls back to Rekognition with a warning rather than failing the run.
    """
    raw = (os.getenv("WAGONEYE_OCR_ENGINE") or ENGINE_REKOGNITION).strip().lower()
    if raw in (ENGINE_REKOGNITION, "aws", "detect_text"):
        return ENGINE_REKOGNITION
    if raw in (ENGINE_EASYOCR, "local", "easy"):
        return ENGINE_EASYOCR
    log.warning("[FEAT/ocr] unknown WAGONEYE_OCR_ENGINE=%r -- using %s",
                raw, ENGINE_REKOGNITION)
    return ENGINE_REKOGNITION


#: Plate class names the detector may emit, normalised to one canonical name.
#: `wagon_number_update.pt` / `wagon_id_counting.pt` emit BOTH `wagon_id` (the
#: 11-digit wagon plate) and `loco_no` (the 5-digit locomotive plate), so one
#: detector serves both OCR paths.  (V4: the right_top model says "wagonno".)
WAGON_NUMBER_CLASS_ALIASES = {"wagonno": "wagon_id", "wagon_number": "wagon_id",
                              "wagon_id": "wagon_id"}
_CANONICAL_PLATE_CLASS = "wagon_id"

#: Locomotive-plate class names (the 5-digit number on the loco face).
LOCO_NUMBER_CLASS_ALIASES = {"loco_no": "loco_no", "locono": "loco_no",
                             "loco_number": "loco_no", "engine_head": "loco_no"}
_CANONICAL_LOCO_CLASS = "loco_no"


def plate_classes_resolvable(known_names) -> Dict[str, bool]:
    """Which OCR paths this plate detector's class list can actually serve.

    ``{"wagon": bool, "loco": bool}``.  The WAGON path is always servable -- an
    unrecognised vocabulary falls back to "every box is a wagon-plate candidate".
    The LOCO path is servable ONLY when a class maps to a known loco alias,
    because it refuses to guess (see `_is_plate_class`).

    Exposed so `scripts/verify_runtime.py` can fail a bad model swap up front
    instead of letting Stage 3 read nothing hours into a batch.
    """
    known = {str(n).lower() for n in (known_names or ())}
    return {
        "wagon": True,
        "loco": bool(known & set(LOCO_NUMBER_CLASS_ALIASES)),
    }


#: One warning per process per model, not one per wagon.
_WARNED_NO_LOCO_CLASS: set = set()


def _warn_if_loco_unservable(known_names: set) -> bool:
    """True when the loco path can run.  Logs ONCE per distinct class list if not.

    Without this the failure is silent: `_is_plate_class` drops every detection,
    Stage 3 records no loco number, and the report simply shows "-" with nothing
    in the log to say why.
    """
    if plate_classes_resolvable(known_names)["loco"]:
        return True
    key = ",".join(sorted(known_names))
    if key not in _WARNED_NO_LOCO_CLASS:
        _WARNED_NO_LOCO_CLASS.add(key)
        log.warning(
            "loco-number OCR DISABLED: the plate model's classes %s contain none "
            "of the known locomotive labels %s, and the loco path will not guess "
            "(a misread 11-digit wagon plate must never be reported as a loco "
            "number).  Rename the model's loco class or add its label to "
            "LOCO_NUMBER_CLASS_ALIASES; wagon-number OCR is unaffected.",
            sorted(known_names) or "(none)",
            sorted(LOCO_NUMBER_CLASS_ALIASES))
    return False


def _is_plate_class(class_name: Optional[str], known_names: set,
                    *, kind: str = "wagon") -> bool:
    """True when a detection should be treated as a plate of the given `kind`.

    `kind="wagon"` keeps the 11-digit wagon plate; `kind="loco"` keeps the 5-digit
    locomotive plate.  A model whose class list we recognise is filtered to that
    class.  A model that emits an unmapped/unknown label is trusted wholesale for
    the WAGON path only -- a single-purpose plate detector's every box is a wagon
    plate candidate (V4's permissive rule); the loco path never guesses, because a
    misread wagon plate would be reported as a loco number.
    """
    aliases = (LOCO_NUMBER_CLASS_ALIASES if kind == "loco"
               else WAGON_NUMBER_CLASS_ALIASES)
    canonical = (_CANONICAL_LOCO_CLASS if kind == "loco"
                 else _CANONICAL_PLATE_CLASS)
    if not class_name:
        return kind != "loco"
    normalised = aliases.get(class_name.lower())
    if normalised is None:
        if kind == "loco":
            return False        # never guess a loco plate
        # This model doesn't use the known vocabulary at all -> accept everything.
        return not (known_names & set(WAGON_NUMBER_CLASS_ALIASES))
    return normalised == canonical


# -----------------------------------------------------------------------------
# Per-process singleton (easyocr Reader is heavy; load once)
# -----------------------------------------------------------------------------

_OCR_SINGLETON: Optional[WagonNumberOCR] = None


def _get_ocr() -> Optional[WagonNumberOCR]:
    global _OCR_SINGLETON
    if _OCR_SINGLETON is not None:
        return _OCR_SINGLETON
    try:
        _OCR_SINGLETON = WagonNumberOCR(
            # CUDA if available, else CPU.  Was hardcoded True, which errors /
            # silently degrades on a CPU-only EC2 host; now matches the
            # resolved device (still GPU on a GPU box -- identical behaviour).
            use_gpu=(DEVICE == "cuda"),
            min_confidence=0.30,        # legacy default for cross-frame aggregation
            resize_factor=3.0,
        )
        if getattr(_OCR_SINGLETON, "reader", None) is None:
            _OCR_SINGLETON = None
    except Exception as e:
        log.warning("[FEAT/ocr] WagonNumberOCR init failed: %s", e)
        _OCR_SINGLETON = None
    return _OCR_SINGLETON


# -----------------------------------------------------------------------------
# Load state (drives the Rekognition triplet order)
# -----------------------------------------------------------------------------

def _is_wagon_loaded(states_root: str, gw_id: str) -> bool:
    """Read this wagon's fused-authority load result: RIGHT_UP_TOP else LEFT_UP_TOP.

    Deterministic: the wagon-wise scheduler finalizes LOAD for this wagon before
    OCR runs, so the JSON is fully written.  An absent / failed / unreadable load
    result degrades to ``False`` (empty), which selects the Start+3/+5/+7 primary
    triplet -- the same default the V4 engine uses for an unclassified rake.
    """
    for camera in (C.CAMERA_RIGHT_UP_TOP, C.CAMERA_LEFT_UP_TOP):
        p = os.path.join(states_root, "load", camera, f"{gw_id}.json")
        if not os.path.isfile(p):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("status") != C.STATUS_OK:
            continue
        status = payload.get("load_status")
        if status in (C.LOAD_LOADED, C.LOAD_EMPTY):
            return status == C.LOAD_LOADED
    return False


# -----------------------------------------------------------------------------
# Rekognition engine -- per-wagon driver
# -----------------------------------------------------------------------------

def _detect_plates_for_wagon(
    yolo_model, cache_root: str, gw_id: str, camera_id: str,
    det_confidence: float, kind: str = "wagon",
) -> List[Dict[str, Any]]:
    """Run the plate detector over one wagon's cached frames.

    Returns ``[{"frame": int, "confidence": float, "bbox": [x1,y1,x2,y2]}, ...]``
    for plates of the given `kind` (``"wagon"`` or ``"loco"``).  Frames are NOT
    retained -- only the 3-6 frames later chosen for a sheet are re-decoded, so
    peak memory stays flat regardless of wagon length.
    """
    known = {str(n).lower() for n in model_class_names(yolo_model).values()}
    names = model_class_names(yolo_model)
    if kind == "loco" and not _warn_if_loco_unservable(known):
        return []
    detections: List[Dict[str, Any]] = []
    for fi, _frame, boxes, confs, clss in iter_wagon_detections(
            yolo_model, cache_root, gw_id, camera_id, trim_stable=True, fp16=True):
        if boxes is None or len(boxes) == 0:
            continue
        for bbox, conf, cls_id in zip(boxes, confs, clss):
            if float(conf) < det_confidence:
                continue
            cls_name = names.get(int(cls_id))
            if not _is_plate_class(str(cls_name) if cls_name else None, known,
                                   kind=kind):
                continue
            detections.append({
                "frame": int(fi),
                "confidence": float(conf),
                "bbox": [float(bbox[0]), float(bbox[1]),
                         float(bbox[2]), float(bbox[3])],
            })
    return detections


def _frame_path_index(cache_root: str, gw_id: str, camera_id: str) -> Dict[int, str]:
    """Map frame index -> cached JPEG path for one (wagon, camera)."""
    index: Dict[int, str] = {}
    for p in list_wagon_frames(cache_root, gw_id, camera_id, trim_stable=False):
        try:
            fi = int(os.path.basename(p).split("_")[1].split(".")[0])
        except (IndexError, ValueError):
            continue
        index[fi] = p
    return index


def _process_one_wagon_rekognition(
    yolo_model, reader: RekognitionWagonNumberOCR, cache_root: str,
    states_root: str, gw_id: str, camera_id: str, det_confidence: float,
    gap_tolerance: int, kind: str = "wagon",
) -> Dict[str, Any]:
    """Detect plate bands for one wagon/loco and read the number via Rekognition."""
    import cv2

    detections = _detect_plates_for_wagon(
        yolo_model, cache_root, gw_id, camera_id, det_confidence, kind=kind)
    frame_paths = _frame_path_index(cache_root, gw_id, camera_id)
    n_frames = len(frame_paths)

    if not detections:
        return {"frame_count": n_frames, "detections": 0, "bands": [],
                "result": None}

    bands = group_detections_into_bands(detections, gap_tolerance=gap_tolerance)

    def _load_frame(frame_idx: int):
        path = frame_paths.get(frame_idx)
        if path is None:
            return None
        return cv2.imread(path)

    if kind == "loco":
        result = reader.read_loco_number(bands=bands, frame_loader=_load_frame)
    else:
        is_loaded = _is_wagon_loaded(states_root, gw_id)
        result = reader.read_wagon_number(
            bands=bands, frame_loader=_load_frame, is_loaded=is_loaded)
        result["is_loaded"] = is_loaded
    return {"frame_count": n_frames, "detections": len(detections),
            "bands": bands, "result": result, "frame_paths": frame_paths}


def _persist_rekognition_evidence(
    *, evidence_root: str, gw_id: str, camera_id: str, outcome: Dict[str, Any],
    timer: Optional[FeatureTimer],
) -> Dict[str, str]:
    """Write the OCR evidence bundle for one wagon.

    Filenames are deliberately unchanged from the easyocr engine
    (``best_frame.jpg`` / ``number_crop.jpg``) so every existing consumer -- the
    camera + combined reports and the per-camera inspection JSON gallery -- keeps
    resolving.  ``ocr_sheet.jpg`` is added: the EXACT preprocessed image that was
    sent to Rekognition (the V4 engine's
    ``wagon_numbers/wagon_number_segment_NNN.jpg`` artifact).
    """
    import cv2

    result = outcome.get("result") or {}
    frame_paths = outcome.get("frame_paths") or {}
    best_frame_idx = result.get("best_frame")
    bbox = result.get("best_bbox")
    sheet = result.get("_sheet")

    frame_img = None
    if best_frame_idx is not None and frame_paths.get(best_frame_idx):
        frame_img = cv2.imread(frame_paths[best_frame_idx])

    if frame_img is None and sheet is None:
        return {}

    final_dir = os.path.join(evidence_root, gw_id, FEATURE_NAME, camera_id)
    paths: Dict[str, str] = {}
    crop_img = crop_plate(frame_img, bbox) if (frame_img is not None and bbox) else None

    with phase(timer, "evidence"), atomic_camera_evidence(
            evidence_root, gw_id, FEATURE_NAME, camera_id) as ev_tmp:
        if frame_img is not None:
            label = (f"OCR {result.get('display_number', '-')} "
                     f"{float(result.get('ocr_confidence') or 0.0):.2f}")
            annotated = (draw_annotated_bbox(frame_img, bbox, label=label,
                                            color=(0, 255, 0))
                         if bbox else frame_img)
            save_jpeg(os.path.join(ev_tmp, "best_frame.jpg"), annotated)
            paths["best_frame"] = os.path.join(final_dir, "best_frame.jpg")
        if crop_img is not None:
            save_jpeg(os.path.join(ev_tmp, "number_crop.jpg"), crop_img)
            paths["number_crop"] = os.path.join(final_dir, "number_crop.jpg")
        if sheet is not None:
            save_jpeg(os.path.join(ev_tmp, "ocr_sheet.jpg"), sheet)
            paths["ocr_sheet"] = os.path.join(final_dir, "ocr_sheet.jpg")
        write_metadata(os.path.join(ev_tmp, "metadata.json"), {
            "global_id":        gw_id,
            "feature":          FEATURE_NAME,
            "camera_id":        camera_id,
            "engine":           result.get("engine", ENGINE_REKOGNITION),
            "frame_idx":        best_frame_idx,
            "bbox":             bbox,
            "full_number":      result.get("raw_number"),
            "display_number":   result.get("display_number"),
            "ocr_confidence":   result.get("ocr_confidence"),
            "kind":             result.get("kind", "wagon"),
            "is_valid_11_digit": result.get("is_valid_11_digit"),
            "is_valid_5_digit": result.get("is_valid_5_digit"),
            "is_full_length":   result.get("is_valid_11_digit")
                                or result.get("is_valid_5_digit"),
            "fallback_triggered": result.get("fallback_triggered"),
            "band_id":          result.get("band_id"),
            "band_count":       result.get("band_count"),
            "rekognition_calls": result.get("rekognition_calls"),
            "is_loaded":        result.get("is_loaded"),
            "aggregated_winner": result.get("wagon_identifier"),
            "aggregated_confidence": result.get("wagon_identifier_confidence"),
        })
    return paths


# -----------------------------------------------------------------------------
# easyocr engine -- per-wagon driver (legacy, unchanged behaviour)
# -----------------------------------------------------------------------------

def _process_one_wagon(
    yolo_model,
    ocr: WagonNumberOCR,
    cache_root: str,
    gw_id: str,
    det_confidence: float,
) -> Dict[str, Any]:
    """Iterate cached RIGHT_UP frames, run YOLO + easyocr, aggregate."""
    aggregator = WagonNumberAggregator(AggregatorConfig(
        min_frame_count=2,
        min_confidence=0.3,
        require_validation=True,
    ))

    used = 0
    raw_candidates: List[Dict[str, Any]] = []
    best = BestFrameTracker()    # remembers the highest-conf OCR snapshot

    for fi, frame in iter_wagon_frames(cache_root, gw_id, C.CAMERA_RIGHT_UP, trim_stable=True):
        used += 1

        # Stage A: YOLO detection -- locate wagon-number bbox regions.
        # (FP16 requested via precision_kwargs -> CUDA uses the supported FP16
        # mechanism; CPU passes nothing, so the deprecated `half=` arg is never
        # sent.  On CPU this is identical to the old half=True call, which
        # ultralytics ran as FP32 anyway.)
        try:
            results = yolo_model(frame, verbose=False, device=DEVICE,
                                 **precision_kwargs(DEVICE, fp16=True))[0]
        except Exception:
            continue
        if results.boxes is None or len(results.boxes) == 0:
            continue

        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()

        # Stage B: per-detection OCR pipeline (preprocess + easyocr +
        # validate + reconstruct)
        for bbox, yolo_conf in zip(boxes, confs):
            if float(yolo_conf) < det_confidence:
                continue
            bbox_list = [float(b) for b in bbox]
            crop = crop_bbox(frame, bbox_list, pad=10)
            if crop is None or crop.size == 0:
                continue
            try:
                wagon_num = ocr.reconstruct_wagon_number(
                    crop, float(yolo_conf), debug=False,
                )
            except Exception:
                continue
            if wagon_num is None:
                continue
            aggregator.add_wagon_number(wagon_num, frame_idx=fi)

            full = getattr(wagon_num, "full_number", None)
            if full:
                ocr_conf = float(getattr(wagon_num, "ocr_confidence", 0.0))
                raw_candidates.append({
                    "frame_idx":       int(fi),
                    "full_number":     str(full),
                    "ocr_confidence":  ocr_conf,
                    "yolo_confidence": float(getattr(wagon_num, "yolo_confidence", 0.0)),
                    "bbox":            bbox_list,
                })
                # Track best snapshot:  prefer full-length (11-digit) numbers
                # and within that bucket, highest OCR confidence.
                is_full = int(len(str(full)) == C.WAGON_NUMBER_LENGTH)
                score = is_full * 10.0 + ocr_conf
                best.update(
                    score=score, frame=frame, bbox=bbox_list,
                    frame_idx=fi,
                    full_number=str(full),
                    ocr_confidence=ocr_conf,
                    yolo_confidence=float(yolo_conf),
                    is_full_length=bool(is_full),
                )

    # Stage C: pick the dominant aggregated wagon number
    aggregated = aggregator.get_aggregated_numbers()
    return {
        "frame_count": used,
        "aggregated":  aggregated,
        "raw":         raw_candidates,
        "best":        best,
    }


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
    wagon_ids: Optional[List[str]] = None,  # None = all wagons; subset = wagon-wise
    det_confidence: float = C.CONF_OCR_BOX,
    wagon_number_length: int = C.WAGON_NUMBER_LENGTH,
    every_nth: int = 1,
    max_frames: int = 0,
    engine: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, str]:
    """Run OCR on every wagon (RIGHT_UP only), writing the per-camera layout
    wagon_states/ocr/RIGHT_UP/<gw>.json and evidence/<gw>/ocr/RIGHT_UP/."""
    del every_nth, max_frames, wagon_number_length  # legacy code uses its own thresholds

    # OCR authority is RIGHT_UP only.  If a caller scopes to other cameras,
    # there is nothing for OCR to do.
    camera_id = C.CAMERA_RIGHT_UP
    if cameras is not None and camera_id not in cameras:
        return {}

    engine = (engine or resolve_engine())
    # RIGHT_UP plate detector: wagon_number_update.pt (the V4 right_up model),
    # falling back to the older wagon_id_counting.pt when that is all a checkout
    # has.  Resolved through core.constants so model_sync + the completion marker
    # agree on which file this run used.
    model_path = C.feature_model_path(feature_models_dir, C.MODEL_WAGON_NUMBER)
    _t_ml = time.time()
    yolo_model = load_yolo(model_path)

    rek_reader: Optional[RekognitionWagonNumberOCR] = None
    ocr: Optional[WagonNumberOCR] = None
    if engine == ENGINE_REKOGNITION:
        from core import rekognition as REK
        client = REK.get_client()
        if client is not None:
            rek_reader = RekognitionWagonNumberOCR(
                client, max_calls=REK.MAX_CALLS_PER_WAGON,
                gap_tolerance=_gap_tolerance(), logger=log)
        else:
            log.warning("[FEAT/ocr] Rekognition unavailable -- falling back to "
                        "easyocr for this run")
            engine = ENGINE_EASYOCR
    if engine == ENGINE_EASYOCR:
        ocr = _get_ocr()
    _model_load_s = time.time() - _t_ml

    wagons = (state.wagons if wagon_ids is None
              else [w for w in state.wagons if w.global_id in wagon_ids])
    if not wagons:
        return {}
    feature_out = feature_camera_dir(output_dir, FEATURE_NAME, camera_id)
    timer = FeatureTimer(FEATURE_NAME, logger=log, total_units=len(wagons))
    timer.set_model_load(_model_load_s)
    summary: Dict[str, str] = {}

    engine_ready = (rek_reader is not None if engine == ENGINE_REKOGNITION
                    else ocr is not None)
    if yolo_model is None:
        log.warning("[FEAT/ocr] %s missing -- NO_DATA for all wagons.", model_path)
    if not engine_ready:
        log.warning("[FEAT/ocr] engine %s unavailable -- NO_DATA for all wagons.",
                    engine)

    if verbose:
        log.info("[FEAT/ocr] start: %d wagons (RIGHT_UP only)  engine=%s  "
                 "model_load=%.2fs", len(wagons), engine, _model_load_s)

    for gw in wagons:
        gw_id = gw.global_id
        t0 = time.time()
        try:
            if yolo_model is None or not engine_ready:
                payload = empty_payload(
                    gw_id, FEATURE_NAME, C.NO_DATA,
                    wagon_identifier=C.NO_DATA,
                    wagon_identifier_confidence=0.0,
                    candidates=[], supporting_cameras=[],
                    engine=engine,
                    error="detector or OCR engine unavailable",
                )
                write_per_wagon_json(feature_out, gw_id, payload)
                summary[gw_id] = C.NO_DATA
                continue

            # An ENGINE carries a 5-digit LOCO number, not an 11-digit wagon
            # number.  Read it via the loco path (same detector's `loco_no`
            # class, middle-frame triplet, 5-digit validator) so
            # `loco_number_results` is populated instead of empty.
            if gw.classification == C.CLASS_ENGINE:
                if engine == ENGINE_REKOGNITION:
                    status, payload = _run_wagon_rekognition(
                        yolo_model=yolo_model, reader=rek_reader,
                        cache_root=cache_root, states_root=output_dir,
                        evidence_root=evidence_root, gw_id=gw_id,
                        camera_id=camera_id, det_confidence=det_confidence,
                        timer=timer, verbose=verbose, kind="loco")
                    write_per_wagon_json(feature_out, gw_id, payload)
                    summary[gw_id] = status
                    continue
                # easyocr has no loco-plate pipeline -- record the entry as before.
                payload = empty_payload(
                    gw_id, FEATURE_NAME, C.STATUS_OK,
                    wagon_identifier=C.NO_DATA,
                    wagon_identifier_confidence=0.0,
                    candidates=[], supporting_cameras=[C.CAMERA_RIGHT_UP],
                    engine=engine, kind="loco",
                    display_number="-", is_valid_5_digit=False,
                    skipped_reason="loco OCR requires the rekognition engine",
                )
                write_per_wagon_json(feature_out, gw_id, payload)
                summary[gw_id] = C.STATUS_OK
                continue

            # A BRAKE_VAN carries neither plate; running OCR on it is pure noise
            # (and, with Rekognition, cost).  Record the entry and skip.
            if gw.classification == C.CLASS_BRAKE_VAN:
                payload = empty_payload(
                    gw_id, FEATURE_NAME, C.STATUS_OK,
                    wagon_identifier=C.NO_DATA,
                    wagon_identifier_confidence=0.0,
                    candidates=[],
                    supporting_cameras=[C.CAMERA_RIGHT_UP],
                    engine=engine,
                    display_number="-",
                    is_valid_11_digit=False,
                    skipped_reason=f"classification={gw.classification}",
                )
                write_per_wagon_json(feature_out, gw_id, payload)
                summary[gw_id] = C.STATUS_OK
                continue

            if engine == ENGINE_REKOGNITION:
                status, payload = _run_wagon_rekognition(
                    yolo_model=yolo_model, reader=rek_reader,
                    cache_root=cache_root, states_root=output_dir,
                    evidence_root=evidence_root, gw_id=gw_id,
                    camera_id=camera_id, det_confidence=det_confidence,
                    timer=timer, verbose=verbose)
            else:
                status, payload = _run_wagon_easyocr(
                    yolo_model=yolo_model, ocr=ocr, cache_root=cache_root,
                    evidence_root=evidence_root, gw_id=gw_id,
                    camera_id=camera_id, det_confidence=det_confidence,
                    timer=timer, verbose=verbose)
            write_per_wagon_json(feature_out, gw_id, payload)
            summary[gw_id] = status
        except Exception as e:
            payload = empty_payload(
                gw_id, FEATURE_NAME, C.STATUS_FAILED,
                wagon_identifier=C.NO_DATA,
                engine=engine,
                error=f"{type(e).__name__}: {e}",
                traceback=traceback.format_exc(limit=2),
            )
            write_per_wagon_json(feature_out, gw_id, payload)
            summary[gw_id] = C.STATUS_FAILED
            log.error("[FEAT/ocr/%s] FAILED: %s", gw_id, e)
        finally:
            timer.stamp(gw_id, t0, camera_id)

    n_ok = sum(1 for v in summary.values() if v == C.STATUS_OK)
    if verbose:
        timer.log_summary(ok=n_ok, total=len(summary))
    return summary


def _gap_tolerance() -> int:
    """Frame gap that separates two plate bands (V4 right_up.yaml default: 8)."""
    raw = os.getenv("WAGONEYE_OCR_GAP_TOLERANCE")
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            pass
    return 8


# -----------------------------------------------------------------------------
# Per-wagon engine wrappers (return (status, payload))
# -----------------------------------------------------------------------------

def _run_wagon_rekognition(
    *, yolo_model, reader, cache_root: str, states_root: str,
    evidence_root: Optional[str], gw_id: str, camera_id: str,
    det_confidence: float, timer: FeatureTimer, verbose: bool,
    kind: str = "wagon",
):
    valid_key = "is_valid_5_digit" if kind == "loco" else "is_valid_11_digit"
    with phase(timer, "inference"):
        outcome = _process_one_wagon_rekognition(
            yolo_model, reader, cache_root, states_root, gw_id, camera_id,
            det_confidence, _gap_tolerance(), kind=kind)

    n_frames = outcome["frame_count"]
    if n_frames == 0:
        return C.STATUS_NO_FRAMES, empty_payload(
            gw_id, FEATURE_NAME, C.STATUS_NO_FRAMES,
            wagon_identifier=C.NO_DATA, wagon_identifier_confidence=0.0,
            candidates=[], supporting_cameras=[], engine=ENGINE_REKOGNITION,
            kind=kind, display_number="-", **{valid_key: False})

    result = outcome.get("result")
    if result is None:
        # Frames existed but the detector found no plate at all.
        return C.STATUS_OK, {
            "global_id": gw_id, "feature": FEATURE_NAME, "camera_id": camera_id,
            "status": C.STATUS_OK, "engine": ENGINE_REKOGNITION, "kind": kind,
            "wagon_identifier": C.NO_DATA, "wagon_identifier_confidence": 0.0,
            "raw_number": "", "display_number": "-", valid_key: False,
            "fallback_triggered": False, "band_id": 0, "band_count": 0,
            "rekognition_calls": 0, "detections": 0,
            "candidates": [], "supporting_cameras": [C.CAMERA_RIGHT_UP],
            "frame_count": n_frames, "evidence": {},
            "no_plate_detected": True,
        }

    evidence_paths: Dict[str, str] = {}
    if evidence_root:
        evidence_paths = _persist_rekognition_evidence(
            evidence_root=evidence_root, gw_id=gw_id, camera_id=camera_id,
            outcome=outcome, timer=timer)

    # The in-memory sheet must never reach the JSON.
    result = {k: v for k, v in result.items() if k != "_sheet"}
    ident = result.get("wagon_identifier", C.NO_DATA)
    payload: Dict[str, Any] = {
        "global_id": gw_id,
        "feature": FEATURE_NAME,
        "camera_id": camera_id,
        "status": C.STATUS_OK,
        "candidates": ([{
            "full_number": result.get("raw_number", ""),
            "observations": 1,
            "mean_conf": result.get("ocr_confidence", 0.0),
            "yolo_conf": 0.0,
            "is_full_length": bool(result.get(valid_key)),
        }] if result.get("raw_number") else []),
        "detections": outcome.get("detections", 0),
        "supporting_cameras": [C.CAMERA_RIGHT_UP],
        "frame_count": n_frames,
        "evidence": evidence_paths,
    }
    payload.update(result)
    if verbose:
        log.info("  [ocr/%s] %s (conf=%.2f, bands=%d, calls=%d, frames=%d)",
                 gw_id, ident, float(result.get("ocr_confidence") or 0.0),
                 result.get("band_count", 0), result.get("rekognition_calls", 0),
                 n_frames)
    return C.STATUS_OK, payload


def _run_wagon_easyocr(
    *, yolo_model, ocr, cache_root: str, evidence_root: Optional[str],
    gw_id: str, camera_id: str, det_confidence: float,
    timer: FeatureTimer, verbose: bool,
):
    with phase(timer, "inference"):
        outcome = _process_one_wagon(
            yolo_model, ocr, cache_root, gw_id, det_confidence)
    used = outcome["frame_count"]
    aggregated = outcome["aggregated"]

    if used == 0:
        return C.STATUS_NO_FRAMES, empty_payload(
            gw_id, FEATURE_NAME, C.STATUS_NO_FRAMES,
            wagon_identifier=C.NO_DATA, wagon_identifier_confidence=0.0,
            candidates=[], supporting_cameras=[], engine=ENGINE_EASYOCR,
            display_number="-", is_valid_11_digit=False)

    # Build serialized candidate list from the aggregator's output
    candidates_out: List[Dict[str, Any]] = []
    for agg in aggregated:
        candidates_out.append({
            "full_number":     str(getattr(agg, "wagon_number", "")),
            "observations":    int(getattr(agg, "frame_count", 0)),
            "mean_conf":       float(getattr(agg, "avg_confidence", 0.0)),
            "yolo_conf":       float(getattr(agg, "avg_yolo_confidence",
                                      getattr(agg, "yolo_confidence", 0.0))),
            "is_full_length":  len(str(getattr(agg, "wagon_number", "")))
                               == C.WAGON_NUMBER_LENGTH,
        })

    # Aggregator already enforces min_frame_count + min_confidence.
    # The "best" candidate is the one with the highest combined
    # (observations, mean_conf) score.
    candidates_out.sort(
        key=lambda c: (
            -int(c["is_full_length"]),
            -c["observations"],
            -c["mean_conf"],
            c["full_number"],
        )
    )

    if candidates_out and candidates_out[0]["is_full_length"]:
        top = candidates_out[0]
        ident = top["full_number"]
        conf  = top["mean_conf"]
    else:
        ident = C.NO_DATA
        conf  = 0.0

    # Persist best-frame evidence:  full annotated frame + tight crop of the
    # wagon-number plate.
    evidence_paths: Dict[str, str] = {}
    best_obj = outcome.get("best")
    if evidence_root and best_obj is not None and best_obj.has_data():
        final_dir = os.path.join(evidence_root, gw_id, FEATURE_NAME, camera_id)
        crop_img = safe_crop(best_obj.frame, best_obj.bbox, pad=4)
        with phase(timer, "evidence"), atomic_camera_evidence(
                evidence_root, gw_id, FEATURE_NAME, camera_id) as ev_tmp:
            annotated = draw_annotated_bbox(
                best_obj.frame, best_obj.bbox,
                label=f"OCR {best_obj.meta.get('full_number','?')} "
                      f"{best_obj.meta.get('ocr_confidence',0.0):.2f}",
                color=(0, 255, 0),
            )
            save_jpeg(os.path.join(ev_tmp, "best_frame.jpg"), annotated)
            if crop_img is not None:
                save_jpeg(os.path.join(ev_tmp, "number_crop.jpg"), crop_img)
            write_metadata(os.path.join(ev_tmp, "metadata.json"), {
                "global_id":       gw_id,
                "feature":         FEATURE_NAME,
                "camera_id":       camera_id,
                "engine":          ENGINE_EASYOCR,
                "frame_idx":       best_obj.frame_idx,
                "bbox":            best_obj.bbox,
                "full_number":     best_obj.meta.get("full_number"),
                "ocr_confidence":  best_obj.meta.get("ocr_confidence"),
                "yolo_confidence": best_obj.meta.get("yolo_confidence"),
                "is_full_length":  best_obj.meta.get("is_full_length"),
                "aggregated_winner": ident,
                "aggregated_confidence": conf,
            })
        evidence_paths["best_frame"] = os.path.join(final_dir, "best_frame.jpg")
        if crop_img is not None:
            evidence_paths["number_crop"] = os.path.join(final_dir, "number_crop.jpg")

    is_valid = ident != C.NO_DATA
    payload: Dict[str, Any] = {
        "global_id":   gw_id,
        "feature":     FEATURE_NAME,
        "camera_id":   camera_id,
        "status":      C.STATUS_OK,
        "engine":      ENGINE_EASYOCR,
        "wagon_identifier":            ident,
        "wagon_identifier_confidence": round(float(conf), 4),
        # V4-parity fields so the per-camera inspection JSON is engine-agnostic
        "raw_number":       candidates_out[0]["full_number"] if candidates_out else "",
        "display_number":   ident if is_valid else "-",
        "is_valid_11_digit": bool(is_valid),
        "fallback_triggered": False,
        "candidates":  candidates_out[:8],
        "raw_candidates_first_8":      outcome["raw"][:8],
        "supporting_cameras": [C.CAMERA_RIGHT_UP],
        "frame_count": used,
        "evidence":    evidence_paths,
    }
    if verbose:
        log.info("  [ocr/%s] %s (conf=%.2f, candidates=%d, frames=%d)",
                 gw_id, ident, conf, len(candidates_out), used)
    return C.STATUS_OK, payload
