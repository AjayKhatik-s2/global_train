"""Thin bridge: Global Train state  ->  legacy per-camera report inputs.

The ONLY job of this module is to expose the new Global Train artifacts
(`GlobalTrainState` + fused `UnifiedWagonState` + per-feature `wagon_states/`
+ `evidence/`) in the *exact dict shape* the unmodified old production report
generators already consume -- one `data` dict per camera:

    {
      "wagon_summary":   [ {wagon_number, start_frame, end_frame,
                            is_loaded, ocr_wagon_number, is_manipulated,
                            classification, global_id}, ... ],   # one per wagon
      "doors":   [ {wagon_number, door_number, state, confidence,
                    open_event_raised, local_snapshot_path, global_id}, ... ],
      # top cameras use "damages" instead of "doors":
      "damages": [ {wagon_number, damage_number, state, confidence,
                    local_snapshot_path, global_id}, ... ],
      "state_counts":     {OPEN, CLOSED, PARTIAL CLOSED, DAMAGE},
      "source_video_url": ..., "main_report_url": ..., "tracked_video_url": ...,
      "session_id":       ...,
    }

It performs NO inference, NO video decode, NO layout.  Snapshot paths and
per-camera frame ranges are resolved with the existing pure helpers in
`_evidence_lookup`.  Because every camera is numbered from the SAME global
wagon list, the four camera lists are already row-aligned -- the old combined
generator's cross-camera alignment therefore runs as a no-op (offset 0),
reproducing the production layout exactly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from core import constants as C
from core.global_state_loader import GlobalTrainState
from core.unified_wagon_state import UnifiedWagonState

from . import _evidence_lookup as ev


# ---------------------------------------------------------------------------
# State-string mapping: v4 canonical -> the exact strings the old generators'
# substring/format logic expects (DoorState.value / raw damage class_name).
# ---------------------------------------------------------------------------

_DOOR_STATE_TO_LEGACY = {
    C.DOOR_OPEN:    "OPEN",
    C.DOOR_CLOSED:  "CLOSED",
    C.DOOR_PARTIAL: "PARTIAL_CLOSED",   # -> "PARTIAL CLOSED" after _format
    C.DOOR_DAMAGED: "DAMAGE",
}

_SIDE_FOR_CAMERA = {C.CAMERA_RIGHT_UP: "right", C.CAMERA_LEFT_UP: "left"}

# Segment-type label the old TOP report uses for non-wagon overview pages.
_SEGMENT_TYPE = {C.CLASS_ENGINE: "engine", C.CLASS_BRAKE_VAN: "brakevan"}


class _LegacyOcr:
    """Minimal stand-in for the old OCR `WagonNumber` object.

    The old generators read `.is_valid` (+ structured parts when valid) or fall
    back to `.full_number`.  We expose exactly those attributes so the ported
    code renders the identifier without modification.  When the reconstructed
    identifier is a full 5-part IR number (``TT-RR-YY-NNNNN-C``) it is parsed
    into parts and marked valid; otherwise `is_valid=False` and the raw string
    is available as `full_number`.
    """

    __slots__ = ("is_valid", "full_number", "wagon_type", "owning_railway",
                 "year_of_manufacture", "individual_number", "check_digit",
                 "wagon_number", "is_manipulated")

    def __init__(self, identifier: str):
        self.full_number = str(identifier)
        self.is_manipulated = False
        parts = str(identifier).split("-")
        if len(parts) == 5 and all(parts):
            (self.wagon_type, self.owning_railway, self.year_of_manufacture,
             self.individual_number, self.check_digit) = parts
            self.wagon_number = parts[3]
            self.is_valid = True
        else:
            self.wagon_type = self.owning_railway = self.year_of_manufacture = ""
            self.individual_number = self.check_digit = ""
            self.wagon_number = str(identifier)
            self.is_valid = False


def _make_ocr(identifier: Optional[str]):
    if identifier in (None, "", C.NO_DATA):
        return None
    return _LegacyOcr(identifier)


def _legacy_door_state(v4_state: str) -> str:
    return _DOOR_STATE_TO_LEGACY.get(v4_state, str(v4_state or "").upper())


def _is_detected(state: Optional[str]) -> bool:
    return state not in (None, "", C.NO_DATA)


def _local_frames(meta: Dict[str, Any], gw) -> Sequence[int]:
    """(start_frame, end_frame) in THIS camera's local frame space."""
    return ev.wagon_local_frames(
        gw.start_time, gw.end_time,
        float(meta.get("fps") or 0.0), int(meta.get("total_frames") or 0),
    )


def _load_image(path: Optional[str]):
    """Load an evidence snapshot as a BGR numpy array (old passed the door/damage
    snapshot in-memory, not as a path).  cv2 is imported lazily so the reportlab-
    only report path never eagerly pulls in OpenCV; returns None on any failure."""
    if not path:
        return None
    try:
        import cv2
        return cv2.imread(path)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Per-camera builders
# ---------------------------------------------------------------------------

def _side_camera_payload(
    *, camera_id: str, state: GlobalTrainState,
    unified: Dict[str, UnifiedWagonState],
    evidence_root: Optional[str], cam_meta: Dict[str, Any],
) -> Dict[str, Any]:
    side = _SIDE_FOR_CAMERA[camera_id]
    is_right = camera_id == C.CAMERA_RIGHT_UP

    wagon_summary: List[Dict[str, Any]] = []
    doors: List[Dict[str, Any]] = []
    counts = {"OPEN": 0, "CLOSED": 0, "PARTIAL CLOSED": 0, "DAMAGE": 0}

    for idx, gw in enumerate(state.wagons, start=1):
        u = unified.get(gw.global_id)
        sf, ef = _local_frames(cam_meta, gw)

        ocr = _make_ocr(u.wagon_identifier) if (is_right and u is not None) else None
        cls = (u.classification if u else gw.classification)

        wagon_summary.append({
            "wagon_number":     idx,
            "global_id":        gw.global_id,
            "start_frame":      int(sf),
            "end_frame":        int(ef),
            "classification":   cls,
            "is_non_wagon":     cls in (C.CLASS_ENGINE, C.CLASS_BRAKE_VAN),
            "segment_type":     _SEGMENT_TYPE.get(cls),
            "is_loaded":        False,
            "ocr_wagon_number": ocr,
            "is_manipulated":   False,
        })

        if u is None:
            continue
        d_state = getattr(u, f"{side}_door", C.NO_DATA)
        d_conf = float(getattr(u, f"{side}_door_confidence", 0.0) or 0.0)
        if not _is_detected(d_state):
            continue

        legacy_state = _legacy_door_state(d_state)
        # tally for the per-camera Detection Summary
        fmt = legacy_state.upper().replace("_", " ")
        if "OPEN" in fmt:
            counts["OPEN"] += 1
        elif "PARTIAL" in fmt:
            counts["PARTIAL CLOSED"] += 1
        elif "DAMAGE" in fmt:
            counts["DAMAGE"] += 1
        elif "CLOSED" in fmt:
            counts["CLOSED"] += 1

        # snapshot only meaningful for open/damage (the images page filters anyway)
        # `local_snapshot_path` -> the full annotated frame (combined report reads
        # it as a path).  `snapshot` -> the OLD refined+annotated door CROP loaded
        # as a numpy array, which is what the per-camera side generator embeds on
        # its door-detail / priority pages (it reads door['snapshot'], not a path).
        snap = ev.evidence_snapshot(evidence_root, gw.global_id, "door",
                                    f"{side}_best", camera_id=camera_id)
        crop = ev.evidence_snapshot(evidence_root, gw.global_id, "door",
                                    f"{side}_snapshot", camera_id=camera_id)
        doors.append({
            "wagon_number":        idx,
            "global_id":           gw.global_id,
            "door_id":             idx,          # one door per wagon-side; idx is unique
            "door_number":         1,
            "state":               legacy_state,
            "confidence":          d_conf,
            "open_event_raised":   True,
            "local_snapshot_path": snap,
            "snapshot":            _load_image(crop),
        })

    return {"wagon_summary": wagon_summary, "doors": doors, "state_counts": counts}


def _top_camera_payload(
    *, camera_id: str, state: GlobalTrainState,
    unified: Dict[str, UnifiedWagonState],
    evidence_root: Optional[str], wagon_states_root: Optional[str],
    cam_meta: Dict[str, Any],
) -> Dict[str, Any]:
    wagon_summary: List[Dict[str, Any]] = []
    damages: List[Dict[str, Any]] = []
    counts = {"OPEN": 0, "CLOSED": 0, "PARTIAL CLOSED": 0, "DAMAGE": 0}

    for idx, gw in enumerate(state.wagons, start=1):
        u = unified.get(gw.global_id)
        sf, ef = _local_frames(cam_meta, gw)
        is_loaded = bool(u and u.load_status == C.LOAD_LOADED)

        cls = (u.classification if u else gw.classification)
        wagon_summary.append({
            "wagon_number":   idx,
            "global_id":      gw.global_id,
            "start_frame":    int(sf),
            "end_frame":      int(ef),
            "classification": cls,
            "is_non_wagon":   cls in (C.CLASS_ENGINE, C.CLASS_BRAKE_VAN),
            "segment_type":   _SEGMENT_TYPE.get(cls),
            "is_loaded":      is_loaded,
        })

        # damage tracks for THIS top camera only (never the sibling top camera)
        for snap_path, tr in ev.damage_track_snapshots(
                evidence_root, gw.global_id, camera_id=camera_id):
            cls = str(tr.get("class_name") or "damage").lower()
            # `snapshot` = the OLD annotate-then-crop report image (numpy); the
            # top generator reads damage['snapshot'] for its damage-detail /
            # priority pages.  `local_snapshot_path` (annotated full frame) is
            # kept for the combined report, which reads it as a path.
            crop = ev.evidence_snapshot(
                evidence_root, gw.global_id, "damage",
                f"track_{int(tr.get('track_idx') or 0)}_snapshot", camera_id=camera_id)
            damages.append({
                "wagon_number":        idx,
                "global_id":           gw.global_id,
                "damage_id":           int(tr.get("track_idx") or (len(damages) + 1)),
                "damage_number":       int(tr.get("track_idx") or (len(damages) + 1)),
                "state":               cls,
                "confidence":          float(tr.get("best_confidence") or 0.0),
                "local_snapshot_path": snap_path,
                "snapshot":            _load_image(crop),
                "frame_idx":           tr.get("best_frame_idx"),
                "bbox":                tr.get("bbox"),
            })
            counts["DAMAGE"] += 1

    return {"wagon_summary": wagon_summary, "damages": damages, "state_counts": counts}


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def build_camera_payloads(
    *,
    state: GlobalTrainState,
    unified: Dict[str, UnifiedWagonState],
    evidence_root: Optional[str] = None,
    wagon_states_root: Optional[str] = None,
    per_camera_tracking_path: Optional[str] = None,
    source_video_urls: Optional[Dict[str, str]] = None,
    tracked_video_urls: Optional[Dict[str, str]] = None,
    camera_pdf_urls: Optional[Dict[str, str]] = None,
    session_id: str = "",
) -> Dict[str, Dict[str, Any]]:
    """Return {camera_id -> legacy `data` dict} for all four cameras.

    Cameras with no reconstructed contribution still get a fully-shaped dict
    (empty doors/damages, wagon_summary sized to the global wagon count) so the
    old generators render them consistently.
    """
    source_video_urls = source_video_urls or {}
    tracked_video_urls = tracked_video_urls or {}
    camera_pdf_urls = camera_pdf_urls or {}
    per_cam_meta = ev.load_per_camera_meta(per_camera_tracking_path)

    out: Dict[str, Dict[str, Any]] = {}
    for cam in C.ALL_CAMERAS:
        meta = per_cam_meta.get(cam, {})
        if cam in C.SIDE_CAMERAS:
            payload = _side_camera_payload(
                camera_id=cam, state=state, unified=unified,
                evidence_root=evidence_root, cam_meta=meta,
            )
        else:
            payload = _top_camera_payload(
                camera_id=cam, state=state, unified=unified,
                evidence_root=evidence_root, wagon_states_root=wagon_states_root,
                cam_meta=meta,
            )
        payload["source_video_url"] = source_video_urls.get(cam)
        payload["main_report_url"] = camera_pdf_urls.get(cam)
        payload["tracked_video_url"] = tracked_video_urls.get(cam)
        payload["session_id"] = session_id
        out[cam] = payload
    return out


def split_for_combined(
    payloads: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Map the {camera -> data} dict to the old generator's kwargs names."""
    return {
        "left_data":     payloads.get(C.CAMERA_LEFT_UP, {}),
        "right_data":    payloads.get(C.CAMERA_RIGHT_UP, {}),
        "top_data":      payloads.get(C.CAMERA_RIGHT_UP_TOP, {}),
        "left_top_data": payloads.get(C.CAMERA_LEFT_UP_TOP, {}),
    }
