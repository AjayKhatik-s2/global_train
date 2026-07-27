"""Global Train state -> per-camera INSPECTION-REPORT view model.

Feeds `reporting.inspection_report`, which reproduces the per-camera
"TRAIN INSPECTION REPORT" layout (title page -> WAGON STATUS SUMMARY ->
DAMAGE / DOOR DETECTED -> per-wagon problem frames -> LOCOMOTIVES ->
per-wagon frame pages).

Like `_legacy_data_adapter`, this module performs NO inference, NO video
decode, and NO layout.  It only reads artifacts already on disk:

    wagon_states/<feature>/<CAMERA>/<gw>.json     per-camera feature results
    evidence/<gw>/<feature>/<CAMERA>/...          per-camera snapshots + metadata
    wagon_cache/<gw>/<camera_folder>/frame_*.jpg  materialized frames
    global_state/per_camera_tracking.json         per-camera fps / total_frames

Every value is READ from a camera's OWN namespace -- a top camera never
reports the sibling top camera's damage, and a side camera never reports the
other side's door.  That mirrors the authority rules the rest of the pipeline
already enforces.

Fields with no source in the v4 artifacts are reported honestly rather than
invented (see `CameraStyle.rake_for` / `resolve_direction`):

  * `direction`  -- optical-flow travel direction is computed only in the raw
                    `train_extraction` stage and is not carried into any
                    finalized artifact (the same gap `delivery.dashboard_ingest`
                    documents).  Defaults to "unknown", which renders as
                    "DIRECTION UNKNOWN".  Override per camera with
                    WAGONEYE_CAMERA_DIRECTION_<CAMERA_ID>.
  * `station`    -- not present anywhere in the pipeline.  Defaults to
                    HAZARIBAGH; override with WAGONEYE_STATION.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core import constants as C
from core.global_state_loader import GlobalTrainState
from core.unified_wagon_state import UnifiedWagonState

from . import _evidence_lookup as ev


# ---------------------------------------------------------------------------
# Per-camera presentation style
# ---------------------------------------------------------------------------

# Which travel direction means LOADED for each camera.  The two top cameras do
# NOT share a convention -- RIGHT_UP_TOP is physically inverted relative to the
# others, so left-to-right means EMPTY there.
LOADED_DIRECTION: Dict[str, str] = {
    C.CAMERA_RIGHT_UP:     "left-to-right",
    C.CAMERA_LEFT_UP:      "left-to-right",
    C.CAMERA_RIGHT_UP_TOP: "right-to-left",   # inverted convention
    C.CAMERA_LEFT_UP_TOP:  "left-to-right",
}

CAMERA_LABEL: Dict[str, str] = {
    C.CAMERA_RIGHT_UP:     "RIGHT UP",
    C.CAMERA_LEFT_UP:      "LEFT UP",
    C.CAMERA_RIGHT_UP_TOP: "RIGHT TOP",
    C.CAMERA_LEFT_UP_TOP:  "LEFT TOP",
}

# 'side' cameras report door state; 'top' cameras report damage.
CAMERA_FLAVOUR: Dict[str, str] = {
    C.CAMERA_RIGHT_UP:     "side",
    C.CAMERA_LEFT_UP:      "side",
    C.CAMERA_RIGHT_UP_TOP: "top",
    C.CAMERA_LEFT_UP_TOP:  "top",
}

DEFAULT_STATION = "HAZARIBAGH"

# Status cell colours (hex, applied to the STATUS column of the summary table).
STATUS_COLOR = {
    "DAMAGE":          "#ff5252",
    "DOOR OPEN":       "#ffb74d",
    "PROBABLE DAMAGE": "#ffb74d",
    "OK":              "#81c784",
}


@dataclass
class CameraStyle:
    """Per-camera display strings + direction -> rake-type mapping."""
    camera_id: str
    camera_label: str
    station_name: str
    flavour: str                      # "side" | "top"
    loaded_direction: str

    def rake_for(self, direction: str) -> tuple:
        """Return (rake_type, hex_color, arrow) for a travel direction.

        An unknown / unavailable direction yields DIRECTION UNKNOWN rather than
        guessing a rake type."""
        d = (direction or "").strip().lower()
        if d not in ("left-to-right", "right-to-left"):
            return ("DIRECTION UNKNOWN", "#808080", "<->")
        if d == self.loaded_direction:
            return ("LOADED RAKE", "#00008b", "->")
        return ("EMPTY RAKE", "#006400", "<-")


def camera_style(camera_id: str, *, station: Optional[str] = None) -> CameraStyle:
    return CameraStyle(
        camera_id=camera_id,
        camera_label=CAMERA_LABEL.get(camera_id, camera_id.replace("_", " ")),
        station_name=station or os.getenv("WAGONEYE_STATION") or DEFAULT_STATION,
        flavour=CAMERA_FLAVOUR.get(camera_id, "side"),
        loaded_direction=LOADED_DIRECTION.get(camera_id, "left-to-right"),
    )


def resolve_direction(camera_id: str, supplied: Optional[str] = None) -> str:
    """Travel direction for one camera.

    v4 does not persist optical-flow direction anywhere downstream of raw
    extraction, so this is "unknown" unless a caller supplies it or the
    WAGONEYE_CAMERA_DIRECTION_<CAMERA_ID> env var is set."""
    if supplied:
        return supplied
    return os.getenv(f"WAGONEYE_CAMERA_DIRECTION_{camera_id}", "unknown")


# ---------------------------------------------------------------------------
# View-model rows
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    """One GlobalWagon as seen by ONE camera."""
    segment_id: int                   # 1-based wagon index (GW_n -> n)
    global_id: str
    segment_type: str                 # wagon | wagon_loaded | engine | brakevan
    start_frame: int                  # this camera's LOCAL frame space
    end_frame: int
    directory: str                    # wagon_cache/<gw>/<camera_folder>

    @property
    def is_wagon(self) -> bool:
        return self.segment_type in ("wagon", "wagon_loaded")


@dataclass
class DamageRow:
    """This camera's own verdict for one wagon."""
    wagon_id: int
    damage_detected: bool = False
    probable_damage_detected: bool = False
    door_status: str = ""             # "open" when this side's door is OPEN


@dataclass
class ProblemFrame:
    """One annotated snapshot to render under 'Problem Frames'."""
    wagon_id: int
    problem_type: str                 # raw detector class name
    frame_number: int
    image_path: Optional[str]


@dataclass
class LocoRow:
    loco_id: int
    frame_path: Optional[str]


@dataclass
class CameraInspectionModel:
    """Everything `inspection_report` needs to render one camera."""
    style: CameraStyle
    direction: str
    raw_video_name: str
    generated_at: datetime
    trimmed_video_url: Optional[str]
    segments: List[Segment] = field(default_factory=list)
    damage_rows: Dict[int, DamageRow] = field(default_factory=dict)
    problem_frames: List[ProblemFrame] = field(default_factory=list)
    locos: List[LocoRow] = field(default_factory=list)

    # -- derived ---------------------------------------------------------
    @property
    def total_wagons(self) -> int:
        return sum(1 for s in self.segments if s.is_wagon)

    @property
    def damaged_wagons(self) -> int:
        return sum(1 for s in self.segments
                   if s.is_wagon
                   and self.damage_rows.get(s.segment_id, DamageRow(s.segment_id)).damage_detected)

    def status_for(self, segment_id: int) -> str:
        """Summary-table status for one wagon, per this camera's flavour."""
        row = self.damage_rows.get(segment_id) or DamageRow(segment_id)
        if self.style.flavour == "top":
            if row.damage_detected:
                return "DAMAGE"
            if row.probable_damage_detected:
                return "PROBABLE DAMAGE"
            return "OK"
        if row.damage_detected:
            return "DAMAGE"
        if row.door_status == "open":
            return "DOOR OPEN"
        return "OK"

    def problem_frames_for(self, segment_id: int) -> List[ProblemFrame]:
        return [p for p in self.problem_frames if p.wagon_id == segment_id]


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _segment_type(gw, unified: Dict[str, UnifiedWagonState]) -> str:
    cls = gw.classification
    if cls == C.CLASS_ENGINE:
        return "engine"
    if cls == C.CLASS_BRAKE_VAN:
        return "brakevan"
    u = unified.get(gw.global_id)
    if u is not None and u.load_status == C.LOAD_LOADED:
        return "wagon_loaded"
    return "wagon"


def _build_segments(
    *, state: GlobalTrainState, unified: Dict[str, UnifiedWagonState],
    camera_id: str, cache_root: Optional[str], cam_meta: Dict[str, Any],
) -> List[Segment]:
    folder = C.CAMERA_FOLDER.get(camera_id, camera_id.lower())
    fps = float(cam_meta.get("fps") or 0.0)
    total = int(cam_meta.get("total_frames") or 0)
    out: List[Segment] = []
    for gw in state.wagons:
        sf, ef = ev.wagon_local_frames(gw.start_time, gw.end_time, fps, total)
        out.append(Segment(
            segment_id=int(gw.wagon_index),
            global_id=gw.global_id,
            segment_type=_segment_type(gw, unified),
            start_frame=int(sf),
            end_frame=int(ef),
            directory=(os.path.join(cache_root, gw.global_id, folder)
                       if cache_root else ""),
        ))
    return out


def _build_side(
    *, state: GlobalTrainState, camera_id: str,
    wagon_states_root: Optional[str], evidence_root: Optional[str],
) -> tuple:
    """Door verdicts + problem frames for one SIDE camera."""
    side = "right" if camera_id == C.CAMERA_RIGHT_UP else "left"
    rows: Dict[int, DamageRow] = {}
    problems: List[ProblemFrame] = []

    for gw in state.wagons:
        wid = int(gw.wagon_index)
        payload = ev.read_wagon_feature_json(
            wagon_states_root, "door", gw.global_id, camera_id=camera_id)
        door_state = str(payload.get("door_state") or C.NO_DATA).upper()
        rows[wid] = DamageRow(
            wagon_id=wid,
            damage_detected=(door_state == C.DOOR_DAMAGED),
            door_status=("open" if door_state == C.DOOR_OPEN else ""),
        )

        # A problem page is emitted for every wagon that produced a door
        # snapshot -- the caption carries the RAW detector class, so a
        # closed-door frame is shown as closed rather than suppressed.
        meta = ev.evidence_metadata(evidence_root, gw.global_id, "door",
                                    camera_id=camera_id)
        info = (meta.get("sides") or {}).get(side) or {}
        img = ev.evidence_snapshot(evidence_root, gw.global_id, "door",
                                   f"{side}_best", camera_id=camera_id)
        if img:
            problems.append(ProblemFrame(
                wagon_id=wid,
                problem_type=str(info.get("raw_class")
                                 or info.get("state") or "door"),
                frame_number=int(info.get("frame_idx") or 0),
                image_path=img,
            ))
    return rows, problems


def _build_top(
    *, state: GlobalTrainState, camera_id: str,
    wagon_states_root: Optional[str], evidence_root: Optional[str],
) -> tuple:
    """Damage verdicts + problem frames for one TOP camera."""
    rows: Dict[int, DamageRow] = {}
    problems: List[ProblemFrame] = []

    for gw in state.wagons:
        wid = int(gw.wagon_index)
        payload = ev.read_wagon_feature_json(
            wagon_states_root, "damage", gw.global_id, camera_id=camera_id)
        rows[wid] = DamageRow(
            wagon_id=wid,
            damage_detected=(payload.get("damage_status") == C.DAMAGE_PRESENT),
            # v4 has no "probable damage" tier -- the damage tracker emits a
            # confirmed track or nothing.  Never fabricated.
            probable_damage_detected=False,
        )
        for img, tr in ev.damage_track_snapshots(
                evidence_root, gw.global_id, camera_id=camera_id):
            problems.append(ProblemFrame(
                wagon_id=wid,
                problem_type=str(tr.get("class_name") or "damage"),
                frame_number=int(tr.get("best_frame_idx") or 0),
                image_path=img,
            ))
    return rows, problems


def _build_locos(
    *, state: GlobalTrainState, camera_id: str, cache_root: Optional[str],
    cam_meta: Dict[str, Any],
) -> List[LocoRow]:
    fps = float(cam_meta.get("fps") or 0.0)
    total = int(cam_meta.get("total_frames") or 0)
    out: List[LocoRow] = []
    n = 0
    for gw in state.wagons:
        if gw.classification != C.CLASS_ENGINE:
            continue
        n += 1
        out.append(LocoRow(
            loco_id=n,
            frame_path=ev.midpoint_cache_path(
                cache_root=cache_root, gw_id=gw.global_id, camera_id=camera_id,
                wagon_start_time=gw.start_time, wagon_end_time=gw.end_time,
                local_fps=fps, local_total_frames=total,
            ),
        ))
    return out


def _parse_batch_timestamp(batch_key: str) -> datetime:
    """batch_key is a canonical YYYYMMDD_HHMMSS train timestamp."""
    try:
        return datetime.strptime(batch_key, "%Y%m%d_%H%M%S")
    except (ValueError, TypeError):
        return datetime.now(timezone.utc).replace(tzinfo=None)


def build_model(
    *,
    camera_id: str,
    state: GlobalTrainState,
    unified: Dict[str, UnifiedWagonState],
    batch_key: str,
    cache_root: Optional[str] = None,
    wagon_states_root: Optional[str] = None,
    evidence_root: Optional[str] = None,
    per_camera_tracking_path: Optional[str] = None,
    video_path: Optional[str] = None,
    trimmed_video_url: Optional[str] = None,
    direction: Optional[str] = None,
    station: Optional[str] = None,
    generated_at: Optional[datetime] = None,
) -> CameraInspectionModel:
    """Assemble the inspection view model for ONE camera."""
    style = camera_style(camera_id, station=station)
    cam_meta = ev.load_per_camera_meta(per_camera_tracking_path).get(camera_id, {})

    segments = _build_segments(
        state=state, unified=unified, camera_id=camera_id,
        cache_root=cache_root, cam_meta=cam_meta,
    )
    if style.flavour == "side":
        damage_rows, problems = _build_side(
            state=state, camera_id=camera_id,
            wagon_states_root=wagon_states_root, evidence_root=evidence_root)
    else:
        damage_rows, problems = _build_top(
            state=state, camera_id=camera_id,
            wagon_states_root=wagon_states_root, evidence_root=evidence_root)

    raw_name = (os.path.splitext(os.path.basename(video_path))[0]
                if video_path else f"{camera_id}_{batch_key}")

    return CameraInspectionModel(
        style=style,
        direction=resolve_direction(camera_id, direction),
        raw_video_name=raw_name,
        generated_at=generated_at or _parse_batch_timestamp(batch_key),
        trimmed_video_url=trimmed_video_url,
        segments=segments,
        damage_rows=damage_rows,
        problem_frames=problems,
        locos=_build_locos(state=state, camera_id=camera_id,
                           cache_root=cache_root, cam_meta=cam_meta),
    )
