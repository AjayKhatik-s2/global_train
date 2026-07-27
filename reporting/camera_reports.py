"""Stage 5 per-camera reports -- THIN wrapper over the OLD production
per-camera generators.

Side cameras (RIGHT_UP / LEFT_UP)  -> `report_generator.DoorReportGenerator`
Top  cameras (RIGHT_UP_TOP / LEFT_UP_TOP) -> `damage_report_generator.DamageReportGenerator`

Both old generators are used UNMODIFIED (only the /tmp portability fix + the
`require_open_event` parameter that folds the LEFT_UP vs RIGHT_UP behavioural
difference into one class).  The Global Train state is exposed in the old
per-camera `data` dict shape by `_legacy_data_adapter`; each camera's wagon
overview frames are extracted by the old generator directly from that camera's
local video (`video_paths[camera]`) at the per-camera local frame ranges the
adapter computed.

`build_all(...)` keeps the orchestrator-facing name + `CAMERA_FILE` mapping so
`orchestrator/lifecycle_runner.stage_reports` is unchanged apart from passing
`video_paths` through.  Returns {camera_id -> pdf_path|None}.
"""

from __future__ import annotations

import os
import time
import traceback
from typing import Dict, List, Optional

from core import constants as C
from core.global_state_loader import GlobalTrainState
from core.unified_wagon_state import UnifiedWagonState

from . import _legacy_data_adapter as LDA
from .report_generator import DoorReportGenerator
from .damage_report_generator import DamageReportGenerator


# Per-camera output filenames (read back by stage_reports for the combined
# report's sibling links).  Same names the previous stage used.
CAMERA_FILE = {
    C.CAMERA_RIGHT_UP:     "right_up_report.pdf",
    C.CAMERA_LEFT_UP:      "left_up_report.pdf",
    C.CAMERA_RIGHT_UP_TOP: "right_up_top_report.pdf",
    C.CAMERA_LEFT_UP_TOP:  "left_up_top_report.pdf",
}


def _build_one(
    *, camera_id: str, payload: Dict, output_pdf: str,
    video_path: Optional[str], source_video_url: Optional[str],
    logo_path: Optional[str],
) -> Optional[str]:
    """Render ONE camera's PDF with the matching old generator.

    The old SageMaker `generate_report` calls also passed an event count
    (`open_door_events` / `damage_events`) and a `full_wagon_summary` (all
    segments incl. engine/brake-van).  We reproduce that exact call signature:
    the event count is the number of OPEN/DAMAGE anomalies, and
    `full_wagon_summary` is the adapter's wagon_summary (already all segments,
    with the `is_non_wagon` flag that drives the non-wagon pages).  Both old
    generators recompute their displayed counts internally and fall back to
    `wagon_summary` when `full_wagon_summary` is absent, so these carry no
    rendering change for engine/brake-van-free trains -- proven byte-identical
    in scratchpad/compare_{camera,top}_report.py -- but the call now matches
    old production exactly."""
    video_path = video_path or ""
    wagon_summary = payload.get("wagon_summary", [])
    state_counts = payload.get("state_counts", {})
    if camera_id in C.SIDE_CAMERAS:
        doors = payload.get("doors", [])
        open_events = sum(1 for d in doors if str(d.get("state", "")).upper() == "OPEN")
        gen = DoorReportGenerator(
            output_path=output_pdf, video_path=video_path,
            source_video_url=source_video_url, logo_path=logo_path,
            require_open_event=(camera_id == C.CAMERA_LEFT_UP),
        )
        gen.generate_report(
            doors=doors,
            state_counts=state_counts,
            processing_time=0.0,
            door_open_events=open_events,
            wagon_summary=wagon_summary,
            full_wagon_summary=wagon_summary,
        )
    else:
        damages = payload.get("damages", [])
        damage_events = sum(1 for d in damages if str(d.get("state", "")).upper() == "DAMAGE")
        gen = DamageReportGenerator(
            output_path=output_pdf, video_path=video_path,
            source_video_url=source_video_url, logo_path=logo_path,
        )
        gen.generate_report(
            damages=damages,
            state_counts=state_counts,
            processing_time=0.0,
            damage_events=damage_events,
            wagon_summary=wagon_summary,
            full_wagon_summary=wagon_summary,
        )
    return output_pdf if os.path.isfile(output_pdf) else None


def build_all(
    *,
    state: GlobalTrainState,
    unified: Dict[str, UnifiedWagonState],
    evidence_root: str,
    output_dir: str,
    batch_key: str,
    wagon_states_root: Optional[str] = None,
    cache_root: Optional[str] = None,
    per_camera_tracking_path: Optional[str] = None,
    video_paths: Optional[Dict[str, str]] = None,
    source_video_urls: Optional[Dict[str, str]] = None,
    logo_path: Optional[str] = None,
    cameras: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, Optional[str]]:
    """Build camera-wise PDFs (old layout).  Returns {camera_id -> path|None}.

    `cameras` restricts regeneration to a subset (a late camera rebuilds only
    its own PDF).  Independent failures never block the other cameras."""
    del cache_root  # accepted for signature parity; frames come from video_paths
    os.makedirs(output_dir, exist_ok=True)
    video_paths = video_paths or {}
    source_video_urls = source_video_urls or {}
    t0 = time.time()

    payloads = LDA.build_camera_payloads(
        state=state, unified=unified,
        evidence_root=evidence_root, wagon_states_root=wagon_states_root,
        per_camera_tracking_path=per_camera_tracking_path,
        source_video_urls=source_video_urls, session_id=batch_key,
    )

    target = set(cameras) if cameras is not None else set(C.ALL_CAMERAS)
    out: Dict[str, Optional[str]] = {}
    for camera_id in C.ALL_CAMERAS:
        if camera_id not in target:
            continue
        path = os.path.join(output_dir, CAMERA_FILE[camera_id])
        try:
            out[camera_id] = _build_one(
                camera_id=camera_id, payload=payloads.get(camera_id, {}),
                output_pdf=path, video_path=video_paths.get(camera_id),
                source_video_url=source_video_urls.get(camera_id),
                logo_path=logo_path,
            )
        except Exception as e:
            print(f"[CAMERA_REPORT/{camera_id}] FAILED: {type(e).__name__}: {e}")
            traceback.print_exc(limit=3)
            out[camera_id] = None

    if verbose:
        n_ok = sum(1 for v in out.values() if v)
        print(f"[STAGE5] camera reports: {n_ok}/{len(out)} ok "
              f"in {time.time() - t0:.1f}s")
    return out
