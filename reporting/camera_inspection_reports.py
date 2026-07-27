"""Stage 5a-bis -- per-camera "TRAIN INSPECTION REPORT" PDFs.

This is a SECOND, additive per-camera report.  It does not replace, modify, or
read anything produced by `camera_reports.build_all` (the old-production
per-camera PDFs) or by `combined_train_report` -- those keep their filenames,
their layout, and their documented byte-parity guarantee.

    camera_reports.build_all             -> <cam>_report.pdf             (existing)
    camera_inspection_reports.build_all  -> <cam>_inspection_report.pdf  (this)

Rendering happens in `inspection_report`; the v4 -> view-model mapping is in
`_inspection_adapter`.  Both are pure consumers of already-finalized artifacts:
no model load, no video decode, no mutation of sealed state.

A per-camera failure is contained -- it is logged and that camera returns None,
leaving the other three cameras and every other report unaffected.
"""

from __future__ import annotations

import os
import time
import traceback
from typing import Dict, List, Optional

from core import constants as C
from core.global_state_loader import GlobalTrainState
from core.logging_setup import get_logger
from core.unified_wagon_state import UnifiedWagonState

from . import _inspection_adapter as IA
from . import inspection_report

log = get_logger("reporting.inspection")


# Output filenames -- deliberately distinct from camera_reports.CAMERA_FILE so
# the two report families can coexist in the same reports/ directory.
CAMERA_INSPECTION_FILE = {
    C.CAMERA_RIGHT_UP:     "right_up_inspection_report.pdf",
    C.CAMERA_LEFT_UP:      "left_up_inspection_report.pdf",
    C.CAMERA_RIGHT_UP_TOP: "right_up_top_inspection_report.pdf",
    C.CAMERA_LEFT_UP_TOP:  "left_up_top_inspection_report.pdf",
}


def build_all(
    *,
    state: GlobalTrainState,
    unified: Dict[str, UnifiedWagonState],
    output_dir: str,
    batch_key: str,
    cache_root: Optional[str] = None,
    wagon_states_root: Optional[str] = None,
    evidence_root: Optional[str] = None,
    per_camera_tracking_path: Optional[str] = None,
    video_paths: Optional[Dict[str, str]] = None,
    source_video_urls: Optional[Dict[str, str]] = None,
    directions: Optional[Dict[str, str]] = None,
    station: Optional[str] = None,
    logo_path: Optional[str] = None,
    cameras: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, Optional[str]]:
    """Build per-camera inspection PDFs.  Returns {camera_id -> path | None}.

    `cameras` restricts regeneration to a subset so a late-arriving camera
    rebuilds only its own PDF (same contract as `camera_reports.build_all`).
    """
    os.makedirs(output_dir, exist_ok=True)
    video_paths = video_paths or {}
    source_video_urls = source_video_urls or {}
    directions = directions or {}
    target = set(cameras) if cameras is not None else set(C.ALL_CAMERAS)
    t0 = time.time()

    out: Dict[str, Optional[str]] = {}
    for camera_id in C.ALL_CAMERAS:
        if camera_id not in target:
            continue
        path = os.path.join(output_dir, CAMERA_INSPECTION_FILE[camera_id])
        try:
            model = IA.build_model(
                camera_id=camera_id, state=state, unified=unified,
                batch_key=batch_key, cache_root=cache_root,
                wagon_states_root=wagon_states_root,
                evidence_root=evidence_root,
                per_camera_tracking_path=per_camera_tracking_path,
                video_path=video_paths.get(camera_id),
                trimmed_video_url=source_video_urls.get(camera_id),
                direction=directions.get(camera_id),
                station=station,
            )
            inspection_report.build(model=model, output_path=path,
                                    logo_path=logo_path)
            out[camera_id] = path if os.path.isfile(path) else None
        except Exception as e:
            log.error("[INSPECTION/%s] FAILED: %s: %s",
                      camera_id, type(e).__name__, e)
            log.debug("%s", traceback.format_exc(limit=4))
            out[camera_id] = None

    if verbose:
        n_ok = sum(1 for v in out.values() if v)
        log.info("[STAGE5] camera inspection reports: %d/%d ok in %.1fs",
                 n_ok, len(out), time.time() - t0)
    return out
