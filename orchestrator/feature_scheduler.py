"""The SINGLE Stage-3 scheduler shared by every execution mode.

There is exactly one feature-scheduling implementation in the pipeline -- this
one -- so `--auto` (incremental `lifecycle_runner.advance`) and
`--local-only`/`--once`/`--batch` (`master_runner.process_batch`) can never
diverge.  It is WAGON-WISE: the Global Wagon is the fundamental scheduling unit.

    FOR EACH Global Wagon (GW_1 .. GW_N):
        run every applicable feature across that wagon's cameras
            LOAD -> DAMAGE -> DOOR -> OCR
        (LOAD precedes DAMAGE so the loaded-wagon floor filter always sees this
         wagon's freshly-written load JSON)
    -> next wagon

Cameras only provide evidence for the wagon.  The feature ALGORITHMS are reused
UNCHANGED -- each processor is simply scoped to one wagon via `wagon_ids=[gw]`
(its per-(wagon,camera) code path, fresh tracker, evidence/snapshot selection,
and confidence are byte-for-byte identical to processing the whole camera; only
the outer iteration order differs).  Models load once (features._common
load_yolo cache).  Fusion/reports/videos/JSON/S3 are untouched downstream.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from core import constants as C

# LOAD must precede DAMAGE (floor filter dependency); door/ocr are independent.
WAGON_FEATURE_ORDER = ("load", "damage", "door", "ocr")

# Which cameras each feature may use (the processor filters to these anyway;
# OCR authority is RIGHT_UP only).
FEATURE_CAMERAS: Dict[str, List[str]] = {
    "door":   list(C.SIDE_CAMERAS),
    "ocr":    [C.CAMERA_RIGHT_UP],
    "load":   list(C.TOP_CAMERAS),
    "damage": list(C.TOP_CAMERAS),
}


def run_features_wagon_wise(
    *,
    state,
    cache_root: str,
    feat_models_dir: str,
    states_root: str,
    evidence_root: str,
    feature_cameras: Dict[str, List[str]],
    verbose: bool = True,
    log=None,
    on_error: Optional[Callable[[str, List[str], str, Exception], None]] = None,
) -> Dict[str, Dict[str, str]]:
    """Run Stage-3 features WAGON-WISE.

    Args:
        feature_cameras: {feature_key -> [camera_id, ...]} -- the work to run.
            A feature absent/empty here is simply not scheduled.  Callers decide
            this set (all present cameras for a full run; marker-filtered subsets
            for the incremental path).
        on_error: optional callback(feature, cameras, gw_id, exc) invoked when a
            feature raises for a wagon (used by the incremental path to mark the
            (camera, feature) FAILED).  Default: log + continue (fail-open).

    Returns {feature_key -> {gw_id -> status}} aggregated across all wagons.
    """
    from features.door   import processor as door_proc
    from features.load   import processor as load_proc
    from features.damage import processor as damage_proc
    from features.ocr    import processor as ocr_proc

    fmap = {
        "door":   door_proc.run,
        "load":   load_proc.run,
        "damage": damage_proc.run,
        "ocr":    ocr_proc.run,
    }
    active = [f for f in WAGON_FEATURE_ORDER if feature_cameras.get(f)]
    summary: Dict[str, Dict[str, str]] = {f: {} for f in active}
    n = len(state.wagons)
    if log is not None:
        log.info("[STAGE3] WAGON-WISE scheduling: %d wagons x features=%s "
                 "(each wagon runs all its features before the next)", n, active)

    for i, gw in enumerate(state.wagons, start=1):
        gwid = gw.global_id
        ran: List[str] = []
        for feat in active:                      # load -> damage -> door -> ocr
            cams = feature_cameras[feat]
            try:
                res = fmap[feat](
                    state=state, cache_root=cache_root,
                    feature_models_dir=feat_models_dir,
                    output_dir=states_root, evidence_root=evidence_root,
                    cameras=cams, wagon_ids=[gwid], verbose=False,
                )
                summary[feat].update(res or {})
                ran.append(feat)
            except Exception as e:               # never let one wagon abort the rest
                if on_error is not None:
                    on_error(feat, cams, gwid, e)
                elif log is not None:
                    log.error("[STAGE3/%s/%s] crashed: %s", feat, gwid, e,
                              exc_info=True)
        if log is not None:
            log.info("[STAGE3/wagon] %s (%d/%d): %s", gwid, i, n,
                     " ".join(ran) or "(none)")
    return summary
