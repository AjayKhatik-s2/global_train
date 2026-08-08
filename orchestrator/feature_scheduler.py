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

import os
from typing import Any, Callable, Dict, List, Optional

from core import constants as C


def wagon_workers() -> int:
    """How many wagons to process at once.  1 (the default) = sequential.

    Stage 3 was strictly one wagon at a time, which on the 16-vCPU production
    box left ~13 cores idle.  Wagons are independent, so they parallelise
    cleanly -- but this stays OFF by default because it changes the memory
    profile (one YOLO instance per feature PER THREAD; damage.pt alone is
    ~200 MB) and because the sequential path is the one that has actually
    produced a correct report.

    WAGONEYE_STAGE3_WAGON_WORKERS=N to opt in, "auto" for cpu_count//4 capped
    at 4 -- each worker already drives several torch threads, so oversubscribing
    makes things slower, not faster.
    """
    raw = (os.getenv("WAGONEYE_STAGE3_WAGON_WORKERS") or "").strip().lower()
    if not raw:
        return 1
    if raw == "auto":
        return max(1, min(4, (os.cpu_count() or 4) // 4))
    try:
        return max(1, int(raw))
    except ValueError:
        return 1

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
    n = len(state.wagons)
    workers = wagon_workers()

    def _one_wagon(gw):
        """All of ONE wagon's features, in order.  Returns (gwid, per_feat, ran, errs).

        Errors are RETURNED, not reported here, so `on_error` (which mutates the
        manifest) always runs on the calling thread.
        """
        gwid = gw.global_id
        per_feat: Dict[str, Dict[str, str]] = {}
        ran: List[str] = []
        errs: List[tuple] = []
        for feat in active:                      # load -> damage -> door -> ocr
            cams = feature_cameras[feat]
            try:
                res = fmap[feat](
                    state=state, cache_root=cache_root,
                    feature_models_dir=feat_models_dir,
                    output_dir=states_root, evidence_root=evidence_root,
                    cameras=cams, wagon_ids=[gwid], verbose=False,
                )
                per_feat[feat] = res or {}
                ran.append(feat)
            except Exception as e:               # never let one wagon abort the rest
                errs.append((feat, cams, gwid, e))
        return gwid, per_feat, ran, errs

    def _report(i, gwid, per_feat, ran, errs, summary):
        for feat, res in per_feat.items():
            summary[feat].update(res)
        for feat, cams, gid, e in errs:
            if on_error is not None:
                on_error(feat, cams, gid, e)
            elif log is not None:
                log.error("[STAGE3/%s/%s] crashed: %s", feat, gid, e, exc_info=True)
        if log is not None:
            log.info("[STAGE3/wagon] %s (%d/%d): %s", gwid, i, n,
                     " ".join(ran) or "(none)")

    summary: Dict[str, Dict[str, str]] = {f: {} for f in active}

    if workers <= 1:
        # ---- sequential (default): unchanged from the original implementation
        if log is not None:
            log.info("[STAGE3] WAGON-WISE scheduling: %d wagons x features=%s "
                     "(each wagon runs all its features before the next)", n, active)
        for i, gw in enumerate(state.wagons, start=1):
            gwid, per_feat, ran, errs = _one_wagon(gw)
            _report(i, gwid, per_feat, ran, errs, summary)
        return summary

    # ---- parallel: WAGONS concurrently, each wagon's features still in order
    #
    # Wagons are independent -- every processor is scoped by `wagon_ids=[gw]` and
    # writes its own per-wagon JSON.  The one ordering constraint (LOAD before
    # DAMAGE, for the floor filter) is WITHIN a wagon and is preserved above.
    #
    # Results are merged in WAGON ORDER after the pool drains, so the returned
    # summary is identical regardless of completion order.
    import os as _os
    from concurrent.futures import ThreadPoolExecutor

    if log is not None:
        log.info("[STAGE3] WAGON-WISE scheduling: %d wagons x features=%s "
                 "(PARALLEL: %d wagons at a time; features within a wagon stay "
                 "ordered)", n, active, workers)

    prev = _os.environ.get("_WAGONEYE_PARALLEL_MODELS")
    _os.environ["_WAGONEYE_PARALLEL_MODELS"] = "1"   # per-thread YOLO instances
    try:
        with ThreadPoolExecutor(max_workers=workers,
                                thread_name_prefix="stage3") as pool:
            results = list(pool.map(_one_wagon, state.wagons))
    finally:
        if prev is None:
            _os.environ.pop("_WAGONEYE_PARALLEL_MODELS", None)
        else:
            _os.environ["_WAGONEYE_PARALLEL_MODELS"] = prev

    for i, (gwid, per_feat, ran, errs) in enumerate(results, start=1):
        _report(i, gwid, per_feat, ran, errs, summary)
    return summary
