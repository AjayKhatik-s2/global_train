"""
run_global_count.py  --  Wagon Eye Phase-1 (standalone)
========================================================

Self-contained CLI entry point for EC2 / any Linux server deployment.

CONVENTIONS (just-drop-files-and-run)
-------------------------------------
Place the 4 trimmed train videos in ./inputs/ with these exact names:
    inputs/right_up.mp4
    inputs/left_up.mp4
    inputs/right_up_top.mp4
    inputs/left_up_top.mp4

Place the 4 YOLO model weights in ./models/ with these exact names:
    models/right_up_wagon_gap.pt     (used by RIGHT_UP -- master)
    models/left_up_wagon_gap.pt      (used by LEFT_UP)
    models/top_gap.pt                (used by RIGHT_UP_TOP and LEFT_UP_TOP)
    models/side_classification.pt    (used by RIGHT_UP for ENGINE/WAGON/BRAKE_VAN)

Then run:
    python run_global_count.py

Outputs land in ./results/ (configurable with --output).

OVERRIDES
---------
You can override any path explicitly:
    python run_global_count.py \
        --right_up      /abs/path/cam_right_up.mp4 \
        --left_up       /abs/path/cam_left_up.mp4 \
        --right_up_top  /abs/path/cam_right_up_top.mp4 \
        --left_up_top   /abs/path/cam_left_up_top.mp4 \
        --models-dir    /abs/path/models \
        --output        /abs/path/results

WHAT THIS PRODUCES
------------------
    results/
        global_train_state.json          <-- canonical Phase-1 output
        per_camera_tracking.json
        processed_videos/
            RIGHT_UP_processed.mp4
            LEFT_UP_processed.mp4
            RIGHT_UP_TOP_processed.mp4
            LEFT_UP_TOP_processed.mp4
        frames/
            RIGHT_UP/
                GW_1/  frame_000000.jpg, frame_000001.jpg, ...
                GW_2/  ...
            LEFT_UP/   ...
            RIGHT_UP_TOP/  ...
            LEFT_UP_TOP/   ...

WHAT THIS DOES NOT DO
---------------------
No door / damage / OCR detection.  No PDF or email.  No S3 upload.
Phase-1 is gap counting + global synchronization + classification only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

from global_train_state import (
    GlobalTrainState,
    LocalCameraTracks,
    SegmentClass,
    MASTER_CAMERA,
    ALL_CAMERAS,
    CAMERA_LEFT_UP,
    CAMERA_RIGHT_UP,
    CAMERA_RIGHT_UP_TOP,
    CAMERA_LEFT_UP_TOP,
    TOP_CAMERAS,
    _MasterClassification,
    summarize_state,
)
from tracker_engine import GapTracker, MasterClassifier, segments_from_gaps
import global_alignment as ga
import video_segmenter as vs
import gap_cache as gc

# ---- Global Train construction (ported from wagon_count_global) --------------
# These five modules ARE the replacement: they sit between gap tracking and
# segmentation, and they are the reason the reference counts wagons correctly.
# Our tracker/alignment/segmenter were AST-compared against the reference's and
# proved supersets (0 functions present there and absent here), so they stay.
import fragment_stitching as fstitch      # STEP 1a  rejoin fragmented gap tracks
import gap_validation as gval             # STEP 1b  candidate -> valid boundary
import temporal_classification as tcls    # STEP 2b  support-camera labels
import train_structure as ts              # STEP 3   wagon window + GW renumber
import global_fusion as gf                # STEP 3   master-fixed fusion


# =============================================================================
# Auto-discovery: default file conventions
# =============================================================================

DEFAULT_INPUT_FILENAMES = {
    CAMERA_RIGHT_UP:     "right_up.mp4",
    CAMERA_LEFT_UP:      "left_up.mp4",
    CAMERA_RIGHT_UP_TOP: "right_up_top.mp4",
    CAMERA_LEFT_UP_TOP:  "left_up_top.mp4",
}

# Some users may use these alternative names; we'll fall back to them.
_INPUT_FALLBACK_PATTERNS = {
    CAMERA_RIGHT_UP:     ["right_up.mp4", "RIGHT_UP.mp4", "cam_right_up.mp4"],
    CAMERA_LEFT_UP:      ["left_up.mp4", "LEFT_UP.mp4", "cam_left_up.mp4"],
    CAMERA_RIGHT_UP_TOP: ["right_up_top.mp4", "RIGHT_UP_TOP.mp4", "cam_right_up_top.mp4"],
    CAMERA_LEFT_UP_TOP:  ["left_up_top.mp4", "LEFT_UP_TOP.mp4", "cam_left_up_top.mp4"],
}


def _here() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _utc_now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _resolve_input(explicit: Optional[str], inputs_dir: str, camera_id: str) -> str:
    """Return a path to the camera's video.

    Search order:
        1. explicit (if provided)
        2. inputs_dir/<filename>  for each fallback name
    """
    if explicit:
        if not os.path.exists(explicit):
            raise FileNotFoundError(f"--{camera_id.lower()} path does not exist: {explicit}")
        return os.path.abspath(explicit)

    for name in _INPUT_FALLBACK_PATTERNS[camera_id]:
        p = os.path.join(inputs_dir, name)
        if os.path.exists(p):
            return os.path.abspath(p)

    raise FileNotFoundError(
        f"No input video found for {camera_id}. "
        f"Looked in {inputs_dir} for: "
        f"{_INPUT_FALLBACK_PATTERNS[camera_id]}. "
        f"Either drop the file there or pass --{camera_id.lower()} <path>."
    )


def _resolve_optional_input(explicit: Optional[str], inputs_dir: str,
                            camera_id: str) -> Optional[str]:
    """Like _resolve_input but returns None (instead of raising) when the
    camera's video is absent -- used by the master-first subset flow where
    only the currently-arrived cameras are processed."""
    if explicit:
        return os.path.abspath(explicit) if os.path.exists(explicit) else None
    for name in _INPUT_FALLBACK_PATTERNS[camera_id]:
        p = os.path.join(inputs_dir, name)
        if os.path.exists(p):
            return os.path.abspath(p)
    return None


# Phase-2/v4 alias map: the new package prefers shorter model names.
# When resolving the canonical wagon_count names, we ALSO accept the
# shorter aliases (right_up_gap.pt / left_up_gap.pt) so the new
# `models/reconstruction/` directory can use either convention.
_MODEL_ALIASES = {
    "right_up_wagon_gap.pt": ("right_up_gap.pt",),
    "left_up_wagon_gap.pt":  ("left_up_gap.pt",),
}


def _resolve_model(name: str, models_dir: str) -> str:
    # 1) try the canonical name first
    p = os.path.join(models_dir, name)
    if os.path.exists(p):
        return os.path.abspath(p)
    # 2) try any registered short-name aliases
    for alias in _MODEL_ALIASES.get(name, ()):
        ap = os.path.join(models_dir, alias)
        if os.path.exists(ap):
            return os.path.abspath(ap)
    aliases = _MODEL_ALIASES.get(name, ())
    looked = [p] + [os.path.join(models_dir, a) for a in aliases]
    raise FileNotFoundError(
        f"Model not found: {name}. Looked at: {looked}. "
        f"Drop the .pt file in {models_dir} or pass --models-dir <path>."
    )


# =============================================================================
# Per-camera processing
# =============================================================================

def _process_side_camera(
    camera_id: str, video_path: str, gap_model_path: str,
    confidence: float, min_height_ratio: float,
    keep_raw_detections: bool, verbose: bool,
) -> LocalCameraTracks:
    tracker = GapTracker(
        camera_id=camera_id, model_path=gap_model_path,
        confidence=confidence, min_height_ratio=min_height_ratio,
        verbose=verbose,
    )
    return tracker.process_video(video_path, keep_raw_detections=keep_raw_detections)


def _process_top_camera(
    camera_id: str, video_path: str, top_gap_model_path: str,
    confidence: float, min_height_ratio: float,
    keep_raw_detections: bool, verbose: bool,
) -> LocalCameraTracks:
    tracker = GapTracker(
        camera_id=camera_id, model_path=top_gap_model_path,
        confidence=confidence, min_height_ratio=min_height_ratio,
        verbose=verbose,
    )
    return tracker.process_video(video_path, keep_raw_detections=keep_raw_detections)


def _classify_master_pre_fusion(
    master_tracks: LocalCameraTracks,
    side_classification_model_path: str,
    num_samples: int,
    verbose: bool,
):
    pre_segments = segments_from_gaps(master_tracks.gaps, master_tracks.total_frames)
    if not pre_segments:
        if verbose:
            print("[CLASSIFY] no pre-fusion segments to classify")
        return []
    if verbose:
        print(f"[CLASSIFY] classifying {len(pre_segments)} pre-fusion segments on "
              f"{os.path.basename(master_tracks.video_path)}")
    clf = MasterClassifier(side_classification_model_path, num_samples=num_samples, verbose=verbose)
    return clf.classify_segments(master_tracks.video_path, pre_segments)


def _classify_top_regions(
    top_tracks: LocalCameraTracks,
    master_fps: float,
    top_classification_model_path: str,
    num_samples: int,
    verbose: bool,
) -> List[_MasterClassification]:
    """Run top_classification.pt over ONE top camera and return the per-segment
    ENGINE/WAGON/BRAKE_VAN labels re-expressed in MASTER frames.

    Segments come from the top camera's OWN gaps (its local structure); the
    labels are the extra semantic evidence fused into the Global Train.  Frames
    are converted top->master via the shared-t=0 timebase so the labels line up
    with each GlobalWagon's master-frame window in ``fuse_semantic_labels``.
    """
    local = _classify_top_local(top_tracks, top_classification_model_path,
                                num_samples, verbose)
    return _rescale_top_to_master(local, top_fps=top_tracks.fps,
                                  master_fps=master_fps)


def _classify_top_local(
    top_tracks: LocalCameraTracks,
    top_classification_model_path: str,
    num_samples: int,
    verbose: bool,
) -> List[_MasterClassification]:
    """The inference half of `_classify_top_regions`, in the top camera's OWN frames.

    Split out so incremental per-camera extraction can run it the moment that
    top camera arrives -- the rescale below needs the MASTER's fps, which is
    unknown while cameras are still landing.
    """
    segs = segments_from_gaps(top_tracks.gaps, top_tracks.total_frames)
    if not segs:
        return []
    clf = MasterClassifier(top_classification_model_path, num_samples=num_samples,
                           verbose=verbose, tag=f"TOP:{top_tracks.camera_id}")
    return clf.classify_segments(top_tracks.video_path, segs)   # top-frame ranges


def _rescale_top_to_master(
    local: List[_MasterClassification], *, top_fps: float, master_fps: float,
) -> List[_MasterClassification]:
    """The arithmetic half: top frames -> master frames via the shared t=0 timebase.

    Identical to the expression this was extracted from; deferred so it can be
    applied at assembly time to a cached top camera.
    """
    fps = top_fps if top_fps > 0 else master_fps
    scale = (master_fps / fps) if fps > 0 else 1.0
    return [_MasterClassification(
        segment_index=c.segment_index,
        start_frame=int(round(c.start_frame * scale)),
        end_frame=int(round(c.end_frame * scale)),
        label=c.label,
        confidence=c.confidence,
    ) for c in local]


# =============================================================================
# Incremental per-camera gap extraction (AUTO/S3 only)
# =============================================================================

#: Per-side-camera gap model. Module level so the full run and the incremental
#: per-camera run cannot drift apart on model selection.
_SIDE_GAP_MODEL = {
    CAMERA_RIGHT_UP: "right_up_wagon_gap.pt",
    CAMERA_LEFT_UP:  "left_up_wagon_gap.pt",
}


def _gap_model_for(camera_id: str, models_dir: str) -> str:
    """The SAME model selection the full run uses -- one source of truth."""
    if camera_id in (CAMERA_RIGHT_UP_TOP, CAMERA_LEFT_UP_TOP):
        return _resolve_model("top_gap.pt", models_dir)
    return _resolve_model(_SIDE_GAP_MODEL[camera_id], models_dir)


def _extract_one_camera(
    camera_id: str, video_path: str, args, verbose: bool,
) -> tuple:
    """Gap-extract ONE camera by calling the existing Stage-1 code paths.

    Returns `(tracks, master_classifications, top_local_classifications)`.

    This deliberately contains no detection, tracking, NMS, merge or
    ownership logic of its own -- it dispatches to `_process_side_camera` /
    `_process_top_camera` / `_classify_master_pre_fusion` /
    `_classify_top_local`, the very functions the full run calls, so there is
    exactly one implementation of gap detection in the repository.
    """
    keep_raw = not args.no_raw_detections
    gap_model = _gap_model_for(camera_id, args.models_dir)

    if camera_id in (CAMERA_RIGHT_UP_TOP, CAMERA_LEFT_UP_TOP):
        tracks = _process_top_camera(
            camera_id, video_path, gap_model,
            confidence=args.top_confidence,
            min_height_ratio=args.top_min_height_ratio,
            keep_raw_detections=keep_raw, verbose=verbose,
        )
    else:
        tracks = _process_side_camera(
            camera_id, video_path, gap_model,
            confidence=args.side_confidence,
            min_height_ratio=args.side_min_height_ratio,
            keep_raw_detections=keep_raw, verbose=verbose,
        )

    # The master's own pre-fusion classification depends only on its own video
    # and its own gaps, so it belongs to this camera's work.
    master_cls: List[_MasterClassification] = []
    if camera_id == args.master_camera:
        try:
            master_cls = _classify_master_pre_fusion(
                tracks, _resolve_model("side_classification.pt", args.models_dir),
                num_samples=args.classification_samples, verbose=verbose)
        except Exception as e:
            print(f"WARNING: master classification failed: {e}", file=sys.stderr)

    # Top semantic evidence, in this camera's OWN frames (rescaled at assembly).
    top_local: List[_MasterClassification] = []
    if camera_id in TOP_CAMERAS:
        try:
            top_local = _classify_top_local(
                tracks, _resolve_model("top_classification.pt", args.models_dir),
                num_samples=args.classification_samples, verbose=verbose)
        except FileNotFoundError:
            print("[STAGE1] top_classification.pt absent; top semantic evidence "
                  "skipped for this camera")
        except Exception as e:
            print(f"WARNING: top classification failed for {camera_id}: {e}",
                  file=sys.stderr)

    return tracks, master_cls, top_local


def _derive_wagon_window(master: LocalCameraTracks, classifications, verbose=False):
    """Derive the wagon window from the CURRENT master gaps + classifications.

    Runtime-derived only: the window comes from the classified segments that the
    master's own validated gaps define.  No frame numbers or timestamps are
    assumed, so it holds for any train.

    `build_global_wagons` is called here as the shared segment primitive -- the
    same one `train_structure` uses -- not as a fusion algorithm.
    """
    if not classifications:
        return None
    try:
        segments = ga.build_global_wagons(
            list(master.gaps),
            master_total_frames=master.total_frames, master_fps=master.fps,
            initial_classifications=list(classifications),
            support_camera_ids=[])
        return ts.get_master_wagon_window(segments, verbose=verbose)
    except Exception:
        return None


def _run_camera_only(args, verbose: bool) -> int:
    """`--camera-only CAM`: extract + persist one camera's gaps, then exit.

    Produces NO GlobalTrainState and performs no fusion -- calling this once
    per camera must never assemble a Global Train.  Assembly stays in the
    normal (non-camera-only) path, gated by the caller's existing policy.
    """
    camera_id = args.camera_only
    if not args.gap_cache:
        print("ERROR: --camera-only requires --gap-cache", file=sys.stderr)
        return 4

    explicit = {
        CAMERA_RIGHT_UP:     args.right_up,
        CAMERA_LEFT_UP:      args.left_up,
        CAMERA_RIGHT_UP_TOP: args.right_up_top,
        CAMERA_LEFT_UP_TOP:  args.left_up_top,
    }[camera_id]
    video_path = _resolve_optional_input(explicit, args.inputs_dir, camera_id)
    if not video_path:
        print(f"ERROR: no video for {camera_id}", file=sys.stderr)
        return 4

    identity = None
    if args.source_identity:
        try:
            identity = json.loads(args.source_identity)
        except (ValueError, TypeError) as e:
            print(f"ERROR: --source-identity is not valid JSON: {e}", file=sys.stderr)
            return 4

    os.makedirs(args.gap_cache, exist_ok=True)
    print("=" * 70)
    print(f"  WAGON EYE - PER-CAMERA GAP EXTRACTION ({camera_id})")
    print("=" * 70)
    print(f"  video      : {video_path}")
    print(f"  gap cache  : {args.gap_cache}")

    gc.write_state(args.gap_cache, camera_id, gc.GapState.PROCESSING,
                   identity, updated_at=_utc_now_iso())
    t0 = time.time()
    try:
        tracks, master_cls, top_local = _extract_one_camera(
            camera_id, video_path, args, verbose)
    except Exception as e:
        traceback.print_exc()
        gc.write_state(args.gap_cache, camera_id, gc.GapState.FAILED, identity,
                       error=f"{type(e).__name__}: {e}", updated_at=_utc_now_iso())
        print(f"ERROR: gap extraction failed for {camera_id}: {e}", file=sys.stderr)
        return 3

    # The master's classifications ride along inside LocalCameraTracks, exactly
    # as the full run attaches them before fusion.
    if master_cls:
        tracks.classifications = master_cls

    gc.write_result(args.gap_cache, camera_id, tracks, identity=identity,
                    top_local_classifications=top_local,
                    produced_at=_utc_now_iso())
    gc.write_state(args.gap_cache, camera_id, gc.GapState.COMPLETED, identity,
                   gap_count=len(tracks.gaps), updated_at=_utc_now_iso())
    print(f"[GAP] {camera_id} complete: gaps={len(tracks.gaps)} "
          f"wagons={tracks.local_wagon_count} fps={tracks.fps:.2f} "
          f"frames={tracks.total_frames} in {time.time() - t0:.1f}s")
    return 0


# =============================================================================
# CLI
# =============================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    here = _here()
    default_inputs = os.path.join(here, "inputs")
    default_models = os.path.join(here, "models")
    default_output = os.path.join(here, "results")

    p = argparse.ArgumentParser(
        prog="run_global_count.py",
        description="Phase-1 global wagon counting + classification (standalone).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Default file conventions:\n"
            "  inputs/{right_up,left_up,right_up_top,left_up_top}.mp4\n"
            "  models/right_up_wagon_gap.pt   (RIGHT_UP -- master)\n"
            "  models/left_up_wagon_gap.pt    (LEFT_UP)\n"
            "  models/top_gap.pt              (RIGHT_UP_TOP + LEFT_UP_TOP)\n"
            "  models/side_classification.pt  (RIGHT_UP classification)\n"
            "Drop the files in ./inputs and ./models, then run with no args."
        ),
    )

    p.add_argument("--right_up",     default=None, help="Override path to RIGHT_UP video (master)")
    p.add_argument("--left_up",      default=None, help="Override path to LEFT_UP video")
    p.add_argument("--right_up_top", default=None, help="Override path to RIGHT_UP_TOP video")
    p.add_argument("--left_up_top",  default=None, help="Override path to LEFT_UP_TOP video")

    # Master-first incremental reconstruction: only the cameras actually
    # present are processed; the master defaults to RIGHT_UP.  A non-RIGHT_UP
    # master (LEFT_UP fallback) is UNVALIDATED and must be opted into.
    p.add_argument("--master-camera", default=CAMERA_RIGHT_UP,
                   choices=list(ALL_CAMERAS),
                   help="Camera to use as the master timeline (default RIGHT_UP)")
    p.add_argument("--allow-fallback-master", action="store_true",
                   help="Permit a non-RIGHT_UP master (LEFT_UP fallback). "
                        "OFF by default -- side_classification.pt is unvalidated "
                        "on LEFT_UP.")

    p.add_argument("--inputs-dir",   default=default_inputs,
                   help=f"Directory containing the 4 input videos (default: {default_inputs})")
    p.add_argument("--models-dir",   default=default_models,
                   help=f"Directory containing the 4 .pt models (default: {default_models})")
    p.add_argument("--output", "-o", default=default_output,
                   help=f"Output root directory (default: {default_output})")

    p.add_argument("--side-confidence", type=float, default=0.4,
                   help="Confidence threshold for the side gap models "
                        "right_up_wagon_gap.pt and left_up_wagon_gap.pt "
                        "(default: 0.4)")
    p.add_argument("--top-confidence",  type=float, default=0.4,
                   help="Confidence threshold for top_gap.pt (default: 0.4)")
    p.add_argument("--side-min-height-ratio", type=float, default=0.35,
                   help="Min bbox height / frame height for SIDE gap detections "
                        "(default: 0.35). Tall gaps are typical on side cameras.")
    p.add_argument("--top-min-height-ratio",  type=float, default=0.05,
                   help="Min bbox height / frame height for TOP gap detections "
                        "(default: 0.05). Top-camera gaps are thin horizontal "
                        "strips, so this MUST be much smaller than the side ratio.")
    p.add_argument("--classification-samples", type=int, default=5,
                   help="Frames per segment for side_classification.pt vote (default: 5)")

    # ---- fragment reassembly: rebuild physical gaps before validating them ---
    # Defaults are taken from the ported modules themselves, so this CLI can
    # never drift from the algorithm's own tuned values.
    _fs = fstitch.DEFAULT_FRAGMENT_STITCH
    p.add_argument("--no-fragment-stitching", action="store_true",
                   help="Disable fragment reassembly and validate each tracker "
                        "fragment separately.  One physical gap split across "
                        "several short tracks is then rejected piece by piece.")
    p.add_argument("--stitch-max-seam-sec", type=float,
                   default=_fs.max_seam_seconds,
                   help=f"Largest temporal hole between two fragments of the "
                        f"same physical gap, in SECONDS "
                        f"(default {_fs.max_seam_seconds})")
    p.add_argument("--stitch-seam-tolerance", type=float,
                   default=_fs.seam_speed_tolerance,
                   help=f"How far a seam jump may exceed what the local advance "
                        f"rate predicts (default {_fs.seam_speed_tolerance})")
    p.add_argument("--stitch-max-seam-frac", type=float,
                   default=_fs.max_seam_frac,
                   help=f"Hard cap on seam displacement as a FRACTION of frame "
                        f"width (default {_fs.max_seam_frac})")

    # ---- gap validation: raw YOLO gaps are CANDIDATES, not boundaries --------
    _gv = gval.DEFAULT_GAP_VALIDATION
    p.add_argument("--no-gap-validation", action="store_true",
                   help="Disable motion/temporal gap validation and treat every "
                        "tracked candidate as a wagon boundary (the behaviour "
                        "before the wagon_count_global replacement)")
    p.add_argument("--gap-min-track-sec", type=float,
                   default=_gv.min_track_seconds)
    p.add_argument("--gap-max-track-gap-sec", type=float,
                   default=_gv.max_detection_gap_seconds)
    p.add_argument("--gap-min-motion-frac", type=float,
                   default=_gv.min_motion_frac)
    p.add_argument("--gap-static-max-frac", type=float,
                   default=_gv.static_max_motion_frac)
    p.add_argument("--gap-min-motion-frac-sec", type=float,
                   default=_gv.min_motion_frac_per_sec)
    p.add_argument("--gap-max-motion-frac-sec", type=float,
                   default=_gv.max_motion_frac_per_sec)
    p.add_argument("--gap-min-separation-sec", type=float,
                   default=_gv.min_separation_seconds)
    p.add_argument("--gap-motion-tolerance", type=float,
                   default=_gv.train_motion_tolerance)
    p.add_argument("--gap-min-confidence", type=float,
                   default=_gv.min_mean_confidence)
    p.add_argument("--gap-min-monotonic", type=float,
                   default=_gv.min_monotonic_fraction)

    # ---- DEPRECATED absolute-unit gap thresholds ----------------------------
    # Superseded by the seconds / frame-width-fraction flags above, which
    # generalize across trains and camera geometry.  Kept so an existing caller
    # does not crash: a value given here is applied verbatim as a PER-CAMERA
    # override at resolve() time and is never stored on the shared config, so it
    # cannot leak from one camera's geometry into another's.
    p.add_argument("--gap-min-track-frames", type=int, default=None)
    p.add_argument("--gap-max-track-gap", type=int, default=None)
    p.add_argument("--gap-min-motion-px", type=float, default=None)
    p.add_argument("--gap-static-max-px", type=float, default=None)
    p.add_argument("--gap-min-motion-px-sec", type=float, default=None)
    p.add_argument("--gap-max-motion-px-sec", type=float, default=None)

    p.add_argument("--no-wagon-recovery", action="store_true",
                   help="Disable the WAGON_ACTIVE second validation pass.  By "
                        "default a master candidate inside the confirmed wagon "
                        "region that failed only a SOFT gate (speed, trajectory "
                        "noise, weaker confidence) is re-examined and accepted "
                        "if it still clears every HARD gate.")

    # ---- master-fixed cross-camera fusion -----------------------------------
    p.add_argument("--no-wagon-only", action="store_true",
                   help="Align support cameras over their whole footage instead "
                        "of only their wagon region.")
    p.add_argument("--fusion-non-strict", action="store_true",
                   help="Downgrade master-fixed fusion invariant violations from "
                        "errors to warnings.")
    p.add_argument("--offset-search", type=float,
                   default=gf.DEFAULT_CONFIG.offset_search_s,
                   help="Half-width of the per-camera clock-offset search, in "
                        f"seconds (default {gf.DEFAULT_CONFIG.offset_search_s})")
    p.add_argument("--offset-min-margin", type=float,
                   default=gf.DEFAULT_CONFIG.offset_min_margin_ratio,
                   help="Minimum margin between the best and runner-up offset "
                        "for it to count as RESOLVED "
                        f"(default {gf.DEFAULT_CONFIG.offset_min_margin_ratio})")
    p.add_argument("--match-tolerance", type=float,
                   default=gf.DEFAULT_CONFIG.match_tolerance_s,
                   help="How close an aligned support gap must be to a master "
                        f"gap to match (default {gf.DEFAULT_CONFIG.match_tolerance_s}s)")

    # ---- DEPRECATED: trust-weighted insertion (pre-replacement fusion) -------
    # Kept only so existing callers/scripts do not crash on an unknown flag.
    # Master-fixed fusion never inserts a support-derived gap, so these have NO
    # effect; passing one is reported once at startup.
    p.add_argument("--fuse-min-support", type=int, default=2,
                   help="Min supporting cameras for inserting a missed gap (default: 2)")
    p.add_argument("--fuse-max-spread",  type=float, default=1.5,
                   help="Max time spread within a fusion cluster (default: 1.5s)")
    p.add_argument("--fuse-min-conf",    type=float, default=0.4,
                   help="Min mean confidence to insert a fused gap (default: 0.4)")

    p.add_argument("--no-videos", action="store_true", help="Skip overlay video rendering")
    p.add_argument("--no-frames", action="store_true", help="Skip per-wagon frame extraction")
    p.add_argument("--every-nth-frame", type=int, default=1,
                   help="Keep 1 of every N frames during extraction (default: 1)")
    p.add_argument("--no-raw-detections", action="store_true",
                   help="Don't keep raw per-frame detections in memory (saves RAM)")
    p.add_argument("--quiet", action="store_true", help="Reduce log verbosity")

    # ---- incremental per-camera gap extraction (AUTO/S3 pipeline only) ----
    #
    # Omit BOTH of these and this script behaves exactly as it always has:
    # every present camera is inferred in-process, then fused.  LOCAL mode
    # never passes them.
    p.add_argument("--gap-cache", default=None, metavar="DIR",
                   help="Directory of per-camera gap results. In normal mode, a "
                        "camera whose cached result matches its source object is "
                        "LOADED instead of re-inferred.")
    p.add_argument("--camera-only", default=None, choices=list(ALL_CAMERAS),
                   help="Run gap extraction for THIS camera only, write its result "
                        "into --gap-cache, and exit. Performs no fusion and "
                        "produces no GlobalTrainState.")
    p.add_argument("--source-identity", default=None, metavar="JSON",
                   help="JSON object identifying the S3 source (bucket/key/etag/"
                        "size/last_modified) recorded with a --camera-only result "
                        "so a replaced video is never mistaken for a cache hit.")

    p.add_argument("--stage1-debug", action="store_true",
                   help="Debug visualization: overlay the raw per-frame candidate "
                        "detections (cyan) on top of the final tracked gaps in the "
                        "processed videos (also via WAGONEYE_STAGE1_DEBUG=true). "
                        "Production shows only the final accepted gaps.")
    p.add_argument("--stage1-frame-trim-percent", type=float, default=None,
                   help="Ignore the first/last N%% of frames during Stage-1 "
                        "reconstruction only (default from "
                        "WAGONEYE_STAGE1_FRAME_TRIM_PERCENT, itself 4; 0 = off). "
                        "Frame numbering is preserved for all downstream stages.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    verbose = not args.quiet

    # CLI overrides the env var read by GapTracker.process_video (single source
    # of truth for the trim percent).
    if args.stage1_frame_trim_percent is not None:
        os.environ["WAGONEYE_STAGE1_FRAME_TRIM_PERCENT"] = str(
            args.stage1_frame_trim_percent)

    # Per-camera preparation mode: extract + persist ONE camera, no assembly.
    if args.camera_only:
        return _run_camera_only(args, verbose)

    t_start = time.time()
    print("=" * 70)
    print("  WAGON EYE - PHASE 1 GLOBAL TRAIN RECONSTRUCTION")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Resolve inputs (present cameras only) + master selection
    # ------------------------------------------------------------------
    _explicit = {
        CAMERA_RIGHT_UP:     args.right_up,
        CAMERA_LEFT_UP:      args.left_up,
        CAMERA_RIGHT_UP_TOP: args.right_up_top,
        CAMERA_LEFT_UP_TOP:  args.left_up_top,
    }
    present_videos: Dict[str, str] = {}
    for cam in ALL_CAMERAS:
        p = _resolve_optional_input(_explicit[cam], args.inputs_dir, cam)
        if p:
            present_videos[cam] = p

    master_cam = args.master_camera
    if master_cam not in present_videos:
        print(f"ERROR: master camera {master_cam} video is not present; cannot "
              f"reconstruct (present: {sorted(present_videos)})", file=sys.stderr)
        return 4
    if master_cam != CAMERA_RIGHT_UP and not args.allow_fallback_master:
        print(f"ERROR: master {master_cam} != RIGHT_UP requires "
              f"--allow-fallback-master (LEFT_UP classification is unvalidated)",
              file=sys.stderr)
        return 4

    # Resolve only the models the present cameras need (_SIDE_GAP_MODEL is
    # module level so --camera-only picks the identical weights).
    try:
        gap_model: Dict[str, str] = {}
        need_top = any(c in present_videos for c in (CAMERA_RIGHT_UP_TOP, CAMERA_LEFT_UP_TOP))
        top_gap_path = _resolve_model("top_gap.pt", args.models_dir) if need_top else None
        for cam in present_videos:
            if cam in (CAMERA_RIGHT_UP_TOP, CAMERA_LEFT_UP_TOP):
                gap_model[cam] = top_gap_path
            else:
                gap_model[cam] = _resolve_model(_SIDE_GAP_MODEL[cam], args.models_dir)
        # classification runs on the master video
        side_cls_path = _resolve_model("side_classification.pt", args.models_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    missing_at_reconstruction = [c for c in ALL_CAMERAS if c not in present_videos]
    print(f"  master camera            : {master_cam}"
          f"{'  (FALLBACK)' if master_cam != CAMERA_RIGHT_UP else ''}")
    for cam in ALL_CAMERAS:
        tag = present_videos.get(cam, "<absent>")
        print(f"  {cam:<24} : {tag}")
    print(f"  output root              : {args.output}")
    print()

    os.makedirs(args.output, exist_ok=True)
    processed_videos_dir = os.path.join(args.output, "processed_videos")
    frames_root = os.path.join(args.output, "frames")
    os.makedirs(processed_videos_dir, exist_ok=True)
    os.makedirs(frames_root, exist_ok=True)

    keep_raw = not args.no_raw_detections
    stage1_debug = bool(args.stage1_debug) or os.getenv(
        "WAGONEYE_STAGE1_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")
    if stage1_debug:
        print("[STAGE1] debug visualization ENABLED "
              "(raw candidate detections overlaid on final gaps)")

    # ------------------------------------------------------------------
    # STEP 1 -- per-camera gap tracking (present cameras only)
    # ------------------------------------------------------------------
    print("-" * 70)
    print("  STEP 1  Per-camera gap tracking")
    print("-" * 70)
    tracks: Dict[str, LocalCameraTracks] = {}
    # Top cameras' own-frame semantic labels, loaded from cache when available;
    # rescaled to master frames in STEP 2b once the master's fps is known.
    cached_top_local: Dict[str, List[_MasterClassification]] = {}
    try:
        for cam in ALL_CAMERAS:
            if cam not in present_videos:
                continue
            # A camera already gap-extracted incrementally is LOADED, not
            # re-inferred.  Identity is not re-checked here: the caller only
            # populates the cache dir for the exact objects in this batch, and
            # it verified the match before scheduling extraction.
            if args.gap_cache:
                cached, tops = gc.load_tracks(args.gap_cache, cam)
                if cached is not None:
                    # The cached video_path came from the extraction run; point
                    # it at this run's copy so overlay rendering still opens it.
                    cached.video_path = present_videos[cam]
                    tracks[cam] = cached
                    if tops:
                        cached_top_local[cam] = tops
                    print(f"[STEP1] {cam}: loaded {len(cached.gaps)} gap(s) from "
                          f"cache (no re-inference)")
                    continue
            if cam in (CAMERA_RIGHT_UP_TOP, CAMERA_LEFT_UP_TOP):
                tracks[cam] = _process_top_camera(
                    cam, present_videos[cam], gap_model[cam],
                    confidence=args.top_confidence,
                    min_height_ratio=args.top_min_height_ratio,
                    keep_raw_detections=keep_raw, verbose=verbose,
                )
            else:
                tracks[cam] = _process_side_camera(
                    cam, present_videos[cam], gap_model[cam],
                    confidence=args.side_confidence,
                    min_height_ratio=args.side_min_height_ratio,
                    keep_raw_detections=keep_raw, verbose=verbose,
                )
    except Exception as e:
        print(f"ERROR: per-camera tracking failed: {e}", file=sys.stderr)
        traceback.print_exc()
        return 3

    print()
    print("  Local counts after Step 1:")
    for cam in ALL_CAMERAS:
        if cam not in tracks:
            continue
        t = tracks[cam]
        print(f"    {cam:<14}  wagons={t.local_wagon_count:>3}   gaps={len(t.gaps):>3}   "
              f"fps={t.fps:.2f}   frames={t.total_frames}")
    print()

    # ------------------------------------------------------------------
    # STEP 1a -- FRAGMENT REASSEMBLY  (before validation, not inside it)
    #
    # One physical gap can leave the tracker as several short tracks: when a
    # detection is missed the object reappears beyond the association gate, so
    # the track closes and a new id opens.  Validation would then judge each
    # piece separately, reject each as too short, and lose the gap they jointly
    # prove.  Reassembly restores the physical object FIRST, so every existing
    # gate then applies to the whole gap.  Nothing is accepted here and no
    # threshold is relaxed -- this layer only decides which observations belong
    # together.
    #
    # Runs over PRESENT cameras only (the reference assumes all four).
    # ------------------------------------------------------------------
    print("-" * 70)
    print("  STEP 1a  Fragment reassembly (tracker fragments -> physical gaps)")
    print("-" * 70)
    stitch_cfg = fstitch.FragmentStitchConfig(
        enabled=not args.no_fragment_stitching,
        max_seam_seconds=float(args.stitch_max_seam_sec),
        seam_speed_tolerance=float(args.stitch_seam_tolerance),
        max_seam_frac=float(args.stitch_max_seam_frac),
    )
    stitching: Dict[str, fstitch.StitchResult] = {}
    for cam in ALL_CAMERAS:
        if cam not in tracks:
            continue
        t = tracks[cam]
        # Geometry per camera: seam limits resolve from THIS camera's own width
        # and frame rate, so nothing measured on one geometry leaks into another.
        sres = fstitch.reassemble_fragments(
            t.gaps, cam, stitch_cfg, frame_width=t.width, fps=t.fps,
            verbose=verbose)
        stitching[cam] = sres
        t.gaps = sres.events
    print()

    # ------------------------------------------------------------------
    # STEP 1b -- GAP VALIDATION
    #
    # A raw YOLO gap detection is a CANDIDATE, not a wagon boundary.  Each
    # tracked candidate is checked for temporal persistence, detection
    # continuity, real motion, plausible speed, trajectory consistency,
    # direction and confidence, and duplicates are collapsed.  The train is
    # moving, so a detection pinned to one pixel column is background, not a
    # gap between wagons.
    #
    # Detection and tracking themselves are UNCHANGED: this filters the
    # GapEvents the existing tracker already emitted.
    # ------------------------------------------------------------------
    print("-" * 70)
    print("  STEP 1b  Gap validation (candidates -> valid wagon boundaries)")
    print("-" * 70)
    # Config holds ONLY camera-independent units (seconds / frame-width
    # fractions / ratios); per-camera geometry is applied at resolve() time.
    gv_cfg = gval.GapValidationConfig(
        enabled=not args.no_gap_validation,
        min_track_seconds=float(args.gap_min_track_sec),
        max_detection_gap_seconds=float(args.gap_max_track_gap_sec),
        min_motion_frac=float(args.gap_min_motion_frac),
        static_max_motion_frac=float(args.gap_static_max_frac),
        min_motion_frac_per_sec=float(args.gap_min_motion_frac_sec),
        max_motion_frac_per_sec=float(args.gap_max_motion_frac_sec),
        min_separation_seconds=float(args.gap_min_separation_sec),
        min_monotonic_fraction=float(args.gap_min_monotonic),
        min_mean_confidence=float(args.gap_min_confidence),
        train_motion_tolerance=float(args.gap_motion_tolerance),
    )
    # Deprecated absolute-unit flags -> per-camera overrides, converted with each
    # camera's own width/fps at resolve() time.  Nothing absolute is stored on
    # the config, so a value given for one geometry cannot leak into another.
    gv_overrides: Dict[str, float] = {}
    for flag, attr, target in (
        ("--gap-min-track-frames", "gap_min_track_frames", "min_track_frames"),
        ("--gap-max-track-gap", "gap_max_track_gap", "max_detection_gap_frames"),
        ("--gap-min-motion-px", "gap_min_motion_px", "min_motion_px"),
        ("--gap-static-max-px", "gap_static_max_px", "static_max_motion_px"),
        ("--gap-min-motion-px-sec", "gap_min_motion_px_sec", "min_motion_px_per_sec"),
        ("--gap-max-motion-px-sec", "gap_max_motion_px_sec", "max_motion_px_per_sec"),
    ):
        value = getattr(args, attr, None)
        if value is not None:
            gv_overrides[target] = float(value)
            print(f"NOTE: {flag} is deprecated -- thresholds are now expressed "
                  f"in seconds and frame-width fractions so they generalize "
                  f"across trains and camera geometry.  Your value is applied "
                  f"verbatim as a per-camera override for this run.",
                  file=sys.stderr)

    gap_validation: Dict[str, gval.GapValidationResult] = {}
    for cam in ALL_CAMERAS:
        if cam not in tracks:
            continue
        t = tracks[cam]
        raw_n = sum(len(v) for v in (t.raw_frame_detections or {}).values())
        res = gval.validate_gap_events(t.gaps, cam, gv_cfg,
                                       raw_detection_count=raw_n, verbose=verbose,
                                       frame_width=t.width, fps=t.fps,
                                       absolute_overrides=gv_overrides or None)
        gap_validation[cam] = res
        # Replace the camera's gap list with the validated subset and restore
        # track_id as a contiguous temporal rank, as the tracker produces.
        t.gaps = gval.renumber_gap_events(res.accepted)

    print()
    print("  Validated counts after Step 1b:")
    for cam in ALL_CAMERAS:
        if cam not in tracks:
            continue
        t = tracks[cam]
        r = gap_validation[cam]
        print(f"    {cam:<14}  raw_det={r.raw_detection_count:>4}  "
              f"candidates={r.tracked_candidate_count:>3}  "
              f"valid_gaps={len(t.gaps):>3}  rejected={len(r.rejected):>3}")
    print()

    # ------------------------------------------------------------------
    # STEP 2 -- master classification (on the chosen master video)
    # ------------------------------------------------------------------
    print("-" * 70)
    print(f"  STEP 2  {master_cam} master classification (ENGINE / WAGON / BRAKE_VAN)")
    print("-" * 70)
    master = tracks[master_cam]
    if master.classifications:
        # Produced during this master's incremental extraction and restored with
        # its tracks -- re-running would spend the same inference twice.
        initial_classifications = list(master.classifications)
        print(f"[STEP2] loaded {len(initial_classifications)} pre-fusion "
              f"classification(s) from cache (no re-inference)")
    else:
        try:
            initial_classifications = _classify_master_pre_fusion(
                master, side_cls_path,
                num_samples=args.classification_samples, verbose=verbose,
            )
        except Exception as e:
            print(f"WARNING: master classification failed: {e}", file=sys.stderr)
            traceback.print_exc()
            initial_classifications = []

    # ------------------------------------------------------------------
    # STEP 2b -- TOP-camera semantic classification (top_classification.pt)
    # Extra evidence for the Global Train's ENGINE/WAGON/BRAKE_VAN labels.
    # Runs only when the model is present (on EC2); skipped gracefully locally.
    # ------------------------------------------------------------------
    top_classifications: Dict[str, List[_MasterClassification]] = {}
    # Cached top cameras only need the frame rescale -- the inference already
    # happened when that camera arrived.  Same arithmetic, same result.
    for cam, local in cached_top_local.items():
        if cam in tracks:
            top_classifications[cam] = _rescale_top_to_master(
                local, top_fps=tracks[cam].fps, master_fps=tracks[master_cam].fps)
            print(f"[STEP2b] {cam}: rescaled {len(local)} cached top "
                  f"classification(s) to master frames (no re-inference)")
    try:
        top_cls_path = _resolve_model("top_classification.pt", args.models_dir)
    except FileNotFoundError:
        top_cls_path = None   # optional evidence; absent locally, present on EC2
    if top_cls_path:
        print()
        print("-" * 70)
        print("  STEP 2b  Top-camera classification (top_classification.pt)")
        print("-" * 70)
        for cam in TOP_CAMERAS:
            if cam not in tracks or cam in top_classifications:
                continue
            try:
                top_classifications[cam] = _classify_top_regions(
                    tracks[cam], master_fps=tracks[master_cam].fps,
                    top_classification_model_path=top_cls_path,
                    num_samples=args.classification_samples, verbose=verbose,
                )
            except Exception as e:
                print(f"WARNING: top classification failed for {cam}: {e}",
                      file=sys.stderr)
    elif not top_classifications:
        print("[STAGE1] top_classification.pt not found in models dir; "
              "top semantic evidence disabled (gap-only classification).")

    # ------------------------------------------------------------------
    # STEP 2b(ii) -- SUPPORT WAGON REGIONS
    #
    #   LEFT_UP       -> side_classification.pt   (a side view, same geometry)
    #   RIGHT_UP_TOP  -> top_classification.pt
    #   LEFT_UP_TOP   -> top_classification.pt
    #
    # Identifies each support camera's OWN engine / wagon / brake-van span, so
    # engine and brake-van observations are kept OUT of wagon synchronization.
    # Support cameras are never counting authorities: under master-fixed fusion
    # this changes which support gap matches which master gap (evidence
    # association), never how many global gaps exist.
    #
    # Present cameras only.  A camera whose model is missing, or whose
    # classification raises, gets an explanatory LocalWagonRegion rather than
    # silently aligning over its whole footage.
    # ------------------------------------------------------------------
    support_regions: Dict[str, ts.LocalWagonRegion] = {}
    classification_models: Dict[str, str] = {
        master_cam: os.path.basename(side_cls_path) if side_cls_path else "",
    }
    temporal_results: Dict[str, Any] = {}
    tc_cfg = tcls.DEFAULT_TEMPORAL_CLASSIFICATION

    print()
    print("-" * 70)
    print("  STEP 2b(ii)  Support wagon regions (per-camera engine/wagon span)")
    print("-" * 70)
    _clf_cache: Dict[str, Any] = {}
    for cam in ALL_CAMERAS:
        if cam not in tracks or cam == master_cam:
            continue
        want = ts.CAMERA_CLASSIFICATION_MODEL.get(cam)
        path = top_cls_path if want == ts.TOP_CLASSIFICATION_MODEL else side_cls_path
        if not path:
            classification_models[cam] = f"{want} (MISSING)"
            support_regions[cam] = ts.LocalWagonRegion(
                camera_id=cam, classifier_model=f"{want} (missing)",
                reason=f"{want} not available; camera not classified")
            print(f"  [CLASSIFY/{cam}] SKIPPED -- {want} is not available")
            continue
        try:
            if path not in _clf_cache:
                # Load each model ONCE and reuse it across cameras.
                _clf_cache[path] = ts.load_segment_classifier(
                    path, num_samples=args.classification_samples, verbose=verbose)
            clf, mapping = _clf_cache[path]
            classification_models[cam] = os.path.basename(path)

            t = tracks[cam]
            segs = segments_from_gaps(t.gaps, t.total_frames)
            labels: List[str] = []
            if segs:
                cls = clf.classify_segments(t.video_path, segs)
                # Same temporal smoothing the master gets, so a support camera's
                # region is not moved by a single bad observation.
                cls, tres = tcls.apply_temporal_classification(
                    cls, t.fps, camera_id=cam, cfg=tc_cfg,
                    sample_history=getattr(clf, "sample_history", None),
                    verbose=verbose)
                temporal_results[cam] = tres
                labels = [c.label for c in cls]
            support_regions[cam] = ts.build_local_wagon_region(
                cam, segs, labels, t.fps,
                classifier_model=os.path.basename(path),
                unmapped_classes=mapping.unmapped, verbose=verbose)
        except Exception as e:
            print(f"WARNING: support classification failed for {cam}: {e}",
                  file=sys.stderr)
            support_regions[cam] = ts.LocalWagonRegion(
                camera_id=cam, classifier_model=os.path.basename(str(path)),
                reason=f"classification error: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # STEP 2c -- WAGON_ACTIVE RECOVERY  (second validation pass)
    #
    # Validation had to run in STEP 1b, before classification, because
    # classification needs the segments that validated gaps define.  So the
    # first pass could not know the train state.  Now that the wagon window
    # exists, re-examine the master candidates that fell INSIDE it and failed
    # only a SOFT gate (speed vs the local reference, absolute speed band,
    # sub-floor displacement, weaker confidence, noisier trajectory).
    #
    # Every HARD gate still rejects: untracked, insufficient confirmation,
    # mostly-blind track, isolated static artefact, wrong direction, duplicate,
    # minimum-separation duplicate.  Recovery also re-checks duplicate and
    # separation against the accepted set, so it cannot crowd an existing gap.
    #
    # This is what makes a genuine wagon gap inside the wagon run count
    # immediately, instead of waiting for a further classification event.
    # ------------------------------------------------------------------
    recovery = None
    if (not args.no_gap_validation and not args.no_wagon_recovery
            and gap_validation.get(master_cam)):
        print()
        print("-" * 70)
        print("  STEP 2c  WAGON_ACTIVE recovery (soft-failed gaps inside the "
              "wagon region)")
        print("-" * 70)
        _win = _derive_wagon_window(master, initial_classifications, verbose=False)
        if _win is not None and _win.wagon_start_frame is not None:
            print(f"  wagon window (runtime-derived): frames "
                  f"{_win.wagon_start_frame}-{_win.wagon_end_frame}")
            recovery = gval.recover_wagon_active_candidates(
                gap_validation[master_cam].rejected,
                master.gaps,
                _win.wagon_start_frame, _win.wagon_end_frame,
                master_cam, gv_cfg,
                frame_width=master.width, fps=master.fps,
                absolute_overrides=gv_overrides or None, verbose=verbose)
            if recovery.recovered:
                master.gaps = gval.renumber_gap_events(
                    list(master.gaps) + list(recovery.recovered))
                # The master gap sequence changed, so the segments and therefore
                # the classification must be rebuilt from it.
                print(f"  recovered {len(recovery.recovered)} gap(s) -> "
                      f"re-deriving master segments and classification")
                try:
                    initial_classifications = _classify_master_pre_fusion(
                        master, side_cls_path,
                        num_samples=args.classification_samples, verbose=False)
                    if initial_classifications:
                        initial_classifications, tres2 = \
                            tcls.apply_temporal_classification(
                                initial_classifications, master.fps,
                                camera_id=master_cam, cfg=tc_cfg, verbose=False)
                        temporal_results[master_cam] = tres2
                except Exception as e:
                    print(f"WARNING: re-classification after recovery failed: "
                          f"{e}", file=sys.stderr)
            else:
                print("  no gap recovered -- every wagon-window candidate either "
                      "passed already or failed a hard gate")
        else:
            print("  no wagon window derived -- recovery skipped")

    # ------------------------------------------------------------------
    # STEP 3 -- cross-camera fusion (support = present non-master cameras)
    # ------------------------------------------------------------------
    print()
    print("-" * 70)
    print("  STEP 3  Cross-camera gap fusion")
    print("-" * 70)
    support = [tracks[c] for c in ALL_CAMERAS if c in tracks and c != master_cam]
    support_present = [c for c in ALL_CAMERAS if c in tracks and c != master_cam]

    # MASTER-FIXED FUSION -- the single authoritative construction path.
    #
    # The global gap sequence IS the master's validated gap sequence.  Support
    # cameras are aligned to it only to attach evidence: they cannot create,
    # delete, split or merge a global gap, so the wagon count is independent of
    # both the support detections and the camera-offset estimation.
    #
    # This REPLACES the previous trust-weighted fusion, in which >=2 agreeing
    # support cameras could INSERT a master gap and therefore change the wagon
    # count.  `global_alignment.assemble_global_train_state` is deliberately no
    # longer called from anywhere -- there is exactly one algorithm here now.
    fusion_cfg = gf.FusionConfig(
        offset_search_s=float(args.offset_search),
        offset_min_margin_ratio=float(args.offset_min_margin),
        match_tolerance_s=float(args.match_tolerance),
        strict_invariants=not args.fusion_non_strict,
    )
    state: GlobalTrainState = gf.assemble_global_train_state_master_fixed(
        master_tracks=master,
        support_tracks=support,
        initial_classifications=initial_classifications,
        config=fusion_cfg,
        verbose=verbose,
        wagon_regions=support_regions,
        wagon_only=not args.no_wagon_only,
    )

    # ---- validation / stitching diagnostics onto the state ----
    state.gap_validation_statistics = {
        cam: gap_validation[cam].to_dict(include_rejections=False)
        for cam in ALL_CAMERAS if cam in gap_validation
    }
    state.gap_rejection_details = {
        cam: gap_validation[cam].to_dict(include_rejections=True)["rejections"]
        for cam in ALL_CAMERAS
        if cam in gap_validation and gap_validation[cam].rejected
    }
    state.gap_validation_config = gv_cfg.describe()
    state.fragment_stitching = {
        cam: stitching[cam].to_dict()
        for cam in ALL_CAMERAS if cam in stitching
    }

    # ---- Master-first reconstruction provenance ----
    # Under master-fixed fusion support cameras never insert a gap, so recoveries
    # is always 0 and `support_fusion_used` is False by construction.  The fields
    # are kept because lifecycle_runner + the report metadata read them.
    recoveries = len(state.corrections_applied)
    if not support_present:
        recon_mode = "MASTER_ONLY"
    elif recoveries > 0:
        recon_mode = "MASTER_WITH_FUSED_SUPPORT"
    else:
        recon_mode = "MASTER_WITH_SUPPORT_AVAILABLE"
    state.participating_cameras = [c for c in ALL_CAMERAS if c in present_videos]
    state.missing_at_reconstruction = missing_at_reconstruction
    state.reconstruction_mode = recon_mode
    state.support_cameras_present = support_present
    state.support_gap_recoveries = recoveries
    state.support_fusion_used = recoveries > 0
    state.fallback_master_used = (master_cam != CAMERA_RIGHT_UP)
    state.reconstruction_confidence = 1.0 if master_cam == CAMERA_RIGHT_UP else 0.6
    state.sealed_at = _utc_now_iso()
    state.sealing_reason = (
        f"reconstructed(master={master_cam}, support_present={support_present}, "
        f"recoveries={recoveries}, mode={recon_mode})"
    )

    # ------------------------------------------------------------------
    # STEP 4 -- write JSON
    # ------------------------------------------------------------------
    state_json_path = os.path.join(args.output, "global_train_state.json")
    with open(state_json_path, "w", encoding="utf-8") as f:
        f.write(state.to_json())
    print()
    print(f"[OUTPUT] wrote {state_json_path}")

    tracking_dump = {
        cam: tracks[cam].to_dict(include_classifications=(cam == master_cam))
        for cam in ALL_CAMERAS if cam in tracks
    }
    if initial_classifications:
        tracking_dump[master_cam]["pre_fusion_classifications"] = [
            c.to_dict() for c in initial_classifications
        ]
    tracking_path = os.path.join(args.output, "per_camera_tracking.json")
    with open(tracking_path, "w", encoding="utf-8") as f:
        json.dump(tracking_dump, f, indent=2)
    print(f"[OUTPUT] wrote {tracking_path}")

    # ------------------------------------------------------------------
    # STEP 5 -- overlay videos
    # ------------------------------------------------------------------
    if not args.no_videos:
        print()
        print("-" * 70)
        print("  STEP 5  Overlay videos")
        print("-" * 70)
        for cam in ALL_CAMERAS:
            if cam not in tracks:
                continue
            try:
                out_mp4 = os.path.join(processed_videos_dir, f"{cam}_processed.mp4")
                vs.render_processed_video(
                    local_tracks=tracks[cam],
                    state=state,
                    output_path=out_mp4,
                    draw_raw_detections=keep_raw,
                    verbose=verbose,
                    debug=stage1_debug,
                )
            except Exception as e:
                print(f"WARNING: render failed for {cam}: {e}", file=sys.stderr)
                state.add_note(f"render_failed:{cam}:{e}")

    # ------------------------------------------------------------------
    # STEP 6 -- wagon-wise frame extraction
    # ------------------------------------------------------------------
    if not args.no_frames:
        print()
        print("-" * 70)
        print("  STEP 6  Per-wagon frame extraction")
        print("-" * 70)
        for cam in ALL_CAMERAS:
            if cam not in tracks:
                continue
            try:
                vs.extract_wagon_frames(
                    local_tracks=tracks[cam],
                    state=state,
                    output_root=frames_root,
                    every_nth_frame=args.every_nth_frame,
                    verbose=verbose,
                )
            except Exception as e:
                print(f"WARNING: frame extraction failed for {cam}: {e}", file=sys.stderr)
                state.add_note(f"frame_extraction_failed:{cam}:{e}")

    # ------------------------------------------------------------------
    # STEP 7 -- final summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t_start
    print()
    print(summarize_state(state))
    print(f"  total elapsed: {elapsed:.1f}s")
    print(f"  output root  : {os.path.abspath(args.output)}")
    print()

    # Re-write JSON so any added notes are persisted
    with open(state_json_path, "w", encoding="utf-8") as f:
        f.write(state.to_json())

    return 0


if __name__ == "__main__":
    sys.exit(main())
