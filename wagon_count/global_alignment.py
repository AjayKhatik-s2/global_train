"""
global_alignment.py  --  Phase-1 cross-camera gap fusion (standalone)
=====================================================================

This is the standalone Phase-1 version of the alignment module: it carries
ONLY the gap-level fusion logic.  The legacy v3 functions that depend on
RIGHT_UP/train_session.CameraEvidence have been removed -- they belong to
the production reporting pipeline, which is out of scope for this package.

Pipeline:
    1) match_support_to_master       support gap -> closest master gap
    2) cluster_unmatched_supports    group leftover support gaps in time
    3) decide_inserted_gaps          accept clusters with quorum + confidence
    4) fuse_master_timeline          combine real master gaps + accepted inserts
    5) build_global_wagons           emit GW_1..GW_N with inherited classification
    6) assemble_global_train_state   end-to-end with deterministic fallback

Determinism: deterministic sort keys throughout; no randomness; same input
yields the same output.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, replace
from typing import List, Dict, Optional, Tuple, Any

from global_train_state import (
    GapEvent,
    LocalCameraTracks,
    SegmentClass,
    GlobalWagon,
    GlobalTrainState,
    GapCorrection,
    _MasterClassification,
    MASTER_CAMERA,
    ALL_CAMERAS,
)


# -----------------------------------------------------------------------------
# Core temporal-IoU math (kept here so this module has no external deps
# beyond global_train_state).
# -----------------------------------------------------------------------------

def compute_temporal_iou(
    a_start: float, a_end: float,
    b_start: float, b_end: float,
) -> Tuple[float, float]:
    """Return (IoU, overlap_seconds) of two closed intervals."""
    if a_end <= a_start or b_end <= b_start:
        return 0.0, 0.0
    inter_start = max(a_start, b_start)
    inter_end = min(a_end, b_end)
    overlap = max(0.0, inter_end - inter_start)
    if overlap <= 0.0:
        return 0.0, 0.0
    union = (a_end - a_start) + (b_end - b_start) - overlap
    if union <= 0.0:
        return 0.0, 0.0
    return overlap / union, overlap


# -----------------------------------------------------------------------------
# Tuning knobs
# -----------------------------------------------------------------------------

PHASE1_DEFAULTS = {
    # Matching a support gap to a master gap
    "match_time_window_sec": 1.0,
    "match_min_iou": 0.2,

    # Inserting a missed master gap (gap-recovery quorum) -- DISABLED under the
    # canonical rule (kept for reference only; see fuse_master_timeline).
    "insert_min_support": 2,
    "insert_max_spread_sec": 1.5,
    "insert_min_confidence": 0.4,
    "insert_min_distance_to_master_sec": 1.0,

    # --- Confidence-weighted boundary refinement (Stage-1 redesign) ----------
    # A trusted TOP camera may NUDGE a canonical boundary toward its own gap
    # when it agrees within `boundary_refine_window_sec`; the nudge is a
    # weighted average and is clamped to `boundary_refine_max_shift_sec` so a
    # refiner can never move a boundary far enough to reorder or drop one.
    "boundary_refine_window_sec": 0.5,
    "boundary_refine_max_shift_sec": 0.5,
}


# -----------------------------------------------------------------------------
# Confidence-weighted camera trust (Stage-1 redesign)
# -----------------------------------------------------------------------------
# RIGHT_UP is the CANONICAL master: it alone defines the wagon count, the Global
# Wagon IDs, and the initial boundaries.  RIGHT_UP_TOP and LEFT_UP_TOP are
# TRUSTED REFINERS: they may only nudge an existing boundary's position (never
# add/delete a wagon).  LEFT_UP has trust 0.0 for gap detection -- its gaps are
# IGNORED for boundaries; it contributes only the train start/end envelope
# (weight 1.0 there) plus downstream feature evidence.  Override per camera via
# config["gap_trust_weights"] or the WAGONEYE_GAP_TRUST_<CAMERA> env var.
DEFAULT_GAP_TRUST_WEIGHTS: Dict[str, float] = {
    "RIGHT_UP":     1.0,   # canonical master
    "RIGHT_UP_TOP": 0.9,   # trusted refiner
    "LEFT_UP_TOP":  0.9,   # trusted refiner
    "LEFT_UP":      0.0,   # NOT a gap source (start/end + features only)
}


def resolve_gap_trust_weights(config: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """Merge default trust weights with config + WAGONEYE_GAP_TRUST_<CAM> env."""
    w = dict(DEFAULT_GAP_TRUST_WEIGHTS)
    if config and isinstance(config.get("gap_trust_weights"), dict):
        for k, v in config["gap_trust_weights"].items():
            try:
                w[k] = float(v)
            except (TypeError, ValueError):
                pass
    for cam in ALL_CAMERAS:
        env = os.environ.get(f"WAGONEYE_GAP_TRUST_{cam}")
        if env is not None:
            try:
                w[cam] = float(env)
            except ValueError:
                pass
    return w


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _gap_to_interval(g: GapEvent) -> Tuple[float, float, float]:
    return g.start_time, g.end_time, g.center_time


def _interval_iou(a_s: float, a_e: float, b_s: float, b_e: float) -> float:
    iou, _ = compute_temporal_iou(a_s, a_e, b_s, b_e)
    return iou


# -----------------------------------------------------------------------------
# Step A -- match each support gap to a master gap
# -----------------------------------------------------------------------------

def match_support_to_master(
    master_gaps: List[GapEvent],
    support_gaps: List[GapEvent],
    match_time_window_sec: float = 1.0,
    match_min_iou: float = 0.2,
) -> Tuple[Dict[int, int], List[GapEvent]]:
    """Match each support gap to its best master gap.

    Returns
    -------
    matched : dict   support.track_id -> master.track_id
    leftover : list  support gaps that could not be matched
    """
    matched: Dict[int, int] = {}
    leftover: List[GapEvent] = []
    if not master_gaps:
        return matched, list(support_gaps)

    m_intervals = [(g.track_id, g.start_time, g.end_time, g.center_time) for g in master_gaps]

    for sg in support_gaps:
        best_score = -1.0
        best_master_id = -1
        s_s, s_e, s_c = _gap_to_interval(sg)
        for (m_id, m_s, m_e, m_c) in m_intervals:
            iou = _interval_iou(s_s, s_e, m_s, m_e)
            dt = abs(s_c - m_c)
            time_score = max(0.0, 1.0 - dt / max(match_time_window_sec, 1e-3))
            score = max(iou, time_score) if (iou >= match_min_iou or dt <= match_time_window_sec) else -1.0
            if score > best_score:
                best_score = score
                best_master_id = m_id

        if best_master_id >= 0 and best_score >= 0.0:
            matched[sg.track_id] = best_master_id
        else:
            leftover.append(sg)

    return matched, leftover


# -----------------------------------------------------------------------------
# Step B -- cluster unmatched supports across cameras
# -----------------------------------------------------------------------------

def cluster_unmatched_supports(
    leftovers_per_camera: Dict[str, List[GapEvent]],
    spread_sec: float,
) -> List[List[GapEvent]]:
    """Sweep over the union of leftover gaps sorted by center_time."""
    all_gaps: List[GapEvent] = []
    for cam, gs in leftovers_per_camera.items():
        all_gaps.extend(gs)
    if not all_gaps:
        return []
    all_gaps.sort(key=lambda g: (g.center_time, g.camera_id, g.track_id))

    clusters: List[List[GapEvent]] = []
    current: List[GapEvent] = [all_gaps[0]]
    cluster_center = all_gaps[0].center_time

    for g in all_gaps[1:]:
        if abs(g.center_time - cluster_center) <= spread_sec:
            current.append(g)
            cluster_center = sum(x.center_time for x in current) / len(current)
        else:
            clusters.append(current)
            current = [g]
            cluster_center = g.center_time
    clusters.append(current)
    return clusters


# -----------------------------------------------------------------------------
# Step C -- decide which clusters become inserted master gaps
# -----------------------------------------------------------------------------

def decide_inserted_gaps(
    clusters: List[List[GapEvent]],
    master_gaps: List[GapEvent],
    *,
    min_support: int,
    max_spread_sec: float,
    min_confidence: float,
    min_distance_to_master_sec: float,
    master_fps: float,
) -> List[GapCorrection]:
    inserted: List[GapCorrection] = []
    if not clusters:
        return inserted

    master_centers = [g.center_time for g in master_gaps]

    for cluster in clusters:
        cams = {g.camera_id for g in cluster}
        if len(cams) < min_support:
            continue
        centers = sorted(g.center_time for g in cluster)
        spread = centers[-1] - centers[0]
        if spread > max_spread_sec:
            continue
        mean_conf = float(sum(g.confidence for g in cluster) / len(cluster))
        if mean_conf < min_confidence:
            continue
        center = sum(centers) / len(centers)
        if master_centers and min(abs(center - mc) for mc in master_centers) < min_distance_to_master_sec:
            continue

        inserted.append(GapCorrection(
            inserted_at_master_time=center,
            inserted_at_master_frame=int(round(center * master_fps)),
            supporting_cameras=sorted(cams),
            mean_confidence=mean_conf,
            time_spread_sec=spread,
            contributing_track_ids={g.camera_id: g.track_id for g in cluster},
        ))

    inserted.sort(key=lambda c: c.inserted_at_master_time)
    return inserted


# -----------------------------------------------------------------------------
# Step D -- fuse the corrected master gap list
# -----------------------------------------------------------------------------

def fuse_master_timeline(
    master_tracks: LocalCameraTracks,
    support_tracks: List[LocalCameraTracks],
    *,
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Tuple[List[GapEvent], List[GapCorrection], Dict[str, List[GapEvent]]]:
    """DEPRECATED / NOT INVOKED.  Support-gap insertion is disabled under the
    canonical-master rule: RIGHT_UP is the sole authority for the wagon count
    and numbering (`assemble_global_train_state` builds from master gaps only).
    Retained for reference; do not call it to build wagons -- doing so would let
    support cameras change the canonical count."""
    cfg = dict(PHASE1_DEFAULTS)
    if config:
        cfg.update(config)

    master_gaps = list(master_tracks.gaps)
    master_fps = master_tracks.fps

    leftovers_per_cam: Dict[str, List[GapEvent]] = {}
    for st in support_tracks:
        matched, leftover = match_support_to_master(
            master_gaps, st.gaps,
            match_time_window_sec=cfg["match_time_window_sec"],
            match_min_iou=cfg["match_min_iou"],
        )
        leftovers_per_cam[st.camera_id] = leftover
        if verbose:
            print(f"[FUSE/{st.camera_id}] matched={len(matched)}  leftover={len(leftover)}  "
                  f"(of {len(st.gaps)} support gaps)")

    clusters = cluster_unmatched_supports(leftovers_per_cam, spread_sec=cfg["insert_max_spread_sec"])
    if verbose:
        print(f"[FUSE] {len(clusters)} cross-camera cluster(s) of unmatched support gaps")

    inserts = decide_inserted_gaps(
        clusters,
        master_gaps,
        min_support=cfg["insert_min_support"],
        max_spread_sec=cfg["insert_max_spread_sec"],
        min_confidence=cfg["insert_min_confidence"],
        min_distance_to_master_sec=cfg["insert_min_distance_to_master_sec"],
        master_fps=master_fps,
    )
    if verbose:
        print(f"[FUSE] {len(inserts)} gap(s) will be inserted into master timeline")
        for c in inserts:
            print(f"   + insert @ t={c.inserted_at_master_time:.2f}s  "
                  f"f={c.inserted_at_master_frame}  "
                  f"supports={'/'.join(c.supporting_cameras)}  "
                  f"conf={c.mean_confidence:.2f}  spread={c.time_spread_sec:.2f}s")

    next_synth_id = -1
    synth_gaps: List[GapEvent] = []
    for c in inserts:
        f = c.inserted_at_master_frame
        synth_gaps.append(GapEvent(
            track_id=next_synth_id,
            camera_id=f"FUSED({'+'.join(c.supporting_cameras)})",
            start_frame=max(0, f - 1),
            end_frame=f + 1,
            confidence=c.mean_confidence,
            hit_count=len(c.contributing_track_ids),
            center_x_trajectory=[],
            fps=master_fps,
            temporal_consistency_score=1.0,
            class_label="gap_inserted",
        ))
        next_synth_id -= 1

    fused = sorted(master_gaps + synth_gaps, key=lambda g: g.center_time)
    return fused, inserts, leftovers_per_cam


# -----------------------------------------------------------------------------
# Ownership-based boundary assignment
# -----------------------------------------------------------------------------

def ownership_transition_frame(gap: GapEvent, frame_width: int) -> Optional[int]:
    """OWNERSHIP transition between the wagon before this gap and the wagon after.

    A gap is visible over a RANGE of frames as it sweeps across the image.  The
    boundary between the two wagons is NOT the gap's temporal centre -- it is the
    frame at which image ownership flips, i.e. where the gap's image-plane centre
    crosses the frame MIDLINE so that the majority of the visible image goes from
    the previous wagon to the next.  Combines both cues the caller asked for:

      * spatial  -- the gap's centre_x vs the image midline (majority owner);
      * temporal -- the train's travel direction (sign of the centre_x drift),
                    used to pick the crossing consistent with that direction.

    Uses the gap's per-hit trajectory (``hit_frames`` + ``bbox_history``).
    Returns ``None`` when the gap never crosses the midline (caller keeps the
    gap's temporal centre).  Frames strictly BEFORE the returned frame belong to
    the previous wagon, frames from it ONWARD to the next -- every frame is owned
    by exactly one wagon; none is shared or discarded.
    """
    hf = getattr(gap, "hit_frames", None)
    bh = getattr(gap, "bbox_history", None)
    if not hf or not bh or frame_width <= 0 or len(hf) != len(bh):
        return None
    mid = frame_width / 2.0
    cx = [(float(b[0]) + float(b[2])) / 2.0 for b in bh]
    direction = 1.0 if cx[-1] >= cx[0] else -1.0     # travel direction across image
    for i in range(len(cx) - 1):
        a, b = cx[i], cx[i + 1]
        if a == mid:
            return int(hf[i])
        if (a - mid) * (b - mid) < 0 and (b - a) * direction >= 0:
            denom = (b - a)
            t = (mid - a) / denom if denom != 0 else 0.0
            return int(round(hf[i] + t * (hf[i + 1] - hf[i])))
    return None


# -----------------------------------------------------------------------------
# Step E -- rebuild GlobalWagons, inheriting RIGHT_UP classification
# -----------------------------------------------------------------------------

def build_global_wagons(
    fused_gaps: List[GapEvent],
    *,
    master_total_frames: int,
    master_fps: float,
    initial_classifications: List[_MasterClassification],
    support_camera_ids: List[str],
    master_camera_id: str = MASTER_CAMERA,
    frame_width: int = 0,
) -> List[GlobalWagon]:
    if master_total_frames <= 0:
        return []

    # Split adjacent wagons at the OWNERSHIP-TRANSITION frame (image-majority
    # flip) rather than the gap's temporal centre; fall back to the centre when
    # the gap never crosses the midline.  Each boundary is clamped to stay
    # strictly between its neighbours, so there is exactly ONE boundary per gap
    # and the wagon COUNT + ordering are preserved (only WHERE the split lands
    # changes).  Every frame lands in exactly one contiguous [start, end] span.
    ordered = sorted(fused_gaps, key=lambda g: g.center_frame)
    n_ord = len(ordered)
    boundaries: List[int] = []
    prev_b = 0
    n_ownership = 0
    for i, g in enumerate(ordered):
        center = int(round(g.center_frame))
        of = ownership_transition_frame(g, frame_width)
        if of is not None:
            n_ownership += 1
        f = of if of is not None else center
        nxt_center = (int(round(ordered[i + 1].center_frame))
                      if i + 1 < n_ord else master_total_frames)
        lo, hi = prev_b + 1, nxt_center - 1
        f = max(lo, min(hi, f)) if lo <= hi else center
        f = max(0, min(master_total_frames - 1, f))
        boundaries.append(f)
        prev_b = f
    if frame_width > 0 and n_ord:
        print(f"[STAGE1] Ownership boundaries: {n_ownership}/{n_ord} split at the "
              f"image-majority crossing (rest kept the gap centre); every frame "
              f"belongs to exactly one wagon")

    def label_for_frame(frame_idx: int) -> Tuple[str, float]:
        for c in initial_classifications:
            if c.start_frame <= frame_idx <= c.end_frame:
                return c.label, c.confidence
        if not initial_classifications:
            return SegmentClass.UNKNOWN, 0.0
        nearest = min(initial_classifications,
                      key=lambda c: min(abs(c.start_frame - frame_idx),
                                        abs(c.end_frame - frame_idx)))
        return nearest.label, nearest.confidence

    segs: List[Tuple[int, int]] = []
    prev = 0
    for b in boundaries:
        if b <= prev:
            continue
        segs.append((prev, b - 1))
        prev = b
    if prev <= master_total_frames - 1:
        segs.append((prev, master_total_frames - 1))

    # --- Startup false-engine guard: delay initialization -------------------
    # NEVER assume the first detected segment is the ENGINE.  If the LEADING
    # segment classifies as UNKNOWN -- i.e. the model gave no stable, confident
    # evidence (low-confidence loco-front, bare track, or background as the
    # train enters frame; an uncertain 'engine' read is demoted to UNKNOWN
    # upstream in tracker_engine._label_to_class) -- it is a phantom leading
    # wagon.  Drop ONLY that single leading segment so GW_1 re-bases onto the
    # first stably-classified real wagon.  We never drop a real
    # WAGON/ENGINE/BRAKE_VAN segment, so real wagons are never removed and
    # never renumbered relative to each other -- only the phantom disappears.
    # Skipped entirely when classification was unavailable (would label
    # everything UNKNOWN) or when only one segment exists (never empty a train).
    if len(segs) > 1 and initial_classifications:
        lead_sf, lead_ef = segs[0]
        lead_label, _lead_conf = label_for_frame((lead_sf + lead_ef) // 2)
        if lead_label == SegmentClass.UNKNOWN:
            segs = segs[1:]

    wagons: List[GlobalWagon] = []
    fused_sorted = sorted(fused_gaps, key=lambda g: g.center_frame)
    for i, (sf, ef) in enumerate(segs, start=1):
        center_frame = (sf + ef) // 2
        label, conf = label_for_frame(center_frame)
        gw = GlobalWagon(
            global_id=f"GW_{i}",
            wagon_index=i,
            start_frame_master=sf,
            end_frame_master=ef,
            start_time=sf / master_fps if master_fps > 0 else 0.0,
            end_time=(ef + 1) / master_fps if master_fps > 0 else 0.0,
            classification=label,
            classification_confidence=conf,
            supporting_cameras=[master_camera_id]
            + [c for c in support_camera_ids if c != master_camera_id],
        )

        leading = None
        trailing = None
        for g in fused_sorted:
            cf = int(round(g.center_frame))
            # Gap whose center is at or before sf IS the leading boundary
            # of this segment.  Strict `<` would miss the boundary-frame case.
            if cf <= sf:
                leading = g
            elif cf > ef and trailing is None:
                trailing = g
                break
        if leading is not None:
            gw.leading_gap = {
                "source": "master" if leading.track_id > 0 else "fused",
                "camera_id": leading.camera_id,
                "track_id": leading.track_id,
                "center_time": round(leading.center_time, 4),
            }
        else:
            gw.leading_gap = {"source": "video_start"}
        if trailing is not None:
            gw.trailing_gap = {
                "source": "master" if trailing.track_id > 0 else "fused",
                "camera_id": trailing.camera_id,
                "track_id": trailing.track_id,
                "center_time": round(trailing.center_time, 4),
            }
        else:
            gw.trailing_gap = {"source": "video_end"}

        if (leading is not None and leading.track_id < 0) or \
           (trailing is not None and trailing.track_id < 0):
            parent_idx = next(
                (c.segment_index for c in initial_classifications
                 if c.start_frame <= sf <= c.end_frame),
                None,
            )
            if parent_idx is not None:
                gw.split_from_global_id = f"PRE_SEG_{parent_idx}"

        wagons.append(gw)

    return wagons


# -----------------------------------------------------------------------------
# Step F -- pure-master fallback
# -----------------------------------------------------------------------------

def build_wagons_pure_master(
    master_tracks: LocalCameraTracks,
    initial_classifications: List[_MasterClassification],
) -> List[GlobalWagon]:
    fused = sorted(master_tracks.gaps, key=lambda g: g.center_time)
    return build_global_wagons(
        fused,
        master_total_frames=master_tracks.total_frames,
        master_fps=master_tracks.fps,
        initial_classifications=initial_classifications,
        support_camera_ids=[master_tracks.camera_id],
        master_camera_id=master_tracks.camera_id,
        frame_width=master_tracks.width,
    )


# -----------------------------------------------------------------------------
# Step F2 -- confidence-weighted boundary refinement (trusted TOP cameras)
# -----------------------------------------------------------------------------

def _nearest_wagon_id(wagons: List[GlobalWagon], frame: float) -> str:
    if not wagons:
        return "GW_?"
    return min(wagons, key=lambda w: abs(w.start_frame_master - frame)).global_id


def projection_camera_envelope(tracks: LocalCameraTracks) -> Tuple[int, int]:
    """Train start/end FRAME envelope for a projection-only camera (LEFT_UP).

    LEFT_UP's individual gaps are NOT trusted for boundaries, but the OUTER span
    of its detected activity reliably brackets where the train enters and leaves
    view.  Falls back to the whole clip if it detected nothing."""
    if tracks.gaps:
        return (min(int(g.start_frame) for g in tracks.gaps),
                max(int(g.end_frame) for g in tracks.gaps))
    return 0, max(0, tracks.total_frames - 1)


def refine_master_boundaries(
    master_tracks: LocalCameraTracks,
    refiner_tracks: List[LocalCameraTracks],
    weights: Dict[str, float],
    *,
    window_sec: float,
    max_shift_sec: float,
) -> Tuple[List[GapEvent], List[Dict[str, Any]]]:
    """Refine every canonical (master) boundary with the TRUSTED refiner cameras.

    For each master gap we form a trust-weighted average of its center with any
    refiner gap that agrees within ``window_sec``.  Cameras share t=0 after the
    Stage-1 frame trim, so the comparison + averaging happen in the master time
    domain.  The resulting shift is clamped to ``max_shift_sec`` AND to stay
    strictly between the neighbouring boundaries, so the wagon COUNT and ordering
    are preserved exactly -- a refiner can only MOVE a boundary, never create,
    delete, split, or merge one.

    Returns ``(refined_gaps, logs)`` where ``logs[i]`` records the refined frame,
    the applied shift, and per-refiner the matching-gap offset (or ``None``).
    """
    m_fps = master_tracks.fps if master_tracks.fps > 0 else 1.0
    # The canonical master ALWAYS anchors its own boundary with weight 1.0.  The
    # trust table applies to the *refiner* role only (so LEFT_UP=0.0 disables it
    # as a support gap source, yet it keeps full authority if it is ever forced
    # to be the fallback master).
    w_master = 1.0
    ordered = sorted(master_tracks.gaps, key=lambda g: g.center_time)
    n = len(ordered)
    max_shift = max_shift_sec * m_fps
    refined: List[GapEvent] = []
    logs: List[Dict[str, Any]] = []
    prev_center = 0.0
    for i, g in enumerate(ordered):
        F, T = g.center_frame, g.center_time
        num, den = w_master * F, w_master
        matches: Dict[str, Optional[float]] = {}
        for rt in refiner_tracks:
            w_c = float(weights.get(rt.camera_id, 0.0))
            if w_c <= 0.0 or not rt.gaps:
                matches[rt.camera_id] = None
                continue
            best = min(rt.gaps, key=lambda gc: abs(gc.center_time - T))
            if abs(best.center_time - T) <= window_sec:
                f_c = best.center_time * m_fps          # -> master frames
                num += w_c * f_c
                den += w_c
                matches[rt.camera_id] = f_c - F
            else:
                matches[rt.camera_id] = None
        F_new = num / den if den > 0 else F
        # clamp the magnitude of the nudge ...
        F_new = max(F - max_shift, min(F + max_shift, F_new))
        # ... and keep boundaries strictly ordered (count + sequence preserved)
        nxt = ordered[i + 1].center_frame if i + 1 < n else float(master_tracks.total_frames)
        lo, hi = prev_center + 1.0, nxt - 1.0
        F_new = max(lo, min(hi, F_new)) if lo <= hi else F
        prev_center = F_new
        shift = F_new - F
        refined.append(replace(g, start_frame=int(round(g.start_frame + shift)),
                               end_frame=int(round(g.end_frame + shift))))
        logs.append({"orig": F, "refined": F_new, "shift": shift, "matches": matches})
    return refined, logs


# -----------------------------------------------------------------------------
# Step F3 -- semantic-label fusion (top_classification.pt evidence)
# -----------------------------------------------------------------------------

def _top_label_for_window(cls_list: List[_MasterClassification],
                          sf: int, ef: int) -> Tuple[Optional[str], float]:
    """Max-overlap top-camera class label over a wagon's master-frame window."""
    best, best_ov = None, 0
    for c in cls_list:
        ov = min(ef, c.end_frame) - max(sf, c.start_frame)
        if ov > best_ov:
            best_ov, best = ov, c
    if best is None or best_ov <= 0:
        return None, 0.0
    return best.label, float(best.confidence)


def fuse_semantic_labels(
    wagons: List[GlobalWagon],
    top_classifications: Dict[str, List[_MasterClassification]],
    weights: Dict[str, float],
    *,
    verbose: bool = True,
) -> None:
    """Refine each ``GlobalWagon.classification`` with the TOP cameras' semantic
    evidence (top_classification.pt) IN PLACE.

    RIGHT_UP (side_classification.pt) is the weight-1.0 anchor; each top camera
    adds a trust-weighted vote for the class it read over the same master-frame
    window.  The wagon COUNT, IDs, and boundaries are never touched -- only the
    class label + its confidence.  Physical structure is then enforced: ENGINE
    only in a LEADING contiguous run and BRAKE_VAN only in a TRAILING contiguous
    run, so a stray engine/brake-van read can never contaminate the wagon region
    (requirement: "prevent wagon creation inside engine or brake van regions").
    """
    tops = {c: cl for c, cl in (top_classifications or {}).items() if cl}
    if not wagons:
        return
    for w in wagons:
        votes: Dict[str, float] = {}
        src: Dict[str, Any] = {}
        base_conf = max(float(w.classification_confidence or 0.0), 0.5)
        votes[w.classification] = (votes.get(w.classification, 0.0)
                                   + float(weights.get(MASTER_CAMERA, 1.0)) * base_conf)
        src[MASTER_CAMERA] = {"label": w.classification,
                              "confidence": round(float(w.classification_confidence or 0.0), 3)}
        for cam, cls_list in tops.items():
            w_c = float(weights.get(cam, 0.0))
            if w_c <= 0.0:
                continue
            lbl, conf = _top_label_for_window(cls_list, w.start_frame_master, w.end_frame_master)
            if not lbl:
                continue
            votes[lbl] = votes.get(lbl, 0.0) + w_c * conf
            src[cam] = {"label": lbl, "confidence": round(conf, 3)}
        final = max(sorted(votes), key=lambda k: votes[k])   # deterministic argmax
        tot = sum(votes.values())
        w.classification = final
        w.classification_confidence = float(votes[final] / tot) if tot > 0 else 0.0
        w.classification_sources = src

    n = len(wagons)
    i = 0
    while i < n and wagons[i].classification == SegmentClass.ENGINE:
        i += 1
    for w in wagons[i:]:
        if w.classification == SegmentClass.ENGINE:      # engine can't be mid-train
            w.classification = SegmentClass.WAGON
    j = n - 1
    while j >= 0 and wagons[j].classification == SegmentClass.BRAKE_VAN:
        j -= 1
    for w in wagons[:j + 1]:
        if w.classification == SegmentClass.BRAKE_VAN:   # brake van can't be mid-train
            w.classification = SegmentClass.WAGON


# -----------------------------------------------------------------------------
# Step G -- end-to-end
# -----------------------------------------------------------------------------

def assemble_global_train_state(
    *,
    master_tracks: LocalCameraTracks,
    support_tracks: List[LocalCameraTracks],
    initial_classifications: List[_MasterClassification],
    top_classifications: Optional[Dict[str, List[_MasterClassification]]] = None,
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> GlobalTrainState:
    cfg = dict(PHASE1_DEFAULTS)
    if config:
        cfg.update(config)

    per_local_counts: Dict[str, int] = {master_tracks.camera_id: master_tracks.local_wagon_count}
    per_gap_counts: Dict[str, int] = {master_tracks.camera_id: len(master_tracks.gaps)}
    per_status: Dict[str, str] = {master_tracks.camera_id: "ok"}
    for st in support_tracks:
        per_local_counts[st.camera_id] = st.local_wagon_count
        per_gap_counts[st.camera_id] = len(st.gaps)
        per_status[st.camera_id] = "ok"

    if verbose:
        print(f"[FUSE] master({master_tracks.camera_id}) wagons={master_tracks.local_wagon_count} "
              f"gaps={len(master_tracks.gaps)}")
        for st in support_tracks:
            print(f"[FUSE] support({st.camera_id}) wagons={st.local_wagon_count} "
                  f"gaps={len(st.gaps)}")

    # =====================================================================
    # CONFIDENCE-WEIGHTED RECONSTRUCTION (Stage-1 redesign)
    #   RIGHT_UP      -> CANONICAL master: sole authority for the wagon COUNT,
    #                    the Global Wagon IDs, and the initial boundaries.
    #   RIGHT_UP_TOP, -> TRUSTED REFINERS (trust > 0): may only NUDGE an existing
    #   LEFT_UP_TOP      boundary's position; NEVER add/delete/split/merge one.
    #   LEFT_UP       -> trust 0.0 for gaps: its gaps are IGNORED for boundaries;
    #                    it supplies only the train start/end envelope + the
    #                    downstream feature evidence.
    # Support cameras can NEVER change the wagon count or numbering, so GW_i is
    # one-to-one with the master's wagon sequence.  Support-gap INSERTION
    # (fuse_master_timeline) stays disabled.
    # =====================================================================
    weights = resolve_gap_trust_weights(config)
    master_id = master_tracks.camera_id
    refiner_tracks = [st for st in support_tracks if weights.get(st.camera_id, 0.0) > 0.0]
    projection_tracks = [st for st in support_tracks if weights.get(st.camera_id, 0.0) <= 0.0]

    fallback_used = False
    fallback_reason = ""
    refine_logs: List[Dict[str, Any]] = []
    try:
        refined_gaps, refine_logs = refine_master_boundaries(
            master_tracks, refiner_tracks, weights,
            window_sec=cfg["boundary_refine_window_sec"],
            max_shift_sec=cfg["boundary_refine_max_shift_sec"],
        )
        wagons = build_global_wagons(
            refined_gaps,
            master_total_frames=master_tracks.total_frames,
            master_fps=master_tracks.fps,
            initial_classifications=initial_classifications,
            support_camera_ids=[st.camera_id for st in support_tracks],
            master_camera_id=master_id,
            frame_width=master_tracks.width,
        )
    except Exception as e:
        fallback_used = True
        fallback_reason = f"reconstruction error: {type(e).__name__}: {e}"
        if verbose:
            print(f"[STAGE1] {fallback_reason} -- falling back to raw master gaps")
        refine_logs = []
        wagons = build_wagons_pure_master(master_tracks, initial_classifications)
    if not wagons and master_tracks.total_frames > 0 and master_tracks.gaps:
        fallback_used = True
        if not fallback_reason:
            fallback_reason = "master produced no wagons"

    total = len(wagons)

    # --- Semantic refinement: fold top_classification.pt evidence into each
    # wagon's class label (ENGINE/WAGON/BRAKE_VAN) WITHOUT changing count/ids/
    # boundaries.  RIGHT_UP stays the weight-1.0 anchor; the top cameras add
    # trust-weighted votes and pin the engine/brake-van regions to the ends.
    if top_classifications:
        try:
            fuse_semantic_labels(wagons, top_classifications, weights, verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"[STAGE1] semantic fusion skipped: {type(e).__name__}: {e}")

    # LEFT_UP (and any weight-0 camera): train envelope only -> stored as a note
    notes: List[str] = []
    for pt in projection_tracks:
        s0, s1 = projection_camera_envelope(pt)
        per_status[pt.camera_id] = "projected_only"
        notes.append(f"{pt.camera_id}_train_envelope_frames={s0}..{s1}")

    # ---------------- [STAGE1] validation logs ----------------
    if verbose:
        print(f"[STAGE1] Canonical camera: {master_id}")
        print(f"[STAGE1] Canonical wagon count: {total} ({master_id})")
        seen = [master_id] + [s.camera_id for s in support_tracks]
        print("[STAGE1] Trust weights (gap): "
              + ", ".join(f"{c}={weights.get(c, 0.0):.1f}" for c in ALL_CAMERAS if c in seen))
        for lg in refine_logs:
            gw = _nearest_wagon_id(wagons, lg["refined"])
            parts = [f"{master_id}: accepted"]
            for rt in refiner_tracks:
                off = lg["matches"].get(rt.camera_id)
                parts.append(f"{rt.camera_id}: "
                             + (f"matched ({off:+.0f}f)" if off is not None else "no-match"))
            for pt in projection_tracks:
                parts.append(f"{pt.camera_id}: projected-only (gap ignored)")
            tail = f"shift {lg['shift']:+.0f}f" if abs(lg["shift"]) >= 0.5 else "unchanged"
            print(f"[STAGE1] Boundary {gw} @f{int(round(lg['refined']))}: "
                  + " | ".join(parts) + f" -> {tail}")
        for pt in projection_tracks:
            s0, s1 = projection_camera_envelope(pt)
            print(f"[STAGE1] {pt.camera_id}: projected-only "
                  f"(train envelope frames {s0}..{s1}; gaps excluded from boundaries)")

    # refiner corroboration summary (audit only -- cannot change count/numbering)
    for rt in refiner_tracks:
        matched, leftover = match_support_to_master(
            master_tracks.gaps, rt.gaps,
            match_time_window_sec=cfg["match_time_window_sec"],
            match_min_iou=cfg["match_min_iou"])
        boundaries_matched = len(set(matched.values()))
        matched_wagons = max(0, total - max(0, len(master_tracks.gaps) - boundaries_matched))
        if matched_wagons < total:
            per_status[rt.camera_id] = "missing_evidence"
        if verbose:
            extra = f"  (+{len(leftover)} unmatched evidence)" if leftover else ""
            print(f"[STAGE1] {rt.camera_id} matched: {matched_wagons}/{total}{extra}")

    if verbose and top_classifications:
        eng = [w.global_id for w in wagons if w.classification == SegmentClass.ENGINE]
        bv = [w.global_id for w in wagons if w.classification == SegmentClass.BRAKE_VAN]
        n_wag = sum(1 for w in wagons if w.classification == SegmentClass.WAGON)
        print(f"[STAGE1] Semantic evidence: top_classification.pt on "
              f"{sorted(top_classifications)}")
        print(f"[STAGE1] Engine region: {eng or '(none)'} | "
              f"Brake-van region: {bv or '(none)'} | WAGON wagons: {n_wag}")
        if wagons:
            print(f"[STAGE1] Train start: {wagons[0].global_id} "
                  f"({wagons[0].classification}) @f{wagons[0].start_frame_master} | "
                  f"Train end: {wagons[-1].global_id} ({wagons[-1].classification}) "
                  f"@f{wagons[-1].end_frame_master}")

    if verbose:
        print(f"[STAGE1] Global boundaries finalized -- Final Global Train: "
              f"{total} wagons ({master_id} canonical)")

    state = GlobalTrainState(
        total_wagons=total,
        wagons=wagons,
        master_camera=master_tracks.camera_id,
        master_fps=master_tracks.fps,
        master_total_frames=master_tracks.total_frames,
        per_camera_local_counts=per_local_counts,
        per_camera_gap_counts=per_gap_counts,
        per_camera_status=per_status,
        # support cameras never insert a gap; refiners only nudge positions
        corrections_applied=[],
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        notes=notes,
    )
    return state
