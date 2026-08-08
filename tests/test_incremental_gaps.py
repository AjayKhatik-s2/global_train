"""Incremental per-camera gap extraction (AUTO/S3 only).

Covers the four properties the change has to hold:
    1. a cached camera rebuilds to an IDENTICAL LocalCameraTracks
    2. the same S3 object is never gap-extracted twice, and a replaced object IS
    3. a still-uploading object is not claimed
    4. two workers cannot claim the same camera

No model inference anywhere -- these are pure serialization/state tests.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
# wagon_count is standalone: its modules import each other by bare name.
_WC = os.path.join(_REPO, "wagon_count")
if _WC not in sys.path:
    sys.path.insert(0, _WC)

from core import constants as C                                    # noqa: E402
from orchestrator import gap_extraction as GX                      # noqa: E402


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def _tracks(camera=C.CAMERA_RIGHT_UP, n_gaps=3):
    from global_train_state import GapEvent, LocalCameraTracks, _MasterClassification
    gaps = []
    for i in range(n_gaps):
        gaps.append(GapEvent(
            track_id=i + 1, camera_id=camera,
            start_frame=100 * i, end_frame=100 * i + 40,
            confidence=0.8123456, hit_count=33,
            center_x_trajectory=[10.5 + i, 20.25 + i, 31.125 + i],
            fps=25.0, temporal_consistency_score=0.777777,
            hit_frames=[100 * i, 100 * i + 20, 100 * i + 40],
            bbox_history=[[1.5, 2.5, 3.5, 4.5], [5.5, 6.5, 7.5, 8.5],
                          [9.5, 10.5, 11.5, 12.5]],
            class_label="gap",
        ))
    return LocalCameraTracks(
        camera_id=camera, video_path="/tmp/x.mp4", fps=25.0, total_frames=900,
        width=1920, height=1080, gaps=gaps,
        classifications=[_MasterClassification(0, 0, 99, "ENGINE", 0.912345)],
        raw_frame_detections={7: [{"bbox": [1.0, 2.0, 3.0, 4.0], "conf": 0.5}]},
    )


class _Slot:
    def __init__(self, camera_id, bucket="b", s3_key="k/v.mp4", etag="e1",
                 file_size=1234, last_modified="2026-08-04T00:00:00+00:00"):
        self.camera_id = camera_id
        self.bucket = bucket
        self.s3_key = s3_key
        self.etag = etag
        self.file_size = file_size
        self.last_modified = last_modified
        self.local_path = None


class _Manifest:
    def __init__(self, slots):
        self.cameras = {s.camera_id: s for s in slots}
        self.global_state_version = None


# -----------------------------------------------------------------------------
# 1. lossless round-trip -- a cached camera must be indistinguishable
# -----------------------------------------------------------------------------

def test_tracks_round_trip_is_lossless():
    from global_train_state import LocalCameraTracks
    original = _tracks()
    restored = LocalCameraTracks.from_cache_dict(original.to_cache_dict())

    assert restored.camera_id == original.camera_id
    assert restored.fps == original.fps
    assert restored.total_frames == original.total_frames
    assert restored.width == original.width and restored.height == original.height
    assert restored.local_wagon_count == original.local_wagon_count
    assert len(restored.gaps) == len(original.gaps)
    for a, b in zip(original.gaps, restored.gaps):
        assert b.track_id == a.track_id
        assert b.start_frame == a.start_frame and b.end_frame == a.end_frame
        assert b.confidence == a.confidence            # NOT rounded
        assert b.hit_count == a.hit_count
        assert b.center_x_trajectory == a.center_x_trajectory
        assert b.fps == a.fps
        assert b.temporal_consistency_score == a.temporal_consistency_score
        assert b.hit_frames == a.hit_frames
        assert b.bbox_history == a.bbox_history
        assert b.class_label == a.class_label


def test_round_trip_preserves_what_the_public_json_drops():
    """`to_dict` is lossy on purpose; the cache pair must not be."""
    from global_train_state import LocalCameraTracks
    original = _tracks()
    public = original.to_dict()["gaps"][0]
    assert "center_x_trajectory" not in public      # dropped by the report shape
    assert public["confidence"] == round(original.gaps[0].confidence, 4)

    restored = LocalCameraTracks.from_cache_dict(original.to_cache_dict())
    assert restored.gaps[0].center_x_trajectory == original.gaps[0].center_x_trajectory
    assert restored.gaps[0].confidence == original.gaps[0].confidence


def test_raw_frame_detections_survive_so_the_overlay_is_unchanged():
    """video_segmenter draws these; losing them would change the debug overlay."""
    from global_train_state import LocalCameraTracks
    original = _tracks()
    restored = LocalCameraTracks.from_cache_dict(original.to_cache_dict())
    assert restored.raw_frame_detections == original.raw_frame_detections
    assert list(restored.raw_frame_detections)[0] == 7        # int key, not "7"


def test_classifications_survive_so_step2_is_not_rerun():
    from global_train_state import LocalCameraTracks
    original = _tracks()
    restored = LocalCameraTracks.from_cache_dict(original.to_cache_dict())
    assert len(restored.classifications) == 1
    assert restored.classifications[0].label == "ENGINE"
    assert restored.classifications[0].confidence == 0.912345   # unrounded


def test_cache_write_then_load_returns_equivalent_tracks(tmp_path):
    import gap_cache as gc
    d = str(tmp_path)
    ident = gc.make_identity(bucket="b", s3_key="k/v.mp4", etag="e1", file_size=9)
    gc.write_result(d, C.CAMERA_RIGHT_UP, _tracks(), identity=ident)
    tracks, tops = gc.load_tracks(d, C.CAMERA_RIGHT_UP, ident)
    assert tracks is not None
    assert len(tracks.gaps) == 3
    assert tops == []


def test_top_local_classifications_are_stored_unscaled(tmp_path):
    """The master-frame rescale needs master fps, unknown at extraction time."""
    import gap_cache as gc
    from global_train_state import _MasterClassification
    d = str(tmp_path)
    local = [_MasterClassification(0, 0, 50, "WAGON", 0.9)]
    gc.write_result(d, C.CAMERA_RIGHT_UP_TOP, _tracks(C.CAMERA_RIGHT_UP_TOP),
                    identity=None, top_local_classifications=local)
    _, tops = gc.load_tracks(d, C.CAMERA_RIGHT_UP_TOP)
    assert [(t.start_frame, t.end_frame) for t in tops] == [(0, 50)]


def test_rescale_matches_the_original_expression():
    """The extracted rescale half must be arithmetically identical."""
    from run_global_count import _rescale_top_to_master
    from global_train_state import _MasterClassification
    local = [_MasterClassification(0, 100, 200, "WAGON", 0.9)]
    out = _rescale_top_to_master(local, top_fps=25.0, master_fps=50.0)
    assert (out[0].start_frame, out[0].end_frame) == (200, 400)
    # top_fps of 0 falls back to master fps -> scale 1.0, as before
    same = _rescale_top_to_master(local, top_fps=0.0, master_fps=50.0)
    assert (same[0].start_frame, same[0].end_frame) == (100, 200)


# -----------------------------------------------------------------------------
# 2. never process the same object twice; a replacement IS a new input
# -----------------------------------------------------------------------------

def test_identity_matches_on_etag():
    a = GX.slot_identity(_Slot(C.CAMERA_RIGHT_UP, etag="abc"))
    b = GX.slot_identity(_Slot(C.CAMERA_RIGHT_UP, etag="abc"))
    assert GX.identity_matches(a, b)


def test_quoted_and_bare_etag_are_the_same_object():
    a = GX.slot_identity(_Slot(C.CAMERA_RIGHT_UP, etag='"abc"'))
    b = GX.slot_identity(_Slot(C.CAMERA_RIGHT_UP, etag="abc"))
    assert GX.identity_matches(a, b)


def test_a_changed_etag_is_a_different_input():
    a = GX.slot_identity(_Slot(C.CAMERA_RIGHT_UP, etag="abc"))
    b = GX.slot_identity(_Slot(C.CAMERA_RIGHT_UP, etag="def"))
    assert not GX.identity_matches(a, b)


def test_a_different_key_is_never_the_same_object():
    a = GX.slot_identity(_Slot(C.CAMERA_RIGHT_UP, s3_key="k/a.mp4", etag=None))
    b = GX.slot_identity(_Slot(C.CAMERA_RIGHT_UP, s3_key="k/b.mp4", etag=None))
    assert not GX.identity_matches(a, b)


def test_without_etags_size_and_mtime_decide():
    a = GX.slot_identity(_Slot(C.CAMERA_RIGHT_UP, etag=None, file_size=10))
    b = GX.slot_identity(_Slot(C.CAMERA_RIGHT_UP, etag=None, file_size=10))
    c = GX.slot_identity(_Slot(C.CAMERA_RIGHT_UP, etag=None, file_size=11))
    assert GX.identity_matches(a, b)
    assert not GX.identity_matches(a, c)


def test_completed_requires_both_state_and_result(tmp_path):
    """A state file without a result must NOT count as done."""
    d = str(tmp_path)
    ident = GX.slot_identity(_Slot(C.CAMERA_RIGHT_UP))
    GX.write_gap_state(d, C.CAMERA_RIGHT_UP, "completed", ident, gap_count=3)
    assert not GX.is_completed(d, C.CAMERA_RIGHT_UP, ident)   # result missing
    open(os.path.join(d, f"{C.CAMERA_RIGHT_UP}.result.json"), "w").write("{}")
    assert GX.is_completed(d, C.CAMERA_RIGHT_UP, ident)


def test_completed_for_one_object_is_not_completed_for_another(tmp_path):
    d = str(tmp_path)
    old = GX.slot_identity(_Slot(C.CAMERA_RIGHT_UP, etag="old"))
    new = GX.slot_identity(_Slot(C.CAMERA_RIGHT_UP, etag="new"))
    GX.write_gap_state(d, C.CAMERA_RIGHT_UP, "completed", old, gap_count=3)
    open(os.path.join(d, f"{C.CAMERA_RIGHT_UP}.result.json"), "w").write("{}")
    assert GX.is_completed(d, C.CAMERA_RIGHT_UP, old)
    assert not GX.is_completed(d, C.CAMERA_RIGHT_UP, new)


def test_a_failed_camera_is_not_treated_as_completed(tmp_path):
    d = str(tmp_path)
    ident = GX.slot_identity(_Slot(C.CAMERA_RIGHT_UP))
    GX.write_gap_state(d, C.CAMERA_RIGHT_UP, "failed", ident, error="boom")
    assert not GX.is_completed(d, C.CAMERA_RIGHT_UP, ident)


def test_result_is_rejected_when_the_identity_does_not_match(tmp_path):
    import gap_cache as gc
    d = str(tmp_path)
    a = gc.make_identity(bucket="b", s3_key="k/v.mp4", etag="e1")
    b = gc.make_identity(bucket="b", s3_key="k/v.mp4", etag="e2")
    gc.write_result(d, C.CAMERA_RIGHT_UP, _tracks(), identity=a)
    assert gc.read_result(d, C.CAMERA_RIGHT_UP, a) is not None
    assert gc.read_result(d, C.CAMERA_RIGHT_UP, b) is None


def test_corrupt_cache_reads_as_absent_not_fatal(tmp_path):
    import gap_cache as gc
    d = str(tmp_path)
    open(gc.result_path(d, C.CAMERA_RIGHT_UP), "w").write("{ truncated")
    assert gc.read_result(d, C.CAMERA_RIGHT_UP) is None
    tracks, tops = gc.load_tracks(d, C.CAMERA_RIGHT_UP)
    assert tracks is None and tops == []


# -----------------------------------------------------------------------------
# 3. partially-uploaded objects are never claimed
# -----------------------------------------------------------------------------

def test_first_sighting_is_never_stable(tmp_path, monkeypatch):
    monkeypatch.setenv("WAGONEYE_GAP_STABILITY_SECONDS", "30")
    d = str(tmp_path)
    ident = GX.slot_identity(_Slot(C.CAMERA_RIGHT_UP))
    assert GX.check_stable(d, C.CAMERA_RIGHT_UP, ident) is False


def test_unchanged_object_becomes_stable_after_the_window(tmp_path, monkeypatch):
    monkeypatch.setenv("WAGONEYE_GAP_STABILITY_SECONDS", "0")
    d = str(tmp_path)
    ident = GX.slot_identity(_Slot(C.CAMERA_RIGHT_UP))
    GX.check_stable(d, C.CAMERA_RIGHT_UP, ident)          # record first sighting
    assert GX.check_stable(d, C.CAMERA_RIGHT_UP, ident) is True


def test_a_changing_object_restarts_the_clock(tmp_path, monkeypatch):
    """Still uploading: the ETag moves between polls, so it is never claimed."""
    monkeypatch.setenv("WAGONEYE_GAP_STABILITY_SECONDS", "0")
    d = str(tmp_path)
    first = GX.slot_identity(_Slot(C.CAMERA_RIGHT_UP, etag="e1", file_size=100))
    grown = GX.slot_identity(_Slot(C.CAMERA_RIGHT_UP, etag="e2", file_size=200))
    GX.check_stable(d, C.CAMERA_RIGHT_UP, first)
    # object changed -> not stable on this sighting, clock restarted
    assert GX.check_stable(d, C.CAMERA_RIGHT_UP, grown) is False
    # unchanged since -> now stable
    assert GX.check_stable(d, C.CAMERA_RIGHT_UP, grown) is True


def test_tiny_file_fails_validation(tmp_path, monkeypatch):
    monkeypatch.setenv("WAGONEYE_GAP_MIN_VIDEO_BYTES", "1024")
    p = tmp_path / "truncated.mp4"
    p.write_bytes(b"\x00" * 10)
    assert GX._validate(str(p)) is False


# -----------------------------------------------------------------------------
# 4. concurrent workers
# -----------------------------------------------------------------------------

def test_only_one_worker_can_claim_a_camera(tmp_path):
    d = str(tmp_path)
    assert GX.acquire(d, C.CAMERA_RIGHT_UP) is True
    assert GX.acquire(d, C.CAMERA_RIGHT_UP) is False      # second worker blocked
    GX.release(d, C.CAMERA_RIGHT_UP)
    assert GX.acquire(d, C.CAMERA_RIGHT_UP) is True       # released -> claimable


def test_different_cameras_are_claimed_independently(tmp_path):
    d = str(tmp_path)
    assert GX.acquire(d, C.CAMERA_RIGHT_UP) is True
    assert GX.acquire(d, C.CAMERA_LEFT_UP) is True


def test_a_stale_lock_is_broken(tmp_path, monkeypatch):
    """A killed worker must not block its camera forever."""
    monkeypatch.setenv("WAGONEYE_GAP_LOCK_STALE_SECONDS", "60")
    d = str(tmp_path)
    assert GX.acquire(d, C.CAMERA_RIGHT_UP) is True
    lock = GX._lock_file(d, C.CAMERA_RIGHT_UP)
    os.utime(lock, (0, 0))                                # ancient
    assert GX.acquire(d, C.CAMERA_RIGHT_UP) is True       # broken and reclaimed


def test_release_of_an_unheld_lock_is_harmless(tmp_path):
    GX.release(str(tmp_path), C.CAMERA_RIGHT_UP)          # must not raise


# -----------------------------------------------------------------------------
# readiness reporting -- "all four for Global Train" vs "one to start extraction"
# -----------------------------------------------------------------------------

def test_readiness_reports_each_camera(tmp_path):
    d = str(tmp_path)
    m = _Manifest([_Slot(C.CAMERA_RIGHT_UP), _Slot(C.CAMERA_LEFT_UP)])
    ident = GX.slot_identity(m.cameras[C.CAMERA_RIGHT_UP])
    GX.write_gap_state(d, C.CAMERA_RIGHT_UP, "completed", ident, gap_count=5)
    open(os.path.join(d, f"{C.CAMERA_RIGHT_UP}.result.json"), "w").write("{}")

    ready = GX.readiness(m, d)
    assert ready[C.CAMERA_RIGHT_UP] == "READY"
    assert ready[C.CAMERA_LEFT_UP] == "PENDING"
    assert ready[C.CAMERA_RIGHT_UP_TOP] == "MISSING"
    assert ready[C.CAMERA_LEFT_UP_TOP] == "MISSING"


def test_readiness_surfaces_a_failed_camera(tmp_path):
    d = str(tmp_path)
    m = _Manifest([_Slot(C.CAMERA_RIGHT_UP)])
    GX.write_gap_state(d, C.CAMERA_RIGHT_UP, "failed",
                       GX.slot_identity(m.cameras[C.CAMERA_RIGHT_UP]), error="x")
    assert GX.readiness(m, d)[C.CAMERA_RIGHT_UP] == "FAILED"


def test_one_camera_ready_does_not_require_the_others(tmp_path):
    """The separation the change exists for: extraction never waits for four."""
    d = str(tmp_path)
    m = _Manifest([_Slot(C.CAMERA_RIGHT_UP)])
    ident = GX.slot_identity(m.cameras[C.CAMERA_RIGHT_UP])
    GX.write_gap_state(d, C.CAMERA_RIGHT_UP, "completed", ident, gap_count=7)
    open(os.path.join(d, f"{C.CAMERA_RIGHT_UP}.result.json"), "w").write("{}")
    assert GX.is_completed(d, C.CAMERA_RIGHT_UP, ident)


# -----------------------------------------------------------------------------
# LOCAL mode and the reconstruction contract are untouched
# -----------------------------------------------------------------------------

def test_local_mode_passes_no_gap_cache():
    """`run()` only adds --gap-cache when a caller supplies one; LOCAL never does."""
    import inspect
    from reconstruction import runner as R
    sig = inspect.signature(R.run)
    assert sig.parameters["gap_cache_dir"].default is None


def test_gap_cache_flag_is_omitted_without_a_cache_dir():
    import inspect
    from reconstruction import runner as R
    src = inspect.getsource(R.run)
    assert "if gap_cache_dir:" in src
    assert '"--gap-cache", gap_cache_dir' in src


def test_incremental_extraction_can_be_disabled(monkeypatch):
    """Escape hatch: off => seal runs the unchanged full-inference path."""
    monkeypatch.setenv("WAGONEYE_INCREMENTAL_GAPS", "0")
    assert GX.enabled() is False
    monkeypatch.setenv("WAGONEYE_INCREMENTAL_GAPS", "1")
    assert GX.enabled() is True


def test_camera_only_never_assembles_a_global_train():
    """Per-camera runs must not seal -- that would reseal once per camera."""
    import inspect
    import run_global_count as RGC
    src = inspect.getsource(RGC._run_camera_only)
    assert "assemble_global_train_state" not in src
    assert "global_train_state.json" not in src


def test_camera_only_requires_a_gap_cache_dir():
    import run_global_count as RGC

    class _Args:
        camera_only = C.CAMERA_RIGHT_UP
        gap_cache = None
    assert RGC._run_camera_only(_Args(), verbose=False) == 4


def test_camera_only_reuses_the_existing_detectors():
    """No second gap-detection implementation."""
    import inspect
    import run_global_count as RGC
    src = inspect.getsource(RGC._extract_one_camera)
    assert "_process_side_camera" in src
    assert "_process_top_camera" in src
    assert "GapTracker(" not in src          # never constructs its own tracker


def test_side_gap_model_mapping_is_shared():
    """Full run and per-camera run must select identical weights."""
    import run_global_count as RGC
    assert RGC._SIDE_GAP_MODEL[C.CAMERA_RIGHT_UP] == "right_up_wagon_gap.pt"
    assert RGC._SIDE_GAP_MODEL[C.CAMERA_LEFT_UP] == "left_up_wagon_gap.pt"
    assert RGC._gap_model_for.__module__ == RGC.__name__
