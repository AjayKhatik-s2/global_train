"""Stage-1 detector batching must be output-identical to the single-frame path.

Stage 1 produces the SEALED CANONICAL wagon count / GW ids / boundaries, so the
batching opt-in is only safe if it changes nothing but speed.  These tests drive
the real `GapTracker.process_video` loop over a synthetic video with a FAKE model,
so the comparison is exact (no BLAS jitter) and isolates the loop logic itself:
frame ordering, the trim window, the frame_limit, and the per-frame tracking.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "wagon_count"))

cv2 = pytest.importorskip("cv2")

import tracker_engine as TE  # noqa: E402


# ---------------------------------------------------------------------------
# a fake YOLO whose output depends only on the frame's OWN content, so a
# single-frame call and a batched call must agree exactly
# ---------------------------------------------------------------------------

class _FakeBoxes:
    def __init__(self, rows):
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    @property
    def xyxy(self):
        return _Arr(np.array([r[0] for r in self._rows], dtype=np.float32).reshape(-1, 4))

    @property
    def conf(self):
        return _Arr(np.array([r[1] for r in self._rows], dtype=np.float32))

    @property
    def cls(self):
        return _Arr(np.array([r[2] for r in self._rows], dtype=np.float32))


class _Arr:
    def __init__(self, a):
        self._a = a

    def cpu(self):
        return self

    def numpy(self):
        return self._a


class _FakeResult:
    def __init__(self, rows):
        self.boxes = _FakeBoxes(rows)


class FakeModel:
    """Emits one gap box whose x follows the frame's encoded marker value.

    Accepts either a single frame or a LIST of frames, mirroring ultralytics.
    Records how many calls were made so batching can be proven to have happened.
    """
    names = {0: "gap"}

    def __init__(self):
        self.calls = 0
        self.frames_seen = 0

    def _one(self, frame):
        # the synthetic video encodes the frame index in pixel [0,0]
        marker = int(frame[0, 0, 0])
        cx = 100.0 + marker * 3.0
        return _FakeResult([([cx - 20, 40, cx + 20, 200], 0.9, 0)])

    def __call__(self, frame, verbose=False, device=None):
        self.calls += 1
        if isinstance(frame, list):
            self.frames_seen += len(frame)
            return [self._one(f) for f in frame]
        self.frames_seen += 1
        return [self._one(frame)]


def _make_video(path, n_frames=60, w=320, h=240):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (w, h))
    for i in range(n_frames):
        f = np.full((h, w, 3), 30, np.uint8)
        f[0, 0] = (i % 256, 0, 0)          # encode the frame index
        vw.write(f)
    vw.release()
    return path


def _tracker(monkeypatch, model):
    """Build a GapTracker without touching a real .pt or ultralytics."""
    monkeypatch.setattr(TE.os.path, "exists", lambda p: True)
    monkeypatch.setattr(TE, "_resolve_device", lambda force=None: "cpu")

    t = TE.GapTracker.__new__(TE.GapTracker)
    t.camera_id = "RIGHT_UP"
    t.model = model
    t.device = "cpu"
    t.confidence = 0.4
    t.min_height_ratio = 0.05
    t.match_distance_px = 80.0
    t.min_hits = 3
    t.max_miss = 30
    t.verbose = False
    t.class_names = dict(model.names)
    t._is_single_class_model = True
    # process_video normally zeroes these; set them so the detector helpers can
    # also be called directly.
    for attr in ("_diag_total_yolo_boxes", "_diag_after_class", "_diag_after_conf",
                 "_diag_kept", "_diag_after_nms", "_diag_tracks_created",
                 "_diag_tracks_confirmed", "_diag_tracks_merged"):
        setattr(t, attr, 0)
    return t


def _run(video, monkeypatch, batch, **kw):
    monkeypatch.setenv("WAGONEYE_STAGE1_INFER_BATCH", str(batch))
    monkeypatch.setattr(TE, "STAGE1_INFER_BATCH", batch)
    model = FakeModel()
    tr = _tracker(monkeypatch, model)
    tracks = tr.process_video(video, **kw)
    return tracks, model


def _signature(tracks):
    """Everything Stage 1 seals: ordering, frames, spans, geometry."""
    return [
        (g.track_id, g.start_frame, g.end_frame, round(g.center_time, 6),
         g.hit_count, tuple(g.hit_frames),
         tuple(tuple(round(v, 6) for v in b) for b in g.bbox_history))
        for g in tracks.gaps
    ]


# ---------------------------------------------------------------------------

def test_default_is_unbatched():
    """Production must keep the original one-frame-per-call path by default."""
    assert TE._stage1_infer_batch() == 1


def test_env_parsing(monkeypatch):
    monkeypatch.setenv("WAGONEYE_STAGE1_INFER_BATCH", "16")
    assert TE._stage1_infer_batch() == 16
    monkeypatch.setenv("WAGONEYE_STAGE1_INFER_BATCH", "0")
    assert TE._stage1_infer_batch() == 1        # clamped
    monkeypatch.setenv("WAGONEYE_STAGE1_INFER_BATCH", "nonsense")
    assert TE._stage1_infer_batch() == 1        # never crashes the stage


def test_batched_output_is_identical(tmp_path, monkeypatch):
    v = _make_video(str(tmp_path / "v.mp4"))
    base, m1 = _run(v, monkeypatch, 1)
    batched, m8 = _run(v, monkeypatch, 8)

    assert _signature(base) == _signature(batched)
    assert base.total_frames == batched.total_frames
    assert base.fps == batched.fps
    assert len(base.gaps) == len(batched.gaps)


def test_batching_actually_batches(tmp_path, monkeypatch):
    v = _make_video(str(tmp_path / "v.mp4"), n_frames=60)
    _, m1 = _run(v, monkeypatch, 1)
    _, m8 = _run(v, monkeypatch, 8)
    # same frames inferred, far fewer model calls
    assert m1.frames_seen == m8.frames_seen
    assert m8.calls < m1.calls
    assert m8.calls <= m1.calls / 4


def test_identical_under_frame_trim(tmp_path, monkeypatch):
    """The trim window must land on the same ORIGINAL frame numbers either way."""
    v = _make_video(str(tmp_path / "v.mp4"), n_frames=80)
    monkeypatch.setenv("WAGONEYE_STAGE1_FRAME_TRIM_PERCENT", "10")
    base, _ = _run(v, monkeypatch, 1)
    monkeypatch.setenv("WAGONEYE_STAGE1_FRAME_TRIM_PERCENT", "10")
    batched, _ = _run(v, monkeypatch, 8)
    assert _signature(base) == _signature(batched)
    # and the trim really applied (no gap starts at frame 0)
    if base.gaps:
        assert min(g.start_frame for g in base.gaps) > 0


def test_identical_under_frame_limit(tmp_path, monkeypatch):
    v = _make_video(str(tmp_path / "v.mp4"), n_frames=80)
    base, _ = _run(v, monkeypatch, 1, frame_limit=25)
    batched, _ = _run(v, monkeypatch, 8, frame_limit=25)
    assert _signature(base) == _signature(batched)
    for g in base.gaps:
        assert g.end_frame < 25


def test_identical_when_batch_exceeds_frame_count(tmp_path, monkeypatch):
    v = _make_video(str(tmp_path / "v.mp4"), n_frames=7)
    base, _ = _run(v, monkeypatch, 1)
    batched, _ = _run(v, monkeypatch, 64)
    assert _signature(base) == _signature(batched)


def test_raw_detections_keyed_by_original_frame(tmp_path, monkeypatch):
    v = _make_video(str(tmp_path / "v.mp4"), n_frames=40)
    base, _ = _run(v, monkeypatch, 1, keep_raw_detections=True)
    batched, _ = _run(v, monkeypatch, 8, keep_raw_detections=True)
    assert set(base.raw_frame_detections) == set(batched.raw_frame_detections)
    for k in base.raw_frame_detections:
        assert base.raw_frame_detections[k] == batched.raw_frame_detections[k]


def test_single_frame_and_batched_postprocess_share_one_code_path(monkeypatch):
    """_detect_gaps and _detect_gaps_batched must not diverge in filtering."""
    model = FakeModel()
    tr = _tracker(monkeypatch, model)
    frames = []
    for i in (1, 2, 3):
        f = np.full((240, 320, 3), 30, np.uint8)
        f[0, 0] = (i, 0, 0)
        frames.append(f)
    one_by_one = [tr._detect_gaps(f, 240) for f in frames]
    batched = tr._detect_gaps_batched(frames, 240)
    assert one_by_one == batched
