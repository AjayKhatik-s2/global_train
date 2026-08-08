"""Consumer discovery must be bounded and stable.

Two production bugs are pinned here, both seen on the first `--auto` run:

1. NO RECENCY WINDOW.  `list_candidate_videos` returned every trimmed clip the
   extractor had ever produced (~17,600) and the scheduler opened a batch for
   each -- it would have tried to inspect the whole archive: weeks of CPU, a full
   disk, and thousands of emails and dashboard posts.

2. ETAG THRASH.  Both `..._train.mp4` and `..._train_incomplete.mp4` exist for the
   same train, so both classified to the same (camera, timestamp) and each poll
   overwrote the other:
       [ATTACH] .../RIGHT_UP ETag changed A -> B: rebuild that camera
       [ATTACH] .../RIGHT_UP ETag changed B -> A: rebuild that camera
   ...forever, re-triggering a camera rebuild every tick.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import constants as C
from orchestrator import train_batch_manager as TBM


def _ago(minutes):
    return datetime.now(timezone.utc) - timedelta(minutes=minutes)


class FakeS3:
    """list_objects_v2 over a fixed object set, prefix-filtered."""

    def __init__(self, objs):      # objs: (key, minutes_ago, etag)
        self.objs = objs

    def list_objects_v2(self, **kw):
        pfx = kw.get("Prefix", "")
        return {"Contents": [{"Key": k, "LastModified": _ago(m), "ETag": f'"{e}"'}
                             for k, m, e in self.objs if k.startswith(pfx)],
                "IsTruncated": False}


RU = "camera_CCTV_HZBN_DHN_2_RIGHT_UP"
LU = "camera_CCTV_HZBN_DHN_1_LEFT_UP"


@pytest.fixture(autouse=True)
def _prefixes(monkeypatch):
    monkeypatch.setattr(C, "S3_INPUT_PREFIXES", [RU, LU])
    monkeypatch.setattr(C, "S3_INPUT_BUCKET", "biro-wagon-pre-processed-video-copy")


# --- bug 1: recency window ---------------------------------------------------

def test_default_consumer_lookback_is_60_minutes(monkeypatch):
    monkeypatch.delenv("WAGONEYE_CONSUMER_LOOKBACK_MINUTES", raising=False)
    assert TBM.consumer_lookback_minutes() == 60.0


def test_archive_is_not_requeued(monkeypatch):
    """The exact failure: months of history must not become candidates."""
    monkeypatch.delenv("WAGONEYE_CONSUMER_LOOKBACK_MINUTES", raising=False)
    s3 = FakeS3([
        (f"{RU}/{RU}_20260803_120000_train.mp4", 5, "fresh"),
        (f"{RU}/{RU}_20260414_144045_train.mp4", 60 * 24 * 111, "april"),
        (f"{LU}/{LU}_20260719_081514_train.mp4", 60 * 24 * 15, "july"),
    ])
    got = TBM.list_candidate_videos(s3)
    assert [c.etag for c in got] == ["fresh"]


def test_window_is_configurable(monkeypatch):
    monkeypatch.setenv("WAGONEYE_CONSUMER_LOOKBACK_MINUTES", "2880")   # 48h
    s3 = FakeS3([
        (f"{RU}/{RU}_20260803_120000_train.mp4", 10, "a"),
        (f"{RU}/{RU}_20260801_120000_train.mp4", 60 * 40, "b"),        # 40h
        (f"{RU}/{RU}_20260414_144045_train.mp4", 60 * 24 * 111, "old"),
    ])
    assert sorted(c.etag for c in TBM.list_candidate_videos(s3)) == ["a", "b"]


def test_zero_disables_the_window(monkeypatch):
    monkeypatch.setenv("WAGONEYE_CONSUMER_LOOKBACK_MINUTES", "0")
    s3 = FakeS3([(f"{RU}/{RU}_20260414_144045_train.mp4", 60 * 24 * 111, "old")])
    assert len(TBM.list_candidate_videos(s3)) == 1


def test_bad_value_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("WAGONEYE_CONSUMER_LOOKBACK_MINUTES", "nonsense")
    assert TBM.consumer_lookback_minutes() == 60.0


def test_window_must_exceed_the_final_camera_deadline(monkeypatch):
    """A late camera has to still be discoverable when it lands."""
    from core import config as CFG
    monkeypatch.delenv("WAGONEYE_CONSUMER_LOOKBACK_MINUTES", raising=False)
    assert TBM.consumer_lookback_minutes() > CFG.FINAL_CAMERA_WAIT_MINUTES


# --- bug 2: one candidate per (camera, timestamp) ----------------------------

def test_complete_clip_beats_incomplete(monkeypatch):
    monkeypatch.delenv("WAGONEYE_CONSUMER_LOOKBACK_MINUTES", raising=False)
    ts = "20260803_120000"
    s3 = FakeS3([
        (f"{LU}/{LU}_{ts}_train_incomplete.mp4", 6, "incomplete"),
        (f"{LU}/{LU}_{ts}_train.mp4", 5, "complete"),
    ])
    got = TBM.list_candidate_videos(s3)
    assert len(got) == 1
    assert got[0].etag == "complete"


def test_complete_wins_even_if_older(monkeypatch):
    """Completeness outranks upload time -- an incomplete re-upload must not win."""
    monkeypatch.delenv("WAGONEYE_CONSUMER_LOOKBACK_MINUTES", raising=False)
    ts = "20260803_120000"
    s3 = FakeS3([
        (f"{LU}/{LU}_{ts}_train.mp4", 50, "complete"),
        (f"{LU}/{LU}_{ts}_train_incomplete.mp4", 1, "incomplete"),
    ])
    assert TBM.list_candidate_videos(s3)[0].etag == "complete"


def test_duplicate_slot_resolves_to_exactly_one(monkeypatch):
    """No slot may yield two candidates -- that is what caused the ETag thrash."""
    monkeypatch.delenv("WAGONEYE_CONSUMER_LOOKBACK_MINUTES", raising=False)
    ts = "20260803_120000"
    s3 = FakeS3([
        (f"{RU}/{RU}_{ts}_train.mp4", 9, "older"),
        (f"{RU}/{RU}_{ts}_train_v2.mp4", 3, "newer"),
    ])
    got = TBM.list_candidate_videos(s3)
    assert len(got) == 1
    assert got[0].etag == "newer"          # newest upload wins


def test_discovery_is_stable_across_polls(monkeypatch):
    """Repeated polls must return an IDENTICAL candidate -- no A->B->A flapping."""
    monkeypatch.delenv("WAGONEYE_CONSUMER_LOOKBACK_MINUTES", raising=False)
    ts = "20260803_120000"
    s3 = FakeS3([
        (f"{RU}/{RU}_{ts}_train_incomplete.mp4", 6, "incomplete"),
        (f"{RU}/{RU}_{ts}_train.mp4", 5, "complete"),
    ])
    seen = {tuple((c.camera_id, c.s3_key, c.etag) for c in
                  TBM.list_candidate_videos(s3)) for _ in range(5)}
    assert len(seen) == 1


def test_idle_polls_log_the_skip_once(monkeypatch, caplog):
    """The idle consumer repeated one identical DISCOVERY line every 60s."""
    import logging
    monkeypatch.setenv("WAGONEYE_CONSUMER_LOOKBACK_MINUTES", "10")
    TBM._LAST_DISCOVERY_SKIP[0] = None
    s3 = FakeS3([(f"{RU}/{RU}_20260803_120000_train.mp4", 999, "stale")])
    with caplog.at_level(logging.INFO, logger="wagon_eye.batch_manager"):
        for _ in range(5):
            TBM.list_candidate_videos(s3)
    skips = [r for r in caplog.records if "older than the window" in r.message]
    assert len(skips) == 1
    # a changed count is still reported
    caplog.clear()
    s3b = FakeS3([(f"{RU}/{RU}_20260803_120000_train.mp4", 999, "stale"),
                  (f"{RU}/{RU}_20260803_130000_train.mp4", 998, "stale2")])
    with caplog.at_level(logging.INFO, logger="wagon_eye.batch_manager"):
        TBM.list_candidate_videos(s3b)
    assert [r for r in caplog.records if "older than the window" in r.message]


def test_different_cameras_and_trains_are_kept_separate(monkeypatch):
    monkeypatch.delenv("WAGONEYE_CONSUMER_LOOKBACK_MINUTES", raising=False)
    s3 = FakeS3([
        (f"{RU}/{RU}_20260803_120000_train.mp4", 5, "ru-a"),
        (f"{LU}/{LU}_20260803_120000_train.mp4", 5, "lu-a"),
        (f"{RU}/{RU}_20260803_123000_train.mp4", 4, "ru-b"),
    ])
    got = TBM.list_candidate_videos(s3)
    assert len(got) == 3
    assert {(c.camera_id, c.train_timestamp) for c in got} == {
        ("RIGHT_UP", "20260803_120000"),
        ("LEFT_UP", "20260803_120000"),
        ("RIGHT_UP", "20260803_123000"),
    }


def test_output_order_is_deterministic(monkeypatch):
    monkeypatch.delenv("WAGONEYE_CONSUMER_LOOKBACK_MINUTES", raising=False)
    s3 = FakeS3([
        (f"{RU}/{RU}_20260803_123000_train.mp4", 4, "b"),
        (f"{LU}/{LU}_20260803_120000_train.mp4", 5, "a"),
    ])
    a = [c.s3_key for c in TBM.list_candidate_videos(s3)]
    b = [c.s3_key for c in TBM.list_candidate_videos(s3)]
    assert a == b == sorted(a)


def test_incomplete_detection():
    assert TBM._is_incomplete(f"{LU}/x_20260803_120000_train_incomplete.mp4")
    assert not TBM._is_incomplete(f"{LU}/x_20260803_120000_train.mp4")


# --- active-manifest resume must also be bounded by train age ----------------

def test_stale_manifest_window_follows_the_consumer_window(monkeypatch):
    from orchestrator import batch_manifest as BM
    monkeypatch.delenv("WAGONEYE_STALE_MANIFEST_MINUTES", raising=False)
    monkeypatch.delenv("WAGONEYE_CONSUMER_LOOKBACK_MINUTES", raising=False)
    assert BM.stale_manifest_minutes() == TBM.consumer_lookback_minutes()


def test_stale_manifest_window_override(monkeypatch):
    from orchestrator import batch_manifest as BM
    monkeypatch.setenv("WAGONEYE_STALE_MANIFEST_MINUTES", "180")
    assert BM.stale_manifest_minutes() == 180.0
    monkeypatch.setenv("WAGONEYE_STALE_MANIFEST_MINUTES", "nonsense")
    assert BM.stale_manifest_minutes() == TBM.consumer_lookback_minutes()


def test_old_manifests_are_not_resumed(monkeypatch):
    """The incident: 423 stale manifests were reloaded as active and began
    sealing months-old trains."""
    from orchestrator import batch_manifest as BM
    monkeypatch.delenv("WAGONEYE_STALE_MANIFEST_MINUTES", raising=False)

    now_key = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    keys = ["20260414_144045", "20260719_134555", now_key]
    fetched = []

    class S3:
        def list_objects_v2(self, **kw):
            return {"CommonPrefixes": [{"Prefix": f"train_batch/{k}/"} for k in keys],
                    "IsTruncated": False}

    def fake_load(s3, key, bucket=None):
        fetched.append(key)
        m = BM.BatchManifest.new(batch_key=key, train_timestamp=key)
        return m

    monkeypatch.setattr(BM, "load_s3", fake_load)
    got = BM.list_active_manifests(S3(), processed_batches={})

    assert [m.batch_key for m in got] == [now_key]
    # and the old ones were skipped WITHOUT a GetObject
    assert fetched == [now_key]


def test_terminal_batches_still_skipped(monkeypatch):
    from orchestrator import batch_manifest as BM
    now_key = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    class S3:
        def list_objects_v2(self, **kw):
            return {"CommonPrefixes": [{"Prefix": f"train_batch/{now_key}/"}],
                    "IsTruncated": False}

    monkeypatch.setattr(BM, "load_s3",
                        lambda *a, **k: pytest.fail("must not fetch a terminal batch"))
    assert BM.list_active_manifests(S3(), processed_batches={now_key: "completed"}) == []


def test_zero_disables_the_resume_bound(monkeypatch):
    from orchestrator import batch_manifest as BM

    class S3:
        def list_objects_v2(self, **kw):
            return {"CommonPrefixes": [{"Prefix": "train_batch/20260414_144045/"}],
                    "IsTruncated": False}

    monkeypatch.setattr(BM, "load_s3", lambda s3, key, bucket=None:
                        BM.BatchManifest.new(batch_key=key, train_timestamp=key))
    got = BM.list_active_manifests(S3(), processed_batches={},
                                   stale_after_minutes=0)
    assert [m.batch_key for m in got] == ["20260414_144045"]


def test_unparseable_batch_key_is_not_dropped(monkeypatch):
    """Never silently discard a manifest we cannot date."""
    from orchestrator import batch_manifest as BM

    class S3:
        def list_objects_v2(self, **kw):
            return {"CommonPrefixes": [{"Prefix": "train_batch/weird-key/"}],
                    "IsTruncated": False}

    monkeypatch.setattr(BM, "load_s3", lambda s3, key, bucket=None:
                        BM.BatchManifest.new(batch_key=key, train_timestamp=key))
    got = BM.list_active_manifests(S3(), processed_batches={})
    assert [m.batch_key for m in got] == ["weird-key"]


# -----------------------------------------------------------------------------
# camera resolution -- the site names TOP rigs RIGHT_TOP/LEFT_TOP
# -----------------------------------------------------------------------------

def test_top_cameras_resolve_from_the_sites_own_naming():
    """`RIGHT_TOP` contains neither `right_up_top` nor `right_up`.

    Matching basenames against the canonical ids silently dropped both top
    cameras at discovery: every batch formed with just the two side cameras.
    """
    from orchestrator.train_batch_manager import _camera_for_key
    f = "camera_CCTV_HZBN_DHN_5_RIGHT_TOP"
    assert _camera_for_key(f"{f}/{f}_20260808_114950_train.mp4") == "RIGHT_UP_TOP"
    g = "camera_CCTV_HZBN_DHN_6_LEFT_TOP"
    assert _camera_for_key(f"{g}/{g}_20260808_115052_train.mp4") == "LEFT_UP_TOP"


def test_side_cameras_still_resolve():
    from orchestrator.train_batch_manager import _camera_for_key
    f = "camera_CCTV_HZBN_DHN_2_RIGHT_UP"
    assert _camera_for_key(f"{f}/{f}_20260808_114950_train.mp4") == "RIGHT_UP"
    g = "camera_CCTV_HZBN_DHN_1_LEFT_UP"
    assert _camera_for_key(f"{g}/{g}_20260808_114954_train.mp4") == "LEFT_UP"


def test_a_top_name_is_never_claimed_by_the_shorter_side_token():
    """`right_up` must not steal a `right_up_top` filename."""
    from orchestrator.train_batch_manager import _camera_for_key
    assert _camera_for_key("x/RIGHT_UP_TOP_20260808_120000.mp4") == "RIGHT_UP_TOP"
    assert _camera_for_key("x/LEFT_UP_TOP_20260808_120000.mp4") == "LEFT_UP_TOP"


def test_folder_wins_over_a_mangled_filename():
    """Real uploads drop the `camera_` prefix and even carry leading spaces."""
    from orchestrator.train_batch_manager import _camera_for_key
    assert _camera_for_key(
        "camera_CCTV_HZBN_DHN_2_RIGHT_UP/  CCTV_HZBN_DHN_2_RIGHT_UP_20260808_115032.mp4"
    ) == "RIGHT_UP"
    assert _camera_for_key(
        "camera_CCTV_HZBN_DHN_6_LEFT_TOP/CCTV_HZBN_DHN_6_LEFT_TOP_20260808_115052.mp4"
    ) == "LEFT_UP_TOP"


def test_non_video_and_unknown_keys_are_still_rejected():
    from orchestrator.train_batch_manager import _camera_for_key
    assert _camera_for_key("camera_CCTV_HZBN_DHN_2_RIGHT_UP/notes.txt") is None
    assert _camera_for_key("some/other/clip_20260808_120000.mp4") is None


def test_local_scan_accepts_the_sites_top_naming(tmp_path):
    """No hand-renaming needed before a --local-only run."""
    from core.batch import scan_local_video_dir
    for n in ("camera_CCTV_HZBN_DHN_2_RIGHT_UP_20260808_120000.mp4",
              "camera_CCTV_HZBN_DHN_1_LEFT_UP_20260808_120000.mp4",
              "camera_CCTV_HZBN_DHN_5_RIGHT_TOP_20260808_120000.mp4",
              "camera_CCTV_HZBN_DHN_6_LEFT_TOP_20260808_120000.mp4"):
        (tmp_path / n).write_bytes(b"x")
    found = scan_local_video_dir(str(tmp_path))
    assert set(found) == {"RIGHT_UP", "LEFT_UP", "RIGHT_UP_TOP", "LEFT_UP_TOP"}


def test_local_scan_still_accepts_canonical_names(tmp_path):
    from core.batch import scan_local_video_dir
    for n in ("RIGHT_UP.mp4", "LEFT_UP.mp4", "RIGHT_UP_TOP.mp4", "LEFT_UP_TOP.mp4"):
        (tmp_path / n).write_bytes(b"x")
    found = scan_local_video_dir(str(tmp_path))
    assert set(found) == {"RIGHT_UP", "LEFT_UP", "RIGHT_UP_TOP", "LEFT_UP_TOP"}
