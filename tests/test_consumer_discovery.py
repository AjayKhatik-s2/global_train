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
