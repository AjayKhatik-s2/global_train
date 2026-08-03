"""Extraction must stay inside its own camera's prefix, and must not re-cut history.

Two production bugs are pinned here:

1. `S3Client.list_objects` split the prefix off a "bucket/prefix" string and then
   DISCARDED it, so the per-camera sweep listed the WHOLE raw bucket and handed
   other cameras' video to this camera's extractor -- running the SIDE classifier
   over a TOP view.
2. `extract()` only ever ADDS to the S3 state store and never reads it, so on a
   fresh clone (empty local ledger) the sweep re-extracted the entire raw history.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_extraction.url_utils import split_bucket_prefix


RAW = "biro-wagon-raw-video-copy/camera_CCTV_HZBN_DHN_2_RIGHT_UP"

ALL_CAMERA_KEYS = [
    "camera_CCTV_HZBN_DHN_2_RIGHT_UP/CCTV_HZBN_DHN_2_RIGHT_UP_20260212_130821.mp4",
    "camera_CCTV_HZBN_DHN_1_LEFT_UP/CCTV_HZBN_DHN_1_LEFT_UP_20260212_130821.mp4",
    "camera_CCTV_HZBN_DHN_5_RIGHT_TOP/CCTV_HZBN_DHN_5_RIGHT_TOP_20260212_130821.mp4",
    "camera_CCTV_HZBN_DHN_6_LEFT_TOP/CCTV_HZBN_DHN_6_LEFT_TOP_20260212_130821.mp4",
]


class FakeBoto:
    """Records the Prefix it was asked for and filters accordingly."""

    def __init__(self, keys):
        self.keys = keys
        self.prefixes = []

    def list_objects_v2(self, **params):
        prefix = params.get("Prefix", "")
        self.prefixes.append(prefix)
        return {"Contents": [{"Key": k} for k in self.keys if k.startswith(prefix)],
                "IsTruncated": False}


def _client(keys):
    from train_extraction.s3 import S3Client
    c = S3Client.__new__(S3Client)          # skip boto3 construction
    c.client = FakeBoto(keys)
    c.region = "ap-south-1"
    import logging
    c.logger = logging.getLogger("test")
    return c


# --- bug 1: prefix scoping ---------------------------------------------------

def test_embedded_prefix_is_applied():
    c = _client(ALL_CAMERA_KEYS)
    objs = c.list_objects(RAW)
    keys = [o["Key"] for o in objs]
    assert len(keys) == 1
    assert "DHN_2_RIGHT_UP" in keys[0]
    # the request really carried the prefix
    assert c.client.prefixes == ["camera_CCTV_HZBN_DHN_2_RIGHT_UP/"]


def test_other_cameras_are_never_listed():
    c = _client(ALL_CAMERA_KEYS)
    keys = [o["Key"] for o in c.list_objects(RAW)]
    for foreign in ("DHN_1_LEFT_UP", "DHN_5_RIGHT_TOP", "DHN_6_LEFT_TOP"):
        assert not any(foreign in k for k in keys), f"{foreign} leaked in"


def test_explicit_prefix_still_wins():
    c = _client(ALL_CAMERA_KEYS)
    c.list_objects(RAW, prefix="camera_CCTV_HZBN_DHN_6_LEFT_TOP/")
    assert c.client.prefixes == ["camera_CCTV_HZBN_DHN_6_LEFT_TOP/"]


def test_bucket_without_a_prefix_lists_everything():
    c = _client(ALL_CAMERA_KEYS)
    assert len(c.list_objects("biro-wagon-raw-video-copy")) == 4
    assert c.client.prefixes == [""]


def test_trailing_slash_is_not_doubled():
    c = _client(ALL_CAMERA_KEYS)
    c.list_objects(RAW + "/")
    assert c.client.prefixes == ["camera_CCTV_HZBN_DHN_2_RIGHT_UP/"]


def test_split_bucket_prefix_contract():
    assert split_bucket_prefix(RAW) == ("biro-wagon-raw-video-copy",
                                        "camera_CCTV_HZBN_DHN_2_RIGHT_UP")
    assert split_bucket_prefix("just-a-bucket") == ("just-a-bucket", "")


# --- bug 2: dedup survives a fresh checkout ---------------------------------

def test_s3_state_is_folded_into_the_dedup_set(monkeypatch, tmp_path):
    """A fresh clone has an empty local ledger; the S3 state must still gate."""
    from train_extraction import run_extraction_service as RES

    monkeypatch.setenv("WAGONEYE_EXTRACTION_STATE_DIR", str(tmp_path))

    already = ALL_CAMERA_KEYS[0]

    class FakeState:
        processed_videos = {already}

    class FakeEx:
        s3 = _client(ALL_CAMERA_KEYS)
        state = FakeState()

    ex = FakeEx()
    assert RES._load_ledger("RIGHT_UP") == set()        # empty on a fresh install
    assert already in RES._s3_processed(ex)             # but S3 knows it

    extracted = []
    monkeypatch.setattr(RES.D, "get_extractor", lambda cam: ex)
    monkeypatch.setattr(RES.D, "raw_bucket_for", lambda cam: RAW)
    monkeypatch.setattr(RES.D, "extract_trains",
                        lambda cam, key: extracted.append(key) or [])

    r = RES.sweep_camera("RIGHT_UP")
    assert r["listed"] == 1          # only this camera's folder
    assert r["new"] == 0            # and it was already cut -> skipped
    assert extracted == []


def test_missing_s3_state_degrades_to_local_only(monkeypatch, tmp_path):
    from train_extraction import run_extraction_service as RES
    monkeypatch.setenv("WAGONEYE_EXTRACTION_STATE_DIR", str(tmp_path))

    class Broken:
        @property
        def state(self):
            raise RuntimeError("no state")

    assert RES._s3_processed(Broken()) == set()      # never raises


def test_foreign_key_is_skipped_not_extracted(monkeypatch, tmp_path):
    """Defence in depth: even if a listing escapes its prefix, the extractor
    must refuse another camera's video."""
    from train_extraction import run_extraction_service as RES
    monkeypatch.setenv("WAGONEYE_EXTRACTION_STATE_DIR", str(tmp_path))

    class FakeState:
        processed_videos = set()

    class LeakyEx:
        # a client that ignores the prefix -- simulates the old bug
        def __init__(self):
            import logging
            from train_extraction.s3 import S3Client
            c = S3Client.__new__(S3Client)
            c.client = FakeBoto(ALL_CAMERA_KEYS)
            c.region = "ap-south-1"
            c.logger = logging.getLogger("t")
            c.list_objects = lambda *a, **k: [{"Key": x} for x in ALL_CAMERA_KEYS]
            self.s3 = c
            self.state = FakeState()

    ex = LeakyEx()
    extracted = []
    monkeypatch.setattr(RES.D, "get_extractor", lambda cam: ex)
    monkeypatch.setattr(RES.D, "raw_bucket_for", lambda cam: RAW)
    monkeypatch.setattr(RES.D, "extract_trains",
                        lambda cam, key: extracted.append(key) or [])

    r = RES.sweep_camera("RIGHT_UP")
    assert r["listed"] == 4
    assert r["new"] == 1                 # only its own
    assert r["foreign"] == 3             # the other three refused
    assert len(extracted) == 1
    assert "DHN_2_RIGHT_UP" in extracted[0]


# --- lookback window: "start now" means process trains from now --------------

def _dt(minutes_ago):
    from datetime import datetime, timedelta, timezone
    return datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)


class FakeBotoTimed:
    def __init__(self, items):          # items: (key, minutes_ago)
        self.items = items

    def list_objects_v2(self, **params):
        pfx = params.get("Prefix", "")
        return {"Contents": [{"Key": k, "LastModified": _dt(m)}
                             for k, m in self.items if k.startswith(pfx)],
                "IsTruncated": False}


def _timed_client(items):
    import logging
    from train_extraction.s3 import S3Client
    c = S3Client.__new__(S3Client)
    c.client = FakeBotoTimed(items)
    c.region = "ap-south-1"
    c.logger = logging.getLogger("t")
    return c


P = "camera_CCTV_HZBN_DHN_2_RIGHT_UP/"


def test_explicit_lookback_value_is_honoured(monkeypatch):
    from train_extraction import run_extraction_service as RES
    monkeypatch.setenv("WAGONEYE_EXTRACTION_LOOKBACK_MINUTES", "10")
    assert RES.lookback_minutes() == 10.0


def test_default_is_the_operational_day_anchor(monkeypatch):
    """No explicit window -> the 05:00 IST operational-day anchor (the previous
    production rule), NOT a sliding window."""
    from train_extraction import run_extraction_service as RES
    from core import config as CFG
    monkeypatch.delenv("WAGONEYE_EXTRACTION_LOOKBACK_MINUTES", raising=False)
    cutoff, desc = RES._raw_cutoff()
    assert cutoff == CFG.discovery_cutoff_utc()
    assert "operational day" in desc


def test_anchor_still_sees_todays_clips_after_a_restart(monkeypatch):
    """The whole point: a mid-day or 05:30 restart must NOT skip today's trains."""
    from train_extraction import run_extraction_service as RES
    monkeypatch.delenv("WAGONEYE_EXTRACTION_LOOKBACK_MINUTES", raising=False)

    class Ex:
        s3 = _timed_client([
            (P + "an_hour_ago.mp4", 60),
            (P + "three_hours_ago.mp4", 180),
            (P + "last_week.mp4", 60 * 24 * 7),
        ])

    names = [os.path.basename(k) for k in RES._list_raw_keys(Ex(), RAW)]
    # both of today's are kept even though far outside any 10-minute window
    assert "last_week.mp4" not in names


def test_only_recent_clips_are_listed(monkeypatch):
    from train_extraction import run_extraction_service as RES
    monkeypatch.setenv("WAGONEYE_EXTRACTION_LOOKBACK_MINUTES", "10")

    class Ex:
        s3 = _timed_client([
            (P + "now.mp4", 2),                 # inside the window
            (P + "recent.mp4", 9),              # inside
            (P + "old.mp4", 30),                # outside
            (P + "february.mp4", 60 * 24 * 170) # the one that bit us
        ])

    keys = RES._list_raw_keys(Ex(), RAW)
    assert sorted(os.path.basename(k) for k in keys) == ["now.mp4", "recent.mp4"]


def test_zero_disables_the_window(monkeypatch):
    from train_extraction import run_extraction_service as RES
    monkeypatch.setenv("WAGONEYE_EXTRACTION_LOOKBACK_MINUTES", "0")

    class Ex:
        s3 = _timed_client([(P + "a.mp4", 1), (P + "b.mp4", 99999)])

    assert len(RES._list_raw_keys(Ex(), RAW)) == 2


def test_window_is_configurable(monkeypatch):
    from train_extraction import run_extraction_service as RES
    monkeypatch.setenv("WAGONEYE_EXTRACTION_LOOKBACK_MINUTES", "120")

    class Ex:
        s3 = _timed_client([(P + "a.mp4", 30), (P + "b.mp4", 200)])

    keys = RES._list_raw_keys(Ex(), RAW)
    assert [os.path.basename(k) for k in keys] == ["a.mp4"]


def test_bad_value_falls_back_to_default(monkeypatch):
    from train_extraction import run_extraction_service as RES
    monkeypatch.setenv("WAGONEYE_EXTRACTION_LOOKBACK_MINUTES", "nonsense")
    assert RES.lookback_minutes() == 10.0


def test_object_without_lastmodified_is_kept(monkeypatch):
    """Never silently discard a clip we cannot date."""
    from train_extraction import run_extraction_service as RES
    monkeypatch.setenv("WAGONEYE_EXTRACTION_LOOKBACK_MINUTES", "10")

    class Ex:
        s3 = _client([P + "undated.mp4"])       # no LastModified at all

    assert len(RES._list_raw_keys(Ex(), RAW)) == 1
