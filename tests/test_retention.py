"""Tests for delivery/retention.py -- local-disk reclaim after finalization.

The guarantee under test: only LOCAL-ONLY intermediates are ever deleted, only
after the batch is terminal AND successful, and an active batch is never removed.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import config as CFG
from delivery import retention as R


def _batch(root, key, *, status="COMPLETED", age_days=0.0, cache_mb=2):
    d = os.path.join(root, key)
    for sub in CFG.BATCH_SUBDIRS:
        os.makedirs(os.path.join(d, sub), exist_ok=True)
    # a chunky wagon_cache + a small durable report
    blob = b"x" * (cache_mb * 1024 * 1024)
    with open(os.path.join(d, CFG.DIR_WAGON_CACHE, "frame_000001.jpg"), "wb") as f:
        f.write(blob)
    with open(os.path.join(d, CFG.DIR_DOWNLOADS, "right_up.mp4"), "wb") as f:
        f.write(b"y" * 1024)
    with open(os.path.join(d, CFG.DIR_REPORTS, "combined_train_report.json"), "w") as f:
        json.dump({"batch_key": key}, f)
    with open(os.path.join(d, CFG.DIR_EVIDENCE, "keep.jpg"), "wb") as f:
        f.write(b"z")
    with open(os.path.join(d, "manifest.json"), "w") as f:
        json.dump({"batch_key": key, "lifecycle_status": status}, f)
    if age_days:
        old = time.time() - age_days * 86400
        os.utime(d, (old, old))
    return d


def test_only_local_only_dirs_are_deleted(tmp_path):
    root = str(tmp_path)
    d = _batch(root, "B1")
    R.prune_batch_intermediates(d)
    # intermediates gone
    assert not os.path.isdir(os.path.join(d, CFG.DIR_WAGON_CACHE))
    assert not os.path.isdir(os.path.join(d, CFG.DIR_DOWNLOADS))
    assert not os.path.isdir(os.path.join(d, CFG.DIR_ARCHIVE))
    # everything mirrored to S3 survives
    assert os.path.isfile(os.path.join(d, CFG.DIR_REPORTS,
                                       "combined_train_report.json"))
    assert os.path.isfile(os.path.join(d, CFG.DIR_EVIDENCE, "keep.jpg"))
    assert os.path.isdir(os.path.join(d, CFG.DIR_GLOBAL_STATE))
    assert os.path.isdir(os.path.join(d, CFG.DIR_WAGON_STATES))
    assert os.path.isdir(os.path.join(d, CFG.DIR_PROCESSED_VIDEOS))


def test_reports_bytes_freed(tmp_path):
    d = _batch(str(tmp_path), "B1", cache_mb=3)
    freed = R.prune_batch_intermediates(d)
    assert freed[CFG.DIR_WAGON_CACHE] >= 3 * 1024 * 1024


def test_prune_is_idempotent(tmp_path):
    d = _batch(str(tmp_path), "B1")
    R.prune_batch_intermediates(d)
    assert R.prune_batch_intermediates(d) == {}      # nothing left, no error


def test_failed_batch_keeps_intermediates(tmp_path):
    root = str(tmp_path)
    d = _batch(root, "B1", status="FAILED")
    R.run(d, workspace_root=root, terminal_status="failed",
          prune_intermediates=False)
    assert os.path.isdir(os.path.join(d, CFG.DIR_WAGON_CACHE))


def test_successful_batch_prunes(tmp_path):
    root = str(tmp_path)
    d = _batch(root, "B1")
    R.run(d, workspace_root=root, terminal_status="completed",
          prune_intermediates=True)
    assert not os.path.isdir(os.path.join(d, CFG.DIR_WAGON_CACHE))


def test_prune_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("WAGONEYE_PRUNE_INTERMEDIATES", "false")
    root = str(tmp_path)
    d = _batch(root, "B1")
    R.run(d, workspace_root=root, prune_intermediates=True)
    assert os.path.isdir(os.path.join(d, CFG.DIR_WAGON_CACHE))


# --- age-based whole-batch retention -----------------------------------------

def test_retention_off_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("WAGONEYE_BATCH_RETENTION_DAYS", raising=False)
    root = str(tmp_path)
    _batch(root, "OLD", age_days=99)
    assert R.prune_old_batches(root) == []
    assert os.path.isdir(os.path.join(root, "OLD"))


def test_old_terminal_batch_removed(tmp_path):
    root = str(tmp_path)
    _batch(root, "OLD", age_days=10)
    _batch(root, "NEW", age_days=0)
    assert R.prune_old_batches(root, days=7) == ["OLD"]
    assert not os.path.isdir(os.path.join(root, "OLD"))
    assert os.path.isdir(os.path.join(root, "NEW"))


def test_active_batch_is_never_removed_however_old(tmp_path):
    """An in-flight train must not be deleted out from under the scheduler."""
    root = str(tmp_path)
    _batch(root, "ACTIVE", status="WAITING_FOR_LATE_CAMERAS", age_days=90)
    assert R.prune_old_batches(root, days=1) == []
    assert os.path.isdir(os.path.join(root, "ACTIVE"))


def test_orphan_without_manifest_is_reclaimable(tmp_path):
    root = str(tmp_path)
    d = os.path.join(root, "ORPHAN")
    os.makedirs(d)
    old = time.time() - 30 * 86400
    os.utime(d, (old, old))
    assert R.prune_old_batches(root, days=7) == ["ORPHAN"]


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("WAGONEYE_BATCH_RETENTION_DAYS", "14")
    assert R.retention_days() == 14.0
    monkeypatch.setenv("WAGONEYE_BATCH_RETENTION_DAYS", "nonsense")
    assert R.retention_days() == 0.0
    monkeypatch.setenv("WAGONEYE_PRUNE_INTERMEDIATES", "0")
    assert R.prune_intermediates_enabled() is False


def test_run_never_raises_on_a_bad_path():
    assert R.run("/nonexistent/path/xyz") is not None
