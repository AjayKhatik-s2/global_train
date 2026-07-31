"""Local-disk retention for finished batches.

`batch_outputs/<key>/` is ~3 GB per batch, and until now nothing ever removed it,
so a long-running box filled its volume and started failing batches in confusing
ways.  This module reclaims that space WITHOUT weakening any durability guarantee.

What is safe to delete, and why
-------------------------------
Only three subdirectories are ever local-only -- every other one is mirrored to
S3 at finalization (`global_state`, `wagon_states`, `reports`, `evidence`,
`processed_videos`):

    wagon_cache/   the per-wagon JPEG cache -- ~80% of the footprint.  A pure
                   Stage-2 intermediate, fully regenerable from the source video.
    downloads/     the source clips, which still exist in the trimmed S3 bucket.
    archive/       run scratch.

Deleting those is therefore lossless with respect to S3.

When it is safe
---------------
NOT before the batch is terminal.  While a batch is still active, a late camera
triggers `stage_reports`, and the camera reports read quartile frames back out of
`wagon_cache` -- pruning early would silently degrade those PDFs.  So pruning runs
only after `stage_finalize` has uploaded everything and the batch has reached a
terminal state, and only for the SUCCESSFUL terminal states: a failed batch keeps
its intermediates so the failure can still be diagnosed on the box.

Both behaviours are opt-outable and neither can fail a batch -- every error here is
logged and swallowed.
"""

from __future__ import annotations

import os
import shutil
import time
from typing import Dict, List, Optional

from core import config as CFG
from core.logging_setup import get_logger

log = get_logger("delivery.retention")

#: Batch subdirectories that exist ONLY on local disk (never uploaded to S3).
LOCAL_ONLY_SUBDIRS = (CFG.DIR_WAGON_CACHE, CFG.DIR_DOWNLOADS, CFG.DIR_ARCHIVE)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def prune_intermediates_enabled() -> bool:
    """Delete the local-only intermediates once a batch finishes successfully."""
    return _env_bool("WAGONEYE_PRUNE_INTERMEDIATES", True)


def retention_days() -> float:
    """Whole-batch retention in days; 0 (default) disables age-based deletion."""
    return max(0.0, _env_float("WAGONEYE_BATCH_RETENTION_DAYS", 0.0))


def _dir_size_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(root, fn))
            except OSError:
                pass
    return total


def prune_batch_intermediates(batch_root: str) -> Dict[str, int]:
    """Delete `LOCAL_ONLY_SUBDIRS` under `batch_root`.

    Returns ``{subdir: bytes_freed}``.  Never raises: a directory that cannot be
    removed is logged and left alone.
    """
    freed: Dict[str, int] = {}
    for sub in LOCAL_ONLY_SUBDIRS:
        d = os.path.join(batch_root, sub)
        if not os.path.isdir(d):
            continue
        size = _dir_size_bytes(d)
        try:
            shutil.rmtree(d)
            freed[sub] = size
        except OSError as e:
            log.warning("[RETENTION] could not remove %s: %s", d, e)
    if freed:
        log.info("[RETENTION] freed %.2f GB from %s (%s) -- durable artifacts "
                 "remain in S3",
                 sum(freed.values()) / 1e9, os.path.basename(batch_root),
                 ", ".join(sorted(freed)))
    return freed


def prune_old_batches(workspace_root: str,
                      days: Optional[float] = None) -> List[str]:
    """Delete whole batch directories older than `days`.

    Age is taken from the directory mtime.  A batch that is still ACTIVE is
    identified by its `manifest.json` lifecycle status and is never removed, so an
    in-flight train cannot be deleted out from under the scheduler.  Returns the
    batch keys removed.
    """
    if days is None:
        days = retention_days()
    if days <= 0 or not os.path.isdir(workspace_root):
        return []

    cutoff = time.time() - days * 86400.0
    removed: List[str] = []
    for name in sorted(os.listdir(workspace_root)):
        d = os.path.join(workspace_root, name)
        if not os.path.isdir(d):
            continue
        try:
            if os.path.getmtime(d) >= cutoff:
                continue
        except OSError:
            continue
        if _is_active(d):
            log.info("[RETENTION] %s is older than %.0fd but still ACTIVE -- kept",
                     name, days)
            continue
        try:
            shutil.rmtree(d)
            removed.append(name)
        except OSError as e:
            log.warning("[RETENTION] could not remove %s: %s", d, e)
    if removed:
        log.info("[RETENTION] removed %d batch dir(s) older than %.0fd: %s",
                 len(removed), days, ", ".join(removed))
    return removed


def _is_active(batch_root: str) -> bool:
    """True when the batch's manifest says it has NOT reached a terminal state.

    A directory with no readable manifest is treated as INACTIVE (it cannot be
    advanced), so an orphaned dir is still reclaimable.
    """
    p = os.path.join(batch_root, "manifest.json")
    if not os.path.isfile(p):
        return False
    try:
        import json
        with open(p, "r", encoding="utf-8") as f:
            status = (json.load(f) or {}).get("lifecycle_status")
    except (OSError, ValueError):
        return False
    if not status:
        return False
    try:
        from core.lifecycle import is_terminal
        return not is_terminal(status)
    except Exception:
        return False


def run(batch_root: str, *, workspace_root: Optional[str] = None,
        terminal_status: Optional[str] = None,
        prune_intermediates: bool = True) -> Dict[str, object]:
    """Post-finalization disk reclaim.  Never raises.

    `prune_intermediates` is the caller's decision about whether THIS batch
    succeeded -- a failed batch keeps its intermediates for diagnosis.
    """
    result: Dict[str, object] = {"freed": {}, "removed_batches": []}
    try:
        if prune_intermediates and prune_intermediates_enabled():
            result["freed"] = prune_batch_intermediates(batch_root)
        elif prune_intermediates_enabled():
            log.info("[RETENTION] %s ended %s -- intermediates KEPT for diagnosis",
                     os.path.basename(batch_root), terminal_status or "unsuccessfully")
        root = workspace_root or os.path.dirname(os.path.abspath(batch_root))
        result["removed_batches"] = prune_old_batches(root)
    except Exception as e:  # absolute isolation: retention never fails a batch
        log.error("[RETENTION] error (non-fatal): %s", e)
    return result
