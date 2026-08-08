"""Per-camera Stage-1 gap-extraction cache (AUTO/S3 pipeline only).

WHY
---
Stage 1's expensive work is per-camera and embarrassingly independent: gap
detection + tracking over one video, plus that camera's own classification
pass.  Only STEP 3 (`global_alignment.assemble_global_train_state`) needs all
cameras at once.

Historically every camera's inference ran serially inside the one blocking
`run_global_count.py` subprocess at seal time, so the first camera to land sat
idle until the last one arrived.  This module lets the AUTO pipeline run each
camera's gap extraction the moment that camera's S3 object is stable, persist
the complete result, and have the eventual reconstruction consume the persisted
results instead of re-running inference.

WHAT IS PERSISTED
-----------------
The exact existing Stage-1 structures -- `LocalCameraTracks` (fps, frame count,
width/height, every `GapEvent` with track ids, hit frames, bbox history,
center-x trajectory, confidences, temporal scores) plus the camera's own
classifications and the raw per-frame detections the overlay renderer needs.
No new gap representation is invented; this is `to_cache_dict` /
`from_cache_dict` on the existing dataclasses.

TWO FILES PER CAMERA
--------------------
    <cache_dir>/<CAMERA>.result.json   full LocalCameraTracks payload
    <cache_dir>/<CAMERA>.state.json    tiny lifecycle record

They are split on purpose.  The orchestrator (which cannot import wagon_count
-- that package is standalone with bare sibling imports) polls only the small
state file, while the heavy result file is written and read exclusively inside
the Stage-1 subprocess.

IDENTITY
--------
Every record carries the S3 object identity it was produced from (bucket, key,
ETag, size, last-modified).  A cached result is used ONLY when that identity
matches the object currently in the slot; a replaced video with a different
ETag is a different input and is re-extracted from scratch.

This module is deliberately dependency-free (stdlib only) so both the
standalone wagon_count subprocess and any caller can use it.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, Optional


# -----------------------------------------------------------------------------
# Lifecycle states for one (camera, S3 object)
# -----------------------------------------------------------------------------

class GapState:
    DISCOVERED  = "discovered"    # object seen in S3, not yet proven stable
    DOWNLOADING = "downloading"   # claimed, transfer in flight
    DOWNLOADED  = "downloaded"    # local file present and validated
    PROCESSING  = "processing"    # gap extraction running
    COMPLETED   = "completed"     # result.json written and loadable
    FAILED      = "failed"        # extraction raised; retryable

    #: States from which no further work should be scheduled for this identity.
    TERMINAL = (COMPLETED,)
    #: A worker holds the camera in one of these; another worker must not claim.
    IN_FLIGHT = (DOWNLOADING, DOWNLOADED, PROCESSING)


RESULT_SCHEMA = "wagon_eye.gap_cache.v1"


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

def result_path(cache_dir: str, camera: str) -> str:
    return os.path.join(cache_dir, f"{camera}.result.json")


def state_path(cache_dir: str, camera: str) -> str:
    return os.path.join(cache_dir, f"{camera}.state.json")


def lock_path(cache_dir: str, camera: str) -> str:
    return os.path.join(cache_dir, f"{camera}.lock")


# -----------------------------------------------------------------------------
# Object identity
# -----------------------------------------------------------------------------

def make_identity(*, bucket: str = "", s3_key: str = "", etag: Optional[str] = None,
                  file_size: int = 0,
                  last_modified: Optional[str] = None) -> Dict[str, Any]:
    """Normalised identity of the S3 object a gap result was produced from."""
    return {
        "bucket": bucket or "",
        "s3_key": s3_key or "",
        # S3 quotes ETags ("abc123"); strip so a quoted and unquoted form of the
        # same object are not treated as two different inputs.
        "etag": (etag or "").strip('"') or None,
        "file_size": int(file_size or 0),
        "last_modified": last_modified or None,
    }


def identity_matches(a: Optional[Dict[str, Any]],
                     b: Optional[Dict[str, Any]]) -> bool:
    """True when two identities denote the same S3 object version.

    ETag is the authoritative discriminator when both sides have one -- it
    changes whenever the content changes, which is exactly the "replacement
    video must be treated as a new input" rule.  Without an ETag on either
    side we fall back to key + size + last-modified.
    """
    if not a or not b:
        return False
    if a.get("s3_key") != b.get("s3_key") or a.get("bucket") != b.get("bucket"):
        return False
    ea, eb = a.get("etag"), b.get("etag")
    if ea and eb:
        return ea == eb
    return (a.get("file_size") == b.get("file_size")
            and a.get("last_modified") == b.get("last_modified"))


# -----------------------------------------------------------------------------
# Atomic JSON I/O
# -----------------------------------------------------------------------------

def _write_json(path: str, payload: Dict[str, Any]) -> None:
    """Write via a temp file + os.replace so a reader never sees a partial file.

    A crash mid-write would otherwise leave truncated JSON that looks like a
    corrupt cache entry forever.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError, ValueError):
        return None      # unreadable cache == no cache; never fatal


# -----------------------------------------------------------------------------
# State record (small; read by the orchestrator)
# -----------------------------------------------------------------------------

def write_state(cache_dir: str, camera: str, state: str,
                identity: Optional[Dict[str, Any]] = None, *,
                error: Optional[str] = None,
                gap_count: Optional[int] = None,
                owner: Optional[str] = None,
                updated_at: Optional[str] = None) -> None:
    _write_json(state_path(cache_dir, camera), {
        "camera_id": camera,
        "state": state,
        "identity": identity,
        "error": error,
        "gap_count": gap_count,
        "owner": owner,
        "updated_at": updated_at,
    })


def read_state(cache_dir: str, camera: str) -> Optional[Dict[str, Any]]:
    return _read_json(state_path(cache_dir, camera))


def is_completed_for(cache_dir: str, camera: str,
                     identity: Optional[Dict[str, Any]]) -> bool:
    """True when THIS exact object has already been gap-extracted successfully.

    Requires both the state record and the result file, so a state file left
    behind by a half-finished run cannot mask a missing result.
    """
    st = read_state(cache_dir, camera)
    if not st or st.get("state") != GapState.COMPLETED:
        return False
    if not identity_matches(st.get("identity"), identity):
        return False
    return os.path.isfile(result_path(cache_dir, camera))


# -----------------------------------------------------------------------------
# Result record (heavy; read/written only inside the Stage-1 subprocess)
# -----------------------------------------------------------------------------

def write_result(cache_dir: str, camera: str, tracks, *,
                 identity: Optional[Dict[str, Any]] = None,
                 top_local_classifications=None,
                 produced_at: Optional[str] = None) -> str:
    """Persist one camera's complete Stage-1 output.

    `top_local_classifications` are a TOP camera's segment labels in its OWN
    frame numbering.  They are stored unscaled on purpose: converting them to
    master frames needs the master's fps, which is not known while cameras are
    still arriving.  The identical rescale is applied at assembly time.
    """
    payload = {
        "schema": RESULT_SCHEMA,
        "camera_id": camera,
        "identity": identity,
        "produced_at": produced_at,
        "tracks": tracks.to_cache_dict(),
        "top_local_classifications": [c.to_cache_dict() for c
                                      in (top_local_classifications or [])],
    }
    p = result_path(cache_dir, camera)
    _write_json(p, payload)
    return p


def read_result(cache_dir: str, camera: str,
                identity: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Raw cached payload for one camera, or None.

    When `identity` is given the payload is returned ONLY if it was produced
    from that same object version.
    """
    payload = _read_json(result_path(cache_dir, camera))
    if not payload or payload.get("schema") != RESULT_SCHEMA:
        return None
    if identity is not None and not identity_matches(payload.get("identity"), identity):
        return None
    return payload


def load_tracks(cache_dir: str, camera: str,
                identity: Optional[Dict[str, Any]] = None):
    """Rebuild `(LocalCameraTracks, top_local_classifications)` from cache.

    Returns `(None, [])` when there is no usable entry.  Imported lazily so
    this module stays importable without the wagon_count dataclasses on the
    path (the orchestrator only ever touches the state file).
    """
    payload = read_result(cache_dir, camera, identity)
    if not payload:
        return None, []
    from global_train_state import LocalCameraTracks, _MasterClassification
    tracks = LocalCameraTracks.from_cache_dict(payload["tracks"])
    tops = [_MasterClassification.from_cache_dict(c)
            for c in payload.get("top_local_classifications", [])]
    return tracks, tops
