"""Tests for the Stage-1 travel-direction derivation.

`travel_direction` reads the sign of each master gap's centre_x drift off the
per-hit ``bbox_history`` the tracker already produced -- no video re-read, no
model.  Its output vocabulary matches the V4 Train-Inspection-Engine's
optical-flow detector so the per-camera inspection JSON reports ``direction``
(and the side-camera ``rake_status`` derived from it) identically.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# wagon_count/ is a standalone package (its modules import each other by bare
# name), so it goes on the path directly.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "wagon_count"))

from global_alignment import travel_direction          # noqa: E402
from global_train_state import GapEvent, LocalCameraTracks  # noqa: E402


def _gap(track_id, cx_start, cx_end, n=5, fps=25.0):
    """A gap whose bbox sweeps from cx_start to cx_end across n hits."""
    step = (cx_end - cx_start) / max(1, n - 1)
    bbox_history = []
    hit_frames = []
    for i in range(n):
        cx = cx_start + step * i
        bbox_history.append([cx - 5.0, 10.0, cx + 5.0, 90.0])
        hit_frames.append(100 + i)
    return GapEvent(
        track_id=track_id, camera_id="RIGHT_UP",
        start_frame=hit_frames[0], end_frame=hit_frames[-1],
        confidence=0.9, hit_count=n, center_x_trajectory=[], fps=fps,
        temporal_consistency_score=1.0, class_label="wagon_gap",
        hit_frames=hit_frames, bbox_history=bbox_history,
    )


def _tracks(gaps, width=1920, total=1000, fps=25.0):
    return LocalCameraTracks(
        camera_id="RIGHT_UP", video_path="/dev/null", fps=fps,
        total_frames=total, width=width, height=1080, gaps=gaps,
    )


def test_rightward_drift_is_left_to_right():
    tracks = _tracks([_gap(1, 200.0, 1600.0), _gap(2, 300.0, 1700.0)])
    assert travel_direction(tracks) == "left-to-right"


def test_leftward_drift_is_right_to_left():
    tracks = _tracks([_gap(1, 1600.0, 200.0), _gap(2, 1700.0, 300.0)])
    assert travel_direction(tracks) == "right-to-left"


def test_majority_of_gaps_decides():
    # 2 rightward vs 1 leftward -> left-to-right
    tracks = _tracks([_gap(1, 200.0, 1600.0), _gap(2, 300.0, 1700.0),
                      _gap(3, 1700.0, 300.0)])
    assert travel_direction(tracks) == "left-to-right"


def test_a_tie_is_unknown_not_a_guess():
    tracks = _tracks([_gap(1, 200.0, 1600.0), _gap(2, 1600.0, 200.0)])
    assert travel_direction(tracks) == "unknown"


def test_no_gaps_is_unknown():
    assert travel_direction(_tracks([])) == "unknown"


def test_gap_without_a_trajectory_is_ignored():
    bare = GapEvent(
        track_id=1, camera_id="RIGHT_UP", start_frame=0, end_frame=5,
        confidence=0.9, hit_count=2, center_x_trajectory=[], fps=25.0,
        temporal_consistency_score=1.0, class_label="wagon_gap",
    )
    # only the drifting gap votes
    assert travel_direction(_tracks([bare, _gap(2, 200.0, 1600.0)])) == "left-to-right"
    # nothing votes at all
    assert travel_direction(_tracks([bare])) == "unknown"


def test_state_serialises_and_reloads_the_direction(tmp_path):
    """The field must survive the JSON round-trip Stage 1 -> Stage 6."""
    import json
    from core.global_state_loader import load_global_train_state

    doc = {
        "schema": "wagon_eye.global_train_state.v1",
        "master_camera": "RIGHT_UP", "master_fps": 25.0,
        "master_total_frames": 100, "total_wagons": 1,
        "travel_direction": "right-to-left",
        "wagons": [{"global_id": "GW_1", "wagon_index": 1,
                    "start_frame_master": 0, "end_frame_master": 99,
                    "start_time": 0.0, "end_time": 4.0,
                    "classification": "WAGON",
                    "classification_confidence": 0.9}],
    }
    p = tmp_path / "global_train_state.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    assert load_global_train_state(str(p)).travel_direction == "right-to-left"


def test_a_batch_sealed_before_the_field_existed_reads_unknown(tmp_path):
    import json
    from core.global_state_loader import load_global_train_state

    doc = {
        "master_camera": "RIGHT_UP", "master_fps": 25.0,
        "master_total_frames": 100, "total_wagons": 1,
        "wagons": [{"global_id": "GW_1", "wagon_index": 1,
                    "start_frame_master": 0, "end_frame_master": 99,
                    "start_time": 0.0, "end_time": 4.0,
                    "classification": "WAGON",
                    "classification_confidence": 0.9}],
    }
    p = tmp_path / "old_state.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    assert load_global_train_state(str(p)).travel_direction == "unknown"


def test_direction_drives_the_side_rake_status():
    """V4's side flavour infers rake_status from direction: L->R == Loaded."""
    from delivery.inspection_json import _rake_status_from_direction
    assert _rake_status_from_direction("left-to-right") == "Loaded"
    assert _rake_status_from_direction("right-to-left") == "Empty"
    assert _rake_status_from_direction("unknown") == "Unknown"
