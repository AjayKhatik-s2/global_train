"""Per-camera TRAIN INSPECTION REPORT (reporting/camera_inspection_reports).

Covers the properties that matter operationally:
  * every camera renders, with the expected page skeleton
  * per-camera AUTHORITY isolation -- a camera never reports another camera's
    door/damage (the same invariant test_camera_isolation asserts upstream)
  * the direction -> rake-type mapping, including RIGHT_UP_TOP's inverted
    convention and the honest DIRECTION UNKNOWN fallback
  * engine / brake-van segments are excluded from the wagon counts
  * a missing snapshot degrades to a placeholder instead of failing the page
  * the existing camera_reports filenames are never collided with
"""
import json
import os
import sys

import numpy as np
import cv2
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import constants as C
from core.global_state_loader import GlobalTrainState, GlobalWagon
from core.unified_wagon_state import UnifiedWagonState
from reporting import camera_inspection_reports as CIR
from reporting import _inspection_adapter as IA

FPS = 25.0
N_SEGMENTS = 6          # GW_1 engine, GW_2..GW_5 wagons, GW_6 brake van
N_WAGONS = 4


def _jpg(path, color=(60, 60, 60)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, np.full((60, 100, 3), color, np.uint8))


@pytest.fixture
def batch(tmp_path):
    """A finalized-looking batch tree with known per-camera results."""
    root = {k: tmp_path / k for k in
            ("wagon_cache", "evidence", "wagon_states", "reports", "global_state")}
    for p in root.values():
        p.mkdir(parents=True, exist_ok=True)

    wagons, t = [], 0.0
    for i in range(1, N_SEGMENTS + 1):
        cls = (C.CLASS_ENGINE if i == 1 else
               C.CLASS_BRAKE_VAN if i == N_SEGMENTS else C.CLASS_WAGON)
        wagons.append(GlobalWagon(
            global_id=f"GW_{i}", wagon_index=i,
            start_frame_master=int(t * FPS),
            end_frame_master=int((t + 4) * FPS) - 1,
            start_time=t, end_time=t + 4.0,
            classification=cls, classification_confidence=0.9))
        t += 4.0
    state = GlobalTrainState(total_wagons=N_SEGMENTS, wagons=wagons,
                             master_camera=C.CAMERA_RIGHT_UP,
                             master_fps=FPS, master_total_frames=int(t * FPS))

    pct = root["global_state"] / "per_camera_tracking.json"
    pct.write_text(json.dumps({
        cam: {"fps": FPS, "total_frames": int(t * FPS), "width": 100, "height": 60}
        for cam in C.ALL_CAMERAS}))

    # cache frames at the three sampled positions
    for gw in wagons:
        sf, ef = int(gw.start_time * FPS), int(gw.end_time * FPS) - 1
        for cam in C.ALL_CAMERAS:
            for pos in IA_POSITIONS:
                idx = int(sf + (ef - sf) * pos)
                _jpg(str(root["wagon_cache"] / gw.global_id /
                         C.CAMERA_FOLDER[cam] / f"frame_{idx:06d}.jpg"))

    unified = {}
    for gw in wagons:
        u = UnifiedWagonState(global_id=gw.global_id,
                              wagon_index=gw.wagon_index,
                              classification=gw.classification)
        # RIGHT_UP: GW_2 door OPEN.  LEFT_UP: GW_3 door DAMAGED.
        for cam, side in ((C.CAMERA_RIGHT_UP, "right"), (C.CAMERA_LEFT_UP, "left")):
            st, raw = C.DOOR_CLOSED, "closed_door"
            if gw.global_id == "GW_2" and side == "right":
                st, raw = C.DOOR_OPEN, "open_door"
            if gw.global_id == "GW_3" and side == "left":
                st, raw = C.DOOR_DAMAGED, "damage"
            setattr(u, f"{side}_door", st)
            d = root["wagon_states"] / "door" / cam
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{gw.global_id}.json").write_text(json.dumps(
                {"global_id": gw.global_id, "status": C.STATUS_OK,
                 "camera_id": cam, "side": side, "door_state": st}))
            ed = root["evidence"] / gw.global_id / "door" / cam
            _jpg(str(ed / f"{side}_best.jpg"))
            (ed / "metadata.json").write_text(json.dumps(
                {"sides": {side: {"raw_class": raw, "frame_idx": 42, "state": st}}}))
        # damage ONLY on RIGHT_UP_TOP, for GW_3 and GW_5
        for cam in C.TOP_CAMERAS:
            dmg = cam == C.CAMERA_RIGHT_UP_TOP and gw.global_id in ("GW_3", "GW_5")
            d = root["wagon_states"] / "damage" / cam
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{gw.global_id}.json").write_text(json.dumps(
                {"global_id": gw.global_id, "status": C.STATUS_OK, "camera_id": cam,
                 "damage_status": C.DAMAGE_PRESENT if dmg else C.DAMAGE_OK}))
            if dmg:
                u.top_damage = C.DAMAGE_PRESENT
                ed = root["evidence"] / gw.global_id / "damage" / cam
                _jpg(str(ed / "track_1.jpg"))
                (ed / "metadata.json").write_text(json.dumps(
                    {"tracks": [{"track_idx": 1, "class_name": "floor_damage",
                                 "best_confidence": 0.8, "best_frame_idx": 77}]}))
        unified[gw.global_id] = u

    return {"state": state, "unified": unified, "root": root, "pct": str(pct)}


IA_POSITIONS = (0.25, 0.55, 0.80)


def _build(batch, **kw):
    r = batch["root"]
    params = dict(
        state=batch["state"], unified=batch["unified"],
        output_dir=str(r["reports"]), batch_key="20260718_143154",
        cache_root=str(r["wagon_cache"]), wagon_states_root=str(r["wagon_states"]),
        evidence_root=str(r["evidence"]), per_camera_tracking_path=batch["pct"],
        verbose=False,
    )
    params.update(kw)
    return CIR.build_all(**params)


def _model(batch, camera_id, **kw):
    r = batch["root"]
    params = dict(
        camera_id=camera_id, state=batch["state"], unified=batch["unified"],
        batch_key="20260718_143154", cache_root=str(r["wagon_cache"]),
        wagon_states_root=str(r["wagon_states"]),
        evidence_root=str(r["evidence"]), per_camera_tracking_path=batch["pct"],
    )
    params.update(kw)
    return IA.build_model(**params)


def test_all_four_cameras_render(batch):
    out = _build(batch)
    assert set(out) == set(C.ALL_CAMERAS)
    for cam, path in out.items():
        assert path, f"{cam} produced no PDF"
        assert os.path.getsize(path) > 0


def test_filenames_never_collide_with_camera_reports():
    from reporting.camera_reports import CAMERA_FILE
    assert not (set(CAMERA_FILE.values())
                & set(CIR.CAMERA_INSPECTION_FILE.values()))


def test_engine_and_brakevan_excluded_from_wagon_counts(batch):
    m = _model(batch, C.CAMERA_RIGHT_UP)
    assert len(m.segments) == N_SEGMENTS      # every segment gets a page
    assert m.total_wagons == N_WAGONS         # but only wagons are counted
    types = {s.segment_id: s.segment_type for s in m.segments}
    assert types[1] == "engine"
    assert types[N_SEGMENTS] == "brakevan"


def test_side_camera_authority_isolation(batch):
    """RIGHT_UP sees only the right door; LEFT_UP only the left."""
    right = _model(batch, C.CAMERA_RIGHT_UP)
    left = _model(batch, C.CAMERA_LEFT_UP)

    assert right.status_for(2) == "DOOR OPEN"     # GW_2 right door OPEN
    assert right.status_for(3) == "OK"            # GW_3 damage is LEFT's
    assert right.damaged_wagons == 0

    assert left.status_for(3) == "DAMAGE"         # GW_3 left door DAMAGED
    assert left.status_for(2) == "OK"             # GW_2 open door is RIGHT's
    assert left.damaged_wagons == 1


def test_top_camera_authority_isolation(batch):
    """LEFT_UP_TOP must not inherit RIGHT_UP_TOP's damage."""
    rt = _model(batch, C.CAMERA_RIGHT_UP_TOP)
    lt = _model(batch, C.CAMERA_LEFT_UP_TOP)

    assert rt.status_for(3) == "DAMAGE"
    assert rt.status_for(5) == "DAMAGE"
    assert rt.damaged_wagons == 2

    assert lt.damaged_wagons == 0
    assert all(lt.status_for(s.segment_id) == "OK" for s in lt.segments)


def test_rake_type_mapping_including_inverted_top_right(batch):
    lr, rl = "left-to-right", "right-to-left"
    assert _model(batch, C.CAMERA_RIGHT_UP, direction=lr)\
        .style.rake_for(lr)[0] == "LOADED RAKE"
    assert _model(batch, C.CAMERA_LEFT_UP, direction=rl)\
        .style.rake_for(rl)[0] == "EMPTY RAKE"
    # RIGHT_UP_TOP inverts: left-to-right means EMPTY there
    assert _model(batch, C.CAMERA_RIGHT_UP_TOP, direction=lr)\
        .style.rake_for(lr)[0] == "EMPTY RAKE"
    assert _model(batch, C.CAMERA_LEFT_UP_TOP, direction=lr)\
        .style.rake_for(lr)[0] == "LOADED RAKE"


def test_unknown_direction_is_not_guessed(batch):
    """No direction in the artifacts -> DIRECTION UNKNOWN, never a fabricated
    rake type."""
    m = _model(batch, C.CAMERA_LEFT_UP_TOP)
    assert m.direction == "unknown"
    assert m.style.rake_for(m.direction)[0] == "DIRECTION UNKNOWN"


def test_probable_damage_never_fabricated(batch):
    """v4 has no 'probable damage' tier -- it must stay False, not be guessed."""
    m = _model(batch, C.CAMERA_RIGHT_UP_TOP)
    assert all(not r.probable_damage_detected for r in m.damage_rows.values())


def test_problem_frames_are_camera_scoped(batch):
    rt = _model(batch, C.CAMERA_RIGHT_UP_TOP)
    lt = _model(batch, C.CAMERA_LEFT_UP_TOP)
    assert {p.wagon_id for p in rt.problem_frames} == {3, 5}
    assert rt.problem_frames[0].problem_type == "floor_damage"
    assert lt.problem_frames == []


def test_missing_snapshots_do_not_fail_the_build(batch, tmp_path):
    """An evidence/cache tree that does not exist still yields a valid PDF."""
    out = _build(batch, cache_root=str(tmp_path / "nope"),
                 evidence_root=str(tmp_path / "gone"))
    for cam, path in out.items():
        assert path, f"{cam} failed on missing snapshots"
        assert os.path.getsize(path) > 0


def test_subset_rebuild_touches_only_that_camera(batch):
    out = _build(batch, cameras=[C.CAMERA_LEFT_UP])
    assert set(out) == {C.CAMERA_LEFT_UP}
    reports = batch["root"]["reports"]
    assert (reports / CIR.CAMERA_INSPECTION_FILE[C.CAMERA_LEFT_UP]).exists()
    assert not (reports / CIR.CAMERA_INSPECTION_FILE[C.CAMERA_RIGHT_UP]).exists()


def test_page_skeleton_matches_reference_layout(batch):
    """title + summary + divider + per-wagon problem pages + loco + per-segment."""
    PdfReader = pytest.importorskip("PyPDF2").PdfReader
    out = _build(batch)
    r = PdfReader(out[C.CAMERA_RIGHT_UP])
    text = [(p.extract_text() or "") for p in r.pages]

    assert "TRAIN INSPECTION REPORT" in text[0]
    assert "Station" in text[0] and "HAZARIBAGH" in text[0]
    assert "Total Wagons: 4" in text[0].replace("\n", " ")

    joined = " ".join(text)
    assert "WAGON STATUS SUMMARY (Page 1 of 1)" in joined
    assert "DATE-TIME" in joined and "DAMAGED WAGONS" in joined
    assert "SR.NO" in joined and "WAGON ID" in joined
    assert "DAMAGE / DOOR DETECTED" in joined
    assert "Problem Frames" in joined
    assert "LOCOMOTIVES" in joined
    # every segment gets its own page at the end
    assert "ENGINE #1" in joined and "BRAKEVAN #6" in joined
