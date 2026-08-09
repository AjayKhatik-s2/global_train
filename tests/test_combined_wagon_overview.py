"""Combined-report wagon-by-wagon 4-camera overview tests.

Covers the additive combined-report section only: the per-(Global Wagon, camera)
wagon-CENTRE snapshot selection, its failure modes, and the guarantee that the
pre-existing combined-report findings/evidence are untouched.

No .pt models, no inference, no video decode -- the wagon_cache JPEGs are
written directly, which is exactly what Stage 2 materializes.

Run:  python tests/test_combined_wagon_overview.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from core import constants as C
from core.global_state_loader import GlobalTrainState, GlobalWagon
from core.unified_wagon_state import UnifiedWagonState

from reporting import _evidence_lookup as EL
from reporting import _legacy_data_adapter as LDA


# -----------------------------------------------------------------------------
# fixtures
# -----------------------------------------------------------------------------

# Every camera in this file runs a DIFFERENT fps on purpose: the same Global
# Wagon therefore maps to a different local frame range in each camera, which is
# precisely the "cameras are not synchronised" condition the selector must
# handle.  fps/total_frames mirror the real per_camera_tracking.json shape.
_CAM_META = {
    C.CAMERA_RIGHT_UP:     {"fps": 25.0,  "total_frames": 1000},
    C.CAMERA_LEFT_UP:      {"fps": 12.5,  "total_frames": 500},
    C.CAMERA_RIGHT_UP_TOP: {"fps": 30.0,  "total_frames": 1200},
    C.CAMERA_LEFT_UP_TOP:  {"fps": 20.0,  "total_frames": 800},
}

_WAGON_SECONDS = 4.0     # each GW spans 4 s of master clock


def _jpeg(path: str, w: int = 64, h: int = 36) -> None:
    """A small but genuinely decodable JPEG (Stage 2 writes real frames)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    from PIL import Image as _PILImage
    _PILImage.new("RGB", (w, h), (40, 90, 140)).save(path, "JPEG", quality=80)


def _state(n: int = 3) -> GlobalTrainState:
    wagons = []
    for i in range(1, n + 1):
        wagons.append(GlobalWagon(
            global_id=f"GW_{i}", wagon_index=i,
            start_frame_master=int((i - 1) * _WAGON_SECONDS * 25),
            end_frame_master=int(i * _WAGON_SECONDS * 25) - 1,
            start_time=(i - 1) * _WAGON_SECONDS,
            end_time=i * _WAGON_SECONDS,
            classification=(C.CLASS_ENGINE if i == 1 else C.CLASS_WAGON),
            classification_confidence=0.9,
        ))
    return GlobalTrainState(total_wagons=n, wagons=wagons,
                            master_camera=C.CAMERA_RIGHT_UP,
                            master_fps=25.0, master_total_frames=n * 100)


def _unified(state: GlobalTrainState):
    out = {}
    for gw in state.wagons:
        out[gw.global_id] = UnifiedWagonState(
            global_id=gw.global_id, wagon_index=gw.wagon_index,
            classification=gw.classification,
            classification_confidence=gw.classification_confidence,
            confidence=0.8,
        )
    return out


def _tracking_json(root: str, cams=None) -> str:
    cams = cams or list(C.ALL_CAMERAS)
    doc = {c: dict(_CAM_META[c], width=1920, height=1080, gaps=[]) for c in cams}
    p = os.path.join(root, "per_camera_tracking.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(doc, f)
    return p


def _expected_range(camera_id: str, gw: GlobalWagon):
    m = _CAM_META[camera_id]
    return EL.wagon_local_frames(gw.start_time, gw.end_time,
                                 m["fps"], m["total_frames"])


def _materialize(cache_root: str, state: GlobalTrainState, cams=None,
                 skip=()) -> None:
    """Write the wagon_cache exactly as Stage 2 would: one JPEG per frame of
    each (wagon, camera) mapped interval, named by its ORIGINAL local frame."""
    cams = cams or list(C.ALL_CAMERAS)
    for gw in state.wagons:
        for cam in cams:
            if (gw.global_id, cam) in skip or cam in skip:
                continue
            sf, ef = _expected_range(cam, gw)
            folder = C.CAMERA_FOLDER[cam]
            for fi in range(sf, ef + 1):
                _jpeg(os.path.join(cache_root, gw.global_id, folder,
                                   f"frame_{fi:06d}.jpg"))


def _fixture(n=3, cams=None, skip=()):
    root = tempfile.mkdtemp()
    st = _state(n)
    cache = os.path.join(root, "wagon_cache")
    _materialize(cache, st, cams=cams, skip=skip)
    return root, st, cache, _tracking_json(root)


def _flowable_texts(flowables):
    """Depth-first Paragraph text of a flowable tree, in emission order."""
    out = []
    for f in flowables or ():
        if isinstance(f, (list, tuple)):
            out.extend(_flowable_texts(f))
            continue
        cells = getattr(f, "_cellvalues", None)
        if cells is not None:
            for row in cells:
                out.extend(_flowable_texts(row))
            continue
        content = getattr(f, "_content", None)
        if content is not None:
            out.extend(_flowable_texts(content))
            continue
        t = getattr(f, "text", None)
        if isinstance(t, str):
            out.append(t)
    return out


def _overview(st, cache, pcf, evidence_root=None):
    return LDA.build_wagon_overview(
        state=st, unified=_unified(st), cache_root=cache,
        evidence_root=evidence_root, per_camera_tracking_path=pcf,
        verbose=False)


# -----------------------------------------------------------------------------
# (1) all four cameras available -> a centre frame from each
# -----------------------------------------------------------------------------

def test_all_four_cameras_center_selection():
    _root, st, cache, pcf = _fixture()
    ov = _overview(st, cache, pcf)

    assert set(ov) == {"GW_1", "GW_2", "GW_3"}
    for gw in st.wagons:
        panels = ov[gw.global_id]["cameras"]
        assert set(panels) == set(LDA.OVERVIEW_CAMERA_ORDER)
        for cam in LDA.OVERVIEW_CAMERA_ORDER:
            p = panels[cam]
            sf, ef = _expected_range(cam, gw)
            assert p["status"] == EL.OVERVIEW_OK, (gw.global_id, cam, p)
            # exact temporal centre of THIS camera's mapped interval
            assert p["frame"] == (sf + ef) // 2, (gw.global_id, cam, p)
            assert sf <= p["frame"] <= ef
            assert os.path.isfile(p["path"])
            # the filename carries the ORIGINAL source-video frame number
            assert os.path.basename(p["path"]) == f"frame_{p['frame']:06d}.jpg"


# -----------------------------------------------------------------------------
# (2) one camera missing -> placeholder, other three still rendered
# -----------------------------------------------------------------------------

def test_one_camera_missing_never_breaks_the_rest():
    _root, st, cache, pcf = _fixture(skip=(C.CAMERA_LEFT_UP_TOP,))
    ov = _overview(st, cache, pcf)

    for gw in st.wagons:
        panels = ov[gw.global_id]["cameras"]
        gone = panels[C.CAMERA_LEFT_UP_TOP]
        assert gone["status"] == EL.OVERVIEW_NO_FRAMES
        assert gone["path"] is None and gone["frame"] is None
        for cam in (C.CAMERA_RIGHT_UP, C.CAMERA_LEFT_UP, C.CAMERA_RIGHT_UP_TOP):
            assert panels[cam]["status"] == EL.OVERVIEW_OK
            assert os.path.isfile(panels[cam]["path"])

    # a camera absent from per_camera_tracking.json entirely still resolves from
    # the materialized cache alone (no fps/total_frames available)
    pcf3 = _tracking_json(os.path.dirname(cache),
                          cams=[C.CAMERA_RIGHT_UP, C.CAMERA_LEFT_UP,
                                C.CAMERA_RIGHT_UP_TOP])
    ov3 = _overview(st, cache, pcf3)
    assert ov3["GW_2"]["cameras"][C.CAMERA_RIGHT_UP]["status"] == EL.OVERVIEW_OK


# -----------------------------------------------------------------------------
# (3) different camera frame rates -> each camera keeps its OWN numbering
# -----------------------------------------------------------------------------

def test_different_frame_rates_use_own_local_numbering():
    _root, st, cache, pcf = _fixture()
    ov = _overview(st, cache, pcf)
    gw2 = next(w for w in st.wagons if w.global_id == "GW_2")
    panels = ov["GW_2"]["cameras"]

    # same physical wagon, four different local frame numbers
    frames = {c: panels[c]["frame"] for c in LDA.OVERVIEW_CAMERA_ORDER}
    assert frames[C.CAMERA_RIGHT_UP] == 149        # 25   fps
    assert frames[C.CAMERA_LEFT_UP] == 74          # 12.5 fps
    assert frames[C.CAMERA_RIGHT_UP_TOP] == 179    # 30   fps
    assert frames[C.CAMERA_LEFT_UP_TOP] == 119     # 20   fps
    assert len(set(frames.values())) == 4, "cameras must not share a frame index"

    # every one of them lands at the same instant on the master clock
    for cam, fi in frames.items():
        t = fi / _CAM_META[cam]["fps"]
        assert gw2.start_time <= t <= gw2.end_time
        assert abs(t - (gw2.start_time + gw2.end_time) / 2.0) < 0.1


# -----------------------------------------------------------------------------
# (4) unsynchronised cameras: never "frame N from every camera"
# -----------------------------------------------------------------------------

def test_unsynchronized_cameras_are_not_index_aligned():
    _root, st, cache, pcf = _fixture()
    ov = _overview(st, cache, pcf)
    for gw in st.wagons:
        panels = ov[gw.global_id]["cameras"]
        picked = [panels[c]["frame"] for c in LDA.OVERVIEW_CAMERA_ORDER]
        assert len(set(picked)) == len(picked)
        # each frame sits inside ITS OWN camera's mapped interval
        for cam in LDA.OVERVIEW_CAMERA_ORDER:
            sf, ef = _expected_range(cam, gw)
            assert sf <= panels[cam]["frame"] <= ef


# -----------------------------------------------------------------------------
# (5) wagon intervals hard against the video boundaries
# -----------------------------------------------------------------------------

def test_intervals_at_video_boundaries():
    root = tempfile.mkdtemp()
    # GW_1 starts at t=0 (frame 0); GW_2 ends exactly at the last frame.
    st = GlobalTrainState(
        total_wagons=2, master_camera=C.CAMERA_RIGHT_UP,
        master_fps=25.0, master_total_frames=200,
        wagons=[
            GlobalWagon("GW_1", 1, 0, 99, 0.0, 4.0, C.CLASS_WAGON, 0.9),
            GlobalWagon("GW_2", 2, 100, 199, 4.0, 8.0, C.CLASS_WAGON, 0.9),
        ])
    cache = os.path.join(root, "wagon_cache")
    for gw in st.wagons:
        for cam in C.ALL_CAMERAS:
            sf, ef = _expected_range(cam, gw)
            for fi in range(sf, ef + 1):
                _jpeg(os.path.join(cache, gw.global_id, C.CAMERA_FOLDER[cam],
                                   f"frame_{fi:06d}.jpg"))
    pcf = _tracking_json(root)
    ov = _overview(st, cache, pcf)

    first = ov["GW_1"]["cameras"][C.CAMERA_RIGHT_UP]
    last = ov["GW_2"]["cameras"][C.CAMERA_RIGHT_UP]
    assert first["status"] == EL.OVERVIEW_OK and first["frame"] == 49
    assert last["status"] == EL.OVERVIEW_OK and last["frame"] == 149
    # never a boundary frame while an interior frame exists
    assert first["frame"] not in (0, 99)
    assert last["frame"] not in (100, 199)

    # a degenerate single-frame wagon still yields that one frame
    st1 = GlobalTrainState(
        total_wagons=1, master_fps=25.0, master_total_frames=25,
        wagons=[GlobalWagon("GW_1", 1, 0, 0, 0.0, 0.04, C.CLASS_WAGON, 0.9)])
    c1 = os.path.join(root, "cache_single")
    _jpeg(os.path.join(c1, "GW_1", C.CAMERA_FOLDER[C.CAMERA_RIGHT_UP],
                       "frame_000000.jpg"))
    sel = EL.center_cache_frame(
        cache_root=c1, gw_id="GW_1", camera_id=C.CAMERA_RIGHT_UP,
        wagon_start_time=0.0, wagon_end_time=0.04,
        local_fps=25.0, local_total_frames=25)
    assert sel["status"] == EL.OVERVIEW_OK and sel["frame"] == 0


# -----------------------------------------------------------------------------
# (6) exact centre frame undecodable -> nearest valid, deterministically
# -----------------------------------------------------------------------------

def test_undecodable_center_falls_back_to_nearest_valid():
    _root, st, cache, pcf = _fixture(n=1)
    folder = C.CAMERA_FOLDER[C.CAMERA_RIGHT_UP]
    d = os.path.join(cache, "GW_1", folder)

    # centre (49) truncated, 48 corrupt-but-large, 50 intact
    open(os.path.join(d, "frame_000049.jpg"), "wb").write(b"")
    open(os.path.join(d, "frame_000048.jpg"), "wb").write(b"\x00" * 4096)

    sel = EL.center_cache_frame(
        cache_root=cache, gw_id="GW_1", camera_id=C.CAMERA_RIGHT_UP,
        wagon_start_time=0.0, wagon_end_time=4.0,
        local_fps=25.0, local_total_frames=1000)
    assert sel["status"] == EL.OVERVIEW_OK
    assert sel["target_frame"] == 49
    assert sel["frame"] == 50, sel          # 49 unreadable, 48 corrupt -> 50
    assert sel["frame"] != 49

    # every frame unreadable -> explicit status, no crash, no fabricated image
    for name in sorted(os.listdir(d)):
        open(os.path.join(d, name), "wb").write(b"")
    bad = EL.center_cache_frame(
        cache_root=cache, gw_id="GW_1", camera_id=C.CAMERA_RIGHT_UP,
        wagon_start_time=0.0, wagon_end_time=4.0,
        local_fps=25.0, local_total_frames=1000)
    assert bad["status"] == EL.OVERVIEW_NO_READABLE
    assert bad["path"] is None and bad["frame"] is None


def test_wagon_past_video_end_is_never_illustrated():
    """A camera whose clip was cut short must report OUTSIDE_VIDEO_RANGE, not the
    clamped final frame -- that frame shows an EARLIER wagon.

    Reproduces the real batch: LEFT_UP_TOP ran 3120 frames while the rake ran to
    GW_49.  Stage 2 clamps every past-the-end wagon to frame 3119 and its
    last-write-wins gives that single frame to the last one, so GW_49's folder
    genuinely contains frame 3119 -- of a wagon that passed ~60s earlier.
    """
    root = tempfile.mkdtemp()
    cache = os.path.join(root, "wagon_cache")
    folder = C.CAMERA_FOLDER[C.CAMERA_LEFT_UP_TOP]
    # GW_49 spans 220.0-224.0s; the clip holds only 3120 frames (208.0s)
    _jpeg(os.path.join(cache, "GW_49", folder, "frame_003119.jpg"))

    sel = EL.center_cache_frame(
        cache_root=cache, gw_id="GW_49", camera_id=C.CAMERA_LEFT_UP_TOP,
        wagon_start_time=220.0, wagon_end_time=224.0,
        local_fps=15.0, local_total_frames=3120)
    assert sel["status"] == EL.OVERVIEW_OUTSIDE_VIDEO, sel
    assert sel["path"] is None and sel["frame"] is None
    assert sel["start_frame"] == 3300          # unclamped, so the reason is visible

    # a wagon only PARTIALLY past the end still gets its genuine overlap
    for fi in range(3079, 3120):
        _jpeg(os.path.join(cache, "GW_38", folder, f"frame_{fi:06d}.jpg"))
    ok = EL.center_cache_frame(
        cache_root=cache, gw_id="GW_38", camera_id=C.CAMERA_LEFT_UP_TOP,
        wagon_start_time=3079 / 15.0, wagon_end_time=3144 / 15.0,
        local_fps=15.0, local_total_frames=3120)
    assert ok["status"] == EL.OVERVIEW_OK
    assert 3079 <= ok["frame"] <= 3119, ok

    # and a wagon wholly before the clip starts is refused the same way
    before = EL.center_cache_frame(
        cache_root=cache, gw_id="GW_38", camera_id=C.CAMERA_LEFT_UP_TOP,
        wagon_start_time=-8.0, wagon_end_time=-4.0,
        local_fps=15.0, local_total_frames=3120)
    assert before["status"] == EL.OVERVIEW_OUTSIDE_VIDEO


def test_selection_is_deterministic():
    _root, st, cache, pcf = _fixture()
    a = _overview(st, cache, pcf)
    b = _overview(st, cache, pcf)
    for gw_id in a:
        for cam in LDA.OVERVIEW_CAMERA_ORDER:
            assert a[gw_id]["cameras"][cam]["frame"] == b[gw_id]["cameras"][cam]["frame"]
            assert a[gw_id]["cameras"][cam]["path"] == b[gw_id]["cameras"][cam]["path"]


# -----------------------------------------------------------------------------
# (7) existing feature evidence is preserved, not replaced
# -----------------------------------------------------------------------------

def _plant_evidence(evidence_root: str, gw_id: str) -> None:
    d = os.path.join(evidence_root, gw_id, "door", C.CAMERA_LEFT_UP)
    _jpeg(os.path.join(d, "left_best.jpg"))
    _jpeg(os.path.join(d, "left_snapshot.jpg"))
    dd = os.path.join(evidence_root, gw_id, "damage", C.CAMERA_RIGHT_UP_TOP)
    _jpeg(os.path.join(dd, "track_1.jpg"))
    with open(os.path.join(dd, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump({"tracks": [{"track_idx": 1, "class_name": "floor_damage",
                               "best_confidence": 0.88}]}, f)


def test_feature_evidence_preserved_alongside_overview():
    root, st, cache, pcf = _fixture()
    evidence = os.path.join(root, "evidence")
    _plant_evidence(evidence, "GW_2")

    unified = _unified(st)
    unified["GW_2"].left_door = C.DOOR_OPEN
    unified["GW_2"].left_door_confidence = 0.91
    unified["GW_2"].top_damage = C.DAMAGE_PRESENT

    ov = LDA.build_wagon_overview(
        state=st, unified=unified, cache_root=cache, evidence_root=evidence,
        per_camera_tracking_path=pcf, verbose=False)

    ev_items = ov["GW_2"]["feature_evidence"]
    labels = [e["label"] for e in ev_items]
    assert any("DOOR OPEN" in l for l in labels), labels
    assert any("FLOOR DAMAGE" in l for l in labels), labels
    for e in ev_items:
        assert os.path.isfile(e["path"])

    # LAYER SEPARATION: the LEFT_UP overview panel is the wagon-centre cache
    # frame, NOT the door evidence snapshot -- both are present, neither wins.
    panel = ov["GW_2"]["cameras"][C.CAMERA_LEFT_UP]
    door_ev = next(e for e in ev_items if e["feature"] == "door")
    assert panel["status"] == EL.OVERVIEW_OK
    assert panel["path"] != door_ev["path"]
    assert "wagon_cache" in panel["path"] and "evidence" in door_ev["path"]

    # and the fused findings are carried through verbatim
    f = ov["GW_2"]["findings"]
    assert f["left_door"] == C.DOOR_OPEN
    assert abs(f["left_door_confidence"] - 0.91) < 1e-9
    assert f["top_damage"] == C.DAMAGE_PRESENT

    # the UNCHANGED legacy payload still resolves the same door snapshot path
    payloads = LDA.build_camera_payloads(
        state=st, unified=unified, evidence_root=evidence,
        per_camera_tracking_path=pcf, session_id="B")
    left_doors = payloads[C.CAMERA_LEFT_UP]["doors"]
    assert len(left_doors) == 1 and left_doors[0]["state"] == "OPEN"
    assert left_doors[0]["local_snapshot_path"] == door_ev["path"]


# -----------------------------------------------------------------------------
# (8) Global Wagon ordering is canonical
# -----------------------------------------------------------------------------

def test_global_wagon_ordering_is_canonical():
    _root, st, cache, pcf = _fixture(n=5)
    ov = _overview(st, cache, pcf)

    assert list(ov.keys()) == [f"GW_{i}" for i in range(1, 6)]
    assert [ov[k]["order"] for k in ov] == [1, 2, 3, 4, 5]
    assert [ov[k]["wagon_number"] for k in ov] == [1, 2, 3, 4, 5]

    # the generator sorts by `order`, so even a shuffled/round-tripped dict
    # renders GW_1..GW_5 in Global Train order (never by camera/confidence/name)
    from reporting.combined_report_generator import CombinedReportGenerator
    shuffled = {k: ov[k] for k in sorted(ov, reverse=True)}
    assert list(shuffled.keys())[0] == "GW_5"
    gen = CombinedReportGenerator(output_path=os.path.join(_root, "x.pdf"),
                                  logo_path=None)
    els = gen._create_wagon_overview_pages(shuffled, missing_cameras=[],
                                           camera_order=LDA.OVERVIEW_CAMERA_ORDER)
    texts = _flowable_texts(els)
    titles = [t for t in texts if t.startswith("GLOBAL WAGON:")]
    assert titles == [f"GLOBAL WAGON: GW_{i}" for i in range(1, 6)], titles

    # the index rows are in the same canonical order
    idx_rows = [t for t in texts if t in {f"<b>GW_{i}</b>" for i in range(1, 6)}]
    assert idx_rows == [f"<b>GW_{i}</b>" for i in range(1, 6)], idx_rows


# -----------------------------------------------------------------------------
# (9) a frame from GW_n+1 can never be used for GW_n
# -----------------------------------------------------------------------------

def test_no_frame_leak_from_the_next_wagon():
    _root, st, cache, pcf = _fixture(n=3)
    folder = C.CAMERA_FOLDER[C.CAMERA_RIGHT_UP]

    # plant a stray frame that belongs to GW_2's interval inside GW_1's folder
    stray = 150
    _jpeg(os.path.join(cache, "GW_1", folder, f"frame_{stray:06d}.jpg"))

    ov = _overview(st, cache, pcf)
    gw1, gw2, gw3 = st.wagons
    f1 = ov["GW_1"]["cameras"][C.CAMERA_RIGHT_UP]["frame"]
    f2 = ov["GW_2"]["cameras"][C.CAMERA_RIGHT_UP]["frame"]
    f3 = ov["GW_3"]["cameras"][C.CAMERA_RIGHT_UP]["frame"]

    assert f1 != stray
    s1, e1 = _expected_range(C.CAMERA_RIGHT_UP, gw1)
    s2, e2 = _expected_range(C.CAMERA_RIGHT_UP, gw2)
    s3, e3 = _expected_range(C.CAMERA_RIGHT_UP, gw3)
    assert s1 <= f1 <= e1 and s2 <= f2 <= e2 and s3 <= f3 <= e3
    assert f1 < s2 <= f2 < s3 <= f3          # strictly increasing, no overlap

    # ... and the same holds for every camera on every wagon
    for gw in st.wagons:
        for cam in LDA.OVERVIEW_CAMERA_ORDER:
            sf, ef = _expected_range(cam, gw)
            fr = ov[gw.global_id]["cameras"][cam]["frame"]
            for other in st.wagons:
                if other.global_id == gw.global_id:
                    continue
                osf, oef = _expected_range(cam, other)
                assert not (osf <= fr <= oef), (gw.global_id, cam, fr)


# -----------------------------------------------------------------------------
# gap frames are avoided when Stage 1 reported one
# -----------------------------------------------------------------------------

def test_gap_frames_are_avoided():
    _root, st, cache, pcf = _fixture(n=1)
    # a Stage-1 gap sitting right on the wagon centre (read-only use of the
    # existing gap list; no gap detection is performed here)
    sel = EL.center_cache_frame(
        cache_root=cache, gw_id="GW_1", camera_id=C.CAMERA_RIGHT_UP,
        wagon_start_time=0.0, wagon_end_time=4.0,
        local_fps=25.0, local_total_frames=1000,
        gaps=[{"start_frame": 45, "end_frame": 55}])
    assert sel["status"] == EL.OVERVIEW_OK
    assert sel["gap_filtered"] is True
    assert not (45 <= sel["frame"] <= 55)
    assert sel["frame"] in (44, 56)


# -----------------------------------------------------------------------------
# end-to-end: PDF gains wagon pages, keeps every existing finding
# -----------------------------------------------------------------------------

def _extract_pdf_text(path: str) -> str:
    from PyPDF2 import PdfReader
    return "\n".join((p.extract_text() or "") for p in PdfReader(path).pages)


def _page_count(path: str) -> int:
    from PyPDF2 import PdfReader
    return len(PdfReader(path).pages)


def test_end_to_end_pdf_and_json():
    from reporting import combined_train_report as CTR

    root, st, cache, pcf = _fixture(n=3)
    evidence = os.path.join(root, "evidence")
    _plant_evidence(evidence, "GW_2")

    unified = _unified(st)
    unified["GW_2"].left_door = C.DOOR_OPEN
    unified["GW_2"].left_door_confidence = 0.91
    unified["GW_2"].top_damage = C.DAMAGE_PRESENT
    unified["GW_3"].right_door = C.DOOR_CLOSED
    unified["GW_3"].right_door_confidence = 0.77

    out = os.path.join(root, "reports")
    res = CTR.build(
        state=st, unified=unified, output_dir=out, batch_key="B1",
        evidence_root=evidence, cache_root=cache,
        per_camera_tracking_path=pcf, verbose=False)

    assert res["pdf_path"] and os.path.isfile(res["pdf_path"])
    text = _extract_pdf_text(res["pdf_path"])

    # --- pre-existing sections all still present ---
    assert "COMBINED WAGON EYE REPORT" in text
    assert "WAGON INSPECTION DETAILS" in text
    assert "OPEN" in text                       # GW_2 open door finding
    assert "Damaged Wagon Report" in text       # existing evidence section

    # --- the new section ---
    assert "WAGON-BY-WAGON VISUAL INSPECTION" in text
    for i in (1, 2, 3):
        assert f"GLOBAL WAGON: GW_{i}" in text
    assert "ENGINE" in text                     # GW_1 classification
    for cam in LDA.OVERVIEW_CAMERA_ORDER:
        assert cam in text
    assert "FEATURE EVIDENCE" in text           # layer-2 strip on GW_2

    # --- JSON companion ---
    with open(res["json_path"], encoding="utf-8") as f:
        doc = json.load(f)
    ov = doc["wagon_overview"]
    assert [ov[k]["order"] for k in ov] == [1, 2, 3]
    assert doc["wagon_overview_camera_order"] == list(LDA.OVERVIEW_CAMERA_ORDER)
    assert all(ov[k]["cameras"][c]["status"] == "OK"
               for k in ov for c in LDA.OVERVIEW_CAMERA_ORDER)
    # existing JSON keys untouched
    assert doc["total_wagons"] == 3 and len(doc["wagons"]) == 3
    assert set(doc["cameras"]) == set(C.ALL_CAMERAS)


def test_wagon_pages_in_rendered_pdf():
    """In the RENDERED PDF, GW_n's page carries GW_n's own four frames -- in the
    fixed camera order, one page per wagon, and never a neighbour's frame."""
    import re
    from PyPDF2 import PdfReader
    from reportlab.platypus import SimpleDocTemplate
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.units import inch
    from reporting.combined_report_generator import CombinedReportGenerator

    root, st, cache, pcf = _fixture(n=3)
    ov = _overview(st, cache, pcf)

    pdf = os.path.join(root, "pages.pdf")
    gen = CombinedReportGenerator(output_path=pdf, logo_path=None)
    doc = SimpleDocTemplate(pdf, pagesize=landscape(A4),
                            rightMargin=0.5 * inch, leftMargin=0.5 * inch,
                            topMargin=0.5 * inch, bottomMargin=0.5 * inch)
    doc.build(gen._create_wagon_overview_pages(
        ov, missing_cameras=[], camera_order=LDA.OVERVIEW_CAMERA_ORDER))

    pages = [(p.extract_text() or "") for p in PdfReader(pdf).pages]
    # the section opens with a PageBreak (absorbed in the real report, where it
    # simply closes the Damaged Wagon Report page) -- drop it when built alone
    while pages and not pages[0].strip():
        pages.pop(0)
    # 1 index page + one page per wagon -- no wagon spills onto a second page
    assert len(pages) == 1 + len(st.wagons), len(pages)

    for i, gw in enumerate(st.wagons, start=1):
        page = pages[i]
        assert f"GLOBAL WAGON: {gw.global_id}" in page
        # camera labels appear in the one fixed order, on every page
        pos = [page.index(c) for c in LDA.OVERVIEW_CAMERA_ORDER]
        assert pos == sorted(pos), (gw.global_id, pos)
        # the four stamped source frames are exactly this wagon's selections
        stamped = {int(m) for m in re.findall(r"source frame (\d+)", page)}
        expected = {ov[gw.global_id]["cameras"][c]["frame"]
                    for c in LDA.OVERVIEW_CAMERA_ORDER}
        assert stamped == expected, (gw.global_id, stamped, expected)
        # and not one frame belonging to any other wagon
        for other in st.wagons:
            if other.global_id == gw.global_id:
                continue
            other_frames = {ov[other.global_id]["cameras"][c]["frame"]
                            for c in LDA.OVERVIEW_CAMERA_ORDER}
            assert not (stamped & other_frames), (gw.global_id, other.global_id)


def test_no_overview_payload_is_a_noop():
    """No cache_root -> no wagon pages, and the PDF is the pre-existing report."""
    from reporting import combined_train_report as CTR
    from reporting.combined_report_generator import CombinedReportGenerator

    root, st, cache, pcf = _fixture(n=2)
    unified = _unified(st)

    with_ov = CTR.build(state=st, unified=unified,
                        output_dir=os.path.join(root, "r_with"),
                        batch_key="B", cache_root=cache,
                        per_camera_tracking_path=pcf, verbose=False)
    without = CTR.build(state=st, unified=unified,
                        output_dir=os.path.join(root, "r_without"),
                        batch_key="B", cache_root=None, verbose=False)

    t_with = _extract_pdf_text(with_ov["pdf_path"])
    t_without = _extract_pdf_text(without["pdf_path"])
    assert "GLOBAL WAGON: GW_1" in t_with
    assert "WAGON-BY-WAGON VISUAL INSPECTION" not in t_without
    assert "GLOBAL WAGON:" not in t_without
    assert "COMBINED WAGON EYE REPORT" in t_without   # existing report intact
    assert _page_count(with_ov["pdf_path"]) > _page_count(without["pdf_path"])

    # an explicitly empty payload adds nothing at all
    gen = CombinedReportGenerator(output_path=os.path.join(root, "z.pdf"),
                                  logo_path=None)
    assert gen._create_wagon_overview_pages({}, missing_cameras=[]) == []
    assert gen._create_wagon_overview_pages(None, missing_cameras=[]) == []


def test_zero_wagons_and_missing_cache():
    from reporting import combined_train_report as CTR

    root = tempfile.mkdtemp()
    empty = GlobalTrainState(total_wagons=0, wagons=[], master_fps=25.0)
    res = CTR.build(state=empty, unified={}, output_dir=os.path.join(root, "r0"),
                    batch_key="B0", cache_root=os.path.join(root, "nope"),
                    verbose=False)
    assert res["json_path"] and os.path.isfile(res["json_path"])
    with open(res["json_path"], encoding="utf-8") as f:
        assert json.load(f)["wagon_overview"] == {}

    # wagons present but the cache was pruned -> placeholders everywhere, and
    # the report still builds
    st = _state(2)
    ov = LDA.build_wagon_overview(
        state=st, unified=_unified(st),
        cache_root=os.path.join(root, "pruned"),
        per_camera_tracking_path=None, verbose=False)
    assert len(ov) == 2
    for gw_id in ov:
        for cam in LDA.OVERVIEW_CAMERA_ORDER:
            assert ov[gw_id]["cameras"][cam]["status"] == EL.OVERVIEW_NO_FRAMES
            assert ov[gw_id]["cameras"][cam]["path"] is None

    res2 = CTR.build(state=st, unified=_unified(st),
                     output_dir=os.path.join(root, "r2"), batch_key="B2",
                     cache_root=os.path.join(root, "pruned"), verbose=False)
    assert res2["pdf_path"] and os.path.isfile(res2["pdf_path"])
    t = _extract_pdf_text(res2["pdf_path"])
    assert "NO FRAME AVAILABLE" in t
    assert "GLOBAL WAGON: GW_1" in t


# -----------------------------------------------------------------------------
# camera-wise report inputs are untouched by this change
# -----------------------------------------------------------------------------

def test_camera_payloads_unchanged_by_overview():
    root, st, cache, pcf = _fixture(n=2)
    evidence = os.path.join(root, "evidence")
    _plant_evidence(evidence, "GW_1")
    unified = _unified(st)
    unified["GW_1"].left_door = C.DOOR_OPEN

    before = LDA.build_camera_payloads(
        state=st, unified=unified, evidence_root=evidence,
        per_camera_tracking_path=pcf, session_id="B")
    LDA.build_wagon_overview(state=st, unified=unified, cache_root=cache,
                             evidence_root=evidence,
                             per_camera_tracking_path=pcf, verbose=False)
    after = LDA.build_camera_payloads(
        state=st, unified=unified, evidence_root=evidence,
        per_camera_tracking_path=pcf, session_id="B")

    def _cmp(p):
        return {c: {"wagon_summary": [
            {k: v for k, v in w.items() if k != "ocr_wagon_number"}
            for w in d["wagon_summary"]],
            "doors": [{k: v for k, v in x.items() if k != "snapshot"}
                      for x in d.get("doors", [])],
            "damages": [{k: v for k, v in x.items() if k != "snapshot"}
                        for x in d.get("damages", [])],
            "state_counts": d["state_counts"]} for c, d in p.items()}

    assert _cmp(before) == _cmp(after)
    assert before[C.CAMERA_LEFT_UP]["doors"][0]["state"] == "OPEN"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS  {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
