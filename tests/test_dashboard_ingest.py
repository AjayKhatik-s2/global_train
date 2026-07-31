"""Tests for delivery/dashboard_ingest.py -- the read-only legacy dashboard adapter.

Covers: per-camera authority, legacy wrapper/schema, operational-day date-folder,
missing evidence, idempotent restart, retry behaviour, disabled-by-default, and
the "no writes outside delivery/" guarantee.

No models, no network, no real S3 -- a fake s3 client and a fake requests module
are injected.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from delivery import dashboard_ingest as DI
from delivery import finalization as FIN
from core import constants as C


# -----------------------------------------------------------------------------
# fakes
# -----------------------------------------------------------------------------

class FakeS3:
    def __init__(self):
        self.uploads = []

    def upload_file(self, local, bucket, key, ExtraArgs=None):
        self.uploads.append((bucket, key))


class FakeResp:
    def __init__(self, code, body=None, text=""):
        self.status_code = code
        self._body = body or {}
        self.text = text

    def json(self):
        return self._body


class FakeRequests:
    """Returns queued responses in order; records calls."""
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        return self._responses.pop(0) if self._responses else FakeResp(200, {"run_id": "R"})


# -----------------------------------------------------------------------------
# fixture batch
# -----------------------------------------------------------------------------

def _write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def _touch_jpg(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\xff\xd8\xff\xd9")  # tiny fake JPEG


def make_batch(root):
    """Build a finalized-artifact batch: ENGINE + 2 WAGONs, camera-scoped state.

    Writes everything the exact-V4 builder reads: the sealed GlobalTrainState, the
    fused unified states, each camera's own per-feature JSON, and evidence.
    """
    classes = [C.CLASS_ENGINE, C.CLASS_WAGON, C.CLASS_WAGON]
    wagons_state = [
        {"global_id": f"GW_{i}", "wagon_index": i,
         "start_frame_master": (i - 1) * 100, "end_frame_master": i * 100 - 1,
         "start_time": (i - 1) * 4.0, "end_time": i * 4.0,
         "classification": cls, "classification_confidence": 0.9,
         "supporting_cameras": ["RIGHT_UP", "LEFT_UP", "RIGHT_UP_TOP"]}
        for i, cls in enumerate(classes, start=1)
    ]
    _write(os.path.join(root, "global_state", "global_train_state.json"), {
        "schema": "wagon_eye.global_train_state.v1",
        "master_camera": "RIGHT_UP", "master_fps": 25.0,
        "master_total_frames": 300, "total_wagons": 3,
        "wagons": wagons_state, "travel_direction": "left-to-right",
    })

    ws = os.path.join(root, "wagon_states")
    # RIGHT_UP doors: GW_2 OPEN, GW_3 CLOSED.  LEFT_UP: all CLOSED.
    right = {"GW_2": C.DOOR_OPEN}
    for i in range(1, 4):
        gw = f"GW_{i}"
        _write(os.path.join(ws, "door", "RIGHT_UP", f"{gw}.json"),
               {"status": C.STATUS_OK, "right_door": right.get(gw, C.DOOR_CLOSED)})
        _write(os.path.join(ws, "door", "LEFT_UP", f"{gw}.json"),
               {"status": C.STATUS_OK, "left_door": C.DOOR_CLOSED})
        _write(os.path.join(ws, "load", "RIGHT_UP_TOP", f"{gw}.json"),
               {"status": C.STATUS_OK,
                "load_status": C.LOAD_LOADED if gw == "GW_2" else C.LOAD_EMPTY})
    # RIGHT_UP_TOP damage on GW_2 only (floor)
    _write(os.path.join(ws, "damage", "RIGHT_UP_TOP", "GW_2.json"),
           {"status": C.STATUS_OK, "damage_status": C.DAMAGE_PRESENT,
            "top_damage_details": [{"class_name": "floor_damage",
                                    "confidence": 0.61}]})
    # OCR (RIGHT_UP authority): GW_2 valid, GW_3 an invalid raw read
    _write(os.path.join(ws, "ocr", "RIGHT_UP", "GW_2.json"),
           {"status": C.STATUS_OK, "engine": "rekognition",
            "wagon_identifier": "12345678901",
            "wagon_identifier_confidence": 0.9, "raw_number": "12345678901",
            "display_number": "12345678901", "is_valid_11_digit": True,
            "fallback_triggered": False})
    _write(os.path.join(ws, "ocr", "RIGHT_UP", "GW_3.json"),
           {"status": C.STATUS_OK, "engine": "rekognition",
            "wagon_identifier": C.NO_DATA, "wagon_identifier_confidence": 0.0,
            "raw_number": "1234", "display_number": "-",
            "is_valid_11_digit": False, "fallback_triggered": True})
    # fused unified states
    for i, cls in enumerate(classes, start=1):
        gw = f"GW_{i}"
        _write(os.path.join(ws, "unified", f"{gw}.json"), {
            "global_id": gw, "wagon_index": i, "classification": cls,
            "right_door": right.get(gw, C.DOOR_CLOSED),
            "left_door": C.DOOR_CLOSED,
            "load_status": C.LOAD_LOADED if gw == "GW_2" else C.LOAD_EMPTY,
            "top_damage": C.DAMAGE_PRESENT if gw == "GW_2" else C.DAMAGE_OK,
            "wagon_identifier": "12345678901" if gw == "GW_2" else C.NO_DATA,
        })

    report = {
        "schema": "wagon_eye.combined_report/1",
        "batch_key": "20260408_032134",
        "master_camera": "RIGHT_UP",
        "total_wagons": 3,
        "travel_direction": "left-to-right",
        "source_video_urls": {
            "RIGHT_UP": "s3://in/right_up_20260408_032134.mp4",
            "LEFT_UP": "s3://in/left_up_20260408_032134.mp4",
            "RIGHT_UP_TOP": "s3://in/right_up_top_20260408_032134.mp4",
        },
        "processed_video_urls": {"RIGHT_UP": "https://x/right_processed.mp4"},
        "summary": {"total_wagons": 3, "engine_count": 1, "wagon_count": 2,
                    "loaded": 1, "empty": 1},
        "wagons": [json.loads(json.dumps(w)) for w in [
            {"global_id": "GW_1", "wagon_index": 1, "classification": C.CLASS_ENGINE,
             "wagon_identifier": C.NO_DATA, "top_damage": C.DAMAGE_OK},
            {"global_id": "GW_2", "wagon_index": 2, "classification": C.CLASS_WAGON,
             "wagon_identifier": "12345678901", "top_damage": C.DAMAGE_PRESENT},
            {"global_id": "GW_3", "wagon_index": 3, "classification": C.CLASS_WAGON,
             "wagon_identifier": C.NO_DATA, "top_damage": C.DAMAGE_OK},
        ]],
        "report_meta": {"report_revision": 0, "report_status": "FINAL",
                        "cameras_present": ["RIGHT_UP", "LEFT_UP", "RIGHT_UP_TOP"],
                        "cameras_missing_final": [],
                        "generated_from_global_state_version": "deadbeef"},
    }
    _write(os.path.join(root, "reports", "combined_train_report.json"), report)

    ev = os.path.join(root, "evidence")
    # GW_2 RIGHT_UP door (open) + ocr sheet
    _write(os.path.join(ev, "GW_2", "door", "RIGHT_UP", "metadata.json"),
           {"sides": {"right": {"bbox": [10, 20, 110, 220], "state": "OPEN",
                                "frame_idx": 137, "confidence": 0.83}}})
    _touch_jpg(os.path.join(ev, "GW_2", "door", "RIGHT_UP", "right_best.jpg"))
    _write(os.path.join(ev, "GW_2", "ocr", "RIGHT_UP", "metadata.json"),
           {"full_number": "12345678901", "ocr_confidence": 0.9,
            "engine": "rekognition"})
    _touch_jpg(os.path.join(ev, "GW_2", "ocr", "RIGHT_UP", "best_frame.jpg"))
    _touch_jpg(os.path.join(ev, "GW_2", "ocr", "RIGHT_UP", "ocr_sheet.jpg"))
    # GW_2 LEFT_UP door (closed)
    _write(os.path.join(ev, "GW_2", "door", "LEFT_UP", "metadata.json"),
           {"sides": {"left": {"bbox": [1, 2, 3, 4], "state": "CLOSED",
                               "confidence": 0.7}}})
    # GW_2 TOP load + damage
    _write(os.path.join(ev, "GW_2", "load", "RIGHT_UP_TOP", "metadata.json"),
           {"load_status": "LOADED", "confidence": 0.77})
    _touch_jpg(os.path.join(ev, "GW_2", "load", "RIGHT_UP_TOP", "best_frame.jpg"))
    _write(os.path.join(ev, "GW_2", "damage", "RIGHT_UP_TOP", "metadata.json"),
           {"damage_status": "DAMAGE",
            "tracks": [{"track_idx": 1, "class_name": "floor_damage",
                        "bbox": [5, 6, 55, 66], "frame_idx": 160,
                        "best_confidence": 0.61}]})
    _touch_jpg(os.path.join(ev, "GW_2", "damage", "RIGHT_UP_TOP", "track_1.jpg"))

    # finalization marker already written by stage_finalize
    FIN.write(root, {"batch_key": "20260408_032134", "report_revision": 0,
                     "uploaded": True, "email_sent": True,
                     "upload_urls": {"pdf": "https://x/combined.pdf",
                                     "camera_RIGHT_UP": "https://x/right.pdf"}})
    return report
    return report


def _all_files(root):
    out = {}
    for dp, _, fns in os.walk(root):
        for fn in fns:
            p = os.path.join(dp, fn)
            out[os.path.relpath(p, root).replace(os.sep, "/")] = os.path.getmtime(p)
    return out


# -----------------------------------------------------------------------------
# date-folder (operational-day 05:00 IST rule)
# -----------------------------------------------------------------------------

def test_date_folder_operational_day_rule():
    assert DI.extract_train_timestamp("x_20260408_032134.mp4") == datetime(2026, 4, 8, 3, 21, 34)
    # 03:21 is before 05:00 -> previous day
    assert DI.date_folder(datetime(2026, 4, 8, 3, 21, 34)) == "2026-04-07"
    # 13:34 is after 05:00 -> same day
    assert DI.date_folder(datetime(2026, 4, 8, 13, 34, 56)) == "2026-04-08"
    # exactly 05:00 -> same day
    assert DI.date_folder(datetime(2026, 4, 8, 5, 0, 0)) == "2026-04-08"


# -----------------------------------------------------------------------------
# per-camera authority + legacy wrapper/schema
# -----------------------------------------------------------------------------

def _url_maker(root, camera):
    return DI._UrlMaker(
        s3_client=None, output_bucket=C.S3_OUTPUT_BUCKET, region=C.S3_REGION,
        inspection_bucket="ankit-version-1-prod", batch_key="20260408_032134",
        folder=DI.folder_for(camera), date_folder_str="2026-04-07",
        reuse=True, skip_upload=True)


def _build(root, camera):
    """Build one per-camera document from the fixture batch, in whatever dialect
    the ambient WAGONEYE_INSPECTION_VERSION selects (default: v1)."""
    with open(os.path.join(root, "reports", "combined_train_report.json"),
              encoding="utf-8") as f:
        report = json.load(f)
    return DI.build_inspection_json(camera=camera, batch_root=root,
                                    report_doc=report,
                                    url_maker=_url_maker(root, camera))


@pytest.fixture
def v4(monkeypatch):
    """Force the V4 dialect (the default is v1, for the live V1 dashboard)."""
    monkeypatch.setenv("WAGONEYE_INSPECTION_VERSION", "v4")
    monkeypatch.delenv("WAGONEYE_INSPECTION_STRIP_CAMERA_PREFIX", raising=False)


# V4 inspection_data key ORDER, per flavour (reporting/json_builder.py).
V4_SIDE_KEYS = ["raw_video_name", "identified_by", "upload_timestamp",
                "upload_timestamp_readable", "direction", "rake_status",
                "pdf_report_url", "trimmed_video_url", "detected_video_url",
                "raw_video_urls", "total_wagons", "doors_open",
                "doors_partially_closed", "doors_closed", "damaged_wagons",
                "num_engines", "total_loco_frames", "total_problem_frames",
                "problem_frames_by_type", "wagon_number_results",
                "loco_number_results", "segment_type_map", "wagon_segments",
                "loco_frames", "problem_frames", "damage_model_active"]
V4_TOP_KEYS = ["raw_video_name", "identified_by", "upload_timestamp",
               "upload_timestamp_readable", "direction", "rake_status",
               "pdf_report_url", "trimmed_video_url", "detected_video_url",
               "raw_video_urls", "total_wagons", "wagons_loaded", "wagons_empty",
               "damaged_wagons", "probable_damage_wagons", "floor_dmg_wagons",
               "inner_wall_dmg_wagons", "floor_dmg_probable_wagons",
               "num_engines", "num_brakevans", "total_loco_frames",
               "total_problem_frames", "problem_frames_by_type",
               "wagon_number_results", "loco_number_results",
               "segment_type_map", "wagon_segments", "loco_frames",
               "problem_frames", "damage_model_active"]


def test_envelope_and_v4_key_order_side(tmp_path, v4):
    root = str(tmp_path)
    make_batch(root)
    doc = _build(root, "RIGHT_UP")
    assert set(doc.keys()) == {"camera_id", "version", "inspection_data"}
    keys = [k for k in doc["inspection_data"] if k != "_adapter"]
    assert keys == V4_SIDE_KEYS


def test_camera_id_form_follows_the_version(tmp_path, monkeypatch):
    """version and camera_id must agree, or the dashboard cannot match the
    document to a camera.  v1 (default) keeps the identifier the live feed has
    always used; v4 emits the V4-exact stripped form."""
    root = str(tmp_path)
    make_batch(root)
    monkeypatch.delenv("WAGONEYE_INSPECTION_STRIP_CAMERA_PREFIX", raising=False)

    monkeypatch.delenv("WAGONEYE_INSPECTION_VERSION", raising=False)
    doc = _build(root, "RIGHT_UP")
    assert doc["version"] == "v1"
    assert doc["camera_id"] == "camera_CCTV_HZBN_DHN_2_RIGHT_UP"

    monkeypatch.setenv("WAGONEYE_INSPECTION_VERSION", "v4")
    doc = _build(root, "RIGHT_UP")
    assert doc["version"] == "v4"
    assert doc["camera_id"] == "CCTV_HZBN_DHN_2_RIGHT_UP"


def test_camera_prefix_can_be_pinned_explicitly(tmp_path, monkeypatch):
    root = str(tmp_path)
    make_batch(root)
    monkeypatch.setenv("WAGONEYE_INSPECTION_STRIP_CAMERA_PREFIX", "true")
    assert _build(root, "RIGHT_UP")["camera_id"] == "CCTV_HZBN_DHN_2_RIGHT_UP"


def test_v4_key_order_top(tmp_path, v4):
    root = str(tmp_path)
    make_batch(root)
    keys = [k for k in _build(root, "RIGHT_UP_TOP")["inspection_data"]
            if k != "_adapter"]
    assert keys == V4_TOP_KEYS


def test_side_flavour_doors_and_ocr(tmp_path, v4):
    root = str(tmp_path)
    make_batch(root)
    d = _build(root, "RIGHT_UP")["inspection_data"]
    # the ENGINE segment is excluded from wagon_segments and counted separately
    assert d["total_wagons"] == 2
    assert d["num_engines"] == 1
    assert len(d["wagon_segments"]) == 2
    assert {s["segment_type"] for s in d["wagon_segments"]} == {"wagon"}
    assert d["doors_open"] == 1                     # GW_2 right OPEN
    assert d["doors_closed"] == 1
    assert d["doors_partially_closed"] == 0
    # wagon_count restarts at 1 on the first WAGON (the engine takes none)
    assert [s["wagon_count"] for s in d["wagon_segments"]] == [1, 2]
    # OCR keyed by str(wagon_count); the invalid read is still emitted
    assert d["wagon_number_results"]["1"]["is_valid_11_digit"] is True
    assert d["wagon_number_results"]["1"]["display_number"] == "12345678901"
    assert d["wagon_number_results"]["2"]["is_valid_11_digit"] is False
    assert d["wagon_number_results"]["2"]["original_number"] == "1234"
    assert d["wagon_number_results"]["2"]["is_manipulated"] is True
    # the URL points at the exact image sent to Rekognition
    assert "ocr_sheet.jpg" in d["wagon_number_results"]["1"]["ocr_frame_s3_url"]
    # one open_door problem frame carrying the bbox + frame from door metadata
    pf = [p for p in d["problem_frames"] if p["problem_type"] == "open_door"]
    assert len(pf) == 1
    assert pf[0]["bounding_box"] == [10, 20, 110, 220]
    assert pf[0]["frame_number"] == 137
    assert pf[0]["segment_number"] is None          # V4 side leaves this null
    assert "evidence/GW_2/door/RIGHT_UP/right_best.jpg" in pf[0]["s3_url"]
    assert d["_adapter"]["camera_authority"] == "right_door+ocr+classification"


def test_side_rake_status_and_direction_from_stage1(tmp_path, v4):
    root = str(tmp_path)
    make_batch(root)
    d = _build(root, "RIGHT_UP")["inspection_data"]
    # Stage 1 persists the direction; the side flavour derives rake_status from it
    assert d["direction"] == "left-to-right"
    assert d["rake_status"] == "Loaded"
    assert d["_adapter"]["direction_estimator"] == "stage1_gap_centre_x_drift"


def test_left_up_authority_isolated(tmp_path):
    root = str(tmp_path)
    make_batch(root)
    d = _build(root, "LEFT_UP")["inspection_data"]
    assert d["doors_open"] == 0                     # both left doors CLOSED
    assert d["problem_frames"] == []
    # LEFT_UP has no OCR authority -- it never claims a wagon number
    assert d["wagon_number_results"] == {}
    assert all("wagon_number" not in s for s in d["wagon_segments"])
    assert d["_adapter"]["camera_authority"] == "left_door"


def test_top_flavour_load_split_and_damage(tmp_path, v4):
    root = str(tmp_path)
    make_batch(root)
    d = _build(root, "RIGHT_UP_TOP")["inspection_data"]
    # top flavour splits wagons by load class and votes rake_status from that
    assert (d["wagons_loaded"], d["wagons_empty"]) == (1, 1)
    assert d["rake_status"] == "Loaded"              # loaded >= empty
    assert {s["segment_type"] for s in d["wagon_segments"]} == {"wagon_loaded",
                                                               "wagon_empty"}
    assert [s["load_status"] for s in d["wagon_segments"]] == ["loaded", "empty"]
    assert d["floor_dmg_wagons"] == 1
    assert d["inner_wall_dmg_wagons"] == 0
    assert d["damaged_wagons"] == 1
    # no "probable" class exists in damage.pt -> always 0, never invented
    assert d["floor_dmg_probable_wagons"] == 0
    assert d["probable_damage_wagons"] == 0
    assert d["num_brakevans"] == 0
    # segment_type_map carries wagon_count on the top flavour only
    assert "wagon_count" in d["segment_type_map"]["2"]
    pf = d["problem_frames"]
    assert len(pf) == 1 and pf[0]["problem_type"] == "floor_dmg"
    assert pf[0]["bounding_box"] == [5, 6, 55, 66]
    assert pf[0]["load_status"] == "loaded"
    assert d["_adapter"]["camera_authority"] == "load(primary)+top_damage"


def test_top_camera_sees_only_its_own_damage(tmp_path):
    root = str(tmp_path)
    make_batch(root)
    # LEFT_UP_TOP has no damage JSON of its own -> reports none, not RIGHT_UP_TOP's
    d = _build(root, "LEFT_UP_TOP")["inspection_data"]
    assert d["floor_dmg_wagons"] == 0
    assert d["damaged_wagons"] == 0
    assert d["problem_frames"] == []


def test_segment_type_map_v4_vocabulary(tmp_path):
    root = str(tmp_path)
    make_batch(root)
    side = _build(root, "RIGHT_UP")["inspection_data"]["segment_type_map"]
    top = _build(root, "RIGHT_UP_TOP")["inspection_data"]["segment_type_map"]
    # keyed by str(segment_id), covering EVERY segment incl. the engine
    assert set(side) == {"1", "2", "3"}
    assert side["1"] == {"type": "engine", "number": 1}
    assert side["2"]["type"] == "wagon"              # side collapses load classes
    assert top["2"]["type"] == "wagon_loaded"        # top splits them
    assert top["3"]["type"] == "wagon_empty"


def test_all_cameras_describe_the_same_wagon_sequence(tmp_path):
    root = str(tmp_path)
    make_batch(root)
    seqs = set()
    maps = set()
    for cam in C.ALL_CAMERAS:
        d = _build(root, cam)["inspection_data"]
        seqs.add(tuple(s["wagon_count"] for s in d["wagon_segments"]))
        maps.add(tuple(sorted(d["segment_type_map"])))
    # the sealed GlobalTrainState is the single source of the wagon sequence
    assert len(seqs) == 1 and len(maps) == 1


def test_fields_without_a_source_are_empty_not_invented(tmp_path, v4):
    root = str(tmp_path)
    make_batch(root)
    d = _build(root, "RIGHT_UP")["inspection_data"]
    # The fixture's ENGINE has no loco OCR result and no loco evidence, so no
    # loco block is fabricated for it (see test_loco_* for the populated case).
    assert d["loco_frames"] == [] and d["loco_number_results"] == {}
    assert d["total_loco_frames"] == 0
    assert d["damage_model_active"] is True


def test_disabled_damage_feature_marks_model_inactive(tmp_path, v4):
    root = str(tmp_path)
    make_batch(root)
    p = os.path.join(root, "reports", "combined_train_report.json")
    with open(p, encoding="utf-8") as f:
        report = json.load(f)
    for w in report["wagons"]:
        w["top_damage"] = C.DISABLED_DISPLAY
    _write(p, report)
    d = _build(root, "RIGHT_UP_TOP")["inspection_data"]
    assert d["damage_model_active"] is False


def test_missing_evidence_graceful(tmp_path, v4):
    root = str(tmp_path)
    make_batch(root)
    import shutil
    shutil.rmtree(os.path.join(root, "evidence"))
    d = _build(root, "RIGHT_UP")["inspection_data"]
    # door state comes from wagon_states, so the counts survive
    assert d["doors_open"] == 1
    for seg in d["wagon_segments"]:
        assert seg["wagon_frames"] == []            # no invented urls
    pf = [p for p in d["problem_frames"] if p["problem_type"] == "open_door"]
    assert pf and pf[0]["bounding_box"] is None     # metadata absent
    assert pf[0]["s3_url"] is None
    assert pf[0]["is_annotated"] is False


def test_camera_prefix_kept_when_opted_out(tmp_path, monkeypatch):
    root = str(tmp_path)
    make_batch(root)
    monkeypatch.setenv("WAGONEYE_INSPECTION_STRIP_CAMERA_PREFIX", "false")
    assert _build(root, "RIGHT_UP")["camera_id"] == "camera_CCTV_HZBN_DHN_2_RIGHT_UP"


def test_no_global_state_is_reported_not_raised(tmp_path):
    root = str(tmp_path)
    make_batch(root)
    os.remove(os.path.join(root, "global_state", "global_train_state.json"))
    res = DI.run(batch_root=root, s3_client=None, skip_upload=True)
    assert res["cameras"] == {}
    assert "no_global_state" in res.get("error", "")


# -----------------------------------------------------------------------------
# disabled-by-default
# -----------------------------------------------------------------------------

def test_enabled_by_default(monkeypatch):
    # default is now ON
    monkeypatch.delenv("WAGONEYE_DASHBOARD_INGEST_ENABLED", raising=False)
    assert DI.is_enabled() is True


def test_disable_via_env_is_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("WAGONEYE_DASHBOARD_INGEST_ENABLED", "false")
    root = str(tmp_path)
    make_batch(root)
    before = _all_files(root)
    res = DI.run(batch_root=root, s3_client=FakeS3())
    assert res == {"enabled": False, "cameras": {}}
    assert _all_files(root) == before          # nothing written


# -----------------------------------------------------------------------------
# full run + idempotent restart
# -----------------------------------------------------------------------------

def test_run_ingests_each_present_camera_then_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("WAGONEYE_DASHBOARD_INGEST_ENABLED", "true")
    root = str(tmp_path)
    make_batch(root)
    s3 = FakeS3()
    rq = FakeRequests([FakeResp(200, {"run_id": "R1"}),
                       FakeResp(200, {"run_id": "R2"}),
                       FakeResp(200, {"run_id": "R3"})])
    res = DI.run(batch_root=root, s3_client=s3, requests_mod=rq)
    assert res["enabled"] is True
    assert set(res["cameras"]) == {"RIGHT_UP", "LEFT_UP", "RIGHT_UP_TOP"}
    assert all(v["status"] == "ingested" for v in res["cameras"].values())
    assert len(rq.calls) == 3
    # 3 JSON uploads to the inspection bucket
    assert sum(1 for b, k in s3.uploads if b == "ankit-version-1-prod") == 3

    # ---- restart: same artifacts -> no re-upload, no re-POST ----
    rq2 = FakeRequests([])
    s3b = FakeS3()
    res2 = DI.run(batch_root=root, s3_client=s3b, requests_mod=rq2)
    assert all(v["status"] == "already_ingested" for v in res2["cameras"].values())
    assert len(rq2.calls) == 0
    assert s3b.uploads == []

    # marker recorded per-camera dashboard status
    marker = FIN.load(root)
    assert set(marker["dashboard_ingested"]) == {"RIGHT_UP", "LEFT_UP", "RIGHT_UP_TOP"}
    assert marker["dashboard_ingested"]["RIGHT_UP"]["run_id"] == "R1"
    # existing finalization fields preserved
    assert marker["email_sent"] is True and marker["uploaded"] is True


def test_new_revision_reingests(tmp_path, monkeypatch):
    monkeypatch.setenv("WAGONEYE_DASHBOARD_INGEST_ENABLED", "true")
    root = str(tmp_path)
    report = make_batch(root)
    DI.run(batch_root=root, s3_client=FakeS3(),
           requests_mod=FakeRequests([FakeResp(200, {"run_id": "R1"})] * 3))
    # bump revision -> different json hash -> re-ingest
    report["report_meta"]["report_revision"] = 1
    _write(os.path.join(root, "reports", "combined_train_report.json"), report)
    rq = FakeRequests([FakeResp(200, {"run_id": "R9"})] * 3)
    res = DI.run(batch_root=root, s3_client=FakeS3(), requests_mod=rq)
    assert all(v["status"] == "ingested" for v in res["cameras"].values())
    assert len(rq.calls) == 3


# -----------------------------------------------------------------------------
# retry behaviour
# -----------------------------------------------------------------------------

def test_post_retries_on_5xx_then_succeeds(monkeypatch):
    monkeypatch.setattr(DI.time, "sleep", lambda *_: None)
    rq = FakeRequests([FakeResp(503), FakeResp(200, {"run_id": "OK"})])
    out = DI._post_ingest(api_url="http://x", payload={}, idem_key="k",
                          requests_mod=rq, base_delay=0.0)
    assert out["ok"] is True and out["run_id"] == "OK"
    assert len(rq.calls) == 2


def test_post_422_is_permanent(monkeypatch):
    monkeypatch.setattr(DI.time, "sleep", lambda *_: None)
    rq = FakeRequests([FakeResp(422, text="bad")])
    out = DI._post_ingest(api_url="http://x", payload={}, idem_key="k",
                          requests_mod=rq, base_delay=0.0)
    assert out["ok"] is False and out["status_code"] == 422
    assert len(rq.calls) == 1


def test_post_exhausts_retries(monkeypatch):
    monkeypatch.setattr(DI.time, "sleep", lambda *_: None)
    rq = FakeRequests([FakeResp(500), FakeResp(500), FakeResp(500)])
    out = DI._post_ingest(api_url="http://x", payload={}, idem_key="k",
                          requests_mod=rq, base_delay=0.0)
    assert out["ok"] is False
    assert len(rq.calls) == 3


def test_ingest_failure_recorded_not_raised(tmp_path, monkeypatch):
    monkeypatch.setenv("WAGONEYE_DASHBOARD_INGEST_ENABLED", "true")
    monkeypatch.setattr(DI.time, "sleep", lambda *_: None)
    root = str(tmp_path)
    make_batch(root)
    rq = FakeRequests([FakeResp(500)] * 9)     # all cameras fail
    res = DI.run(batch_root=root, s3_client=FakeS3(), requests_mod=rq)
    assert all(v["status"] == "ingest_failed" for v in res["cameras"].values())
    marker = FIN.load(root)
    assert marker["dashboard_ingested"]["RIGHT_UP"]["status"] == "ingest_failed"


# -----------------------------------------------------------------------------
# no writes outside delivery/
# -----------------------------------------------------------------------------

def test_no_writes_outside_delivery(tmp_path, monkeypatch):
    monkeypatch.setenv("WAGONEYE_DASHBOARD_INGEST_ENABLED", "true")
    root = str(tmp_path)
    make_batch(root)
    before = _all_files(root)
    # dry-run: build + record, no upload/POST
    DI.run(batch_root=root, s3_client=FakeS3(), skip_upload=True)
    after = _all_files(root)
    changed = {p for p in after if p not in before or after[p] != before.get(p)}
    assert changed, "adapter should have written something"
    for p in changed:
        assert p.startswith("delivery/"), f"wrote outside delivery/: {p}"


def test_dry_run_does_not_post(tmp_path, monkeypatch):
    monkeypatch.setenv("WAGONEYE_DASHBOARD_INGEST_ENABLED", "true")
    root = str(tmp_path)
    make_batch(root)
    rq = FakeRequests([FakeResp(200, {"run_id": "X"})] * 3)
    s3 = FakeS3()
    res = DI.run(batch_root=root, s3_client=s3, skip_upload=True, requests_mod=rq)
    assert len(rq.calls) == 0
    assert s3.uploads == []
    assert all(v.get("dry_run") for v in res["cameras"].values())


# -----------------------------------------------------------------------------
# dashboard S3 folder per camera -- VERIFIED against the old per-camera pipelines
# -----------------------------------------------------------------------------

def test_inspection_folders_match_the_old_pipeline_constants():
    """Each folder is that camera's own INSPECTION_JSON_FOLDER from
    output_test/<CAMERA>/sagemaker_main.py -- the process that has been feeding
    this dashboard.  The TOP cameras deliberately break the "<side>_up_top"
    pattern (capital T, no "up"); guessing it publishes into folders the
    dashboard never reads."""
    assert DI.folder_for("RIGHT_UP") == "Right_up"
    assert DI.folder_for("LEFT_UP") == "Left_up"
    assert DI.folder_for("RIGHT_UP_TOP") == "Right_Top"
    assert DI.folder_for("LEFT_UP_TOP") == "Left_Top"


def test_no_folder_collisions():
    folders = [DI.folder_for(c) for c in C.ALL_CAMERAS]
    assert len(set(folders)) == 4, folders


def test_folders_are_overridable(monkeypatch):
    monkeypatch.setenv("WAGONEYE_INSPECTION_FOLDERS",
                       '{"RIGHT_UP_TOP":"RTop","LEFT_UP_TOP":"LTop"}')
    assert DI.folder_for("RIGHT_UP_TOP") == "RTop"
    assert DI.folder_for("LEFT_UP_TOP") == "LTop"
    assert DI.folder_for("RIGHT_UP") == "Right_up"      # untouched keys survive


def test_json_key_layout_matches_the_old_pipeline(tmp_path, monkeypatch):
    """<folder>/<YYYY-MM-DD>/<raw_basename>_inspection.json, 05:00 IST day rule."""
    root = str(tmp_path)
    make_batch(root)
    s3 = FakeS3()
    monkeypatch.setattr(DI, "_post_ingest",
                        lambda **kw: {"ok": True, "status_code": 200,
                                      "run_id": "R", "error": None})
    DI.run(batch_root=root, s3_client=s3, skip_upload=False)
    keys = dict(s3.uploads)
    uploaded = [k for _, k in s3.uploads]
    # batch 20260408_032134 is 03:21 IST -> previous operational day 2026-04-07
    assert any(k.startswith("Right_up/2026-04-07/") for k in uploaded), uploaded
    assert any(k.startswith("Left_up/2026-04-07/") for k in uploaded), uploaded
    assert any(k.startswith("Right_Top/2026-04-07/") for k in uploaded), uploaded
    assert all(k.endswith("_inspection.json") for k in uploaded), uploaded


# -----------------------------------------------------------------------------
# schema dialect: a v1 document must carry v1 SHAPES, not just a v1 version field
# -----------------------------------------------------------------------------

# The old per-camera pipeline's inspection_data key list, verbatim and in order
# (output_test/<CAMERA>/sagemaker_main.py::generate_inspection_json).
V1_SIDE_KEYS = [
    "raw_video_name", "identified_by", "upload_timestamp",
    "upload_timestamp_readable", "direction", "rake_status", "pdf_report_url",
    "trimmed_video_url", "detected_video_url", "raw_video_urls", "total_wagons",
    "doors_open", "doors_closed", "damaged_wagons", "num_engines",
    "total_loco_frames", "total_problem_frames", "problem_frames_by_type",
    "wagon_number_results", "loco_number_results", "segment_type_map",
    "wagon_segments", "loco_frames", "problem_frames",
]


def _open_door_pf(inspection_data):
    return [p for p in inspection_data["problem_frames"]
            if p["problem_type"] in ("door_open", "open_door")][0]


def test_v1_document_matches_the_old_pipeline_key_list(tmp_path, monkeypatch):
    root = str(tmp_path)
    make_batch(root)
    monkeypatch.delenv("WAGONEYE_INSPECTION_VERSION", raising=False)
    d = _build(root, "RIGHT_UP")["inspection_data"]
    assert [k for k in d if k != "_adapter"] == V1_SIDE_KEYS


def test_v1_bounding_box_is_the_wrapped_dict(tmp_path, monkeypatch):
    """The V1 dashboard reads bounding_box.bounding_box_coordinates; a bare list
    breaks it."""
    root = str(tmp_path)
    make_batch(root)
    monkeypatch.delenv("WAGONEYE_INSPECTION_VERSION", raising=False)
    bb = _open_door_pf(_build(root, "RIGHT_UP")["inspection_data"])["bounding_box"]
    assert isinstance(bb, dict)
    assert bb["bounding_box_coordinates"] == [10.0, 20.0, 110.0, 220.0]
    assert bb["confidence"] == 0.83
    assert bb["class_name"] == "door_open"


def test_v4_bounding_box_is_the_bare_list(tmp_path, monkeypatch):
    root = str(tmp_path)
    make_batch(root)
    monkeypatch.setenv("WAGONEYE_INSPECTION_VERSION", "v4")
    bb = _open_door_pf(_build(root, "RIGHT_UP")["inspection_data"])["bounding_box"]
    assert bb == [10.0, 20.0, 110.0, 220.0]


def test_open_door_problem_type_differs_by_dialect(tmp_path, monkeypatch):
    root = str(tmp_path)
    make_batch(root)
    monkeypatch.delenv("WAGONEYE_INSPECTION_VERSION", raising=False)
    d = _build(root, "RIGHT_UP")["inspection_data"]
    assert _open_door_pf(d)["problem_type"] == "door_open"
    assert set(d["problem_frames_by_type"]) == {"damage", "door_open"}

    monkeypatch.setenv("WAGONEYE_INSPECTION_VERSION", "v4")
    d = _build(root, "RIGHT_UP")["inspection_data"]
    assert _open_door_pf(d)["problem_type"] == "open_door"
    assert set(d["problem_frames_by_type"]) == {"damage", "open_door",
                                                "closed_door", "partially_closed"}


def test_v1_problem_frame_carries_segment_number(tmp_path, monkeypatch):
    root = str(tmp_path)
    make_batch(root)
    monkeypatch.delenv("WAGONEYE_INSPECTION_VERSION", raising=False)
    assert _open_door_pf(_build(root, "RIGHT_UP")["inspection_data"])["segment_number"] == 1
    monkeypatch.setenv("WAGONEYE_INSPECTION_VERSION", "v4")
    assert _open_door_pf(_build(root, "RIGHT_UP")["inspection_data"])["segment_number"] is None


def test_side_rake_status_polarity_is_dialect_specific():
    """The two sources genuinely disagree: the old pipeline treats right-to-left
    as loaded, V4 treats left-to-right as loaded."""
    from delivery.inspection_json import _rake_status_from_direction as r
    from delivery.inspection_json import SCHEMA_V1, SCHEMA_V4
    assert r("left-to-right", SCHEMA_V4) == "Loaded"
    assert r("right-to-left", SCHEMA_V4) == "Empty"
    assert r("right-to-left", SCHEMA_V1) == "Loaded"
    assert r("left-to-right", SCHEMA_V1) == "Empty"
    assert r("unknown", SCHEMA_V1) == r("unknown", SCHEMA_V4) == "Unknown"


def test_schema_follows_version():
    from delivery.inspection_json import schema_for_version, SCHEMA_V1, SCHEMA_V4
    assert schema_for_version("v1") == SCHEMA_V1
    assert schema_for_version("v4") == SCHEMA_V4
    assert schema_for_version("") == SCHEMA_V4


# -----------------------------------------------------------------------------
# loco-number OCR feed (was structurally empty before)
# -----------------------------------------------------------------------------

def _add_loco_ocr(root, number="23456", valid=True):
    """GW_1 is the ENGINE; give it a loco-path OCR result + evidence."""
    ws = os.path.join(root, "wagon_states")
    _write(os.path.join(ws, "ocr", "RIGHT_UP", "GW_1.json"), {
        "status": C.STATUS_OK, "engine": "rekognition", "kind": "loco",
        "wagon_identifier": number if valid else C.NO_DATA,
        "wagon_identifier_confidence": 0.91,
        "raw_number": number, "display_number": number if valid else "-",
        "is_valid_5_digit": valid, "confidence": 0.91, "ocr_confidence": 0.91,
        "best_frame": 42, "fallback_triggered": False,
    })
    ev = os.path.join(root, "evidence")
    for fn in ("best_frame.jpg", "number_crop.jpg", "ocr_sheet.jpg"):
        _touch_jpg(os.path.join(ev, "GW_1", "ocr", "RIGHT_UP", fn))


def test_loco_number_results_are_populated(tmp_path):
    root = str(tmp_path)
    make_batch(root)
    _add_loco_ocr(root)
    d = _build(root, "RIGHT_UP")["inspection_data"]
    assert d["loco_number_results"], "loco_number_results must not be empty"
    entry = d["loco_number_results"]["1"]          # keyed by str(loco_id)
    assert entry["is_valid_5_digit"] is True
    assert entry["display_number"] == "23456"
    assert entry["raw_number"] == "23456"
    assert entry["ocr_frame_s3_url"] and "ocr_sheet.jpg" in entry["ocr_frame_s3_url"]


def test_loco_frames_and_total_are_populated(tmp_path):
    root = str(tmp_path)
    make_batch(root)
    _add_loco_ocr(root)
    d = _build(root, "RIGHT_UP")["inspection_data"]
    assert len(d["loco_frames"]) == 1
    block = d["loco_frames"][0]
    assert block["loco_id"] == 1
    assert block["loco_number"] == "23456"
    assert len(block["frames"]) == 3               # best_frame + crop + sheet
    assert d["total_loco_frames"] == 3
    for f in block["frames"]:
        assert set(f) == {"position", "filename", "s3_key", "s3_url",
                          "frame_number", "timestamp_sec"}


def test_invalid_loco_read_is_reported_not_hidden(tmp_path):
    root = str(tmp_path)
    make_batch(root)
    _add_loco_ocr(root, number="123", valid=False)
    d = _build(root, "RIGHT_UP")["inspection_data"]
    e = d["loco_number_results"]["1"]
    assert e["is_valid_5_digit"] is False
    assert e["display_number"] == "-"
    assert e["raw_number"] == "123"                # auditable
    assert d["loco_frames"][0]["loco_number"] is None


def test_only_the_ocr_authority_camera_claims_locos(tmp_path):
    root = str(tmp_path)
    make_batch(root)
    _add_loco_ocr(root)
    left = _build(root, "LEFT_UP")["inspection_data"]
    assert left["loco_number_results"] == {}
    assert left["loco_frames"] == []
    assert left["total_loco_frames"] == 0


def test_no_engine_means_no_loco_block(tmp_path):
    """A rake with no ENGINE segment reports no locos -- not a fabricated one."""
    root = str(tmp_path)
    make_batch(root)
    p = os.path.join(root, "global_state", "global_train_state.json")
    with open(p, encoding="utf-8") as f:
        st = json.load(f)
    for w in st["wagons"]:
        w["classification"] = C.CLASS_WAGON
    _write(p, st)
    d = _build(root, "RIGHT_UP")["inspection_data"]
    assert d["loco_frames"] == [] and d["loco_number_results"] == {}
    assert d["total_loco_frames"] == 0


# -----------------------------------------------------------------------------
# probable top damage now reaches the JSON
# -----------------------------------------------------------------------------

def test_probable_damage_is_reported_separately(tmp_path, v4):
    root = str(tmp_path)
    make_batch(root)
    # GW_3 gets a PROBABLE floor damage using damage.pt's real class name
    _write(os.path.join(root, "wagon_states", "damage", "RIGHT_UP_TOP",
                        "GW_3.json"),
           {"status": C.STATUS_OK, "damage_status": C.DAMAGE_PRESENT,
            "top_damage_details": [{"class_name": "Floor__probable_damage",
                                    "confidence": 0.55}]})
    d = _build(root, "RIGHT_UP_TOP")["inspection_data"]
    assert d["floor_dmg_probable_wagons"] == 1
    assert d["probable_damage_wagons"] == 1
    # ... and it must NOT be counted as confirmed damage
    assert d["floor_dmg_wagons"] == 1          # only GW_2's real floor damage
    assert d["damaged_wagons"] == 1
    seg3 = [s for s in d["wagon_segments"] if s["wagon_count"] == 2][0]
    assert seg3["floor_dmg_probable_detected"] is True
    assert seg3["floor_dmg_detected"] is False
    assert seg3["damage_detected"] is False
    assert seg3["probable_damage_detected"] is True
