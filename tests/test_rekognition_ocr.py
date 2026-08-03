"""Tests for the Amazon Rekognition wagon-number OCR path.

Covers the V4-parity behaviours: three-frame sheet selection (loaded vs empty),
vertical sheet assembly, best-valid-line picking off a multi-crop sheet, two-row
plate reassembly, band grouping, the per-wagon call budget, engine selection, and
graceful degradation.

No AWS, no network, no models: a fake DetectText client is injected.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import constants as C
from features.inference_lib.ocr_preprocessor import Preprocessor
from features.inference_lib.rekognition_reader import digits_only, read_digits
from features.inference_lib.rekognition_wagon_number import (
    MIN_FRAMES_FOR_SHEET_SELECTION, RekognitionWagonNumberOCR, crop_plate,
    group_detections_into_bands, is_valid_wagon_number,
)
from features.inference_lib.three_frame_sheet import (
    build_horizontal_sheet, build_vertical_sheet, select_frame_positions,
)


# -----------------------------------------------------------------------------
# fakes
# -----------------------------------------------------------------------------

class FakeRekognition:
    """Returns a fixed set of (top, text, confidence_0_100) LINE detections."""

    def __init__(self, lines, fail=False):
        self.lines = lines
        self.fail = fail
        self.calls = 0

    def detect_text(self, image_bytes):
        self.calls += 1
        if self.fail:
            raise RuntimeError("ProvisionedThroughputExceededException")
        out = []
        for top, text, conf in self.lines:
            # a real response interleaves LINE and WORD; only LINEs are used
            out.append({"Type": "LINE", "DetectedText": text, "Confidence": conf,
                        "Geometry": {"BoundingBox": {"Top": top}}})
            out.append({"Type": "WORD", "DetectedText": text, "Confidence": conf,
                        "Geometry": {"BoundingBox": {"Top": top}}})
        return out


def _dets(frames, conf=0.8, bbox=(0, 0, 40, 20)):
    return [{"frame": f, "confidence": conf, "bbox": list(bbox)} for f in frames]


IMG = np.full((60, 900, 3), 128, np.uint8)


# -----------------------------------------------------------------------------
# validator + digit extraction
# -----------------------------------------------------------------------------

def test_wagon_number_validator_is_11_digits():
    assert is_valid_wagon_number("12345678901")
    assert not is_valid_wagon_number("1234567890")     # 10
    assert not is_valid_wagon_number("123456789012")   # 12
    assert not is_valid_wagon_number("1234567890a")
    assert not is_valid_wagon_number("")
    assert not is_valid_wagon_number(None) if None is None else True


def test_digits_only_strips_non_digits():
    assert digits_only("IR 31234-567890") == "31234567890"
    assert digits_only("") == ""
    assert digits_only("abc") == ""


# -----------------------------------------------------------------------------
# three-frame selection (V4 rules)
# -----------------------------------------------------------------------------

def test_loaded_reads_from_the_end_empty_from_the_start():
    # loaded rake -> End-7/-5/-3 ; empty rake -> Start+3/+5/+7
    assert select_frame_positions(20, "loaded") == [12, 14, 16]
    assert select_frame_positions(20, "empty") == [3, 5, 7]


def test_fallback_is_the_mirror_of_the_primary():
    assert select_frame_positions(20, "loaded", True) == select_frame_positions(20, "empty")
    assert select_frame_positions(20, "empty", True) == select_frame_positions(20, "loaded")


def test_loco_reads_around_the_middle():
    assert select_frame_positions(20, "loco") == [8, 10, 12]


def test_short_band_clamps_instead_of_indexing_out_of_range():
    assert select_frame_positions(2, "empty") == [1, 1, 1]
    assert select_frame_positions(1, "loaded") == [0, 0, 0]
    assert select_frame_positions(0, "empty") == []


def test_unknown_category_is_rejected():
    with pytest.raises(ValueError):
        select_frame_positions(10, "wagon")


# -----------------------------------------------------------------------------
# sheet assembly
# -----------------------------------------------------------------------------

def test_vertical_sheet_stacks_verbatim_with_white_gutters():
    a = np.full((10, 30, 3), 10, np.uint8)
    b = np.full((20, 50, 3), 20, np.uint8)
    sheet = build_vertical_sheet([a, b], spacing=20)
    assert sheet.shape == (10 + 20 + 20, 50, 3)     # widest crop sets the width
    assert (sheet[0:10, 0:30] == 10).all()          # pixel-exact copy
    assert (sheet[30:50, 0:50] == 20).all()
    assert (sheet[10:30, :, :] == 255).all()        # white gutter between them


def test_horizontal_sheet_lays_out_left_to_right():
    a = np.full((10, 30, 3), 10, np.uint8)
    b = np.full((20, 50, 3), 20, np.uint8)
    assert build_horizontal_sheet([a, b], spacing=20).shape == (20, 30 + 50 + 20, 3)


def test_grayscale_crop_is_promoted_to_bgr():
    assert build_vertical_sheet([np.full((10, 10), 5, np.uint8)]).shape == (10, 10, 3)


def test_empty_crop_list_is_rejected():
    with pytest.raises(ValueError):
        build_vertical_sheet([])
    with pytest.raises(ValueError):
        build_vertical_sheet([None])


def test_preprocessor_upscales_to_the_ocr_target_width():
    out = Preprocessor.primary(np.full((10, 40, 3), 7, np.uint8))
    assert out.shape[1] >= Preprocessor.OCR_TARGET_WIDTH


# -----------------------------------------------------------------------------
# reader: turning LINE detections into a number
# -----------------------------------------------------------------------------

def test_three_copies_on_a_sheet_are_not_concatenated():
    """The whole point of the validator: 3x11 digits must not become 33."""
    client = FakeRekognition([(0.1, "31234567890", 88.0),
                              (0.4, "31234567890", 95.0),
                              (0.7, "3123456789", 70.0)])
    num, conf = read_digits(client, IMG, validator=is_valid_wagon_number)
    assert num == "31234567890"
    assert conf == pytest.approx(0.95)          # the best VALID line's confidence


def test_two_row_plate_is_reassembled_top_to_bottom():
    client = FakeRekognition([(0.3, "25759", 80.0), (0.1, "221423", 90.0)])
    num, _ = read_digits(client, IMG, validator=is_valid_wagon_number)
    assert num == "22142325759"                 # re-ordered by bbox Top


def test_nothing_valid_falls_back_to_best_effort_concatenation():
    client = FakeRekognition([(0.1, "123", 50.0), (0.3, "456", 50.0)])
    num, conf = read_digits(client, IMG, validator=is_valid_wagon_number)
    assert num == "123456"                      # reported, but caller sees invalid
    assert not is_valid_wagon_number(num)
    assert conf == pytest.approx(0.5)


def test_no_detections_returns_empty():
    assert read_digits(FakeRekognition([]), IMG,
                       validator=is_valid_wagon_number) == ("", 0.0)


def test_client_error_never_propagates():
    assert read_digits(FakeRekognition([], fail=True), IMG) == ("", 0.0)


def test_missing_client_or_image_returns_empty():
    assert read_digits(None, IMG) == ("", 0.0)
    assert read_digits(FakeRekognition([]), None) == ("", 0.0)


# -----------------------------------------------------------------------------
# band grouping
# -----------------------------------------------------------------------------

def test_frame_gap_starts_a_new_band():
    bands = group_detections_into_bands(
        _dets(range(100, 112)) + _dets(range(200, 208)), gap_tolerance=8)
    assert len(bands) == 2
    assert bands[0]["frame_count"] == 12
    assert bands[1]["frame_count"] == 8


def test_contiguous_detections_stay_one_band():
    assert len(group_detections_into_bands(_dets(range(0, 30)),
                                           gap_tolerance=8)) == 1


def test_band_exposes_both_orderings():
    band = group_detections_into_bands(
        [{"frame": 3, "confidence": 0.9, "bbox": [0, 0, 1, 1]},
         {"frame": 1, "confidence": 0.4, "bbox": [0, 0, 1, 1]},
         {"frame": 2, "confidence": 0.7, "bbox": [0, 0, 1, 1]}])[0]
    assert [f["frame"] for f in band["frames_by_time"]] == [1, 2, 3]
    assert [f["frame"] for f in band["top_frames"]] == [3, 2, 1]
    assert band["best_frame"] == 3 and band["best_confidence"] == 0.9


def test_duplicate_frame_keeps_the_highest_confidence_box():
    band = group_detections_into_bands(
        [{"frame": 5, "confidence": 0.3, "bbox": [0, 0, 1, 1]},
         {"frame": 5, "confidence": 0.8, "bbox": [2, 2, 3, 3]}])[0]
    assert band["frame_count"] == 1
    assert band["best_confidence"] == 0.8
    assert band["best_bbox"] == [2, 2, 3, 3]


def test_no_detections_yields_no_bands():
    assert group_detections_into_bands([]) == []
    assert group_detections_into_bands([{"frame": 1, "confidence": 0.5,
                                         "bbox": None}]) == []


# -----------------------------------------------------------------------------
# crop padding
# -----------------------------------------------------------------------------

def test_crop_pads_by_a_quarter_of_the_box():
    frame = np.full((100, 200, 3), 9, np.uint8)
    crop = crop_plate(frame, [50, 40, 90, 60])      # 40x20 box -> +10x / +5y each side
    assert crop.shape == (20 + 10, 40 + 20, 3)


def test_crop_clamps_at_the_frame_edge():
    frame = np.full((100, 200, 3), 9, np.uint8)
    assert crop_plate(frame, [0, 0, 10, 10]) is not None
    assert crop_plate(frame, [190, 90, 200, 100]) is not None


def test_degenerate_or_missing_bbox_returns_none():
    frame = np.full((100, 200, 3), 9, np.uint8)
    assert crop_plate(frame, [10, 10, 10, 10]) is None
    assert crop_plate(frame, None) is None
    assert crop_plate(None, [0, 0, 5, 5]) is None


# -----------------------------------------------------------------------------
# orchestrator
# -----------------------------------------------------------------------------

def _reader(client, **kw):
    return RekognitionWagonNumberOCR(client, gap_tolerance=8, **kw)


def _frames(rng):
    return {f: np.full((100, 200, 3), 50, np.uint8) for f in rng}


def test_first_valid_sheet_wins_in_one_call():
    bands = group_detections_into_bands(_dets(range(100, 112)))
    frames = _frames(range(100, 112))
    client = FakeRekognition([(0.2, "31234567890", 92.0)])
    res = _reader(client, max_calls=4).read_wagon_number(
        bands=bands, frame_loader=frames.get, is_loaded=True)
    assert res["is_valid_11_digit"] is True
    assert res["wagon_identifier"] == "31234567890"
    assert res["display_number"] == "31234567890"
    assert res["fallback_triggered"] is False
    assert res["rekognition_calls"] == 1          # stops at the first valid read
    assert res["engine"] == "rekognition"


def test_invalid_read_reports_no_data_but_keeps_the_raw_string():
    bands = group_detections_into_bands(_dets(range(100, 112)))
    frames = _frames(range(100, 112))
    res = _reader(FakeRekognition([(0.2, "1234", 50.0)]),
                  max_calls=4).read_wagon_number(
        bands=bands, frame_loader=frames.get, is_loaded=False)
    assert res["is_valid_11_digit"] is False
    assert res["wagon_identifier"] == C.NO_DATA    # never a wrong number
    assert res["display_number"] == "-"
    assert res["raw_number"] == "1234"             # still auditable
    assert res["fallback_triggered"] is True


def test_call_budget_is_never_exceeded():
    bands = group_detections_into_bands(
        _dets(range(0, 20)) + _dets(range(100, 120)) + _dets(range(200, 220)))
    assert len(bands) == 3
    frames = _frames(list(range(0, 20)) + list(range(100, 120)) + list(range(200, 220)))
    client = FakeRekognition([(0.2, "999", 40.0)])   # never validates
    res = _reader(client, max_calls=2).read_wagon_number(
        bands=bands, frame_loader=frames.get, is_loaded=False)
    assert client.calls == 2
    assert res["rekognition_calls"] == 2


def test_short_band_uses_a_single_best_frame_crop():
    bands = group_detections_into_bands(_dets([5, 6]))   # < MIN_FRAMES_FOR_SHEET
    assert bands[0]["frame_count"] < MIN_FRAMES_FOR_SHEET_SELECTION
    client = FakeRekognition([(0.2, "31234567890", 90.0)])
    res = _reader(client, max_calls=4).read_wagon_number(
        bands=bands, frame_loader=_frames([5, 6]).get, is_loaded=False)
    assert res["is_valid_11_digit"] is True
    assert client.calls == 1


def test_unloadable_frames_degrade_without_calling_the_api():
    bands = group_detections_into_bands(_dets(range(100, 112)))
    client = FakeRekognition([(0.2, "31234567890", 90.0)])
    res = _reader(client, max_calls=4).read_wagon_number(
        bands=bands, frame_loader=lambda _i: None, is_loaded=False)
    assert res["is_valid_11_digit"] is False
    assert client.calls == 0                       # no crop -> no request
    assert res["wagon_identifier"] == C.NO_DATA


def test_no_bands_returns_a_no_data_result():
    res = _reader(FakeRekognition([]), max_calls=4).read_wagon_number(
        bands=[], frame_loader=lambda _i: None, is_loaded=False)
    assert res["wagon_identifier"] == C.NO_DATA
    assert res["rekognition_calls"] == 0


def test_loaded_and_empty_pick_different_frames():
    """Load state must actually change which crops are sent."""
    bands = group_detections_into_bands(_dets(range(0, 20)))
    frames = _frames(range(0, 20))
    seen = {}

    for loaded in (True, False):
        picked = []

        def _loader(i, _p=picked):
            _p.append(i)
            return frames.get(i)

        _reader(FakeRekognition([(0.2, "1", 10.0)]), max_calls=1).read_wagon_number(
            bands=bands, frame_loader=_loader, is_loaded=loaded)
        seen[loaded] = picked[:3]

    assert seen[True] != seen[False]
    assert seen[True] == [12, 14, 16]      # loaded: End-7/-5/-3
    assert seen[False] == [3, 5, 7]        # empty:  Start+3/+5/+7


def test_sheet_is_kept_in_memory_but_never_serialised():
    """The processor writes the sheet as evidence, then strips it from the JSON."""
    bands = group_detections_into_bands(_dets(range(100, 112)))
    res = _reader(FakeRekognition([(0.2, "31234567890", 90.0)]),
                  max_calls=4).read_wagon_number(
        bands=bands, frame_loader=_frames(range(100, 112)).get, is_loaded=True)
    assert res["_sheet"] is not None
    assert isinstance(res["_sheet"], np.ndarray)


# -----------------------------------------------------------------------------
# engine selection
# -----------------------------------------------------------------------------

def test_engine_defaults_to_rekognition(monkeypatch):
    from features.ocr import processor as P
    monkeypatch.delenv("WAGONEYE_OCR_ENGINE", raising=False)
    assert P.resolve_engine() == P.ENGINE_REKOGNITION


def test_engine_can_be_forced_to_easyocr(monkeypatch):
    from features.ocr import processor as P
    monkeypatch.setenv("WAGONEYE_OCR_ENGINE", "easyocr")
    assert P.resolve_engine() == P.ENGINE_EASYOCR


def test_unknown_engine_falls_back_to_rekognition(monkeypatch):
    from features.ocr import processor as P
    monkeypatch.setenv("WAGONEYE_OCR_ENGINE", "tesseract")
    assert P.resolve_engine() == P.ENGINE_REKOGNITION


def test_gap_tolerance_env_override(monkeypatch):
    from features.ocr import processor as P
    monkeypatch.setenv("WAGONEYE_OCR_GAP_TOLERANCE", "15")
    assert P._gap_tolerance() == 15
    monkeypatch.setenv("WAGONEYE_OCR_GAP_TOLERANCE", "nonsense")
    assert P._gap_tolerance() == 8          # bad value -> V4 default


def test_right_up_plate_detector_is_the_v4_model():
    from core import camera_features as CF
    assert C.FEATURE_MODEL_BY_KEY["ocr"] == "wagon_number_update.pt"
    assert CF.FEATURE_MODEL_FILENAME["ocr"] == "wagon_number_update.pt"
    # the older name is still accepted so an existing checkout keeps running
    assert C.FEATURE_MODEL_LEGACY["wagon_number_update.pt"] == "wagon_id_counting.pt"


def test_model_path_resolution_prefers_canonical_then_legacy(tmp_path):
    d = str(tmp_path)
    canonical = C.MODEL_WAGON_NUMBER
    legacy = C.MODEL_WAGON_ID_COUNTING
    # neither present -> canonical path reported (so it names the right file)
    assert C.feature_model_path(d, canonical) == os.path.join(d, canonical)
    # only legacy present -> legacy used
    open(os.path.join(d, legacy), "wb").write(b"x")
    assert C.feature_model_path(d, canonical) == os.path.join(d, legacy)
    # canonical present -> wins
    open(os.path.join(d, canonical), "wb").write(b"x")
    assert C.feature_model_path(d, canonical) == os.path.join(d, canonical)


# -----------------------------------------------------------------------------
# load-state lookup (drives the triplet order)
# -----------------------------------------------------------------------------

def test_load_state_read_from_the_wagons_own_load_result(tmp_path):
    from features.ocr import processor as P
    import json
    states = str(tmp_path)

    def _w(cam, gw, status):
        p = os.path.join(states, "load", cam, f"{gw}.json")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        json.dump({"status": C.STATUS_OK, "load_status": status}, open(p, "w"))

    _w(C.CAMERA_RIGHT_UP_TOP, "GW_1", C.LOAD_LOADED)
    assert P._is_wagon_loaded(states, "GW_1") is True
    _w(C.CAMERA_RIGHT_UP_TOP, "GW_2", C.LOAD_EMPTY)
    assert P._is_wagon_loaded(states, "GW_2") is False
    # primary absent -> LEFT_UP_TOP fallback
    _w(C.CAMERA_LEFT_UP_TOP, "GW_3", C.LOAD_LOADED)
    assert P._is_wagon_loaded(states, "GW_3") is True
    # nothing at all -> empty (the V4 default triplet order)
    assert P._is_wagon_loaded(states, "GW_9") is False


# -----------------------------------------------------------------------------
# loco-number OCR (5-digit, middle-frame triplet)
# -----------------------------------------------------------------------------

def test_loco_validator_is_5_digits():
    from features.inference_lib.rekognition_wagon_number import is_valid_loco_number
    assert is_valid_loco_number("12345")
    assert not is_valid_loco_number("1234")
    assert not is_valid_loco_number("123456")
    assert not is_valid_loco_number("1234a")


def test_loco_reads_a_5_digit_number():
    bands = group_detections_into_bands(_dets(range(0, 20)))
    frames = _frames(range(0, 20))
    client = FakeRekognition([(0.2, "23456", 91.0)])
    res = _reader(client, max_calls=4).read_loco_number(
        bands=bands, frame_loader=frames.get)
    assert res["is_valid_5_digit"] is True
    assert res["wagon_identifier"] == "23456"
    assert res["display_number"] == "23456"
    assert res["kind"] == "loco"
    assert "is_valid_11_digit" not in res     # V4 names the flag per digit count


def test_loco_rejects_an_11_digit_read():
    """An 11-digit read on a loco band is a mis-detected wagon plate, not a loco."""
    bands = group_detections_into_bands(_dets(range(0, 20)))
    res = _reader(FakeRekognition([(0.2, "31234567890", 95.0)]),
                  max_calls=4).read_loco_number(
        bands=bands, frame_loader=_frames(range(0, 20)).get)
    assert res["is_valid_5_digit"] is False
    assert res["wagon_identifier"] == C.NO_DATA
    assert res["raw_number"] == "31234567890"      # still auditable


def test_loco_uses_the_middle_triplet():
    """V4 reads a loco at Middle-2/Middle/Middle+2 -- not the start/end offsets."""
    bands = group_detections_into_bands(_dets(range(0, 20)))
    frames = _frames(range(0, 20))
    picked = []

    def _loader(i):
        picked.append(i)
        return frames.get(i)

    _reader(FakeRekognition([(0.2, "1", 10.0)]), max_calls=1).read_loco_number(
        bands=bands, frame_loader=_loader)
    assert picked[:3] == [8, 10, 12]
    assert select_frame_positions(20, "loco") == [8, 10, 12]


def test_loco_plate_class_is_never_guessed():
    """A wagon plate must not be read as a loco number just because the class
    label is unrecognised."""
    from features.ocr import processor as P
    known = {"loco_no", "wagon_id"}
    assert P._is_plate_class("loco_no", known, kind="loco") is True
    assert P._is_plate_class("locono", known, kind="loco") is True
    assert P._is_plate_class("wagon_id", known, kind="loco") is False
    assert P._is_plate_class("something_else", known, kind="loco") is False
    assert P._is_plate_class(None, known, kind="loco") is False
    # the wagon path stays permissive for a single-purpose detector
    assert P._is_plate_class("wagon_id", known, kind="wagon") is True
    assert P._is_plate_class("loco_no", known, kind="wagon") is False
    assert P._is_plate_class(None, known, kind="wagon") is True


def test_deployed_plate_model_serves_both_ocr_paths():
    """The class names `wagon_number_update.pt` actually ships with.

    Verified on the production box: `{0: 'loco_no', 1: 'wagon_id'}`.  Pinning them
    here means a model swap that renames a class fails a test rather than silently
    reading no loco numbers in the field.
    """
    from features.ocr import processor as P
    assert P.plate_classes_resolvable(["loco_no", "wagon_id"]) == {
        "wagon": True, "loco": True}


def test_a_model_without_a_loco_class_disables_only_the_loco_path():
    from features.ocr import processor as P
    assert P.plate_classes_resolvable(["wagonno"]) == {"wagon": True, "loco": False}
    assert P.plate_classes_resolvable([]) == {"wagon": True, "loco": False}
    # case and ordering are irrelevant
    assert P.plate_classes_resolvable(["WAGON_ID", "Loco_No"])["loco"] is True


def test_unservable_loco_model_warns_loudly_instead_of_failing_silently(caplog):
    """The old failure mode: every detection dropped, nothing in the log to say why."""
    import logging
    from features.ocr import processor as P
    P._WARNED_NO_LOCO_CLASS.clear()
    with caplog.at_level(logging.WARNING, logger="features.ocr"):
        assert P._warn_if_loco_unservable({"wagonno"}) is False
        assert P._warn_if_loco_unservable({"wagonno"}) is False   # once per model
    warnings = [r for r in caplog.records if "loco-number OCR DISABLED" in r.message]
    assert len(warnings) == 1
    # a servable model says nothing at all
    caplog.clear()
    assert P._warn_if_loco_unservable({"loco_no", "wagon_id"}) is True
    assert not caplog.records


def test_wagon_number_model_detects_both_plate_classes():
    """wagon_number_update.pt / wagon_id_counting.pt emit loco_no AND wagon_id,
    so one detector serves both OCR paths."""
    from features.ocr import processor as P
    assert "loco_no" in P.LOCO_NUMBER_CLASS_ALIASES
    assert "wagon_id" in P.WAGON_NUMBER_CLASS_ALIASES


# -----------------------------------------------------------------------------
# probable top damage -- damage.pt's real class name
# -----------------------------------------------------------------------------

def test_probable_damage_class_is_recognised():
    """damage.pt emits `Floor__probable_damage` (DOUBLE underscore); getting that
    key wrong is what left floor_dmg_probable structurally zero."""
    assert C.is_probable_damage("Floor__probable_damage")
    assert C.is_probable_damage("floor__probable_damage")
    assert not C.is_probable_damage("Floor_damage")
    assert not C.is_probable_damage("Inner_wall_damage")


def test_probable_damage_maps_to_the_probable_flag():
    from delivery.inspection_json import _DAMAGE_CLASS_TO_FLAG as M
    assert M["floor__probable_damage"] == "floor_dmg_probable_detected"
    assert M["floor_damage"] == "floor_dmg_detected"
    assert M["inner_wall_damage"] == "inner_wall_dmg_detected"


def test_probable_damage_problem_type():
    from delivery.inspection_json import _V4_TOP_PROBLEM_TYPE as M
    assert M["floor__probable_damage"] == "floor_dmg_probable"
