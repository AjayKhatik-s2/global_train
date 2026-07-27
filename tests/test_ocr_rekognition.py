"""OCR via AWS Rekognition — features/inference_lib/ocr + features/ocr.

The ocr package mirrors the V4 Train-Inspection-Engine `inspection/ocr/` folder.
Rekognition bills per image, so the properties that matter are as much about
CALL COUNT and sheet assembly as about the digits themselves:

  * three frames are stacked on ONE sheet -> 3 looks at the plate, 1 API call
  * the triplet offsets and primary/fallback order follow the load state
  * a sheet's LINEs are picked apart, not concatenated (3 crops != 33 digits)
  * a two-row plate on the sheet still reconstructs
  * the 10-39 wagon-type correction runs and reports is_manipulated
  * OCR authority stays RIGHT_UP-only
  * missing client / model degrades to NO_DATA rather than raising
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
from features.inference_lib import ocr as OCRLIB
from features.inference_lib.ocr import three_frame_sheet as TFS
from features.inference_lib.ocr import rekognition_reader as RR
from features.ocr import processor as OCR


# -----------------------------------------------------------------------------
# three_frame_sheet — selection
# -----------------------------------------------------------------------------

def test_triplet_offsets_match_spec():
    n = 30
    assert TFS.select_frame_positions(n, "empty") == [3, 5, 7]
    assert TFS.select_frame_positions(n, "empty", use_fallback=True) == [22, 24, 26]
    assert TFS.select_frame_positions(n, "loaded") == [22, 24, 26]
    assert TFS.select_frame_positions(n, "loaded", use_fallback=True) == [3, 5, 7]
    assert TFS.select_frame_positions(n, "loco") == [13, 15, 17]


def test_triplet_positions_clamp_on_short_segments():
    pos = TFS.select_frame_positions(4, "empty")
    assert pos == [3, 3, 3]                       # clamped, never out of range
    assert all(0 <= p < 4 for p in pos)


def test_unknown_category_rejected():
    with pytest.raises(ValueError):
        TFS.select_frame_positions(10, "banana")


# -----------------------------------------------------------------------------
# three_frame_sheet — assembly
# -----------------------------------------------------------------------------

def test_vertical_sheet_stacks_verbatim_with_gutters():
    a = np.full((20, 50, 3), 10, np.uint8)
    b = np.full((30, 80, 3), 20, np.uint8)
    sheet = TFS.build_vertical_sheet([a, b], spacing=20)
    assert sheet.shape == (20 + 30 + 20, 80, 3)   # widest width, summed height
    # crops copied pixel-exact, left-aligned
    assert np.array_equal(sheet[0:20, 0:50], a)
    assert np.array_equal(sheet[40:70, 0:80], b)
    # gutter is white
    assert np.all(sheet[20:40, :] == 255)


def test_horizontal_sheet_lays_left_to_right():
    a = np.full((20, 50, 3), 10, np.uint8)
    b = np.full((30, 80, 3), 20, np.uint8)
    sheet = TFS.build_horizontal_sheet([a, b], spacing=20)
    assert sheet.shape == (30, 50 + 80 + 20, 3)


def test_sheet_requires_at_least_one_crop():
    with pytest.raises(ValueError):
        TFS.build_vertical_sheet([None, None])


def test_grayscale_crop_promoted_to_bgr():
    sheet = TFS.build_vertical_sheet([np.zeros((10, 10), np.uint8)])
    assert sheet.ndim == 3 and sheet.shape[2] == 3


def test_preprocess_upscales_to_min_width():
    out = OCRLIB.Preprocessor.primary(np.zeros((10, 20, 3), np.uint8))
    assert out.shape[1] >= OCRLIB.Preprocessor.OCR_TARGET_WIDTH


# -----------------------------------------------------------------------------
# rekognition_reader — picking a sheet apart
# -----------------------------------------------------------------------------

class _Client:
    """Returns a fixed LINE list; counts calls."""
    def __init__(self, lines):
        self.lines, self.calls = lines, 0
    def detect_text(self, image_bytes):
        self.calls += 1
        return [{"Type": "LINE", "DetectedText": t, "Confidence": c,
                 "Geometry": {"BoundingBox": {"Top": top}}}
                for top, t, c in self.lines]


IMG = np.full((40, 120, 3), 200, np.uint8)


def test_sheet_lines_are_picked_not_concatenated():
    """Three crops of the same plate must yield ONE 11-digit number."""
    c = _Client([(0.1, "22142319215", 96.0),
                 (0.4, "22142319215", 92.0),
                 (0.7, "22142319215", 90.0)])
    text, conf = RR.read_digits(c, IMG, validator=OCRLIB.is_valid_wagon_number)
    assert text == "22142319215"                  # not 33 digits
    assert conf == pytest.approx(0.96)            # highest-confidence LINE


def test_two_row_plate_reconstructs_from_consecutive_lines():
    """A plate split across two rows joins; separate crops do not."""
    c = _Client([(0.10, "221423", 95.0),
                 (0.18, "19215", 93.0),
                 (0.60, "999", 99.0)])
    text, conf = RR.read_digits(c, IMG, validator=OCRLIB.is_valid_wagon_number)
    assert text == "22142319215"
    assert conf == pytest.approx((95.0 + 93.0) / 2 / 100.0)


def test_without_validator_legacy_concatenation():
    c = _Client([(0.1, "221423", 90.0), (0.5, "19215", 90.0)])
    text, _ = RR.read_digits(c, IMG)
    assert text == "22142319215"


def test_nothing_valid_falls_back_to_concatenation():
    c = _Client([(0.1, "12", 80.0), (0.5, "34", 80.0)])
    text, _ = RR.read_digits(c, IMG, validator=OCRLIB.is_valid_wagon_number)
    assert text == "1234"                         # best-effort, still reported


def test_reader_survives_api_failure():
    class _Boom:
        def detect_text(self, b):
            raise RuntimeError("throttled")
    assert RR.read_digits(_Boom(), IMG) == ("", 0.0)


def test_loco_validator_is_five_digits():
    assert OCRLIB.is_valid_loco_number("44014") is True
    assert OCRLIB.is_valid_loco_number("4401") is False


# -----------------------------------------------------------------------------
# processor integration (stubbed YOLO + stubbed Rekognition)
# -----------------------------------------------------------------------------

class _StubBox:
    def __init__(self, bbox, conf):
        self.xyxy = [_Np(np.array(bbox, dtype=float))]
        self.conf = [np.float32(conf)]
        self.cls = [np.int64(0)]


class _Np:
    def __init__(self, a):
        self._a = a
    def cpu(self):
        return self
    def numpy(self):
        return self._a


class _StubResult:
    def __init__(self, boxes):
        self.boxes = boxes
        self.names = {0: "wagon_id"}


class _StubYolo:
    """One plate detection per frame."""
    names = {0: "wagon_id"}
    def predict(self, source=None, **kw):
        return [_StubResult([_StubBox([10, 10, 60, 30], 0.9)])]


class _SheetClient:
    """Counts DetectText calls; returns 3 identical LINEs (a 3-crop sheet)."""
    def __init__(self, text="22142319215", conf=97.0):
        self.calls = 0
        self.text, self.conf = text, conf
    def detect_text(self, image_bytes):
        self.calls += 1
        return [{"Type": "LINE", "DetectedText": self.text, "Confidence": self.conf,
                 "Geometry": {"BoundingBox": {"Top": t}}} for t in (0.1, 0.4, 0.7)]


@pytest.fixture
def batch(tmp_path):
    cache, states, evid = (tmp_path / n for n in
                           ("wagon_cache", "wagon_states", "evidence"))
    for d in (cache, states, evid):
        d.mkdir(parents=True, exist_ok=True)

    wagons = [GlobalWagon(global_id="GW_1", wagon_index=1,
                          start_frame_master=0, end_frame_master=99,
                          start_time=0.0, end_time=4.0,
                          classification=C.CLASS_WAGON)]
    state = GlobalTrainState(total_wagons=1, wagons=wagons,
                             master_camera=C.CAMERA_RIGHT_UP,
                             master_fps=25.0, master_total_frames=100)

    d = cache / "GW_1" / C.CAMERA_FOLDER[C.CAMERA_RIGHT_UP]
    d.mkdir(parents=True, exist_ok=True)
    for f in range(0, 40):
        cv2.imwrite(str(d / f"frame_{f:06d}.jpg"),
                    np.full((80, 160, 3), 180, np.uint8))
    return {"state": state, "cache": str(cache), "states": str(states),
            "evid": str(evid)}


def _run(batch, monkeypatch, client, model=None):
    monkeypatch.setattr("features._common.load_yolo",
                        lambda p: model if model is not None else _StubYolo())
    monkeypatch.setattr(OCR, "_get_client", lambda: client)
    return OCR.run(state=batch["state"], cache_root=batch["cache"],
                   feature_models_dir="/nonexistent",
                   output_dir=batch["states"], evidence_root=batch["evid"],
                   verbose=False)


def _payload(batch):
    with open(os.path.join(batch["states"], "ocr", C.CAMERA_RIGHT_UP,
                           "GW_1.json")) as f:
        return json.load(f)


def test_one_sheet_one_call_for_three_frames(batch, monkeypatch):
    client = _SheetClient()
    assert _run(batch, monkeypatch, client)["GW_1"] == C.STATUS_OK
    d = _payload(batch)
    assert d["wagon_identifier"] == "22142319215"
    assert d["engine"] == "rekognition"
    assert d["is_manipulated"] is False
    assert d["wagon_identifier_confidence"] == pytest.approx(0.97, abs=1e-3)
    # 34 frames in the stable interior, 1 band -> ONE sheet -> ONE API call
    assert client.calls == 1
    assert d["bands"] == 1
    assert d["fallback_triggered"] is False


def test_fallback_sheet_tried_when_primary_fails(batch, monkeypatch):
    client = _SheetClient(text="263131")             # never valid
    _run(batch, monkeypatch, client)
    d = _payload(batch)
    assert d["wagon_identifier"] == C.NO_DATA
    assert d["wagon_identifier_confidence"] == 0.0
    assert client.calls == 2                         # primary + fallback sheet
    assert d["fallback_triggered"] is True
    # Upstream contract: when NOTHING on the sheet validates, read_digits falls
    # back to concatenating every LINE -- so a 3-crop sheet of the same failed
    # plate reports the text three times over.  Best-effort, never a number.
    assert d["original_number"] == "263131" * 3


def test_wagon_type_correction_flags_manipulation(batch, monkeypatch):
    _run(batch, monkeypatch, _SheetClient(text="91142319215"))
    d = _payload(batch)
    assert d["wagon_identifier"] == "31142319215"    # 9x -> 3x
    assert d["is_manipulated"] is True
    assert d["original_number"] == "91142319215"     # raw pre-correction


def test_evidence_persists_the_sheet_that_was_read(batch, monkeypatch):
    _run(batch, monkeypatch, _SheetClient())
    ev = os.path.join(batch["evid"], "GW_1", "ocr", C.CAMERA_RIGHT_UP)
    assert os.path.isfile(os.path.join(ev, C.OCR_SHEET_FILENAME))
    assert os.path.isfile(os.path.join(ev, "best_frame.jpg"))
    assert os.path.isfile(os.path.join(ev, "number_crop.jpg"))
    with open(os.path.join(ev, "metadata.json")) as f:
        meta = json.load(f)
    # keys dashboard_ingest reads off OCR evidence
    assert meta["engine"] == "rekognition"
    assert meta["is_manipulated"] is False
    assert meta["original_number"] == "22142319215"
    # the verification trail: which file was read, and from which frames
    assert meta["ocr_input_image"] == C.OCR_SHEET_FILENAME
    assert len(meta["sheet_frames"]) == 3
    assert meta["frame_idx"] in meta["sheet_frames"]


def test_sheet_crop_and_frame_are_three_distinct_images(batch, monkeypatch):
    """number_sheet is the OCR input; number_crop/best_frame are its source."""
    _run(batch, monkeypatch, _SheetClient())
    ev = os.path.join(batch["evid"], "GW_1", "ocr", C.CAMERA_RIGHT_UP)
    sheet, crop, frame = (cv2.imread(os.path.join(ev, n)) for n in
                          (C.OCR_SHEET_FILENAME, "number_crop.jpg",
                           "best_frame.jpg"))
    assert frame.shape[:2] == (80, 160)             # the full cached frame
    # the 10,10-60,30 plate box, padded by WagonImageEnhancer's 0.25
    assert crop.shape[:2] == (30, 72)
    # 3 stacked crops + 2 gutters, then upscaled -> taller than one crop
    assert sheet.shape[0] > crop.shape[0] * 3


def test_persisted_sheet_is_byte_identical_to_what_rekognition_read(batch, monkeypatch):
    """The verification claim only holds if the saved file IS the posted image.
    Guards against `read_digits` and `save_jpeg` drifting on JPEG quality."""
    class _Recording(_SheetClient):
        posted = None
        def detect_text(self, image_bytes):
            _Recording.posted = image_bytes
            return super().detect_text(image_bytes)

    _run(batch, monkeypatch, _Recording())
    saved = open(os.path.join(batch["evid"], "GW_1", "ocr", C.CAMERA_RIGHT_UP,
                              C.OCR_SHEET_FILENAME), "rb").read()
    assert saved == _Recording.posted


def test_sheet_still_written_when_source_frame_is_unrecoverable(batch, monkeypatch):
    """An unreadable source frame must not cost the wagon its OCR evidence."""
    monkeypatch.setattr(OCR, "_best_frame_images", lambda *a, **k: (None, None))
    _run(batch, monkeypatch, _SheetClient())
    ev = os.path.join(batch["evid"], "GW_1", "ocr", C.CAMERA_RIGHT_UP)
    for name in (C.OCR_SHEET_FILENAME, "number_crop.jpg", "best_frame.jpg"):
        assert os.path.isfile(os.path.join(ev, name))


def test_authority_is_right_up_only(batch, monkeypatch):
    client = _SheetClient()
    monkeypatch.setattr("features._common.load_yolo", lambda p: _StubYolo())
    monkeypatch.setattr(OCR, "_get_client", lambda: client)
    for cam in (C.CAMERA_LEFT_UP, C.CAMERA_RIGHT_UP_TOP, C.CAMERA_LEFT_UP_TOP):
        assert OCR.run(state=batch["state"], cache_root=batch["cache"],
                       feature_models_dir="/nonexistent",
                       output_dir=batch["states"], cameras=[cam],
                       verbose=False) == {}
    assert client.calls == 0            # never billed for a non-master camera


def test_brake_van_skipped_without_api_calls(batch, monkeypatch):
    batch["state"].wagons[0].classification = C.CLASS_BRAKE_VAN
    client = _SheetClient()
    _run(batch, monkeypatch, client)
    d = _payload(batch)
    assert d["status"] == C.STATUS_OK and d["wagon_identifier"] == C.NO_DATA
    assert "classification=BRAKE_VAN" in d["skipped_reason"]
    assert client.calls == 0


# -----------------------------------------------------------------------------
# loco (ENGINE) branch
# -----------------------------------------------------------------------------

def test_engine_reads_five_digit_loco_number(batch, monkeypatch):
    batch["state"].wagons[0].classification = C.CLASS_ENGINE
    client = _SheetClient(text="44014", conf=98.0)     # valid 5-digit
    assert _run(batch, monkeypatch, client)["GW_1"] == C.STATUS_OK
    d = _payload(batch)
    assert d["segment_role"] == "loco"
    assert d["loco_id"] == 1
    assert d["loco_number"] == "44014"
    assert d["is_valid_5_digit"] is True
    assert d["loco_confidence"] == pytest.approx(0.98, abs=1e-3)
    assert d["fallback_triggered"] is False
    # the Middle-2/Middle/Middle+2 sheet read on the first call
    assert client.calls == 1
    # an engine never claims an 11-digit wagon number
    assert d["wagon_identifier"] == C.NO_DATA


def test_loco_falls_back_to_best_frame(batch, monkeypatch):
    batch["state"].wagons[0].classification = C.CLASS_ENGINE
    client = _SheetClient(text="22142319215")          # 11 digits, never a loco
    _run(batch, monkeypatch, client)
    d = _payload(batch)
    assert d["is_valid_5_digit"] is False
    assert d["loco_number"] == "-"
    assert d["fallback_triggered"] is True
    assert client.calls == 2                           # sheet, then best frame


def test_loco_evidence_gallery_and_metadata(batch, monkeypatch):
    batch["state"].wagons[0].classification = C.CLASS_ENGINE
    _run(batch, monkeypatch, _SheetClient(text="44014"))
    ev = os.path.join(batch["evid"], "GW_1", "ocr", C.CAMERA_RIGHT_UP)
    # 4-position report gallery + the OCR sheet + the fallback crop
    assert os.path.isfile(os.path.join(ev, "loco_001_sheet.jpg"))
    for pos in ("start", "mid1", "mid2", "end"):
        assert os.path.isfile(os.path.join(ev, f"loco_001_{pos}.jpg"))
    with open(os.path.join(ev, "metadata.json")) as f:
        meta = json.load(f)
    assert meta["segment_role"] == "loco"
    assert meta["loco_id"] == 1
    assert meta["loco_number"] == "44014"
    assert meta["is_valid_5_digit"] is True
    assert {r["position"] for r in meta["loco_frames"]} == {
        "start", "mid1", "mid2", "end"}
    # the sheet won -> that is the image the dashboard must show
    assert meta["ocr_input_image"] == "loco_001_sheet.jpg"
    assert meta["ocr_input_role"] == "sheet"
    assert len(meta["sheet_frames"]) == 3


def test_loco_ocr_input_points_at_the_fallback_when_sheet_fails(batch, monkeypatch):
    """When the single-frame fallback produced the reading, the verification
    image must be that frame -- not the sheet that failed."""
    batch["state"].wagons[0].classification = C.CLASS_ENGINE
    _run(batch, monkeypatch, _SheetClient(text="22142319215"))
    ev = os.path.join(batch["evid"], "GW_1", "ocr", C.CAMERA_RIGHT_UP)
    with open(os.path.join(ev, "metadata.json")) as f:
        meta = json.load(f)
    assert meta["ocr_input_role"] == "best"
    assert meta["ocr_input_image"].startswith("loco_001_frame_")
    assert os.path.isfile(os.path.join(ev, meta["ocr_input_image"]))
    assert meta["sheet_filename"] == "loco_001_sheet.jpg"


def test_loco_ids_are_sequential_in_rake_order(batch, monkeypatch):
    """Two engines -> loco_id 1 and 2, in rake order."""
    from core.global_state_loader import GlobalWagon
    second = GlobalWagon(global_id="GW_2", wagon_index=2,
                         start_frame_master=100, end_frame_master=199,
                         start_time=4.0, end_time=8.0,
                         classification=C.CLASS_ENGINE)
    batch["state"].wagons[0].classification = C.CLASS_ENGINE
    batch["state"].wagons.append(second)
    src = os.path.join(batch["cache"], "GW_1", C.CAMERA_FOLDER[C.CAMERA_RIGHT_UP])
    dst = os.path.join(batch["cache"], "GW_2", C.CAMERA_FOLDER[C.CAMERA_RIGHT_UP])
    os.makedirs(dst, exist_ok=True)
    for fn in os.listdir(src):
        cv2.imwrite(os.path.join(dst, fn), np.full((80, 160, 3), 170, np.uint8))

    _run(batch, monkeypatch, _SheetClient(text="44014"))
    for gw, expected in (("GW_1", 1), ("GW_2", 2)):
        with open(os.path.join(batch["states"], "ocr", C.CAMERA_RIGHT_UP,
                               f"{gw}.json")) as f:
            assert json.load(f)["loco_id"] == expected


def test_missing_client_degrades_to_no_data(batch, monkeypatch):
    assert _run(batch, monkeypatch, None)["GW_1"] == C.NO_DATA
    assert _payload(batch)["status"] == C.NO_DATA


def test_load_state_read_across_cameras(batch):
    d = os.path.join(batch["states"], "load", C.CAMERA_RIGHT_UP_TOP)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "GW_1.json"), "w") as f:
        json.dump({"status": C.STATUS_OK, "load_status": C.LOAD_LOADED}, f)
    assert OCR._wagon_is_loaded(batch["states"], "GW_1") is True
