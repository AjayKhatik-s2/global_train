"""Historical (time-range) mode: window resolution, S3 selection, clustering,
and -- most importantly -- proof that the live `--auto` path is unchanged.

No AWS, no models, no inference: the S3 client is a fake and `process_batch` is
stubbed, so these assert the SELECTION layer and the isolation guarantees only.

Run:  python tests/test_historical_mode.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from core import config as CFG
from core import constants as C
from orchestrator import historical_runner as HR
from orchestrator import train_batch_manager as TBM

IST = CFG.IST


# -----------------------------------------------------------------------------
# fakes
# -----------------------------------------------------------------------------

class FakeS3:
    """Minimal stand-in for the boto3 client surface the discovery path uses.

    `get_object` / `put_object` raise: the ONLY reason historical mode would
    call them is to touch `processed_batches.json`, which it must never do.
    """

    def __init__(self, objects):
        # objects: [(key, last_modified, etag, size), ...]
        self.objects = list(objects)
        self.list_calls = []

    def list_objects_v2(self, **kw):
        self.list_calls.append(kw)
        prefix = kw.get("Prefix", "")
        contents = [
            {"Key": k, "LastModified": lm, "ETag": f'"{e}"', "Size": s}
            for (k, lm, e, s) in self.objects if k.startswith(prefix)
        ]
        return {"Contents": contents, "IsTruncated": False}

    def get_object(self, **kw):
        raise AssertionError("historical mode must not read live batch state")

    def put_object(self, **kw):
        raise AssertionError("historical mode must not write live batch state")

    def download_file(self, *a, **kw):
        raise AssertionError("no download expected in these tests")


def _key(camera_id: str, ts: str, suffix: str = "train") -> str:
    """A key in the real production shape: <camera folder>/<basename>_train.mp4."""
    folder = C.CAMERA_S3_FOLDER[camera_id]
    return f"{folder}/{folder}_{ts}_{suffix}.mp4"


def _obj(camera_id, ts, *, etag=None, size=1024, lm=None, suffix="train"):
    k = _key(camera_id, ts, suffix)
    return (k, lm or datetime(2026, 8, 8, 6, 0, tzinfo=timezone.utc),
            etag or k, size)


def _use_prefixes(monkey_prefixes=None):
    """Point discovery at the four real camera folders; returns a restore fn."""
    old_p, old_b = C.S3_INPUT_PREFIXES, C.S3_INPUT_BUCKET
    C.S3_INPUT_PREFIXES = (monkey_prefixes if monkey_prefixes is not None
                           else [C.CAMERA_S3_FOLDER[c] for c in C.ALL_CAMERAS])
    C.S3_INPUT_BUCKET = "test-trimmed-bucket"

    def restore():
        C.S3_INPUT_PREFIXES, C.S3_INPUT_BUCKET = old_p, old_b
    return restore


def _window(start="10:00", end="12:00", date="2026-08-08", tz="Asia/Kolkata"):
    return HR.resolve_window(date=date, start_time=start, end_time=end,
                             timezone_name=tz)


def _discover(objects, window=None, pad=HR.DEFAULT_PAD_MINUTES, prefixes=None):
    restore = _use_prefixes(prefixes)
    try:
        return HR.select_objects(s3_client=FakeS3(objects),
                                 window=window or _window(), pad_minutes=pad)
    finally:
        restore()


# -----------------------------------------------------------------------------
# timezone + window resolution
# -----------------------------------------------------------------------------

def test_timezone_conversion():
    w = _window()
    assert w.start.utcoffset() == timedelta(hours=5, minutes=30)
    assert w.start.astimezone(timezone.utc).hour == 4     # 10:00 IST -> 04:30 UTC
    assert w.start.astimezone(timezone.utc).minute == 30
    assert w.end.astimezone(timezone.utc).hour == 6

    # a DIFFERENT zone must shift the absolute instant
    utc_w = HR.resolve_window(date="2026-08-08", start_time="10:00",
                              end_time="12:00", timezone_name="UTC")
    assert utc_w.start.astimezone(timezone.utc).hour == 10
    assert utc_w.start != w.start

    # ISO form: the offset in the string wins over --timezone
    iso = HR.resolve_window(start_iso="2026-08-08T10:00:00+05:30",
                            end_iso="2026-08-08T12:00:00+05:30",
                            timezone_name="UTC")
    assert iso.start == w.start and iso.end == w.end

    # naive ISO adopts --timezone
    naive = HR.resolve_window(start_iso="2026-08-08T10:00:00",
                              end_iso="2026-08-08T12:00:00",
                              timezone_name="Asia/Kolkata")
    assert naive.start == w.start


def test_window_validation_and_overnight():
    over = _window(start="22:00", end="02:00")
    assert over.rolled_overnight is True
    assert (over.end - over.start) == timedelta(hours=4)

    for kw in (dict(date="2026-08-08", start_time="10:00"),          # no end
               dict(date="2026-08-08", start_time="99:00", end_time="12:00"),
               dict(start_iso="2026-08-08T12:00:00+05:30",
                    end_iso="2026-08-08T10:00:00+05:30"),            # end <= start
               dict(start_iso="2026-08-08T10:00:00+05:30"),          # start w/o end
               dict(date="2026-08-08", start_time="10:00", end_time="12:00",
                    start_iso="2026-08-08T10:00:00+05:30",
                    end_iso="2026-08-08T12:00:00+05:30")):           # both forms
        try:
            HR.resolve_window(**kw)
            raise AssertionError(f"expected ValueError for {kw}")
        except ValueError:
            pass


def test_filename_timestamps_are_ist_matching_the_producer():
    """The tz we attach must equal what the module that WRITES the names uses."""
    from train_extraction.time_utils import parse_timestamp_from_filename
    for ts in ("20260808_101500", "20260808_235959", "20260101_000000"):
        name = f"{C.CAMERA_S3_FOLDER[C.CAMERA_RIGHT_UP]}_{ts}_train.mp4"
        producer = parse_timestamp_from_filename(name)
        ours = HR.filename_timestamp_local(ts)
        assert producer is not None and ours is not None
        assert producer == ours, (ts, producer, ours)
        assert ours.utcoffset() == timedelta(hours=5, minutes=30)


# -----------------------------------------------------------------------------
# selection: boundaries, overlap, padding
# -----------------------------------------------------------------------------

def test_exact_boundary_conditions():
    # pad=0 so coverage is the instant itself and the boundaries are crisp
    objs = [_obj(C.CAMERA_RIGHT_UP, ts) for ts in
            ("20260808_095959",     # 1s before start  -> out
             "20260808_100000",     # exactly start    -> IN (inclusive)
             "20260808_110000",     # middle           -> IN
             "20260808_120000",     # exactly end      -> IN (inclusive)
             "20260808_120001")]    # 1s after end     -> out
    res = _discover(objs, pad=0.0)
    got = sorted(s.train_timestamp for s in res.selected)
    assert got == ["20260808_100000", "20260808_110000", "20260808_120000"], got


def test_clip_starting_before_the_window_is_kept_when_it_can_still_cover_it():
    # A raw clip starts at 09:52 and runs 5 min; a train inside it can still
    # begin at 09:57... but our window starts at 10:00, so only the pad decides.
    objs = [_obj(C.CAMERA_RIGHT_UP, "20260808_095200")]

    assert _discover(objs, pad=0.0).selected == []
    assert _discover(objs, pad=5.0).selected == []          # covers to 09:57 only
    kept = _discover(objs, pad=15.0).selected               # covers to 10:07
    assert len(kept) == 1
    assert "before the window" in kept[0].reason
    assert kept[0].covers_until_local.strftime("%H:%M") == "10:07"

    # a clip starting after the window end is never rescued by padding
    assert _discover([_obj(C.CAMERA_RIGHT_UP, "20260808_130000")],
                     pad=600.0).selected == []


def test_no_operational_day_cutoff_in_historical_mode():
    """The live cutoff would hide the archive; historical must see it."""
    old = datetime(2026, 1, 1, 6, 0, tzinfo=timezone.utc)     # months stale
    objs = [_obj(C.CAMERA_RIGHT_UP, "20260808_101500", lm=old)]
    res = _discover(objs)
    assert len(res.selected) == 1, "historical must not apply the live cutoff"

    # ...and the live path DOES still apply it (unchanged behaviour)
    restore = _use_prefixes()
    try:
        live = TBM.list_candidate_videos(FakeS3(objs))
    finally:
        restore()
    assert live == [], "live discovery must keep its operational-day cutoff"


# -----------------------------------------------------------------------------
# batching
# -----------------------------------------------------------------------------

def test_four_cameras_form_one_batch():
    objs = [_obj(C.CAMERA_RIGHT_UP, "20260808_101500"),
            _obj(C.CAMERA_LEFT_UP, "20260808_101504"),
            _obj(C.CAMERA_RIGHT_UP_TOP, "20260808_101512"),
            _obj(C.CAMERA_LEFT_UP_TOP, "20260808_101527")]
    res = _discover(objs)
    assert len(res.batches) == 1
    b = res.batches[0]
    assert b.is_complete() and b.batch_key == "20260808_101500"
    assert sorted(b.videos) == sorted(C.ALL_CAMERAS)


def test_multiple_trains_stay_separate_batches():
    objs = []
    for ts in ("20260808_101500", "20260808_104500", "20260808_113000"):
        for cam in C.ALL_CAMERAS:
            objs.append(_obj(cam, ts))
    res = _discover(objs)
    assert len(res.batches) == 3
    keys = [b.batch_key for b in res.batches]
    assert keys == sorted(keys) == ["20260808_101500", "20260808_104500",
                                    "20260808_113000"]
    # never merged: each batch holds exactly its own four clips
    for b in res.batches:
        assert b.is_complete()
        assert {cv.train_timestamp for cv in b.videos.values()} == {b.batch_key}


def test_missing_camera_is_marked_not_substituted():
    objs = [_obj(C.CAMERA_RIGHT_UP, "20260808_101500"),
            _obj(C.CAMERA_LEFT_UP, "20260808_101504"),
            _obj(C.CAMERA_RIGHT_UP_TOP, "20260808_101512")]
    # a DIFFERENT train's LEFT_UP_TOP exists but must not be borrowed
    objs.append(_obj(C.CAMERA_LEFT_UP_TOP, "20260808_113000"))
    res = _discover(objs)

    first = next(b for b in res.batches if b.batch_key == "20260808_101500")
    assert first.missing_cameras() == [C.CAMERA_LEFT_UP_TOP]
    assert C.CAMERA_LEFT_UP_TOP not in first.videos
    other = next(b for b in res.batches if b.batch_key == "20260808_113000")
    assert list(other.videos) == [C.CAMERA_LEFT_UP_TOP]


def test_duplicate_objects_are_deduped_like_the_live_path():
    ts = "20260808_101500"
    newer = datetime(2026, 8, 8, 7, 0, tzinfo=timezone.utc)
    older = datetime(2026, 8, 8, 5, 0, tzinfo=timezone.utc)
    objs = [
        _obj(C.CAMERA_RIGHT_UP, ts, suffix="train_incomplete", etag="INC", lm=newer),
        _obj(C.CAMERA_RIGHT_UP, ts, suffix="train", etag="COMPLETE", lm=older),
    ]
    res = _discover(objs)
    assert res.duplicates_dropped == 1
    assert len(res.selected) == 1
    # complete beats incomplete even though the incomplete one is newer
    assert res.selected[0].etag == "COMPLETE"
    assert "_train_incomplete" not in res.selected[0].key


def test_empty_results_are_reported_not_crashed():
    res = _discover([])
    assert res.selected == [] and res.batches == []
    msg = HR._no_match_message(res)
    for expect in ("test-trimmed-bucket", "2026-08-08 10:00:00", "pad"):
        assert expect in msg, msg

    # no prefixes configured at all -> same clean failure, message says so
    res2 = _discover([_obj(C.CAMERA_RIGHT_UP, "20260808_101500")], prefixes=[])
    assert res2.batches == []
    assert "none configured" in HR._no_match_message(res2)


# -----------------------------------------------------------------------------
# manifest
# -----------------------------------------------------------------------------

def test_manifest_contents_and_dry_run(capsys=None):
    objs = [_obj(cam, "20260808_101500", size=5_000_000) for cam in C.ALL_CAMERAS]
    objs += [_obj(cam, "20260808_113000") for cam in
             (C.CAMERA_RIGHT_UP, C.CAMERA_LEFT_UP)]
    res = _discover(objs)
    root = tempfile.mkdtemp()
    man = HR.build_manifest(res, workspace_root=root, dry_run=True)

    assert man["mode"] == "historical" and man["dry_run"] is True
    assert man["batches_discovered"] == 2
    assert man["requested_window"]["timezone"] == "Asia/Kolkata"
    assert man["requested_window"]["start_utc"].startswith("2026-08-08T04:30")
    assert man["search"]["bucket"] == "test-trimmed-bucket"
    assert man["search"]["objects_selected"] == 6

    b0 = man["batches"][0]
    assert b0["batch_key"] == "20260808_101500"
    assert sorted(b0["present_cameras"]) == sorted(C.ALL_CAMERAS)
    ru = b0["cameras"][C.CAMERA_RIGHT_UP]
    assert ru["s3_uri"].startswith("s3://test-trimmed-bucket/")
    assert ru["size_bytes"] == 5_000_000 and ru["train_timestamp"] == "20260808_101500"
    assert ru["clip_start_ist"].endswith("+05:30")
    assert b0["staged_inputs"].endswith(os.path.join("20260808_101500",
                                                     CFG.DIR_DOWNLOADS))
    assert man["batches"][1]["missing_cameras"] == [C.CAMERA_RIGHT_UP_TOP,
                                                    C.CAMERA_LEFT_UP_TOP]

    p = os.path.join(root, "m.json")
    assert HR._write_manifest(man, p) == p
    with open(p, encoding="utf-8") as f:
        assert json.load(f)["batches_discovered"] == 2


def test_dry_run_processes_nothing_and_touches_no_state():
    objs = [_obj(cam, "20260808_101500") for cam in C.ALL_CAMERAS]
    restore = _use_prefixes()
    root = tempfile.mkdtemp()
    called = []
    import orchestrator.master_runner as MR
    orig = MR.process_batch
    MR.process_batch = lambda **kw: called.append(kw)
    try:
        rc = HR.run(s3_client=FakeS3(objs), window=_window(),
                    workspace_root=root, recon_models_dir="/nope",
                    feat_models_dir="/nope", dry_run=True, verbose=False)
    finally:
        MR.process_batch = orig
        restore()
    assert rc == 0
    assert called == [], "--dry-run must not invoke the pipeline"
    # FakeS3.get_object/put_object would have raised -> live state untouched
    assert os.path.isfile(os.path.join(root, HR.HISTORICAL_SUBDIR,
                                       HR.MANIFEST_NAME))


def test_empty_window_exits_cleanly():
    restore = _use_prefixes()
    root = tempfile.mkdtemp()
    try:
        rc = HR.run(s3_client=FakeS3([]), window=_window(), workspace_root=root,
                    recon_models_dir="/nope", feat_models_dir="/nope",
                    dry_run=False, verbose=False)
    finally:
        restore()
    assert rc == 2, "no matching video must fail cleanly, not crash"


# -----------------------------------------------------------------------------
# execution: reuses the existing pipeline, isolated from live
# -----------------------------------------------------------------------------

class _Outcome:
    def __init__(self, status):
        self.final_status = status
        self.report_pdf_path = None
        self.report_json_path = None


def test_run_delegates_to_process_batch_and_isolates_from_live():
    objs = [_obj(cam, ts) for ts in ("20260808_101500", "20260808_113000")
            for cam in C.ALL_CAMERAS]
    restore = _use_prefixes()
    root = tempfile.mkdtemp()
    seen = []
    import orchestrator.master_runner as MR
    orig = MR.process_batch

    def fake(**kw):
        seen.append(kw)
        os.makedirs(os.path.join(kw["workspace_root"], kw["batch"].batch_key,
                                 CFG.DIR_DOWNLOADS), exist_ok=True)
        return _Outcome(C.BATCH_COMPLETED)

    MR.process_batch = fake
    try:
        rc = HR.run(s3_client=FakeS3(objs), window=_window(),
                    workspace_root=root, recon_models_dir="/m/recon",
                    feat_models_dir="/m/feat", verbose=False)
    finally:
        MR.process_batch = orig
        restore()

    assert rc == 0
    assert len(seen) == 2, "one call to the EXISTING pipeline per train"
    hist_root = os.path.join(root, HR.HISTORICAL_SUBDIR)
    for kw in seen:
        # isolation: never the live batch tree
        assert kw["workspace_root"] == hist_root
        assert not kw["workspace_root"].rstrip("/").endswith(CFG.DIR_WAGON_CACHE)
        # delivery off by default
        assert kw["skip_upload"] is True and kw["skip_email"] is True
        # existing config passed straight through
        assert kw["recon_models_dir"] == "/m/recon"
        assert kw["feat_models_dir"] == "/m/feat"
    assert [kw["batch"].batch_key for kw in seen] == ["20260808_101500",
                                                      "20260808_113000"]
    # staged inputs cleaned after success
    for kw in seen:
        assert not os.path.isdir(os.path.join(hist_root, kw["batch"].batch_key,
                                              CFG.DIR_DOWNLOADS))


def test_keep_inputs_and_failure_retention():
    objs = [_obj(cam, "20260808_101500") for cam in C.ALL_CAMERAS]
    import orchestrator.master_runner as MR
    orig = MR.process_batch

    def run_with(status, keep):
        restore = _use_prefixes()
        root = tempfile.mkdtemp()

        def fake(**kw):
            os.makedirs(os.path.join(kw["workspace_root"], kw["batch"].batch_key,
                                     CFG.DIR_DOWNLOADS), exist_ok=True)
            return _Outcome(status)

        MR.process_batch = fake
        try:
            rc = HR.run(s3_client=FakeS3(objs), window=_window(),
                        workspace_root=root, recon_models_dir="/n",
                        feat_models_dir="/n", keep_inputs=keep, verbose=False)
        finally:
            MR.process_batch = orig
            restore()
        d = os.path.join(root, HR.HISTORICAL_SUBDIR, "20260808_101500",
                         CFG.DIR_DOWNLOADS)
        return rc, os.path.isdir(d)

    assert run_with(C.BATCH_COMPLETED, True) == (0, True)      # --keep-inputs
    assert run_with(C.BATCH_COMPLETED, False) == (0, False)    # cleaned
    rc, kept = run_with(C.BATCH_REPORT_FAILED, False)
    assert rc == 3 and kept is True, "a failed batch keeps its inputs"


def test_deliver_flag_enables_upload_and_email():
    objs = [_obj(cam, "20260808_101500") for cam in C.ALL_CAMERAS]
    restore = _use_prefixes()
    root = tempfile.mkdtemp()
    seen = []
    import orchestrator.master_runner as MR
    orig = MR.process_batch
    MR.process_batch = lambda **kw: (seen.append(kw), _Outcome(C.BATCH_COMPLETED))[1]
    try:
        HR.run(s3_client=FakeS3(objs), window=_window(), workspace_root=root,
               recon_models_dir="/n", feat_models_dir="/n", deliver=True,
               verbose=False)
    finally:
        MR.process_batch = orig
        restore()
    assert seen[0]["skip_upload"] is False and seen[0]["skip_email"] is False


# -----------------------------------------------------------------------------
# REGRESSION: the live --auto path is unchanged
# -----------------------------------------------------------------------------

def test_auto_cli_is_unaffected_by_the_new_flags():
    from orchestrator.master_runner import _build_parser
    a = _build_parser().parse_args(["--auto", "--no-interactive"])
    assert a.auto is True
    # every historical flag is inert on the live command line
    assert a.historical is False and a.dry_run is False
    assert a.keep_inputs is False and a.historical_deliver is False
    assert (a.date, a.start_time, a.end_time, a.timezone) == (None,) * 4
    assert (a.start, a.end, a.pad_minutes, a.manifest_out) == (None,) * 4
    # pre-existing flags keep their defaults
    assert a.poll_interval == 60 and a.partial_wait == 30.0
    assert a.source is None and a.skip_upload is False and a.skip_email is False


def test_auto_dispatch_unchanged_end_to_end():
    """`--auto` must still validate as mode='auto' with the ORIGINAL skip flags
    and land in run_auto -- never in the historical branch."""
    import orchestrator.master_runner as MR

    logdir = tempfile.mkdtemp()
    os.environ["WAGONEYE_LOG_DIR"] = logdir
    seen = {}
    orig_auto, orig_hist = MR.run_auto, MR.run_historical
    orig_validate = MR.CFG.validate_config
    orig_summary = MR.CFG.startup_summary

    def _auto(**kw):
        seen["auto"] = kw
        return 0

    def _validate(**kw):
        seen["validate"] = kw
        return []

    def _summary(**kw):
        seen["summary"] = kw
        return ""

    MR.run_auto = _auto
    MR.run_historical = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("--auto must never enter the historical branch"))
    MR.CFG.validate_config = _validate
    MR.CFG.startup_summary = _summary
    try:
        rc = MR.main(["--auto", "--no-interactive", "--skip-model-sync"])
    finally:
        MR.run_auto, MR.run_historical = orig_auto, orig_hist
        MR.CFG.validate_config = orig_validate
        MR.CFG.startup_summary = orig_summary
        os.environ.pop("WAGONEYE_LOG_DIR", None)

    assert rc == 0
    assert seen["summary"]["mode"] == "auto"
    assert seen["validate"] == {"mode": "auto", "skip_upload": False,
                                "skip_email": False}
    kw = seen["auto"]
    assert kw["run_once"] is False and kw["force_batch_key"] is None
    assert kw["poll_interval"] == 60 and kw["partial_wait_minutes"] == 30.0
    assert kw["skip_upload"] is False and kw["skip_email"] is False
    assert kw["source"] is None


def test_historical_dispatch_never_enters_run_auto():
    import orchestrator.master_runner as MR

    logdir = tempfile.mkdtemp()
    os.environ["WAGONEYE_LOG_DIR"] = logdir
    seen = {}
    orig_auto, orig_hist = MR.run_auto, MR.run_historical
    orig_validate = MR.CFG.validate_config

    def _hist(args, **kw):
        seen["hist"] = args
        return 0

    def _validate(**kw):
        seen["validate"] = kw
        return []

    MR.run_auto = lambda **kw: (_ for _ in ()).throw(
        AssertionError("--historical must never enter run_auto"))
    MR.run_historical = _hist
    MR.CFG.validate_config = _validate
    try:
        rc = MR.main(["--historical", "--date", "2026-08-08",
                      "--start-time", "10:00", "--end-time", "12:00",
                      "--no-interactive", "--skip-model-sync"])
    finally:
        MR.run_auto, MR.run_historical = orig_auto, orig_hist
        MR.CFG.validate_config = orig_validate
        os.environ.pop("WAGONEYE_LOG_DIR", None)

    assert rc == 0
    # validated with the one-shot ruleset, delivery off -> email not required
    assert seen["validate"]["mode"] == "once"
    assert seen["validate"]["skip_upload"] is True
    assert seen["validate"]["skip_email"] is True
    assert seen["hist"].date == "2026-08-08"


def test_historical_preserves_existing_processing_flags():
    """--disable-features / --infer-batch / stage-1 trim must survive."""
    from orchestrator.master_runner import _build_parser
    a = _build_parser().parse_args([
        "--historical", "--date", "2026-08-08", "--start-time", "10:00",
        "--end-time", "12:00", "--timezone", "Asia/Kolkata",
        "--disable-features", "ocr", "--infer-batch", "24",
        "--stage1-frame-trim-percent", "5", "--raw-detections",
        "--workspace", "/w", "--keep-inputs", "--pad-minutes", "30",
    ])
    assert a.historical and a.disable_features == "ocr" and a.infer_batch == 24
    assert a.stage1_frame_trim_percent == 5.0 and a.raw_detections is True
    assert a.workspace == "/w" and a.keep_inputs is True and a.pad_minutes == 30.0


def test_live_discovery_module_still_behaves_identically():
    """Importing historical_runner must not perturb the live discovery path."""
    assert TBM.DEFAULT_BATCH_TOLERANCE_SEC == 120
    assert TBM.consumer_lookback_minutes() == 60.0
    cutoff, desc = TBM._discovery_cutoff()
    assert cutoff is not None and "operational day" in desc
    # historical reuses these helpers read-only; they are the live ones
    assert HR.cluster_into_batches.__defaults__[0] == TBM.DEFAULT_BATCH_TOLERANCE_SEC


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS  {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()


# -----------------------------------------------------------------------------
# dashboard ingest (process_batch does not do it; historical must)
# -----------------------------------------------------------------------------

def test_dashboard_ingest_runs_only_when_delivering():
    import orchestrator.master_runner as MR
    from delivery import dashboard_ingest as DI

    objs = [_obj(cam, "20260808_101500") for cam in C.ALL_CAMERAS]
    orig_pb, orig_run = MR.process_batch, DI.run
    calls = []

    class _Out(_Outcome):
        def __init__(self, root, batch):
            super().__init__(C.BATCH_COMPLETED)
            self.batch = batch
            self.report_pdf_url = "https://x/combined.pdf"
            self.camera_pdf_urls = {C.CAMERA_RIGHT_UP: "https://x/right_up.pdf"}

    def go(deliver, send_email=True):
        calls.clear()
        restore = _use_prefixes()
        root = tempfile.mkdtemp()

        def fake_pb(**kw):
            os.makedirs(os.path.join(kw["workspace_root"],
                                     kw["batch"].batch_key), exist_ok=True)
            fake_pb.kw = kw
            return _Out(root, kw["batch"])

        DI.run = lambda **kw: (calls.append(kw), {"enabled": True,
                                                  "cameras": {}})[1]
        MR.process_batch = fake_pb
        try:
            HR.run(s3_client=FakeS3(objs), window=_window(), workspace_root=root,
                   recon_models_dir="/n", feat_models_dir="/n",
                   deliver=deliver, send_email=send_email, verbose=False)
        finally:
            MR.process_batch, DI.run = orig_pb, orig_run
            restore()
        return fake_pb.kw, (calls[0] if calls else None)

    pb_kw, ing = go(deliver=False)
    assert ing is None, "no dashboard ingest without --historical-deliver"
    assert pb_kw["skip_upload"] is True and pb_kw["skip_email"] is True

    pb_kw, ing = go(deliver=True)
    assert ing is not None and ing["skip_upload"] is False
    assert ing["batch_root"].endswith(os.path.join(HR.HISTORICAL_SUBDIR,
                                                   "20260808_101500"))
    assert pb_kw["skip_upload"] is False and pb_kw["skip_email"] is False

    # dashboard yes, email no
    pb_kw, ing = go(deliver=True, send_email=False)
    assert ing is not None
    assert pb_kw["skip_upload"] is False and pb_kw["skip_email"] is True


def test_dashboard_marker_seeds_pdf_urls_and_never_raises():
    from delivery import finalization as FIN
    from delivery import dashboard_ingest as DI

    root = tempfile.mkdtemp()

    class _O:
        final_status = C.BATCH_COMPLETED
        report_pdf_url = "https://x/combined.pdf"
        camera_pdf_urls = {C.CAMERA_RIGHT_UP: "https://x/ru.pdf"}

        class batch:
            batch_key = "20260808_101500"

    orig = DI.run
    DI.run = lambda **kw: {"enabled": True, "cameras": {"RIGHT_UP": {}}}
    try:
        HR._dashboard_ingest(root, _O(), None)
    finally:
        DI.run = orig
    marker = FIN.load(root)
    assert marker and marker["upload_urls"]["pdf"] == "https://x/combined.pdf"
    assert marker["upload_urls"]["camera_RIGHT_UP"] == "https://x/ru.pdf"

    # a raising ingest is swallowed -- it must never fail the batch
    DI.run = lambda **kw: (_ for _ in ()).throw(RuntimeError("boom"))
    try:
        HR._dashboard_ingest(root, _O(), None)      # must not raise
    finally:
        DI.run = orig
