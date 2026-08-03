#!/usr/bin/env python3
"""Deep runtime verification -- surface mid-batch failures BEFORE a train arrives.

`preflight.py` checks the dependencies it knows to name.  This goes further and
exercises what only actually runs deep inside a batch:

    1. IMPORT EVERY MODULE in the pipeline.  A missing or broken import inside,
       say, `reporting/damage_report_generator.py` is invisible until Stage 5 --
       an hour into a batch.  Importing everything up front turns that into an
       immediate, obvious failure.
    2. LOAD EVERY MODEL.  A model file can exist, be the right size, and still be
       unloadable (truncated download, LFS pointer, torch/ultralytics version
       mismatch).  `preflight` only checks presence; this actually constructs it.
    3. ffmpeg on PATH -- needed by the overlay-video compression in Stage 6.
    4. Free disk vs one batch's working set (~3 GB).
    5. Writable workspace + log dir.

Exit 0 = nothing here will break a run.  Non-zero = fix it before a train lands.

    python scripts/verify_runtime.py
    python scripts/verify_runtime.py --skip-models     # fast, imports only
"""

from __future__ import annotations

import argparse
import importlib
import os
import pkgutil
import shutil
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
# wagon_count is a standalone package: its modules import each other by bare name.
_WC = os.path.join(_REPO, "wagon_count")
if _WC not in sys.path:
    sys.path.insert(0, _WC)

_TTY = sys.stdout.isatty()


def _c(code, s):
    return f"\033[{code}m{s}\033[0m" if _TTY else s


def ok(s):   return _c("1;32", s)
def bad(s):  return _c("1;31", s)
def warn(s): return _c("1;33", s)
def hdr(s):  return _c("1;36", s)


#: Packages walked for the import check.
PACKAGES = ("core", "features", "orchestrator", "fusion", "materializer",
            "rendering", "reconstruction", "reporting", "delivery",
            "train_extraction")

#: Modules that are deliberately NOT importable in isolation, with the reason.
SKIP_IMPORT = {
    # wagon_count runs as a subprocess with its own sys.path; its modules import
    # siblings by bare name, so importing them as `wagon_count.x` fails by design.
    "wagon_count",
}


def check_imports() -> list:
    print(hdr("\n[1/5] Import every pipeline module"))
    names: list = []
    for pkg in PACKAGES:
        pkg_dir = os.path.join(_REPO, pkg)
        if not os.path.isdir(pkg_dir):
            continue
        names.append(pkg)
        for mod in pkgutil.walk_packages([pkg_dir], prefix=f"{pkg}."):
            names.append(mod.name)

    failures = []
    for name in sorted(set(names)):
        if name.split(".")[0] in SKIP_IMPORT:
            continue
        try:
            importlib.import_module(name)
        except Exception as e:
            failures.append((name, f"{type(e).__name__}: {e}"))
    total = len(set(names))
    if failures:
        print(f"  [{bad('FAIL')}] {total - len(failures)}/{total} modules import")
        for name, err in failures:
            print(f"         {bad(name)}: {err[:160]}")
    else:
        print(f"  [{ok('PASS')}] all {total} modules import cleanly")

    # wagon_count separately, on its own path (how the subprocess sees it)
    wc_fail = []
    for mod in ("global_train_state", "global_alignment", "tracker_engine",
                "video_segmenter"):
        try:
            importlib.import_module(mod)
        except Exception as e:
            wc_fail.append((mod, f"{type(e).__name__}: {e}"))
    if wc_fail:
        print(f"  [{bad('FAIL')}] wagon_count (Stage 1)")
        for m, e in wc_fail:
            print(f"         {bad(m)}: {e[:160]}")
        failures.extend(wc_fail)
    else:
        print(f"  [{ok('PASS')}] wagon_count Stage-1 modules import cleanly")
    return failures


def check_models(skip: bool) -> list:
    print(hdr("\n[2/5] Load every model (not just check it exists)"))
    if skip:
        print(f"  [{warn('SKIP')}] --skip-models")
        return []
    from core import config as CFG
    from core import constants as C

    targets = [(CFG.RECON_MODELS_DIR, f) for f in C.RECON_MODEL_FILES]
    targets.append((CFG.RECON_MODELS_DIR, C.MODEL_TOP_CLASSIFICATION))
    for key, fn in C.FEATURE_MODEL_BY_KEY.items():
        targets.append((CFG.FEAT_MODELS_DIR, fn))
    if CFG.PIPELINE_SOURCE.requires_extraction:
        targets += [(CFG.EXTRACTION_MODELS_DIR, f) for f in C.EXTRACTION_MODEL_FILES]

    failures = []
    for d, fn in targets:
        path = C.feature_model_path(d, fn)
        label = f"{os.path.basename(d)}/{fn}"
        if not os.path.isfile(path):
            print(f"  [{warn('SKIP')}] {label}  (absent -- optional here)")
            continue
        try:
            from features._common import load_yolo
            m = load_yolo(path)
            if m is None:
                raise RuntimeError("loader returned None (LFS pointer or unreadable)")
            names = getattr(m, "names", {}) or {}
            print(f"  [{ok('PASS')}] {label}  ({len(names)} classes)")
            if fn == C.MODEL_WAGON_NUMBER:
                failures += _check_plate_classes(label, names)
        except Exception as e:
            print(f"  [{bad('FAIL')}] {label}  {type(e).__name__}: {str(e)[:120]}")
            failures.append((label, str(e)))
    return failures


def _check_plate_classes(label: str, names) -> list:
    """The plate detector serves BOTH OCR paths, keyed on its class names.

    The wagon path is permissive (unknown vocabulary -> every box is a candidate),
    but the loco path refuses to guess -- so a model whose loco class is named
    something unexpected reads NOTHING, and does it silently.  Catch that here.
    """
    from features.ocr.processor import (plate_classes_resolvable,
                                        LOCO_NUMBER_CLASS_ALIASES)
    values = list((names or {}).values())
    servable = plate_classes_resolvable(values)
    print(f"           classes: {sorted(str(v) for v in values)}")
    if servable["loco"]:
        print(f"  [{ok('PASS')}] {label}  serves both wagon (11-digit) and "
              f"loco (5-digit) OCR")
        return []
    print(f"  [{warn('WARN')}] {label}  no locomotive class -- loco-number OCR "
          f"will be skipped")
    print(f"           expected one of: {sorted(LOCO_NUMBER_CLASS_ALIASES)}")
    print(f"           wagon-number OCR is unaffected")
    return []


def check_tools() -> list:
    print(hdr("\n[3/5] External tools"))
    failures = []
    for tool, why, fatal in (
        ("ffmpeg", "Stage-6 overlay-video compression + extraction trimming", True),
        ("ffprobe", "video metadata", False),
        ("git-lfs", "model weights are LFS-tracked", False),
    ):
        p = shutil.which(tool)
        if p:
            print(f"  [{ok('PASS')}] {tool}  ({p})")
        elif fatal:
            print(f"  [{bad('FAIL')}] {tool} not on PATH -- {why}")
            failures.append((tool, "missing"))
        else:
            print(f"  [{warn('WARN')}] {tool} not on PATH -- {why}")
    return failures


def check_disk() -> list:
    print(hdr("\n[4/5] Disk headroom (one batch needs ~3 GB)"))
    from core import config as CFG
    failures = []
    for label, path in (("workspace", CFG.WORKSPACE_ROOT), ("logs", CFG.LOG_DIR)):
        try:
            os.makedirs(path, exist_ok=True)
            free_gb = shutil.disk_usage(path).free / 1e9
        except OSError as e:
            print(f"  [{bad('FAIL')}] {label} ({path}): {e}")
            failures.append((label, str(e)))
            continue
        writable = os.access(path, os.W_OK)
        if not writable:
            print(f"  [{bad('FAIL')}] {label} ({path}) not writable")
            failures.append((label, "not writable"))
        elif free_gb < 3:
            print(f"  [{bad('FAIL')}] {label}: only {free_gb:.1f} GB free "
                  f"-- a batch needs ~3 GB")
            failures.append((label, f"{free_gb:.1f} GB free"))
        elif free_gb < 10:
            print(f"  [{warn('WARN')}] {label}: {free_gb:.1f} GB free (tight)")
        else:
            print(f"  [{ok('PASS')}] {label}: {free_gb:.1f} GB free")
    return failures


def check_ocr() -> list:
    print(hdr("\n[5/5] OCR engine reachability"))
    from core import config as CFG
    if CFG.OCR_ENGINE != "rekognition":
        print(f"  [{warn('SKIP')}] engine is {CFG.OCR_ENGINE}")
        return []
    from core import rekognition as REK
    client = REK.get_client()
    if client is None:
        print(f"  [{bad('FAIL')}] no Rekognition client "
              f"(boto3/credentials/region) -- OCR would fall back to easyocr")
        return [("rekognition", "no client")]
    print(f"  [{ok('PASS')}] Rekognition client ready (region={REK.REKOGNITION_REGION}, "
          f"max {REK.MAX_CALLS_PER_WAGON} calls/wagon)")
    return []


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Deep runtime verification")
    ap.add_argument("--skip-models", action="store_true",
                    help="don't actually load the .pt files (faster)")
    args = ap.parse_args(argv)

    print(hdr("=" * 70))
    print(hdr("WagonEye deep runtime verification"))
    print(hdr("=" * 70))

    failures = []
    failures += check_imports()
    failures += check_models(args.skip_models)
    failures += check_tools()
    failures += check_disk()
    failures += check_ocr()

    print(hdr("\n" + "=" * 70))
    if failures:
        print(bad(f"{len(failures)} problem(s) that could break a run:"))
        for name, err in failures:
            print(f"  - {name}: {str(err)[:140]}")
        return 1
    print(ok("ALL CLEAR -- nothing here will break a batch mid-run."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
