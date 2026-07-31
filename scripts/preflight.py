#!/usr/bin/env python3
"""WagonEye v4 pre-flight validator.

Verifies EVERY prerequisite before the pipeline is started and prints a clear
PASS/FAIL report.  Exit code 0 = all good; non-zero = at least one blocker.

Checks:
    1. Python dependencies (import each required package).
    2. Runtime directories exist + are writable (workspace, logs, models, tmp).
    3. Effective configuration summary (env vars + defaults) for the chosen mode.
    4. Model availability + optional S3 auto-sync (reconstruction + the feature
       models for the ENABLED features).  Reports the exact missing filename,
       expected s3://bucket/key, and reason.
    5. AWS / S3 connectivity + IAM (unless --no-aws): default-session credentials,
       output-bucket access, and model-bucket GetObject access.

Usage:
    python scripts/preflight.py                       # local mode, check only
    python scripts/preflight.py --mode auto           # validate continuous run
    python scripts/preflight.py --disable-features door,ocr,load   # Damage-only
    python scripts/preflight.py --sync                # actually DOWNLOAD missing models
    python scripts/preflight.py --no-aws              # skip AWS checks (offline)
"""

from __future__ import annotations

import argparse
import os
import sys

# Make the repo root importable when run as `python scripts/preflight.py`.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# ANSI (auto-disabled when not a TTY).
_TTY = sys.stdout.isatty()
def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _TTY else s
def _ok(s: str) -> str:   return _c("1;32", s)
def _bad(s: str) -> str:  return _c("1;31", s)
def _warn(s: str) -> str: return _c("1;33", s)
def _hdr(s: str) -> str:  return _c("1;36", s)


class Result:
    def __init__(self) -> None:
        self.failures: list = []
        self.warnings: list = []

    def check(self, ok: bool, label: str, detail: str = "") -> None:
        tag = _ok("PASS") if ok else _bad("FAIL")
        print(f"  [{tag}] {label}" + (f"  -- {detail}" if detail else ""))
        if not ok:
            self.failures.append(label)

    def warn(self, label: str, detail: str = "") -> None:
        print(f"  [{_warn('WARN')}] {label}" + (f"  -- {detail}" if detail else ""))
        self.warnings.append(label)


# Packages the pipeline actually imports at runtime (hard blockers).
REQUIRED_IMPORTS = [
    "cv2", "numpy", "scipy", "PIL", "torch", "torchvision",
    "ultralytics", "reportlab", "boto3", "requests",
]
# Declared in requirements.txt but not imported by the current code paths
# (kept for parity / future use).  Missing => WARN, never a blocker.
OPTIONAL_IMPORTS = ["easyocr", "filterpy", "imageio_ffmpeg"]


def check_deps(res: Result) -> None:
    print(_hdr("\n[1/5] Python dependencies"))
    import importlib
    for m in REQUIRED_IMPORTS:
        try:
            importlib.import_module(m)
            res.check(True, m)
        except Exception as e:
            res.check(False, m, f"import failed: {e}  (pip install -r requirements.txt)")
    for m in OPTIONAL_IMPORTS:
        try:
            importlib.import_module(m)
            print(f"  [{_ok('PASS')}] {m} (optional)")
        except Exception:
            hint = ("needed only for the OCR feature; pip install easyocr"
                    if m == "easyocr" else
                    "declared in requirements.txt but not imported by current code")
            res.warn(f"{m} (optional)", hint)


def check_dirs(res: Result) -> None:
    from core import config as CFG
    import tempfile
    print(_hdr("\n[2/5] Runtime directories (exist + writable)"))
    dirs = [
        ("WORKSPACE_ROOT", CFG.WORKSPACE_ROOT), ("LOG_DIR", CFG.LOG_DIR),
        ("MODELS_DIR", CFG.MODELS_DIR),
        ("RECON_MODELS_DIR", CFG.RECON_MODELS_DIR),
        ("FEAT_MODELS_DIR", CFG.FEAT_MODELS_DIR),
        ("TMPDIR", tempfile.gettempdir()),
    ]
    # Only relevant when this process produces its own trimmed clips.
    if CFG.PIPELINE_SOURCE.requires_extraction:
        dirs.append(("EXTRACTION_MODELS_DIR", CFG.EXTRACTION_MODELS_DIR))
    for name, d in dirs:
        try:
            os.makedirs(d, exist_ok=True)
            res.check(os.access(d, os.W_OK), f"{name} ({d})",
                      "" if os.access(d, os.W_OK) else "not writable")
        except OSError as e:
            res.check(False, f"{name} ({d})", str(e))


def check_config(res: Result, mode: str, disabled: list) -> None:
    from core import config as CFG
    from core.feature_config import FeatureConfig
    print(_hdr("\n[3/5] Effective configuration"))
    print(CFG.startup_summary(mode=mode))
    fc = FeatureConfig.from_disabled(disabled)
    print(f"  enabled features         : {fc.enabled_keys()}")
    print(f"  disabled features        : {fc.disabled_keys() or '(none)'}")
    errs = CFG.validate_config(mode=mode, skip_upload=(mode == "local"),
                               skip_email=(mode == "local"))
    for e in errs:
        res.check(False, "config", e)
    if not errs:
        res.check(True, f"config valid for mode={mode}")


def check_models(res: Result, disabled: list, do_sync: bool) -> None:
    from core import model_sync as MS
    from core.feature_config import FeatureConfig
    print(_hdr("\n[4/5] Model availability" + (" + S3 sync" if do_sync else "")))
    enabled = FeatureConfig.from_disabled(disabled).enabled_keys()
    report = MS.verify_and_sync(enabled_features=enabled, download=do_sync)
    for line in report.summary_lines():
        print(line)
    res.check(report.ok, "all required models present",
              "" if report.ok else f"{len(report.missing)} missing "
              "(run with --sync once WAGONEYE_MODELS_S3_BUCKET is set, or "
              "`git lfs pull`)")


def check_aws(res: Result) -> None:
    from core import constants as C
    print(_hdr("\n[5/5] AWS / S3 connectivity + IAM"))
    try:
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError
    except Exception as e:
        res.check(False, "boto3 import", str(e))
        return
    try:
        sts = boto3.client("sts", region_name=C.S3_REGION)
        ident = sts.get_caller_identity()
        res.check(True, "AWS credentials", f"arn={ident.get('Arn','?')}")
    except NoCredentialsError:
        res.check(False, "AWS credentials",
                  "none found -- attach an EC2 IAM role or set ~/.aws/credentials")
        return
    except Exception as e:
        res.check(False, "AWS credentials (STS get-caller-identity)", str(e))
        return

    s3 = boto3.client("s3", region_name=C.S3_REGION)
    # output bucket (upload target)
    if C.S3_OUTPUT_BUCKET:
        try:
            s3.head_bucket(Bucket=C.S3_OUTPUT_BUCKET)
            res.check(True, f"output bucket reachable ({C.S3_OUTPUT_BUCKET})")
        except Exception as e:
            res.check(False, f"output bucket ({C.S3_OUTPUT_BUCKET})",
                      f"{type(e).__name__}: {e}")
    else:
        res.warn("output bucket unset (WAGONEYE_S3_OUTPUT_BUCKET)")

    # model bucket GetObject (IAM check on one recon key)
    if C.MODELS_S3_BUCKET:
        key = (f"{C.MODELS_S3_PREFIX}/reconstruction/{C.MODEL_SIDE_CLASSIFICATION}"
               if C.MODELS_S3_PREFIX
               else f"reconstruction/{C.MODEL_SIDE_CLASSIFICATION}")
        try:
            s3.head_object(Bucket=C.MODELS_S3_BUCKET, Key=key)
            res.check(True, f"model bucket GetObject ({C.MODELS_S3_BUCKET}/{key})")
        except Exception as e:
            code = ""
            try:
                code = e.response.get("Error", {}).get("Code", "")  # type: ignore[attr-defined]
            except Exception:
                pass
            hint = ("object missing -- check MODELS_S3_PREFIX/layout"
                    if code in ("404", "NoSuchKey")
                    else "IAM lacks s3:GetObject" if code in ("403", "AccessDenied")
                    else "")
            res.check(False, f"model bucket access ({C.MODELS_S3_BUCKET}/{key})",
                      f"{code or type(e).__name__} {hint}")
    else:
        res.warn("model bucket unset (WAGONEYE_MODELS_S3_BUCKET)",
                 "auto model-sync disabled; models must be present locally")

    # Rekognition DetectText -- the default OCR engine.  A 1x1 JPEG is the
    # cheapest possible real call and proves both the IAM permission and that
    # DetectText exists in this region.  InvalidImageFormatException also proves
    # both (the request was authorized and reached the service), so it passes.
    from core import config as CFG
    if CFG.OCR_ENGINE != "rekognition":
        print(f"  [{_ok('SKIP')}] Rekognition (WAGONEYE_OCR_ENGINE="
              f"{CFG.OCR_ENGINE})")
        return
    from core.rekognition import REKOGNITION_REGION
    try:
        rek = boto3.client("rekognition", region_name=REKOGNITION_REGION)
    except Exception as e:
        res.check(False, f"Rekognition client ({REKOGNITION_REGION})", str(e))
        return
    tiny_jpeg = bytes.fromhex(
        "ffd8ffe000104a46494600010100000100010000ffdb004300ffffffffffffffff"
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
        "ffffffffffc00011080001000101011100ffc40014000100000000000000000000"
        "0000000000ffda0008010100003f00d2cf20ffd9")
    try:
        rek.detect_text(Image={"Bytes": tiny_jpeg})
        res.check(True, f"Rekognition DetectText ({REKOGNITION_REGION})")
    except Exception as e:
        code = ""
        try:
            code = e.response.get("Error", {}).get("Code", "")  # type: ignore[attr-defined]
        except Exception:
            pass
        if code in ("InvalidImageFormatException", "InvalidParameterException",
                    "ImageTooLargeException"):
            # Reached + authorized the service; it just disliked the probe image.
            res.check(True, f"Rekognition DetectText ({REKOGNITION_REGION})",
                      f"reachable (probe rejected: {code})")
        elif code in ("AccessDeniedException", "AccessDenied", "403"):
            res.check(False, f"Rekognition DetectText ({REKOGNITION_REGION})",
                      "IAM lacks rekognition:DetectText -- grant it, or set "
                      "WAGONEYE_OCR_ENGINE=easyocr")
        else:
            res.check(False, f"Rekognition DetectText ({REKOGNITION_REGION})",
                      f"{code or type(e).__name__}: {e}  (check the region "
                      f"supports DetectText, or set WAGONEYE_OCR_ENGINE=easyocr)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="WagonEye v4 pre-flight validator")
    ap.add_argument("--mode", default="local",
                    choices=["auto", "local", "once", "batch"])
    ap.add_argument("--disable-features", default="",
                    help="comma-separated: door,ocr,load,damage")
    ap.add_argument("--sync", action="store_true",
                    help="actually DOWNLOAD missing models from S3 (not just check)")
    ap.add_argument("--no-aws", action="store_true", help="skip AWS/S3 checks")
    args = ap.parse_args(argv)

    from core.feature_config import parse_disable_arg
    disabled = parse_disable_arg(args.disable_features)

    print(_hdr("=" * 70))
    print(_hdr(f"WagonEye v4 pre-flight  (mode={args.mode}, "
               f"disabled={disabled or '[]'})"))
    print(_hdr("=" * 70))

    res = Result()
    check_deps(res)
    check_dirs(res)
    check_config(res, args.mode, disabled)
    check_models(res, disabled, do_sync=args.sync)
    if not args.no_aws:
        check_aws(res)
    else:
        print(_hdr("\n[5/5] AWS / S3 -- skipped (--no-aws)"))

    print(_hdr("\n" + "=" * 70))
    if res.failures:
        print(_bad(f"PRE-FLIGHT FAILED: {len(res.failures)} blocker(s): "
                   + ", ".join(res.failures)))
        if res.warnings:
            print(_warn(f"warnings: {', '.join(res.warnings)}"))
        return 1
    print(_ok("PRE-FLIGHT PASSED -- all prerequisites satisfied."))
    if res.warnings:
        print(_warn(f"warnings (non-blocking): {', '.join(res.warnings)}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
