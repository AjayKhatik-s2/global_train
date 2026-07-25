"""Reconstruction + feature model availability check (+ optional S3 sync).

PRIMARY model source is the repo itself: the `.pt` weights ship WITH THE CODE
(old pipeline: in each camera folder; this repo: Git LFS -- `git lfs pull`).
There is NO production model bucket.  This module's main job is therefore to
VERIFY that every model a run needs is present locally, and fail fast (naming
the exact missing file) if not -- the same guarantee the old `bootstrap.sh`
Step 5 gave.

OPTIONAL S3 sync: if (and only if) `WAGONEYE_MODELS_S3_BUCKET` is set, a model
missing locally is downloaded into the local model dir from:
    reconstruction:  s3://<MODELS_S3_BUCKET>/<MODELS_S3_PREFIX>/reconstruction/<file>
    features:        s3://<MODELS_S3_BUCKET>/<MODELS_S3_PREFIX>/features/<file>
(This mirrors, for the inference models, what train_extraction/model_store.py
already does for the extractor's classify models via s3:// URIs.)

Which models are required:
    * every reconstruction model, ALWAYS
    * one feature model per ENABLED feature (Damage-only -> just damage.pt)

Failures are never silent: each missing/failed model reports the exact filename,
the expected `s3://bucket/key` (when a bucket is set), and the reason (no bucket
configured -> run `git lfs pull`; NoSuchKey/404; AccessDenied/403; no creds).

Downloads are atomic (`<file>.part` -> rename) so a half-download can never be
loaded.  Nothing here loads a model or runs inference.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from core import config as CFG
from core import constants as C
from core.logging_setup import get_logger

log = get_logger("model_sync")


# ---------------------------------------------------------------------------
# What a run needs
# ---------------------------------------------------------------------------

@dataclass
class ModelReq:
    category: str          # "reconstruction" | "features"
    filename: str          # canonical filename (e.g. damage.pt)
    local_dir: str         # RECON_MODELS_DIR or FEAT_MODELS_DIR
    legacy: Optional[str] = None   # accepted legacy filename fallback

    @property
    def local_path(self) -> str:
        return os.path.join(self.local_dir, self.filename)

    @property
    def s3_key(self) -> str:
        base = f"{C.MODELS_S3_PREFIX}/{self.category}" if C.MODELS_S3_PREFIX \
            else self.category
        return f"{base}/{self.filename}"

    @property
    def s3_uri(self) -> str:
        return f"s3://{C.MODELS_S3_BUCKET}/{self.s3_key}"

    def existing_local(self) -> Optional[str]:
        """Return a present local path (canonical or legacy), else None."""
        if os.path.isfile(self.local_path):
            return self.local_path
        if self.legacy:
            legacy_path = os.path.join(self.local_dir, self.legacy)
            if os.path.isfile(legacy_path):
                return legacy_path
        return None


def required_models(enabled_features: Optional[List[str]] = None) -> List[ModelReq]:
    """Return the ModelReq list for a run.

    `enabled_features` restricts the feature models (default: all four).  The
    reconstruction set is always included.
    """
    reqs: List[ModelReq] = [
        ModelReq("reconstruction", f, CFG.RECON_MODELS_DIR,
                 legacy=C.RECON_MODEL_LEGACY.get(f))
        for f in C.RECON_MODEL_FILES
    ]
    keys = C.FEATURE_MODEL_BY_KEY.keys() if enabled_features is None \
        else [k for k in enabled_features if k in C.FEATURE_MODEL_BY_KEY]
    for k in keys:
        reqs.append(ModelReq("features", C.FEATURE_MODEL_BY_KEY[k],
                             CFG.FEAT_MODELS_DIR))
    return reqs


# ---------------------------------------------------------------------------
# Result of a verify/sync pass
# ---------------------------------------------------------------------------

@dataclass
class ModelStatus:
    req: ModelReq
    present: bool = False
    downloaded: bool = False
    local_path: Optional[str] = None
    error: Optional[str] = None       # human-readable reason when not present


@dataclass
class SyncReport:
    statuses: List[ModelStatus] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(s.present for s in self.statuses)

    @property
    def missing(self) -> List[ModelStatus]:
        return [s for s in self.statuses if not s.present]

    def summary_lines(self) -> List[str]:
        out: List[str] = []
        for s in self.statuses:
            if s.present and s.downloaded:
                out.append(f"  [downloaded] {s.req.category}/{s.req.filename}  <- {s.req.s3_uri}")
            elif s.present:
                out.append(f"  [present]    {s.req.category}/{s.req.filename}  ({s.local_path})")
            else:
                out.append(f"  [MISSING]    {s.req.category}/{s.req.filename}  "
                           f"expected s3 {s.req.s3_uri}  -- {s.error}")
        return out


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def _make_s3_client():
    """Best-effort default-session S3 client (IAM role or ~/.aws creds)."""
    try:
        import boto3
        return boto3.client("s3", region_name=C.S3_REGION)
    except Exception as e:  # pragma: no cover - boto3 always present via reqs
        log.warning("[MODEL_SYNC] could not create S3 client: %s", e)
        return None


def _download_reason(exc: Exception) -> str:
    """Turn a boto3 exception into an operator-actionable reason."""
    code = getattr(getattr(exc, "response", None) or {}, "get", lambda *_: None)("Error")
    err_code = ""
    try:
        err_code = (exc.response.get("Error", {}) or {}).get("Code", "")  # type: ignore[attr-defined]
    except Exception:
        err_code = ""
    if err_code in ("404", "NoSuchKey", "NoSuchBucket"):
        return f"S3 object/bucket not found ({err_code or 'NoSuchKey'}) -- check the key/prefix/bucket"
    if err_code in ("403", "AccessDenied"):
        return ("access denied (403 AccessDenied) -- the instance IAM role/user "
                "lacks s3:GetObject on this key")
    name = type(exc).__name__
    if name in ("NoCredentialsError", "PartialCredentialsError"):
        return "no AWS credentials found (attach an EC2 IAM role or set ~/.aws/credentials)"
    if name == "EndpointConnectionError":
        return "cannot reach S3 endpoint (network/VPC/region issue)"
    return f"{name}: {exc}"


def _download(s3_client, req: ModelReq) -> ModelStatus:
    """Download one model atomically.  Returns a populated ModelStatus."""
    st = ModelStatus(req=req)
    os.makedirs(req.local_dir, exist_ok=True)
    part = req.local_path + ".part"
    try:
        s3_client.download_file(C.MODELS_S3_BUCKET, req.s3_key, part)
        os.replace(part, req.local_path)
        st.present = True
        st.downloaded = True
        st.local_path = req.local_path
        log.info("[MODEL_SYNC] downloaded %s -> %s", req.s3_uri, req.local_path)
    except Exception as e:
        if os.path.exists(part):
            try:
                os.remove(part)
            except OSError:
                pass
        st.error = _download_reason(e)
        log.error("[MODEL_SYNC] FAILED %s : %s", req.s3_uri, st.error)
    return st


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def verify_and_sync(
    *,
    enabled_features: Optional[List[str]] = None,
    s3_client=None,
    download: bool = True,
) -> SyncReport:
    """Verify every required model is present locally; download missing ones.

    * A model already present locally (canonical or legacy name) is `present`,
      never re-downloaded.
    * A missing model is downloaded when `download` and a models bucket is
      configured; otherwise it is reported MISSING with the exact reason.
    """
    report = SyncReport()
    reqs = required_models(enabled_features)
    bucket_set = bool(C.MODELS_S3_BUCKET)
    client = s3_client
    if download and bucket_set and client is None:
        client = _make_s3_client()

    for req in reqs:
        local = req.existing_local()
        if local:
            report.statuses.append(ModelStatus(req=req, present=True,
                                               local_path=local))
            continue
        # missing locally
        if not download:
            report.statuses.append(ModelStatus(
                req=req, present=False,
                error="missing locally (sync disabled)"))
            continue
        if not bucket_set:
            report.statuses.append(ModelStatus(
                req=req, present=False,
                error=("missing locally and WAGONEYE_MODELS_S3_BUCKET is not set "
                       "-- either `git lfs pull` the weights or set the model "
                       "bucket for auto-sync")))
            continue
        if client is None:
            report.statuses.append(ModelStatus(
                req=req, present=False,
                error="missing locally and no S3 client/credentials available"))
            continue
        report.statuses.append(_download(client, req))

    return report


def ensure_models_or_report(
    *,
    enabled_features: Optional[List[str]] = None,
    s3_client=None,
    download: bool = True,
) -> SyncReport:
    """verify_and_sync + log a one-block summary.  Caller decides fail-fast."""
    report = verify_and_sync(enabled_features=enabled_features,
                             s3_client=s3_client, download=download)
    header = ("[MODEL_SYNC] model availability "
              f"(bucket={C.MODELS_S3_BUCKET or '<unset>'}, "
              f"prefix={C.MODELS_S3_PREFIX or '<root>'}):")
    log.info("%s\n%s", header, "\n".join(report.summary_lines()))
    if not report.ok:
        log.error("[MODEL_SYNC] %d model(s) unavailable -- see MISSING lines above.",
                  len(report.missing))
    return report
