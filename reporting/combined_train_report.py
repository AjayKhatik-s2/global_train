"""Stage 5 combined report -- THIN wrapper over the OLD production
`CombinedReportGenerator`.

The previous new-pipeline rewrite (its own layout/branding) has been removed.
This module now does exactly one architectural thing: expose the Global Train
state through `_legacy_data_adapter` in the per-camera dict shape the unmodified
old `combined_report_generator.CombinedReportGenerator.generate()` consumes, and
call it.  The PDF layout, tables, colours, fonts, spacing, summary, and damaged-
wagon image pages are therefore byte-for-byte the old production report.

Public entry `build(...)` keeps its original signature so the orchestrator
(`orchestrator/lifecycle_runner.stage_reports`) is unchanged.  It returns
``{"json_path": ..., "pdf_path": ...|None}``.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from typing import Any, Dict, Optional, Sequence

from core import constants as C
from core.global_state_loader import GlobalTrainState
from core.unified_wagon_state import UnifiedWagonState, summarize_wagons

from . import _legacy_data_adapter as LDA
from .combined_report_generator import CombinedReportGenerator


# Local artifact filenames kept stable (delivery/finalization hashes them; the
# old S3 key/filename scheme is applied in the delivery stage, not here).
PDF_NAME = "combined_train_report.pdf"
JSON_NAME = "combined_train_report.json"


def _build_json(
    *, state: GlobalTrainState, unified: Dict[str, UnifiedWagonState],
    batch_key: str, payloads: Dict[str, Dict[str, Any]],
    report_meta: Optional[Dict[str, Any]], missing_cameras: Sequence[str],
    source_video_urls: Dict[str, str], processed_video_urls: Dict[str, str],
) -> Dict[str, Any]:
    """Canonical machine-readable JSON companion to the PDF.

    Carries the same per-camera legacy `data` dicts fed to the report generator
    (so downstream / dashboard consumers read exactly what the report rendered)
    plus a train-level summary + incremental-lifecycle meta.
    """
    wagons = [unified.get(w.global_id) for w in state.wagons]
    wagons = [u for u in wagons if u is not None]
    return {
        "batch_key": batch_key,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "report_meta": report_meta or {},
        "missing_cameras": list(missing_cameras),
        "summary": summarize_wagons(wagons),
        "master_camera": state.master_camera,
        "total_wagons": state.total_wagons,
        "source_video_urls": dict(source_video_urls or {}),
        "processed_video_urls": dict(processed_video_urls or {}),
        "cameras": {cam: payloads.get(cam, {}) for cam in C.ALL_CAMERAS},
        "wagons": [u.to_dict() for u in wagons],
    }


def build(
    *,
    state: GlobalTrainState,
    unified: Dict[str, UnifiedWagonState],
    output_dir: str,
    batch_key: str,
    source_video_urls: Optional[Dict[str, str]] = None,
    processed_video_urls: Optional[Dict[str, str]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
    evidence_root: Optional[str] = None,
    wagon_states_root: Optional[str] = None,
    cache_root: Optional[str] = None,
    missing_cameras: Optional[Sequence[str]] = None,
    camera_pdf_urls: Optional[Dict[str, str]] = None,
    logo_path: Optional[str] = None,
    report_meta: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Dict[str, Optional[str]]:
    """Build the combined PDF (old layout) + canonical JSON.  Never loads a model."""
    del extra_metadata, cache_root  # accepted for signature parity; not needed here
    os.makedirs(output_dir, exist_ok=True)
    source_video_urls = dict(source_video_urls or {})
    processed_video_urls = dict(processed_video_urls or {})
    camera_pdf_urls = dict(camera_pdf_urls or {})
    missing_cameras = list(missing_cameras or [])

    # 1) Thin translation: Global Train state -> old per-camera `data` dicts.
    payloads = LDA.build_camera_payloads(
        state=state, unified=unified,
        evidence_root=evidence_root, wagon_states_root=wagon_states_root,
        source_video_urls=source_video_urls,
        tracked_video_urls=processed_video_urls,
        camera_pdf_urls=camera_pdf_urls,
        session_id=batch_key,
    )
    sp = LDA.split_for_combined(payloads)

    # 2) Canonical JSON (always written, even if PDF fails).
    json_path = os.path.join(output_dir, JSON_NAME)
    json_doc = _build_json(
        state=state, unified=unified, batch_key=batch_key, payloads=payloads,
        report_meta=report_meta, missing_cameras=missing_cameras,
        source_video_urls=source_video_urls,
        processed_video_urls=processed_video_urls,
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_doc, f, indent=2, default=str)
    if verbose:
        print(f"[STAGE5] wrote {json_path}")

    # 3) Combined PDF via the unmodified old generator.
    pdf_path: Optional[str] = os.path.join(output_dir, PDF_NAME)
    t0 = time.time()
    try:
        gen = CombinedReportGenerator(
            output_path=pdf_path, logo_path=logo_path,
            left_report_url=camera_pdf_urls.get(C.CAMERA_LEFT_UP),
            right_report_url=camera_pdf_urls.get(C.CAMERA_RIGHT_UP),
            top_report_url=camera_pdf_urls.get(C.CAMERA_RIGHT_UP_TOP),
            left_top_report_url=camera_pdf_urls.get(C.CAMERA_LEFT_UP_TOP),
            left_video_url=source_video_urls.get(C.CAMERA_LEFT_UP),
            right_video_url=source_video_urls.get(C.CAMERA_RIGHT_UP),
            top_video_url=source_video_urls.get(C.CAMERA_RIGHT_UP_TOP),
            left_top_video_url=source_video_urls.get(C.CAMERA_LEFT_UP_TOP),
            left_tracked_url=processed_video_urls.get(C.CAMERA_LEFT_UP),
            right_tracked_url=processed_video_urls.get(C.CAMERA_RIGHT_UP),
            top_tracked_url=processed_video_urls.get(C.CAMERA_RIGHT_UP_TOP),
            left_top_tracked_url=processed_video_urls.get(C.CAMERA_LEFT_UP_TOP),
        )
        gen.generate(
            sp["left_data"], sp["right_data"], sp["top_data"], sp["left_top_data"],
            missing_cameras=missing_cameras,
        )
        if not os.path.isfile(pdf_path):
            pdf_path = None
        elif verbose:
            print(f"[STAGE5] wrote {pdf_path}")
    except Exception as e:
        print(f"[STAGE5] combined PDF FAILED: {type(e).__name__}: {e}")
        traceback.print_exc(limit=3)
        pdf_path = None

    if verbose:
        print(f"[STAGE5] combined report done in {time.time() - t0:.1f}s")
    return {"json_path": json_path, "pdf_path": pdf_path}
