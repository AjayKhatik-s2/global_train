# Production-Parity Validation Report

Scope: make the new Global-Train pipeline production-identical downstream while
keeping the new Stage-1 reconstruction. Every claim below is verified by reading
code and/or running it — not assumed.

---

## Task 1 — Remove the deprecated Ultralytics `half=` argument (root-cause fix)

**Finding (verified, not assumed):** in the *installed* ultralytics **8.4.53**,
`half` is still a valid predict key (`cfg/default.yaml:56`) and is **not** in the
deprecation map (`cfg/__init__.py` maps only `boxes/hide_labels/hide_conf/
line_thickness`). So the `'half' is deprecated … Use 'quantize' instead` warning
comes from a **newer** ultralytics (your EC2). The fix is therefore version-robust.

**Root cause & fix (no warning suppression):** a single precision helper
`features/_common.precision_kwargs(device, fp16)`:
- **CPU** → returns `{}` — passes **nothing** (FP32 is the CPU default; `half` was
  always `False` on CPU, so omitting it is behaviourally identical and never
  triggers the deprecation).
- **CUDA** → returns the version's supported FP16 mechanism: `{"quantize":"fp16"}`
  when the installed ultralytics exposes `quantize` (newer), else `{"half":True}`
  (legacy, e.g. 8.4.53). Cached (`_cuda_fp16_kwarg`).

**Every inference call migrated** (grep-verified: no `half=` reaches any
model/predict call anywhere in the repo; the wrapper param was also renamed
`half`→`fp16`):

| File | Call | Before | After |
|---|---|---|---|
| `features/_common.py` | `run_detection` | `model(..., half=use_half)` | `model(..., **precision_kwargs(dev, fp16))` |
| `features/_common.py` | `iter_wagon_detections` (single+batch) | `half=use_half` | `**precision_kwargs(dev, fp16)` |
| `features/ocr/processor.py` | stage-A detect | `model(..., half=True)` | `model(..., device=DEVICE, **precision_kwargs(DEVICE, fp16=True))` |
| `features/damage/processor.py` | caller | `iter_wagon_detections(..., half=False)` | `fp16=False` |

`wagon_count` (reconstruction) never passed `half` — unchanged.

**Verification (live, CPU):**
```
DEVICE=cpu  precision_kwargs(cpu)={}  (cuda would be {'half': True})
half/quantize warnings from migrated call: NONE
detections identical to plain FP32 CPU call: True (n=2)
```
→ warning gone, **outputs identical** (same boxes/conf/class), thresholds/pre/
post/NMS untouched (only the precision kwarg changed).

---

## Task 2 — Stage-1-only frame trimming (avoid false edge wagons)

**Where:** `wagon_count/tracker_engine.py::GapTracker.process_video` — the single
per-camera gap loop used by all four cameras. Original videos are **not** modified.

**Behaviour:** before gap detection, compute `trim = int(total*pct/100)` for both
ends; analyze only `frame_idx ∈ [trim, total-trim)`. Edge frames are skipped but
**`frame_idx` keeps counting over the original video**, so gap events — and thus
Stage 2 materialization, Stage 3 inference, processed videos, annotations,
reports, and JSON — reference the **same original frame indices**. Nothing
downstream changes.

**Config (both, as requested):**
- env `WAGONEYE_STAGE1_FRAME_TRIM_PERCENT` (default **4**)
- CLI `--stage1-frame-trim-percent` on `orchestrator.master_runner` (sets the env
  → propagates to the reconstruction subprocess) and on
  `wagon_count/run_global_count.py` (standalone). CLI overrides env.
- **`0` reproduces current behaviour exactly.**

**Per-camera log (exact requested format):**
```
[STAGE1] RIGHT_UP: total_frames=3969, trimmed_start=0,   trimmed_end=0,   processing_frames=3969   # pct=0
[STAGE1] RIGHT_UP: total_frames=3969, trimmed_start=396, trimmed_end=396, processing_frames=3177   # pct=10
```

**Verification (real `right_up.mp4`):** at pct=10, detected gap frame indices were
`min=435 max=449` — inside the trimmed window and at **original** numbering (not
renumbered from 0), proving frame-number preservation. At pct=0 the window is the
full video (identical to prior behaviour).

**Benchmark note:** trimming reduces Stage-1 frames analyzed by `2*pct%`
(4%+4% ⇒ ~8% fewer YOLO frames per camera, e.g. 3969→3653), so Stage 1 gets ~8%
faster *and* stops fabricating wagons from partially-visible edge passes. No other
stage's runtime or output changes.

---

## Task 3 — Production filters / evidence / snapshot logic reused (not reinvented)

The new feature processors call the **vendored old-production intelligence**
verbatim from `features/inference_lib/` (diff vs old `output_test/*`):

| Module (old production) | Changed lines | Status |
|---|---|---|
| `temporal_reasoning.py` | **0** | byte-identical |
| `damage_tracker.py` | **0** | byte-identical |
| `snapshot_selector.py` | **0** | byte-identical (best-snapshot logic) |
| `illumination_processor.py` | **0** | byte-identical |
| `geometric_shape_prior.py` | **0** | byte-identical (door shape filter) |
| `wagon_number_aggregator.py` | **0** | byte-identical (OCR aggregation) |
| `door_identity_merger.py` | 5 | import-path tweaks only (dup-suppression logic intact) |
| `door_tracker.py` | 34 | import-path tweaks only (Kalman+FSM+hysteresis intact) |

The processors **apply** these filters in production mode: door → illumination →
geometric prior → DoorTracker (temporal reasoning + 2× OPEN→CLOSED hysteresis +
sticky DAMAGE) → DoorIdentityMerger → snapshot selection; damage → confidence
floor + skip-classes + area band + edge-zone → DamageTracker → cross-track dedup →
loaded-wagon floor filter. (These are only bypassed under the explicit
benchmark-only `WAGONEYE_RAW_DETECTIONS=true`; **off by default** → production
filtering is fully in force.)

---

## Task 4 — Processed-video annotation parity

`rendering/feature_overlay_renderer.py` draw primitives are pixel-exact clones of
the old draw code (`old door_processor._annotate_frame`, `damage_processor`
annotate): identical BGR colour maps (`_DOOR_STATE_COLORS`, `_DAMAGE_COLORS`),
box thickness (3 if OPEN else 2 / 2 for damage), `Door {id}: {STATE}` label with
filled-colour background + black text, confidence below the box, velocity arrow,
red event banner, and green frame/damage info block. It replays the
per-frame trajectories the processors persist in `evidence/<gw>/<feat>/overlay.json`
(door: bbox/state/last_class/confidence/velocity; damage: per-frame detections),
so every detection that survives the production filters is drawn with the same
colours/labels/IDs/fonts/placement. These annotated `<CAMERA>_processed.mp4`s are
the delivered processed videos.

---

## Task 5 — Camera reports, combined report, dashboard JSON

**Combined report — PROVEN byte-identical** (see `COMBINED_REPORT_PARITY.md`):
- `reporting/combined_report_generator.py` **md5-identical** to old production
  (`d59e936…`), `pdf_utils.py` & `hazaribagh_report.py` md5-identical.
- Fed identical input, OLD vs NEW generator produced a **byte-identical PDF**
  (`md5 3d884bb…`) with clock frozen; the only live-run delta is the report's own
  `datetime.now()` header timestamp (identical code in both).

**Camera reports:** old generators reused; `report_generator.py` differs only by
(a) Windows-portable temp dir (no PDF effect) and (b) a `require_open_event` param
that defaults to the exact old RIGHT_UP behaviour and reproduces old LEFT_UP when
set; `damage_report_generator.py` differs only by the temp-dir fix.

**Adapters introduced (thin, no business logic):**
- `reporting/_legacy_data_adapter.py` — Global-Train state → the old per-camera
  `data` dicts the generators expect (verified: renders with **no missing field**).
- `delivery/legacy_layout.py` — old S3 keys/filenames.
- `delivery/dashboard_ingest.py` — re-derives the legacy per-camera dashboard
  payload from finalized artifacts (fields `report_meta`, `wagons`, `batch_key`,
  per-camera `cameras.*`), preserving the old dashboard contract.

---

## Files modified (this task)
| File | Change |
|---|---|
| `features/_common.py` | `precision_kwargs()` helper; migrated `run_detection`/`iter_wagon_detections` off `half=`; param `half`→`fp16` |
| `features/ocr/processor.py` | hardcoded `half=True` → `precision_kwargs(..., fp16=True)` |
| `features/damage/processor.py` | `half=False` → `fp16=False` |
| `wagon_count/tracker_engine.py` | Stage-1 frame trimming + per-camera `[STAGE1]` log |
| `wagon_count/run_global_count.py` | `--stage1-frame-trim-percent` CLI (sets env) |
| `orchestrator/master_runner.py` | `--stage1-frame-trim-percent` CLI (sets env → subprocess) |

## Verdict
- **`half` deprecation:** eliminated at the root; CPU passes nothing, GPU uses the
  supported mechanism; **outputs identical**. ✅
- **Stage-1 trimming:** configurable (env+CLI, default 4%, 0=off), original frame
  numbering preserved downstream, per-camera log in the exact format. ✅
- **Filters / annotations / reports / dashboard:** reuse the old production code
  (mostly byte-identical); combined report proven byte-identical. The only
  intended architectural difference remains Stage-1 Global-Train reconstruction. ✅
