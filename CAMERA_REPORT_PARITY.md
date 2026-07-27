# Camera-wise report parity — new Global Train vs old production

**Goal:** the four per-camera reports (`RIGHT_UP`, `LEFT_UP`, `RIGHT_UP_TOP`,
`LEFT_UP_TOP`) must be **identical** to old production. This document is the
line-by-line comparison, the root causes found, the fixes applied (reusing old
implementations verbatim), and the end-to-end verification.

Method: I did not eyeball PDFs. I loaded the **old** generator and the **new**
ported generator side by side, fed them **identical input** under reportlab
invariant mode with a frozen clock, and compared the PDF bytes (md5). Then I
rendered the new reports from the **real batch** (`batch_outputs/20260724_204400`)
and inspected the embedded images.

---

## 1. Per-camera generator files are byte-identical to old production

The two generators are the old production files, ported with only (a) a
`/tmp` → `tempfile` portability fix and (b) a `require_open_event` parameter that
folds the LEFT_UP-vs-RIGHT_UP behavioural difference into one class. To prove the
diffs are cosmetic, both were driven with an identical `data` dict:

| Generator | old md5 | new md5 | Result |
|---|---|---|---|
| `report_generator.DoorReportGenerator` (side) | `50aae601…48a7f` | `50aae601…48a7f` | **BYTE-IDENTICAL** |
| `damage_report_generator.DamageReportGenerator` (top) | `db1367bf…befb9` | `db1367bf…befb9` | **BYTE-IDENTICAL** |

Proofs: `scratchpad/compare_camera_report.py`, `scratchpad/compare_top_report.py`.
→ Layout, fonts, sizes, colours, margins, page/section order, headers/footers,
tables, summary pages, wagon pages, captions, image scaling, statistics and
report naming come straight from old production and are not re-implemented.

## 2. Invocation restored to match the old SageMaker call exactly

Old `sagemaker_main.py` called:

```python
# side  (LEFT_UP/sagemaker_main.py:3013)
generate_report(doors, state_counts, processing_time,
                door_open_events=…, wagon_summary=…, full_wagon_summary=…)
# top   (RIGHT_UP_TOP/sagemaker_main.py:2975)
generate_report(damages, state_counts, processing_time,
                damage_events=…, wagon_summary=…, full_wagon_summary=…)
```

`camera_reports.py` previously omitted `door_open_events`/`damage_events` and
`full_wagon_summary`. Both are now passed (`reporting/camera_reports.py::_build_one`).
Verified **no-op for rendering** (the generator recomputes its displayed counts
from `state_counts`, and `full_wagon_summary` falls back to `wagon_summary`, which
in the adapter already carries every segment incl. `is_non_wagon`): passing the
old arg set vs the previous minimal set produced a **byte-identical** PDF. The
call now matches old production regardless.

## 3. Root cause A — wagon-overview frames (the frame-range wiring)

The side/top wagon-overview pages extract 4 quartile frames per wagon from that
camera's own video (`_extract_wagon_snapshots_quad`). This silently returns
4×`None` when `end_frame - start_frame <= 0`. The per-camera **local** frame
ranges come from `global_state/per_camera_tracking.json` (fps + total_frames per
camera) via `_legacy_data_adapter._local_frames`. With that file, `gw.start_time`
/`gw.end_time` map to real frames (GW_1 → 81–188 @ 15 fps). The lifecycle already
passes it (`lifecycle_runner.stage_reports`, line 427/440).

Verified on the real batch **with** the pcf: `RIGHT_UP` embeds **248** wagon
overview frames (62 wagons × 4 at 960×540) — not the logo-only output produced
when the pcf is absent.

## 4. Root cause B — the door / damage snapshot crops (the real gap)

The per-camera generators embed the door/damage image from an **in-memory numpy
array** (`door['snapshot']` / `damage['snapshot']`), exactly as old production
passed it. The adapter only supplied a file **path**, so every door-detail page
rendered `[Snapshot not available]` and every damage-detail page the same. This
was the actual visible divergence.

Old production built those images with these exact steps, which are now reused
**verbatim**:

**Side / door** — `refined_crop_around_detection` then `annotate_snapshot`
(vendored in `features/inference_lib/snapshot_selector.py`; old
`door_processor.py:1629-1641`): crop centred on the door (40 % expand, min-60 %
frame, fallback ROI) with the bbox + `Door #N: STATE (conf%)` label drawn **on
the crop**.

**Top / damage** — draw a thick red bbox + `<class> (conf%)` label on the best
frame, **then** `_crop_around_detection` (old `damage_processor.py:62-116` +
`1213-1236`, ported verbatim into `features/damage/processor.py`).

### Files changed
- `features/door/processor.py` — imports the vendored old crop/annotate,
  ports old `STATE_COLORS_BGR`, writes `<side>_snapshot.jpg` (the annotated crop),
  registers `evidence_paths[<side>_snapshot]`.
- `features/damage/processor.py` — ports `_crop_around_detection` +
  `_damage_report_snapshot` (annotate-then-crop), writes `track_{i}_snapshot.jpg`,
  registers `evidence_paths[track_{i}_snapshot]`.
- `reporting/_legacy_data_adapter.py` — `_load_image()` (lazy cv2) loads those
  crops into `door['snapshot']` / `damage['snapshot']` (numpy). `local_snapshot_path`
  is unchanged so the combined report is untouched. Also adds `damage_id`.
- `reporting/camera_reports.py` — passes the old arg set (§2).

## 5. Restored old-production selection/rendering components (this pass)

The evidence/selection pipeline (geometric prior, DoorTracker temporal reasoning
+ hysteresis + identity merge, DamageTracker best-snapshot, confidence/edge/quality
scoring, `snapshot_score` = old `_score_detection`, raw bbox) was already restored
(see `VALIDATION_LEFTUP_SNAPSHOTS.md`). This pass restores the two that were still
diverging:

- **Snapshot cropping** — old `refined_crop_around_detection` (door) and
  `_crop_around_detection` (damage), not the new `safe_crop(pad)`.
- **Report-snapshot annotation rendering** — old `annotate_snapshot` (door) and
  the red-bbox + label annotate-then-crop (damage), not `draw_annotated_bbox`.

## 6. End-to-end verification (real batch, real videos)

Traced complete wagons through Stage 2→feature→fusion→adapter→report:

| Camera | Wagons | Door/Damage crop images embedded | `snapshot not available` |
|---|---|---|---|
| `RIGHT_UP` (side) | GW_3, GW_6 | 708×366, 740×374 (annotated crops) + 8×960×540 overview | **0** |
| `RIGHT_UP_TOP` (top) | GW_25, GW_26 | 947×580, 736×479, 639×551 (annotate-then-crop) | **0** |

Before the fix every door/damage detail page showed the placeholder; after, the
annotated crop appears and the placeholder count is 0.
Proofs: `scratchpad/verify_door_crop_render.py`, `scratchpad/verify_damage_crop_render.py`.

## 7. Combined report — not regressed

The combined generator reads doors via `door['local_snapshot_path']` and top
damages via `_local_snapshot_path`; the numpy `snapshot` key is only consumed for
side-camera damages, guarded by `hasattr(snapshot,'shape')`. The camera payloads
are never JSON-serialised. On the current batch (no `_snapshot.jpg` yet)
`build_camera_payloads` + `split_for_combined` run clean with `snapshot=None`.
The combined PDF remains byte-identical to old (`COMBINED_REPORT_PARITY.md`).

## 8. Processed videos

The `_tracked.mp4` overlays are produced by `rendering/feature_overlay_renderer.py`
from the same tracks/raw bboxes, independent of the report path. Raw-bbox parity
(no 15 % expansion) was verified previously (`scratchpad/verify_rawbox.py`); the
report-snapshot changes here do not touch the video overlay code.

## 9. Final checklist — only Stage 1 differs

- [x] Side + top generators **byte-identical** to old production.
- [x] Invocation matches the old SageMaker `generate_report` call.
- [x] Wagon-overview quartile frames render (frame ranges resolved via pcf).
- [x] Door snapshot = old refined+annotated crop; **0 placeholders**.
- [x] Damage snapshot = old annotate-then-crop; **0 placeholders**.
- [x] Snapshot cropping + annotation rendering reuse old code verbatim.
- [x] Combined report unaffected; camera payloads never serialised.
- [x] Reconstruction / Global-Train IDs / fusion identity unchanged — Stage 1 is
      the sole architectural difference; everything downstream consumes the
      GlobalTrainState through the old-production generators.

> Note: the door/damage evidence crops (`<side>_snapshot.jpg`,
> `track_{i}_snapshot.jpg`) are regenerated on the next full feature run. Reports
> built against pre-existing evidence fall back to `snapshot=None` (placeholder),
> exactly as before — no crash, no partial output.
