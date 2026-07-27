# Validation — LEFT_UP detections + snapshot selection restored to old production

Two independent root causes were found by **reading the code and running it** on the
completed batch (`batch_outputs/20260724_204400`), not by patching symptoms.

---

## Issue 2 — LEFT_UP had no detections

### Trace of one LEFT_UP wagon (before fix)
| Stage | Result |
|---|---|
| Stage 1 reconstruction | LEFT_UP wagons produced (GW_1…GW_62) ✓ |
| Stage 2 materialize | `wagon_cache/GW_2/left_up/` = 67 frames (same as right_up) ✓ |
| Feature inference (YOLO) | door_state.pt on LEFT_UP GW_2 = **12 raw boxes** (6 closed, 6 partial) ✓ |
| Confidence floor | 12 → **3** survive (floor was **0.68**) |
| **Geometric prior** | 3 → **0** survive ← **detections disappear here** |
| Tracker / temporal | 0 detections → 0 tracks |
| Fusion | `left_door = CLOSED, conf 0.0, tracks 0` |
| Report | nothing to show |

RIGHT_UP for the same wagons: geometric prior kept **7/7** — so the prior was
rejecting *only* LEFT_UP (side-camera doors hug the frame edge).

### Root cause
The new door processor built its filters with **library defaults**:
`GeometricShapePrior(GeometricPriorConfig())` and `TrackerConfig()`.
Old production built the SAME (byte-identical) classes with **relaxed, tuned
settings** (`door_processor.py:364-374, 394-405`):

| Setting | Old production | New default (bug) |
|---|---|---|
| `require_border_completeness` | **False** | True → dropped LEFT_UP |
| `require_vertical_edges` | **False** | True |
| `min_aspect_ratio` / `max` | 0.15 / 2.0 | 0.3 / 1.2 |
| `min_structure_score` | 0.2 | 0.4 |
| `min_border_edge_ratio` | 0.03 | 0.08 |
| `min_sides_with_edges` | 1 | 2 |
| closed conf floor | **0.50** | 0.68 |
| open conf | 0.60 | 0.80 |
| `n_init` / `min_hits` | 2 / 2 | 3 / 3 |
| open/closed confirmation frames | 3 / 4 | 5 / 5 |

The old comments literally read: *"require_vertical_edges/require_border_completeness
= DISABLE - too strict, filters valid doors."* The new pipeline re-enabled exactly
those filters via the defaults.

### Fix (no algorithm change — config port)
`features/door/processor.py`: construct `GeometricShapePrior` and `TrackerConfig`
with the **exact old production values** (ported verbatim with a code-reference
comment). The `GeometricShapePrior` / `DoorTracker` code is untouched (already
byte-identical to production).

### Verified after fix (LEFT_UP, real cache)
```
GW_2 CLOSED 0.78   GW_3 PARTIAL 0.91   GW_4 CLOSED 0.79
GW_5 CLOSED 0.84   GW_6 OPEN 0.94       (was: all CLOSED 0.0, tracks=0)
```
Evidence snapshots now written and non-blank:
`evidence/GW_6/door/LEFT_UP/left_best.jpg` (960×540, std 28.5), `…/left_crop.jpg`
(cropped around the door bbox).

---

## Issue 1 — snapshot selection not matching old production

### Root cause
The best-snapshot scorer `core/frame_quality.snapshot_score` was a **deliberately
diverged** reimplementation of old `_score_detection` (`door_processor.py:513-560`).
Its own comments admitted the divergence ("ITEM 5: bias evidence selection toward
the LARGEST visible door"):

| Term | Old `_score_detection` | New (bug) |
|---|---|---|
| area optimal ratio | 0.15 | 0.28 |
| area weight | 2.0 | 3.5 |
| extra raw-area tie-break | none | +0.5·raw_area |
| centre / conf / quality weights | 2.5 / 1.0 / 0.5 | 2.5 / 1.0 / 0.5 (same) |
| edge penalty | 0.3 | 0.3 (same) |

Because the area term dominated differently, a **different frame** was chosen as
the best snapshot than old production would pick.

### Fix
`core/frame_quality.py`: reverted `snapshot_score` to the **exact** old formula
`(area·2.0 + center·2.5 + conf·1.0 + quality·0.5)·edge_penalty` with the area term
peaking at 15% — removed the 0.28 peak, the 3.5 weight, and the tie-break term.

### Verified
Numeric equality vs the old `_score_detection` formula on 4 representative bboxes:
all `MATCH=True` (identical to 1e-9). Only the DOOR processor used `snapshot_score`;
load/ocr rank by confidence (unchanged) and damage uses the DamageTracker's own
best-snapshot selection (vendored, byte-identical to production).

---

## Validation checklist
1. **LEFT_UP report vs old** — report generators are byte-identical (see
   `COMBINED_REPORT_PARITY.md`); LEFT_UP now feeds them real detections. ✓
2. **LEFT_UP detections appear** — CLOSED/PARTIAL/OPEN per wagon (incl. OPEN GW_6). ✓
3. **Snapshots correspond to the detection** — crop is taken around the detected
   door bbox; state-bucketed so an OPEN wagon's snapshot shows the OPEN frame. ✓
4. **Same evidence frame in camera + combined** — both resolve the shared
   `evidence/<gw>/door/<CAMERA>/<side>_best.jpg` via `reporting/_evidence_lookup`. ✓
5. **No blank / duplicate / unrelated snapshot** — verified non-blank (std≈29);
   one best frame per state per side (no duplicates). ✓
6. **All production filters applied before snapshot selection** — geometric prior +
   DoorTracker (temporal reasoning, hysteresis, identity merge) now run with the
   exact production config *before* evidence bucketing. ✓
7. This report. ✓

## Files modified
- `features/door/processor.py` — ported old production `GeometricPriorConfig` +
  `TrackerConfig` (relaxed settings + 0.60/0.50 thresholds).
- `core/frame_quality.py` — reverted `snapshot_score` constants/terms to the exact
  old `_score_detection`.

No feature algorithm, tracker, temporal reasoning, fusion, report, dashboard, or
Global-Train logic was changed — only the two mis-defaulted config/scoring values
that had diverged from old production.
