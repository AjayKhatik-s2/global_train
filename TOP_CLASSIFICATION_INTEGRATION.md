# top_classification.pt — Stage-1 semantic integration

`top_classification.pt` is now an **additional evidence source** for Stage-1
reconstruction and a synchronized label in all four processed videos. It does
**not** replace the gap detector and does **not** change the canonical RIGHT_UP
wagon count/numbering or Stage 2–5 logic. (No pipeline/inference was run locally.)

## How the model is integrated

- Runs on **RIGHT_UP_TOP** and **LEFT_UP_TOP** (Step 2b in `run_global_count.py`).
- Reuses the existing, now model-agnostic `MasterClassifier`: for each **segment**
  of a top camera (segments come from that camera's own gaps), it samples N frames
  and majority-votes a class. The name-based label map (`_label_to_class`) already
  covers `engine/loco`, `wagon`, `brake_van/guard_van/tail`, `track/background`, so
  it handles the top model's class list (incl. any extra classes → mapped by name).
- Segment labels are re-expressed in **master frames** (shared-t=0 timebase) so they
  line up with each GlobalWagon's master-frame window.
- The gap detector is untouched — this is parallel evidence. If
  `top_classification.pt` is absent, Step 2b is skipped gracefully (gap-only
  classification, no crash).

## How semantic labels enter the Global Train

`assemble_global_train_state(..., top_classifications=…)` calls
`fuse_semantic_labels` **after** the canonical wagons are built and boundaries
refined:

1. **Trust-weighted vote per wagon.** RIGHT_UP (via `side_classification.pt`) is
   the weight-1.0 anchor; each top camera adds a `gap_trust_weights` vote (0.9)
   for the class it read over that wagon's master-frame window. `argmax` sets
   `GlobalWagon.classification`; the normalized winning weight sets
   `classification_confidence`; the per-camera votes are stored in the new
   `classification_sources` (audit only).
2. **Physical-layout enforcement.** `ENGINE` is kept only in a **leading**
   contiguous run and `BRAKE_VAN` only in a **trailing** contiguous run; any
   engine/brake-van label elsewhere is corrected to `WAGON`. This identifies the
   engine/brake-van regions, keeps them at the ends, and **prevents them from
   contaminating the wagon region** — while never touching the count, IDs, or
   boundaries.

Requirement mapping: *validate train start/end* → `[STAGE1] Train start/end`
log (first/last wagon + class); *identify engine / brake-van region* → the
leading/trailing runs (logged); *prevent wagon in engine/brake-van* → those
segments are classed non-WAGON so they drop out of `regular_wagon_count`
downstream; *increase confidence* → `classification_confidence` now blends
RIGHT_UP + both tops. **Canonical numbering is unchanged** (semantic-only refine).

`[STAGE1]` logs added:
```
[STAGE1] Semantic evidence: top_classification.pt on ['LEFT_UP_TOP', 'RIGHT_UP_TOP']
[STAGE1] Engine region: ['GW_1'] | Brake-van region: ['GW_58'] | WAGON wagons: 56
[STAGE1] Train start: GW_1 (ENGINE) @f0 | Train end: GW_58 (BRAKE_VAN) @f...
```

## How the processed videos render the label

The label is **not** classified per camera for display — it comes from the one
canonical Global Train, so all four videos stay in lockstep even if a camera
briefly misreads a frame:

```
top_classification.pt (tops) ─┐
side_classification.pt (RIGHT_UP)─┼─ fuse_semantic_labels ─ GlobalWagon.classification
                                 │        (canonical, per wagon)
                                 └─ global_train_state.json ─ core.global_state_loader
                                        │
   rendering/feature_overlay_renderer._render_one_camera (per frame):
     wagon = frame_to_wagon[frame_idx]   # master window -> this camera's local frames
     _draw_class_label(frame, wagon.classification)   # "Class: ENGINE/WAGON/BRAKE VAN"
```

- **All four** `*_processed.mp4` now draw a top-left **`Class: <ENGINE|WAGON|BRAKE
  VAN>`** label (colour-coded, distinct from door/damage/gap), synchronized from
  the shared `GlobalWagon.classification`.
- These sit alongside the existing overlays already added: gap boxes, **Gap #/Track/
  Conf**, the **Detected Gaps: k/N** counter, and the `GW_BOUNDARY | GW_n` wagon-id
  banner — so the videos now show gaps + gap IDs + gap count + wagon IDs + train-
  region class together.
- Stage-1's own debug video (`video_segmenter`) already prints the per-wagon
  classification in its info panel.

## Files modified
1. `wagon_count/run_global_count.py` — Step 2b runs `top_classification.pt` on the
   two top cameras (`_classify_top_regions`, master-frame mapped), soft model
   resolution, passes `top_classifications` to `assemble_global_train_state`.
2. `wagon_count/global_alignment.py` — `fuse_semantic_labels` + `_top_label_for_window`;
   `assemble_global_train_state` gains the `top_classifications` param, calls the
   fusion, and logs the semantic `[STAGE1]` lines.
3. `wagon_count/tracker_engine.py` — `MasterClassifier` made model-agnostic with a
   `tag` for logs (RIGHT_UP=`MASTER`, tops=`TOP:<cam>`).
4. `wagon_count/global_train_state.py` — `GlobalWagon.classification_sources` audit
   field (per-camera votes) + serialization.
5. `rendering/feature_overlay_renderer.py` — synchronized `Class:` label on all four
   cameras from `GlobalWagon.classification` (frame→wagon map now built for every
   camera).

## Not changed
- Gap detector / tracker association (only reused).
- Canonical RIGHT_UP wagon count, GW IDs, boundaries.
- Stage 2 (materialization), 3 (features), 4 (fusion), 5 (reports), dashboard —
  they consume `GlobalWagon.classification` exactly as before; only its VALUES are
  now top-informed.

> Working-tree note: this sits on top of the earlier (still-uncommitted) Stage-1
> gap-lifecycle work in the same files (tracker same-frame NMS + duplicate-track
> merge + lifecycle stats; gap-ID/running-counter annotations; `--stage1-debug`).
> Nothing was executed locally; validate on EC2.
