# Stage-1 gap annotations in the processed videos

The four `<CAMERA>_processed.mp4` videos now render the **final accepted Stage-1
reconstruction gaps** on top of the existing feature annotations (door / damage /
load / OCR). One video is now enough to debug the whole pipeline: gap detection,
wagon segmentation, wagon IDs, and every feature.

The Stage-1 reconstruction algorithm is **not modified**. The renderer only
*replays* results Stage-1 already computed.

## What is drawn

| Element | Colour | Source |
|---|---|---|
| Tracked-gap bounding box (interpolated every frame in the gap span) | **cyan** `(255,255,0)` BGR | the camera's final `GapEvent` tracks |
| Gap label `TRACKED_GAP #<id>` + `conf=<c> f=<frame>` | cyan | `track_id`, `confidence`, frame index |
| Fused wagon-boundary flash (top/bottom edge) + `GW_BOUNDARY` banner | **magenta** `(255,0,255)` BGR | fused wagon time windows → local frames |

Cyan and magenta are **not** in the feature palette (door/damage use red / green /
yellow / orange / gray), so gaps stay visually distinct. Feature colours are
unchanged, and the gap overlay is drawn **last** so it overlays — never replaces —
the door/damage boxes.

Only **final accepted** gaps are drawn. `[camera]["gaps"]` are the tracks that
survived Stage-1 temporal filtering; the rejected raw single-frame candidates
(`LocalCameraTracks.raw_frame_detections`) are never serialized, so they can never
be drawn here. Merged/rejected/corrected gaps resolve to the fused boundary set,
which is what the magenta lines come from.

## How the renderer obtains the final gaps (no reconstruction re-run)

Stage-1 already records, per gap track, the exact per-hit image-plane box
trajectory in `GapEvent.bbox_history` (paired with `hit_frames`) and already
serializes each camera's final gaps into `global_state/per_camera_tracking.json`.
The **only** thing missing was that `bbox_history` / `hit_frames` were not in
`GapEvent.to_dict()`. Exposing them (one additive change) makes the persisted
`per_camera_tracking.json` self-sufficient for replay:

```
Stage 1 (unchanged detection/tracking/fusion)
   └─ GapEvent{start/end_frame, confidence, track_id, hit_frames, bbox_history}
        └─ LocalCameraTracks.to_dict → per_camera_tracking.json[camera]["gaps"]   ← now carries bbox_history
   └─ GlobalTrainState.wagons (fused time windows)  → global_train_state.json

Stage 4b renderer (rendering/feature_overlay_renderer.py)
   ├─ loads per_camera_tracking.json (already loaded for fps/total_frames)
   │     → camera_meta["gaps"]  (final tracked gaps + bbox_history)
   ├─ loads global_train_state.json wagons → boundary frames via the SAME
   │     time→local-frame arithmetic Stage 1 uses (_map_wagon_to_local_frames)
   └─ per frame: draw door/damage (existing) → draw gaps LAST (rendering/gap_overlay.py)
```

`rendering/gap_overlay.py::interp_gap_bbox` is a faithful copy of
`wagon_count/video_segmenter._interp_gap_bbox` (linear interpolation between
recorded hits). It re-derives nothing — it interpolates the boxes the tracker
already recorded. The gap **coordinates are identical** to Stage 1's own debug
video because they come from the same `bbox_history` and the same boundary
arithmetic.

> Why not read Stage 1's own gap-annotated video? `video_segmenter` writes to
> `global_state/processed_videos/` and the final videos come from the feature
> renderer (`processed_videos/`) built from the raw input. Compositing onto an
> already-encoded mp4 would double-encode and couple the two renderers; replaying
> the persisted gap data keeps the feature renderer the single producer of the
> final video and avoids any re-encode.

## Files modified

1. **`wagon_count/global_train_state.py`** — `GapEvent.to_dict()` now also emits
   `hit_frames` + `bbox_history` (the already-computed per-hit trajectory). Pure
   exposure; no detection/tracking/fusion logic touched. This flows automatically
   into `per_camera_tracking.json` (written by `LocalCameraTracks.to_dict`).

2. **`rendering/gap_overlay.py`** *(new)* — `interp_gap_bbox`,
   `build_gap_frame_index`, `draw_gap_overlays`. Consumes the serialized gaps +
   fused boundaries and draws the cyan gap box + magenta boundary. Cited port of
   the Stage-1 `video_segmenter` drawing/interpolation (visualization only).

3. **`rendering/feature_overlay_renderer.py`** — imports `gap_overlay`; builds the
   per-camera gap index from `camera_meta["gaps"]` and the boundary-frame set from
   `state.wagons`; draws gaps **after** the feature overlays for all four cameras.
   No feature-drawing code or colour changed.

Stage 1 (`wagon_count/run_global_count.py`, `video_segmenter.py`, tracker/fusion)
is unchanged apart from the additive `to_dict` fields.

## Verification

- **Unit** — `gap_overlay`: `build_gap_frame_index` spans `[start,end]`;
  `interp_gap_bbox` interpolates the exact midpoint and returns `None` outside the
  span; `draw_gap_overlays` writes cyan gap pixels inside a span and magenta pixels
  within ±3 of a boundary, and nothing elsewhere. `GapEvent.to_dict()` now contains
  `hit_frames` + `bbox_history`.
- **Full render path** — rendering RIGHT_UP (side) and RIGHT_UP_TOP (top) from the
  real batch state + real door/damage evidence: **every** sampled gap-span frame
  shows the gap box (RIGHT_UP 59/59 gaps, RIGHT_UP_TOP 62/62) and boundary frames
  show the magenta flash, drawn together with the feature boxes.
- **Real coordinates (all four videos)** — a fresh Stage-1 run regenerates
  `per_camera_tracking.json` with real `bbox_history`; rendering all four cameras
  then shows the gaps at Stage-1's exact coordinates.
  (`RIGHT_UP_processed.mp4`, `LEFT_UP_processed.mp4`, `RIGHT_UP_TOP_processed.mp4`,
  `LEFT_UP_TOP_processed.mp4`.)
