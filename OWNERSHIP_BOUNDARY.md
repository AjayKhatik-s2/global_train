# Ownership-based wagon boundary assignment (Stage 1)

Adjacent wagons are no longer split at the detected gap's **temporal centre**.
They are split at the **ownership-transition frame** — the frame where image
ownership flips from the previous wagon to the next. Every frame is assigned to
exactly one wagon; no gap frame is shared or discarded.

## The decision

A gap is visible over a *range* of frames as it sweeps across the image. For
each gap we compute the transition using both cues:

- **Spatial** — the gap's image-plane centre_x vs the frame **midline**. The
  wagon occupying the majority of the image owns the frame; ownership flips when
  the gap centre crosses the midline.
- **Temporal** — the train's travel direction (sign of the centre_x drift across
  the gap's hits) selects the crossing consistent with that direction.

`ownership_transition_frame(gap, frame_width)` interpolates the gap's per-hit
trajectory (`hit_frames` + `bbox_history`) to the midline crossing. Frames
**strictly before** it belong to the previous wagon; frames **from it onward** to
the next. If a gap never crosses the midline, it falls back to the gap centre.

## What changed (and what didn't)

- **Changed:** `wagon_count/global_alignment.py::build_global_wagons` now derives
  each boundary from `ownership_transition_frame` (fallback = gap centre). Each
  boundary is clamped to stay strictly between its neighbours, so there is
  exactly **one boundary per gap** and the wagon **count, GW IDs, and ordering
  are preserved** — only *where* the split lands moves (a few frames).
- **Unchanged (as required):** the gap detector, tracker, cross-camera
  refinement, wagon numbering, the phantom-lead guard, fusion, and Stage 2–5.
  The ownership frame is computed on RIGHT_UP's trajectory and stored as the
  master-frame boundary → projected to every camera by the existing shared-t=0
  mapping, so all four cameras stay synchronized.

## Verification (model-free, no pipeline/inference/video)

Reconstructing from the real per-camera gaps (with `bbox_history`):

- count centre-based **58** == count ownership **58**; GW ids identical.
- **58/58** boundaries used the image-majority crossing; **53/58** moved vs the
  gap centre.
- Adjacent wagons tile **contiguously with no overlap**, covering through the
  final frame (`GW_58` ends at `master_total-1`). Every frame between the first
  and last wagon belongs to **exactly one** wagon — none shared, none discarded.
- The only uncovered span is the leading phantom `[0, GW_1.start)` — the
  **pre-existing** phantom-lead guard (existing reconstruction logic, untouched).

## Visualization (all four processed videos)

Added so it's easy to verify each frame belongs to exactly one wagon:

- a white **ownership divider** — a vertical line at the gap's live centre_x —
  showing the moving split between the two wagons;
- an **`Owner: <GW>`** tag on the active gap (the wagon owning the current frame);
- the **`GW_BOUNDARY -> <GW>`** magenta flash at the ownership-transition frame
  (it now lands on the ownership frame, not the gap centre);
- alongside the existing `Gap #/Track/Conf`, `Detected Gaps: k/N`, `Class:`, and
  wagon-id overlays.

`rendering/gap_overlay.py` (final videos, all four cameras) and
`wagon_count/video_segmenter.py` (Stage-1 debug video) both draw the divider +
owner; `rendering/feature_overlay_renderer.py` feeds the current owner id.

## Files modified
- `wagon_count/global_alignment.py` — `ownership_transition_frame` +
  ownership-based boundary assignment in `build_global_wagons` (`frame_width`
  threaded from both call sites) + `[STAGE1]` ownership log.
- `rendering/gap_overlay.py` — ownership divider + `Owner:` tag + `GW_BOUNDARY ->`.
- `rendering/feature_overlay_renderer.py` — passes the current owner id.
- `wagon_count/video_segmenter.py` — ownership divider + owner in the debug video.
