# Stage-1: confidence-weighted multi-camera reconstruction

RIGHT_UP is the canonical master. The two TOP cameras are *trusted refiners* that
may only nudge a boundary's position. LEFT_UP is demoted: its gaps are ignored
for boundaries; it contributes only the train start/end envelope + downstream
feature evidence. All contained in Stage 1 (`wagon_count/global_alignment.py`);
Stage 2+ is untouched.

## Trust weights (configurable)

```python
DEFAULT_GAP_TRUST_WEIGHTS = {
    "RIGHT_UP":     1.0,   # canonical master: count + IDs + initial boundaries
    "RIGHT_UP_TOP": 0.9,   # trusted refiner (position only)
    "LEFT_UP_TOP":  0.9,   # trusted refiner (position only)
    "LEFT_UP":      0.0,   # NOT a gap source -> train start/end + features only
}
```
Override per camera via `config["gap_trust_weights"]` or
`WAGONEYE_GAP_TRUST_<CAMERA>` env vars. LEFT_UP still has full weight (1.0) for
its train-envelope + feature roles — the 0.0 applies only to *gap detection*.

## Pipeline (maps to the requested steps)

1. **Trusted gap sources** — RIGHT_UP, RIGHT_UP_TOP, LEFT_UP_TOP keep their
   existing detectors/filters/confidence/refinement (unchanged).
2. **LEFT_UP** — gaps excluded from boundaries; only its train-activity envelope
   `[min gap start, max gap end]` is taken (`projection_camera_envelope`), stored
   in `state.notes` as `LEFT_UP_train_envelope_frames=…`.
3. **Canonical + refine** — RIGHT_UP gaps define the count, `GW_1..GW_N`, and the
   initial boundaries. `refine_master_boundaries` nudges each boundary toward
   agreeing RIGHT_UP_TOP / LEFT_UP_TOP gaps (weighted average, clamped). No
   support camera can create/delete a wagon.
4. **LEFT_UP projection** — canonical wagons are projected into LEFT_UP from the
   finalized master boundaries (Stage 2 already maps `gw.start_time*fps`, which
   under the shared-t=0 trim equals the normalized `train_start + %·train_len`
   projection; the envelope is stored so a normalized projection can be enabled
   if a real offset is ever measured). LEFT_UP never splits/merges/creates.
5. **Final train** = RIGHT_UP (+ RIGHT_UP_TOP + LEFT_UP_TOP refinement); LEFT_UP
   contributes envelope + features only.
6. **Logs + weights** — see below.

### `[STAGE1]` logs (real 4-camera train)
```
[STAGE1] Canonical camera: RIGHT_UP
[STAGE1] Canonical wagon count: 58 (RIGHT_UP)
[STAGE1] Trust weights (gap): RIGHT_UP=1.0, LEFT_UP=0.0, RIGHT_UP_TOP=0.9, LEFT_UP_TOP=0.9
[STAGE1] Boundary GW_10 @f821: RIGHT_UP: accepted | RIGHT_UP_TOP: matched (+0f) | LEFT_UP_TOP: matched (-2f) | LEFT_UP: projected-only (gap ignored) -> shift -1f
...
[STAGE1] LEFT_UP: projected-only (train envelope frames 178..3767; gaps excluded from boundaries)
[STAGE1] RIGHT_UP_TOP matched: 58/58  (+1 unmatched evidence)
[STAGE1] LEFT_UP_TOP matched: 57/58  (+1 unmatched evidence)
[STAGE1] Global boundaries finalized -- Final Global Train: 58 wagons (RIGHT_UP canonical)
```
`(+Nf)` is the matching refiner gap's offset from the canonical boundary in the
**master frame** domain (a boundary is a temporal position; frames is the correct
unit — the example's "px" is the same idea). One compact line per boundary.

## Is it mathematically sound?

**Yes.** The refinement of a boundary at master frame `F` is a trust-weighted
convex combination

```
F' = (1.0·F + Σ_c w_c·f_c) / (1.0 + Σ_c w_c)      over matched refiners c
```

- **Convexity** — `F'` is a weighted average of `{F, f_c}` with positive weights,
  so it lies in their convex hull, i.e. between the smallest and largest agreeing
  position. With the master weight (1.0) dominating each refiner (0.9), `F'` stays
  near `F`. It cannot diverge.
- **Bounded** — the shift is clamped to `boundary_refine_max_shift_sec`, then to
  stay strictly between the neighbouring boundaries.
- **Count/order preserved** — exactly one refined gap per master gap (a bijection),
  and the neighbour clamp keeps boundaries strictly increasing, so
  `build_global_wagons` yields exactly the master count. Verified on real data:
  refined count == master-only count (58), `GW_1..GW_58` unchanged, boundaries
  monotonic.
- **Identity fallback** — if no refiner matches (`den = 1.0`), `F' = F`.
- **Deterministic** — nearest-in-time selection + fixed weights; no randomness.
- **LEFT_UP isolation** — weight 0 ⇒ never enters the sum ⇒ provably cannot move
  any boundary.

## Edge cases

| Case | Behaviour |
|---|---|
| No refiner gap within the window | boundary unchanged (`F'=F`) |
| A refiner outlier gap inside the window | pull limited by its 0.9 weight + the max-shift clamp; a lone outlier can't dominate |
| Two adjacent master gaps very close | neighbour clamp prevents crossing → no dropped/merged wagon |
| Master has 0 gaps (single-wagon train) | nothing to refine; 1 wagon |
| LEFT_UP detected no gaps | envelope falls back to the full clip `[0, total-1]` |
| RIGHT_UP absent → fallback master (e.g. LEFT_UP) | master anchor weight forced to 1.0 → fallback master keeps full authority; refiners still nudge |
| Cameras not perfectly t=0-aligned | out-of-window refiner gaps simply don't match → safe degradation to raw master boundaries (never a wrong nudge) |
| Trust weights overridden | a re-weighted refiner can nudge more/less but the count/order guarantee is structural — it still cannot add/delete a wagon |

## Stage 2 onward — unchanged

- `GlobalTrainState` schema is identical (same fields; `notes` now also carries
  the LEFT_UP envelope — additive, an already-existing field).
- Stage 2 materialization, Stage 3 features, Stage 4 fusion, Stage 5 reports,
  Stage 6 dashboard JSON + processed videos all consume `total_wagons` and
  `state.wagons[].global_id / start_time / end_time` — all present and valid.
  Boundaries are marginally refined (±1–4 frames here) for better alignment, but
  the interface, the wagon count, and the GW IDs are byte-for-byte the same shape.
- No feature, fusion, report, dashboard, or video code was modified. Full test
  suite: 48 passed (1 pre-existing unrelated failure in `test_camera_isolation.py`).

## Files modified
- `wagon_count/global_alignment.py` — trust weights + `resolve_gap_trust_weights`,
  `refine_master_boundaries`, `projection_camera_envelope`, and a rewritten
  `assemble_global_train_state` (canonical build → TOP-camera refinement →
  LEFT_UP envelope → `[STAGE1]` logs). Support-gap insertion stays disabled.
- `tests/test_incremental_lifecycle.py` — canonical-rule invariant (from the
  previous change) still holds under weighting.
