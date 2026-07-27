# RIGHT_UP is the authoritative master for wagon count + numbering

RIGHT_UP now owns the Global Train wagon **count** and **numbering** end-to-end.
Support cameras (LEFT_UP, RIGHT_UP_TOP, LEFT_UP_TOP) can only corroborate and
attach evidence — they can never add, delete, or renumber a Global Wagon.

## Audit — where support cameras could still change the count

The reconstruction already treated RIGHT_UP as master conceptually, but one path
let support cameras change the canonical count:

`wagon_count/global_alignment.py`
- `assemble_global_train_state` called `fuse_master_timeline(master, support)`,
  which ran `decide_inserted_gaps` and appended **synthetic support gaps**
  (`class_label="gap_inserted"`) to the master timeline, then
  `build_global_wagons(fused_gaps)` built wagons from that fused list.
- Effect: when ≥2 support cameras agreed on a gap RIGHT_UP missed, a wagon was
  **inserted** (`support_gap_recoveries`, `reconstruction_mode =
  MASTER_WITH_FUSED_SUPPORT`).

Proof it happened in practice:

| Run | RIGHT_UP local wagons | `support_gap_recoveries` | `total_wagons` |
|---|---|---|---|
| original batch `20260724_204400` | 60 | **3** | 62 |
| earlier re-run | 59 | **1** | 59* |

`total_wagons ≠ RIGHT_UP` → support cameras were changing the count. Everything
else was already master-owned: GW ids are assigned `GW_1..GW_N` by positional
index over the master timeline (`build_global_wagons`), and the master fps /
frames / classifications are the sole basis for segmentation.

## Fix — master-only reconstruction (one function)

`assemble_global_train_state` now builds wagons from the **master's gaps only**
and never inserts support gaps:

```python
wagons = build_global_wagons(
    sorted(master_tracks.gaps, ...),        # RIGHT_UP gaps ONLY
    ..., support_camera_ids=support_ids,    # support listed for evidence, not structure
    master_camera_id=master_tracks.camera_id)
# support matched purely to report corroboration + surface unmatched evidence
corrections_applied = []                    # support NEVER inserts a gap
```

- `fuse_master_timeline` / `decide_inserted_gaps` are **no longer invoked**
  (kept, marked DEPRECATED/NOT INVOKED, so they can't be reintroduced silently).
- Because `corrections_applied` is now always empty, the existing provenance
  logic in `run_global_count.py` auto-sets `support_gap_recoveries = 0`,
  `support_fusion_used = False`, `reconstruction_mode =
  MASTER_WITH_SUPPORT_AVAILABLE` (or `MASTER_ONLY`) — no change needed there.
- GW ids remain `GW_1..GW_N` in the master sequence (one-to-one).

Support matching (`match_support_to_master`) still runs, but only to compute, per
support camera: how many canonical wagons it corroborated (matched X/N) and how
many extra detections it had (unmatched evidence). Neither can change the count.

### Requirement mapping
| # | Requirement | How it holds |
|---|---|---|
| 1 | count == RIGHT_UP | wagons built from master gaps only |
| 2/7 | RIGHT_UP numbering = canonical GW ids | `GW_i` = master positional index |
| 3 | support maps onto existing ids | support never builds/renumbers wagons |
| 4 | support may only add evidence | matching is audit-only; no insertion |
| 5 | extra support wagon → unmatched evidence | leftover gaps reported, never inserted |
| 6 | missing support wagon → keep + mark camera | wagon stays; `per_camera_status = missing_evidence` |
| 8 | Stage 2–6 use canonical ids | all read `state.wagons[].global_id`, now canonical |

Downstream (materialization, features, fusion, reports, dashboard, processed
videos) already key exclusively off `state.wagons[].global_id`; since that list is
now strictly RIGHT_UP-canonical, every stage references the canonical ids with no
change to any feature or report logic.

## Validation logs

`[STAGE1]` lines are emitted during assembly, e.g. on the real 4-camera train:

```
[STAGE1] Master camera: RIGHT_UP
[STAGE1] Canonical wagon count: 58 (RIGHT_UP)
[STAGE1] LEFT_UP matched: 57/58  (+1 unmatched evidence)
[STAGE1] RIGHT_UP_TOP matched: 58/58  (+1 unmatched evidence)
[STAGE1] LEFT_UP_TOP matched: 57/58  (+1 unmatched evidence)
[STAGE1] Final Global Train: 58 wagons (RIGHT_UP canonical)
```

`matched X/N` = canonical wagons that camera corroborated (N minus the master
boundaries it missed); `+K unmatched evidence` = extra detections not present in
RIGHT_UP (evidence only, never a new wagon).

## Verification

Same real gap inputs that previously produced **59** wagons (with `+1` support
recovery) now produce **58** (RIGHT_UP canonical):

- `total == master-only count` (58) with 1 and with 3 support cameras.
- `corrections_applied == 0`; GW ids == `GW_1..GW_58`.
- Ownership scenarios on the real master, varying support:
  - **perfect agreement** → 58, `matched 58/58`, status `ok`.
  - **#5 extra support gap** → still 58, extra = unmatched evidence, ids unchanged.
  - **#6 missing support gap** → still 58 (wagon kept), `matched 57/58`, camera
    `per_camera_status = missing_evidence`.
- Unit test `tests/test_incremental_lifecycle.py::test_master_id_and_fusion_consensus`
  updated to the canonical rule (support never recovers; count == canonical; GW ids
  one-to-one) — passes; full suite: 48 passed (1 pre-existing unrelated failure in
  `test_camera_isolation.py`, a stale `_run_camera_features` reference from an
  earlier scheduler rename, untouched by this change).

## Files modified
- `wagon_count/global_alignment.py` — `assemble_global_train_state` builds from
  master gaps only + emits `[STAGE1]` logs; `fuse_master_timeline` marked
  DEPRECATED/NOT INVOKED.
- `tests/test_incremental_lifecycle.py` — `[C1]` invariant updated to the
  canonical-master rule.

No feature algorithm, fusion feature-attachment, report, dashboard, or video code
was changed — this is purely a reconstruction/global-state ownership rule.
