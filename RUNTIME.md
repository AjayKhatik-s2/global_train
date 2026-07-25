# WagonEye v4 — Runtime Lifecycle & Operational Semantics

How the continuous service behaves in production: what runs, in what order, how
it survives restarts, how it avoids doing the same work twice, how it stops
cleanly, and what happens when something fails. Grounded in the actual modules
(`orchestrator/`, `core/`, `features/`, `train_extraction/`, `delivery/`).

---

## 1. Process topology

One command runs the whole thing:

```bash
python -m orchestrator.master_runner --auto        # + WAGONEYE_PIPELINE_SOURCE
```

`master_runner` is the **high-level coordinator**. It does not contain
extraction logic. Depending on the **pipeline source** (`core/pipeline_source.py`,
env `WAGONEYE_PIPELINE_SOURCE`, or `--source`):

- **`trimmed`** (default) — pure consumer. The input prefixes already hold
  trimmed per-camera train clips (produced by the standalone
  `wagon-eye-extraction.service`, another pipeline, or a manual upload).
- **`raw`** — `master_runner` constructs and owns an **`ExtractionManager`**
  (`orchestrator/extraction_manager.py`). The manager runs the raw→trimmed
  production stage; the coordinator keeps consuming trimmed clips exactly as in
  the pure-consumer path. The two halves are decoupled through S3 (the
  extractor's trimmed bucket == the consumer's `WAGONEYE_S3_INPUT_*`).

There is exactly **one** extraction implementation (`train_extraction/`), used by
both the `ExtractionManager` and the standalone service via the shared
`run_extraction_service.sweep_camera`.

---

## 2. Startup sequence

1. **Logging** — `setup_logging()` installs a rotating file handler
   (`<WAGONEYE_LOG_DIR>/wagon_eye.log`, 50 MB × 10) plus stdout.
2. **Config validation (fail-fast)** — `core.config.validate_config()` refuses to
   start (exit 2) on: negative deadlines, `support_wait > final_wait`, interim
   upload/email without interim generation, empty S3 input prefixes / output
   bucket in `--auto`, missing email endpoint/recipients when email is enabled,
   or a non-writable workspace/log/temp dir.
3. **Redacted startup summary** — effective settings are logged (recipients as
   counts; no secrets), including `pipeline_source`.
4. **Source wiring** — for `raw`, the `ExtractionManager` is started (continuous)
   or run once (`--once`/`--batch`). For `trimmed`, nothing extra.
5. **Signal handlers** — SIGTERM/SIGINT installed for graceful shutdown (§5).
6. **Resume** — `processed_batches.json` is loaded from S3 so terminal batches
   are known before the first poll.

---

## 3. The two loops

### 3a. Production loop (ExtractionManager, `raw` source only)

A daemon thread sweeps every camera every `WAGONEYE_EXTRACTION_POLL_INTERVAL`
seconds. Per camera, `sweep_camera`:

1. **Raw discovery** — lists video keys under the camera's raw bucket/prefix.
2. **Dedup** — skips keys already in the per-camera local ledger
   (`logs/extraction_state/processed_<cam>.json`).
3. **Train-completion detection + extraction** — hands each new key to the
   vendored extractor. The `TrainSegmentFinder` locates complete train passes; a
   clip holding only the *leading* part of a train (continues into the next clip)
   is **held as ongoing** in the extractor's S3 `PipelineStateStore` and not cut
   until its continuation arrives.
4. **Produce** — uploads the trimmed `<...>_train.mp4` clip(s) to the camera's
   trimmed bucket, then records the raw key as processed.

### 3b. Consumption loop (`run_auto`)

Each tick (`WAGONEYE_ACTIVE_BATCH_POLL_INTERVAL`, default 60 s):

1. **Load active manifests** from S3 (all non-terminal batches).
2. **Discover** trimmed candidate videos under the input prefixes
   (`list_candidate_videos`): each is classified to a camera + train timestamp +
   ETag.
3. **Attach** (`_attach_candidate`) — cluster each candidate onto an active batch
   whose canonical train timestamp is within tolerance (±120 s), or open a new
   batch. Ambiguous matches (near-equidistant between two batches) are **held for
   review**, never guessed.
4. **Advance** every active batch's lifecycle state machine one tick (§4).
5. **Checkpoint** — a batch that reached a terminal state is written to
   `processed_batches.json`.
6. **Sleep** `poll_interval`, unless shutdown was requested.

An unhandled error in a tick is logged with a traceback; the loop sleeps and
continues (a bad poll never crashes the daemon).

---

## 4. Per-batch lifecycle (`lifecycle_runner.advance`)

A batch is a **manifest** (`orchestrator/batch_manifest.py`) persisted locally and
to S3 after every transition. `advance()` is a pure state machine that makes as
much safe forward progress as it can each tick:

```
DISCOVERED → COLLECTING_CAMERAS → WAITING_FOR_MASTER / WAITING_FOR_SUPPORT
   → RECONSTRUCTING ──(seal GlobalTrainState once)──▶ GLOBAL_STATE_SEALED
   → PROCESSING_AVAILABLE ──(materialize + features + fuse + interim report)
   → WAITING_FOR_LATE_CAMERAS ⇄ PROCESSING_LATE_CAMERA
   → FINALIZING ──(one upload + one email)──▶ COMPLETED / COMPLETED_PARTIAL
```

Key invariants:

- **RIGHT_UP is the master.** Reconstruction waits for it up to
  `MASTER_WAIT_MINUTES`; once present, a short `SUPPORT_FUSION_WAIT_MINUTES`
  window lets support cameras improve Stage-1 gap recovery, then the
  `GlobalTrainState` is **sealed exactly once** (immutable version hash). Late
  cameras only *attach features* to sealed wagons — Stage 1 never re-runs and
  wagons are never renumbered.
- **Partial completion.** At `FINAL_CAMERA_WAIT_MINUTES` (from first-seen),
  still-missing cameras become `CAMERA_MISSING_FINAL` and the batch finalizes as
  `COMPLETED_PARTIAL`. If RIGHT_UP never arrives by then →
  `FAILED_NO_GLOBAL_STATE` (no report, no email).
- **The six stages** inside a sealed batch: (1) reconstruction → GlobalTrainState,
  (2) materialize wagon cache, (3) feature inference, (4) fusion, (4b) overlay
  render, (5) reports, (6) delivery.

### Stage 3 observability

Each feature processor (door, ocr, load, damage) logs, via the structured logger:

- a **start** line with the one-time `model_load` wall-clock,
- a **progress** line every N wagons (`done/total`, camera, last wagon time,
  elapsed),
- a **DONE** summary with `total`, `model_load`, per-phase totals
  (`inference=…s  evidence=…s`), `ok/total`, and the three slowest wagons.

Load runs to completion before door/ocr/damage (the damage floor-filter reads the
load result); the three then run in parallel, each model loaded once per process.

---

## 5. Graceful shutdown

- `systemctl stop` sends **SIGTERM**. `_request_shutdown` sets a flag; it is
  checked **between batches and after each idle sleep**, so an in-flight batch is
  **never interrupted mid-processing** — the current batch finishes, then the
  process exits 0.
- The `ExtractionManager` stops on the same path: its stop `Event` is set and the
  reused sweep's stop flag is bridged, so an in-flight raw-key extraction breaks
  at the **next key boundary** (current key finishes), then the thread is joined.
- `TimeoutStopSec` in the unit (default 1800 s) must exceed the longest expected
  batch, or systemd will SIGKILL a still-running batch when the timer expires.
- `Restart=on-failure` restarts on a crash but **not** on a clean graceful exit.

---

## 6. Restart behavior (resume, don't redo)

Everything durable enough to resume lives outside the process:

| State | Where | On restart |
|---|---|---|
| Terminal batches | `s3://<out>/master_runner/processed_batches.json` | never reprocessed |
| Active (in-flight) batches | per-batch `manifest.json` (local + S3) | re-loaded and advanced from their last persisted state |
| Sealed GlobalTrainState | `batch_outputs/<key>/global_state/` + `global_state_version` | never re-sealed; late cameras attach only |
| Per-(camera,feature) done | feature completion markers + `materialized_cameras` | up-to-date work is skipped |
| Delivery done | `delivery/finalization.json` | already-done upload/email skipped |
| Extracted raw keys | `logs/extraction_state/processed_<cam>.json` | already-extracted clips not re-cut |
| Ongoing (split) trains | extractor S3 `PipelineStateStore` | continuity preserved across restarts |

Net effect: a `systemctl restart` at any point resumes exactly where it left off
and repeats no completed work.

---

## 7. Duplicate prevention (defense in depth)

Five independent layers ensure a train is processed — and delivered — once:

1. **Extraction** — per-camera local ledger (raw-key dedup) + extractor S3 state
   store (a train split across raw clips is cut once, not twice).
2. **Ingestion** — candidates cluster by train timestamp (±120 s); re-seeing the
   same `(camera, ETag)` is a no-op; a **changed ETag** rebuilds only that
   camera; ambiguous matches are held; **terminal batches are never reopened**.
3. **Batch** — `processed_batches.json` terminal set excludes finished batches
   from rediscovery.
4. **Feature** — each `(camera, feature)` carries a completion marker keyed by
   source ETag + GlobalTrainState version + model hash + thresholds; identical
   identity ⇒ skipped, so a late camera never re-runs or overwrites another
   camera's results.
5. **Delivery** — `finalization.json` records an idempotency key (batch +
   report revision + report hash); **one upload + one email per final revision**,
   and a crash between API-200 and marker-write is the only (rare) resend window.

---

## 8. Failure recovery

| Failure | Behavior |
|---|---|
| Stage 1 (reconstruction) fails | batch → `FAILED_NO_GLOBAL_STATE`; no report, no email; terminal (not retried) |
| RIGHT_UP master never arrives by final deadline | `FAILED_NO_GLOBAL_STATE` |
| A support/top camera never arrives | sealed from present cameras; batch finalizes `COMPLETED_PARTIAL`; a late camera still attaches its features without re-running Stage 1 |
| One feature processor crashes | that feature's wagons get a `FAILED` sentinel (fail-open); other features + the report still run → `COMPLETED_PARTIAL` |
| Stage 2 / 4 exception | batch → `FAILED`; Stage 5 exception → `REPORT_FAILED` (canonical JSON still uploaded; email suppressed) |
| Extraction sweep error for one camera | logged + counted; other cameras continue; **missing extraction model is non-fatal** — the consumer keeps polling trimmed input |
| Delivery / email transient failure | recorded in `finalization.json`; retried idempotently on the next advance/restart |
| Unhandled poll-tick error | logged with traceback; loop sleeps and continues (daemon stays up) |
| Process crash | `Restart=on-failure` restarts; §6 resume applies |

---

## 9. Operator quick reference

```bash
# status / logs
systemctl status wagon-eye
journalctl -u wagon-eye -f
tail -f logs/wagon_eye.log

# control (graceful)
sudo systemctl stop wagon-eye      # finishes current batch, then exits
sudo systemctl restart wagon-eye

# run modes
python -m orchestrator.master_runner --auto                 # continuous
python -m orchestrator.master_runner --auto --source raw     # + inline extraction
python -m orchestrator.master_runner --once                  # one batch, exit
python -m orchestrator.master_runner --batch <key>           # replay one batch
python -m orchestrator.master_runner --local-only            # offline, local_inputs/
```

See `DEPLOYMENT.md` for install, topology (single- vs two-service), and env vars.
