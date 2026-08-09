# WagonEye v4 — EC2 Deployment Guide

This guide takes a fresh Amazon EC2 Linux instance to a running, continuously
polling WagonEye v4 pipeline. The pipeline is **EC2-native**: it needs no
SageMaker notebook, no Jupyter runtime, and no hardcoded paths — it runs from
wherever you clone it.

> All pipeline logic (GlobalTrainState, wagon counting, feature inference,
> fusion, rendering, reports, delivery) is unchanged from the SageMaker
> version. Only the runtime/infrastructure layer changed.

---

## 0. Prerequisites

- An EC2 instance running **Ubuntu 22.04/24.04** or **Amazon Linux 2023**.
  - CPU-only works. For production throughput use a GPU instance
    (e.g. `g4dn.xlarge`) with the NVIDIA driver installed.
- An **IAM instance role** attached to the EC2 instance granting S3 access to
  the input and output buckets (recommended over static keys).
- The **8 model files** (`.pt`) — 4 reconstruction + 4 feature models.

---

## 1. Clone the repository

Clone anywhere — the project auto-detects its own root; no path is hardcoded.

```bash
sudo mkdir -p /opt && cd /opt
git clone <your-repo-url> wagon_eye_v4      # or copy the folder up via scp/rsync
cd wagon_eye_v4
```

(If you use a different location, adjust the paths in
`deploy/wagon-eye.service` accordingly — they're marked `<<EDIT>>`.)

---

## 2. Install dependencies (one command)

```bash
bash scripts/setup_ec2.sh
```

This script:
- installs OS packages: `ffmpeg`, OpenCV/reportlab runtime libs
  (`libgl1`/`mesa-libGL`, `glib2`, `fontconfig` + DejaVu fonts), `python3-venv`,
  `pip`, and a C compiler;
- creates a virtualenv at `.venv/`;
- installs `requirements.txt`;
- **auto-detects a GPU** (`nvidia-smi`) and, if present, reinstalls the CUDA
  build of torch from `https://download.pytorch.org/whl/cu121`;
- creates the runtime directory skeleton (`models/`, `logs/`, `batch_outputs/`,
  `local_inputs/`);
- runs an import + device sanity check.

Force CPU torch even on a GPU box: `WAGONEYE_FORCE_CPU=1 bash scripts/setup_ec2.sh`.
Pick a specific interpreter: `PYTHON_BIN=python3.11 bash scripts/setup_ec2.sh`.

---

## 3. Get the models

The `.pt` weights **ship with the code** (production convention — the old
pipeline kept them in each camera folder; this repo tracks them via **Git LFS**).
On a fresh clone:

```bash
git lfs install && git lfs pull        # pulls the 8 weight files into models/
```

Expected layout (already created by `git lfs pull`):

```
models/reconstruction/     right_up_gap.pt   left_up_gap.pt   top_gap.pt   side_classification.pt
                           (long names right_up_wagon_gap.pt / left_up_wagon_gap.pt also accepted)
models/features/           door_state.pt   loaded.pt   damage.pt   wagon_id_counting.pt
```

Startup **verifies** every required model is present (all reconstruction models
+ one feature model per enabled feature) and **fails fast**, naming any missing
file. Verify ahead of time with the pre-flight (§5):

```bash
python scripts/preflight.py --no-aws          # checks model presence + deps + dirs
```

**Optional S3 sync (only if you host the weights in S3 — there is no production
model bucket):** set `WAGONEYE_MODELS_S3_BUCKET` (+ `WAGONEYE_MODELS_S3_PREFIX`,
default `models`) and startup downloads any *missing* model into the local dirs
from `s3://<bucket>/<prefix>/{reconstruction,features}/<file>` (IAM needs
`s3:GetObject`). Pre-download/verify: `python scripts/preflight.py --sync`.
Bypass all model checks with `--skip-model-sync`. Point the dirs elsewhere with
`WAGONEYE_MODELS_DIR` / `WAGONEYE_RECON_MODELS_DIR` / `WAGONEYE_FEAT_MODELS_DIR`.

---

## 4. Configure the environment

Every setting has a working default (the original production values). Override
only what differs on this host:

```bash
cp deploy/wagon-eye.env.example deploy/wagon-eye.env
nano deploy/wagon-eye.env
```

For **continuous `--auto` mode you must set the input prefixes** so the poller
knows where the already-**trimmed** camera clips land (these are the
extractor's output — see §6b for how they get produced):

```ini
WAGONEYE_S3_INPUT_BUCKET=biro-wagon-pre-processed-video-copy
WAGONEYE_S3_INPUT_PREFIXES=camera_CCTV_HZBN_DHN_2_RIGHT_UP,camera_CCTV_HZBN_DHN_1_LEFT_UP,camera_CCTV_HZBN_DHN_5_RIGHT_TOP,camera_CCTV_HZBN_DHN_6_LEFT_TOP
```

Other common overrides: `WAGONEYE_DEVICE=cpu|cuda`, `WAGONEYE_WORKSPACE_ROOT`,
`WAGONEYE_LOG_DIR`, `WAGONEYE_LOG_LEVEL`, `WAGONEYE_S3_OUTPUT_BUCKET`,
`WAGONEYE_EMAIL_RECEIVER`. Full list with defaults is in
`deploy/wagon-eye.env.example`.

---

## 5. Smoke test before going live (no S3)

Put 4 trimmed videos (filenames containing `right_up` / `left_up` /
`right_up_top` / `left_up_top`) into `local_inputs/`, then:

```bash
source .venv/bin/activate
set -a; source deploy/wagon-eye.env; set +a      # load your overrides
python scripts/preflight.py --mode local --no-aws     # verify deps/dirs/config/models first
python -m orchestrator.master_runner --local-only \
       --local-inputs ./local_inputs --no-interactive
```

Success looks like: `[BATCH <key>] completed (…s)` and a
`batch_outputs/<key>/reports/combined_train_report.pdf` that now includes the
company logo. Check `logs/wagon_eye.log` for the timestamped per-stage trace.

---

## 5b. Validation runbook — Damage feature first

Bring the migrated pipeline up **one feature at a time**. First run enables
ONLY Damage (Door/OCR/Load off); reconstruction, materialization, wagon cache,
fusion, processed videos, reports, dashboard JSON, and delivery all still run.

```bash
# 1. Update the repo + models
cd /opt/wagon_eye_v4 && git fetch && git checkout global-train-old-downstream && git pull
git lfs pull

# 2. Activate the venv
source .venv/bin/activate

# 3. Load production env vars (and keep validation off the live dashboard/email)
set -a; source deploy/wagon-eye.env; set +a
export WAGONEYE_DASHBOARD_INGEST_ENABLED=false

# 4+5. Synchronize + verify models (downloads from S3 only if a bucket is set;
#      otherwise confirms the Git-LFS weights are present) and STOP on any gap:
python scripts/preflight.py --mode auto --disable-features door,ocr,load --sync || exit 1

# 6. (preflight already fails on missing dep/model/env/AWS-permission)

# 7. Start the Damage-only pipeline
#    offline (local_inputs, no S3/email):
python -m orchestrator.master_runner --local-only --local-inputs ./local_inputs \
       --no-interactive --disable-features door,ocr,load
#    OR one real batch from S3 (uploads outputs; no email):
python -m orchestrator.master_runner --once --no-interactive \
       --disable-features door,ocr,load --skip-email
```

**Monitor**

```bash
tail -f logs/wagon_eye.log
grep -E "STAGE 1|materializ|FEAT/damage|\[STAGE5\]|\[LEGACY\]" logs/wagon_eye.log
```

**Stop / restart**

```bash
# foreground: Ctrl-C (finishes the current batch first)
# service:
sudo systemctl stop wagon-eye
sudo systemctl restart wagon-eye
```

**Verify outputs**

```bash
BK=$(ls -t batch_outputs | head -1)
cat  batch_outputs/$BK/global_state/global_train_state.json | python -m json.tool | grep -E 'total_wagons|classification' | head
ls   batch_outputs/$BK/reports/                 # combined_train_report.pdf/.json + *_report.pdf
ls   batch_outputs/$BK/processed_videos/        # <CAMERA>_processed.mp4 (damage boxes/labels)
find batch_outputs/$BK/legacy_output            # old layout: camera_reports/… + combined_reports/…
```

Confirm: reconstruction + wagon IDs (`GW_*`), damage annotations in the
processed videos + evidence crops, camera/combined PDFs, `combined_train_report.json`,
and the old-layout `camera_reports/<CAM>/<DD-MM-YYYY>_<NNN>.*` files.

**Manually upload results if delivery was skipped/failed**

```bash
# idempotent replay (reuses the daily counter):
python -m orchestrator.master_runner --batch $BK --no-interactive --disable-features door,ocr,load
# or push the old-layout tree by hand:
aws s3 cp --recursive batch_outputs/$BK/legacy_output/ s3://$WAGONEYE_S3_OUTPUT_BUCKET/
```

**Progressive enablement** (validate each independently before the next):

```bash
python -m orchestrator.master_runner --local-only --no-interactive --disable-features ocr,load   # + Door
python -m orchestrator.master_runner --local-only --no-interactive --disable-features load        # + OCR
python -m orchestrator.master_runner --local-only --no-interactive                                 # all four
```

---

## 6. Install as a service (continuous mode, restarts on reboot)

```bash
# Edit the three <<EDIT>> placeholders (User, WorkingDirectory, paths):
sudo cp deploy/wagon-eye.service /etc/systemd/system/wagon-eye.service
sudo nano /etc/systemd/system/wagon-eye.service

sudo systemctl daemon-reload
sudo systemctl enable --now wagon-eye        # start now + on every boot
```

`enable` makes it **auto-start after a reboot**. Verify:

```bash
systemctl is-enabled wagon-eye     # -> enabled
systemctl status wagon-eye         # -> active (running)
```

---

## 6b. Automatic ingestion: raw → trimmed → report

`--auto` polls for **already-trimmed** per-camera train clips. Something must
produce those clips by cutting the train pass out of the **raw** CCTV videos.
That producer is `train_extraction` (the vendored, consolidated V4 extractor).
There are two supported topologies — pick one.

**Extraction classify models (required for either topology).** The extractor
needs its own classifiers (empty_track / wagon / engine), which are **separate**
from the Stage-1 reconstruction models. Drop them here:

```
models/extraction/     side_classification.pt   top_classification.pt
```

(Override the location with `WAGONEYE_EXTRACTION_MODELS_DIR`.) Without them the
producer logs `extraction classify model not found` and stays non-fatal — the
consumer keeps polling, but no new trimmed clips are produced.

### Option A — Single service (one `--auto` process does everything)

Set `WAGONEYE_PIPELINE_SOURCE=raw` in `deploy/wagon-eye.env`. The one
`wagon-eye.service` process then owns an **ExtractionManager**
(`orchestrator/extraction_manager.py`) that produces trimmed clips from raw S3
— reusing the exact same extraction code as the standalone service — while
`master_runner` stays the high-level coordinator and consumes those clips. The
operator command stays exactly:

```bash
python -m orchestrator.master_runner --auto      # extract + inspect, one process
```

Minimal `deploy/wagon-eye.env` for single-service:

```ini
WAGONEYE_PIPELINE_SOURCE=raw
WAGONEYE_EXTRACTION_MODELS_DIR=/opt/wagon_eye_v4/models/extraction
# consumer input == extractor output (production trimmed bucket + prefixes):
WAGONEYE_S3_INPUT_BUCKET=biro-wagon-pre-processed-video-copy
WAGONEYE_S3_INPUT_PREFIXES=camera_CCTV_HZBN_DHN_2_RIGHT_UP,camera_CCTV_HZBN_DHN_1_LEFT_UP,camera_CCTV_HZBN_DHN_5_RIGHT_TOP,camera_CCTV_HZBN_DHN_6_LEFT_TOP
```

Raw/trimmed buckets default to the production per-camera layout in
`train_extraction/driver.py::_CAMERA_CONFIG`; override any with
`WAGONEYE_EXTRACTION_<CAM>_RAW_BUCKET` / `_TRIMMED_BUCKET`. Install exactly as in
§6 — no unit change; the env value alone selects the topology. Override at
runtime with `--source raw` / `--source trimmed`.

### Option B — Two services (producer and consumer run independently)

Leave `WAGONEYE_PIPELINE_SOURCE` unset (defaults to `trimmed`). Run the standalone extractor unit
alongside the inspection unit — independent restart/scaling, isolated logs:

```bash
cp deploy/wagon-eye-extraction.env.example deploy/wagon-eye-extraction.env   # edit
sudo cp deploy/wagon-eye-extraction.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now wagon-eye-extraction   # raw -> trimmed producer
sudo systemctl enable --now wagon-eye              # trimmed -> report consumer
journalctl -u wagon-eye-extraction -f
```

Both topologies use **one** extraction implementation and connect only through
S3 (producer's trimmed bucket == consumer's `WAGONEYE_S3_INPUT_*`).

---

## 6c. Historical mode — reprocess a past time range

`--historical` runs the **same** pipeline over already-trimmed clips that are
already in S3, selected by a date + time window. It is an *input-selection*
layer only: reconstruction, materialization, features, fusion, overlays and
reports are the identical code the live service runs (`process_batch`). It does
not enter the `--auto` polling loop, the lifecycle scheduler, or
`processed_batches.json`.

```bash
cd /opt/global_train && source .venv/bin/activate
set -a; source deploy/wagon-eye.env; set +a

# 1. ALWAYS dry-run first: lists what would be processed, downloads nothing
python -m orchestrator.master_runner --historical \
       --date 2026-08-08 --start-time 10:00 --end-time 12:00 \
       --timezone Asia/Kolkata --dry-run

# 2. then run it for real
python -m orchestrator.master_runner --historical \
       --date 2026-08-08 --start-time 10:00 --end-time 12:00 \
       --timezone Asia/Kolkata --disable-features ocr --infer-batch 24
```

ISO timestamps work too, and the offset in the string wins:

```bash
python -m orchestrator.master_runner --historical \
       --start "2026-08-08T10:00:00+05:30" --end "2026-08-08T12:00:00+05:30"
```

**Flags**

| Flag | Meaning |
|---|---|
| `--date / --start-time / --end-time` | window in `--timezone` (default `Asia/Kolkata`) |
| `--start / --end` | ISO-8601 alternative; cannot be mixed with the above |
| `--timezone` | IANA zone name; falls back to a fixed +05:30 if tzdata is absent |
| `--pad-minutes` | how far past its filename timestamp a clip may still hold its train (default 15) |
| `--dry-run` | discover + print + write the manifest, then stop |
| `--keep-inputs` | keep the staged clips after a batch succeeds |
| `--historical-deliver` | enable S3 upload + email (**OFF by default**) |
| `--manifest-out` | where to write the JSON manifest |

All existing processing flags still apply: `--disable-features`, `--infer-batch`,
`--stage1-frame-trim-percent`, `--raw-detections`, `--workspace`,
`--recon-models-dir`, `--feat-models-dir`, `WAGONEYE_DEVICE`, and the rest.

**How clips are matched to the window.** A trimmed clip is named after the *raw*
clip it was cut from, and those digits are **IST wall-clock**
(`train_extraction/time_utils.parse_timestamp_from_filename`). So the filename
timestamp is when the raw recording started, not when the train passed — the
train is somewhere inside `[T, T + clip span]`. Historical mode therefore keeps
a clip when `[T, T + pad]` overlaps your window. If a train seems missing,
widen `--pad-minutes`; the dry-run manifest prints the reason each object was
selected.

**Multiple trains.** Each train pass becomes its own batch (same
±120 s clustering rule as the live path) with its own batch key, output
directory and reports. Trains are never merged into one Global Train.

**Missing cameras** are reported as missing and processed with the existing
partial-camera behaviour. No substitute video is ever used.

**Output location.** `<workspace>/historical/<batch_key>/`, so a historical
re-run can never overwrite the live `batch_outputs/<batch_key>/` tree. The JSON
manifest lands at `<workspace>/historical/historical_manifest.json`. Staged
clips go to `<workspace>/historical/<batch_key>/downloads/` (never
`local_inputs/`) and are removed after a batch succeeds unless `--keep-inputs`;
a **failed** batch always keeps them for diagnosis.

**Delivery is off by default** — reprocessing history should not re-email the
operators or overwrite the delivered artifacts of the original live run. Pass
`--historical-deliver` only when you intend to replace them.

> Stop the service first if the box is busy: `sudo systemctl stop wagon-eye`.
> Historical mode writes to a separate tree and never touches live batch state,
> but the two will still compete for CPU and disk.

**Exit codes:** `0` all batches completed · `2` bad arguments or nothing matched
the window (the message prints the bucket, prefixes and window searched) ·
`3` at least one batch failed.

---

## 7. Monitor

```bash
# Application log (structured, timestamped, rotates at 50 MB × 10):
tail -f logs/wagon_eye.log

# Or via systemd's journal:
journalctl -u wagon-eye -f

# Belt-and-braces raw stdout/stderr captured by the unit:
tail -f logs/service.out logs/service.err
```

Every stage logs a start line, an elapsed-time completion line, and any
warnings/errors with full tracebacks. The full Stage-1 (`wagon_count`)
subprocess trace for each batch is saved at
`batch_outputs/<key>/global_state/stage1_wagon_count.log`.

---

## 8. Control the service

```bash
sudo systemctl stop wagon-eye       # SIGTERM: finishes the CURRENT batch, then exits
sudo systemctl restart wagon-eye
sudo systemctl start wagon-eye
```

Graceful stop waits for the in-flight batch. If your batches can run longer
than `TimeoutStopSec` (default 1800 s in the unit), raise it, or systemd will
SIGKILL the batch when the timer expires.

Manual (non-service) alternatives:

```bash
python -m orchestrator.master_runner --auto        # continuous, foreground
python -m orchestrator.master_runner --once         # one batch then exit
python -m orchestrator.master_runner --batch <key>  # replay a specific batch
```

---

## 9. Verify it's actually working

1. `systemctl status wagon-eye` → `active (running)`.
2. `grep 'logging initialized' logs/wagon_eye.log` → confirms log rotation is set up.
3. Drop a complete set of 4 videos into the input prefix(es); within one poll
   interval (`--poll-interval`, default 60 s) the log shows
   `[BATCH] discovered … batch(es)` → the 6 stage lines → `[BATCH <key>] completed`.
4. Check the output bucket for `train_batch/<key>/reports/combined_train_report.pdf`
   and the archived tree; confirm the notification email arrived.
5. Confirm the device line at startup:
   `WagonEye v4 orchestrator starting (device=cuda)` on a GPU box.

---

## Reboot behavior & disk notes

- **After a reboot**, `systemd` restarts the service automatically (because of
  `enable`). Processed-batch state lives in S3
  (`s3://<bucket>/master_runner/processed_batches.json`), so no batch is
  reprocessed after a restart.
- **Disk growth**: each batch leaves its full working tree under
  `batch_outputs/<key>/` (downloads, wagon_cache JPEGs, evidence, processed
  videos). Nothing prunes these automatically. Add a retention policy suited to
  your audit requirements, e.g. a cron job:
  ```bash
  # delete batch working dirs older than 14 days
  find /opt/wagon_eye_v4/batch_outputs -mindepth 1 -maxdepth 1 -type d -mtime +14 -exec rm -rf {} +
  ```

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `--auto` idles, "WAGONEYE_S3_INPUT_PREFIXES is empty" | Set the input prefixes in `deploy/wagon-eye.env` (step 4). |
| `train_batch_manager not importable` | Should not happen — the module now ships in `orchestrator/`. Ensure the repo wasn't partially copied. |
| `libGL.so.1: cannot open shared object` | OS libs missing; re-run `scripts/setup_ec2.sh` (installs `libgl1`/`mesa-libGL`). |
| Reports render without fonts / boxes | Install `fontconfig` + DejaVu fonts (setup script does this). |
| Runs on CPU on a GPU box | Driver/CUDA torch not installed; `pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision`, or set `WAGONEYE_DEVICE=cuda`. |
| S3 `AccessDenied` | Attach an IAM instance role with read on the input bucket + write on the output bucket. |

---

## Incremental lifecycle (v4 async-camera) — deployment notes

The `--auto` service is now a **manifest-driven, multi-batch scheduler** that
handles cameras arriving at different times. No change to service names or paths:
the existing `wagon-eye.service` unit and `EnvironmentFile=` layout are unchanged.
An old env file that does not set the new variables uses safe defaults (all new
`WAGONEYE_*` keys are optional).

**New environment variables** (see `deploy/wagon-eye.env.example` for full comments):

| Variable | Default | Meaning |
|---|---|---|
| `WAGONEYE_MASTER_WAIT_MINUTES` | `10` | wait for RIGHT_UP before it's "late" |
| `WAGONEYE_SUPPORT_FUSION_WAIT_MINUTES` | `3` | support window (armed on RIGHT_UP arrival); must be ≤ final wait |
| `WAGONEYE_FINAL_CAMERA_WAIT_MINUTES` | `30` | hard close → `COMPLETED_PARTIAL` |
| `WAGONEYE_ENABLE_LEFT_UP_FALLBACK_MASTER` | `false` | experimental; keep off |
| `WAGONEYE_GENERATE_INTERIM_REPORTS` | `true` | regenerate reports on disk as cameras arrive |
| `WAGONEYE_UPLOAD_INTERIM_REPORTS` | `false` | interim reports are local-only unless set |
| `WAGONEYE_EMAIL_INTERIM_REPORTS` | `false` | one email at closure unless set |
| `WAGONEYE_LATE_CAMERA_POLICY` | `IGNORE` | terminal batches never reopened |
| `WAGONEYE_ACTIVE_BATCH_POLL_INTERVAL` | `60` | scheduler poll cadence (s) |
| `WAGONEYE_MANIFEST_S3_PREFIX` | *(empty)* | manifest S3 prefix override |
| `WAGONEYE_PIPELINE_SOURCE` | `trimmed` | `raw` = `--auto` owns an ExtractionManager (raw→trimmed) in-process (§6b) |
| `WAGONEYE_EXTRACTION_POLL_INTERVAL` | `60` | ExtractionManager / standalone extractor sweep cadence (s) |
| `WAGONEYE_EXTRACTION_MODELS_DIR` | `models/extraction` | extractor classify models dir (§6b) |

**Startup validation.** The orchestrator validates configuration before polling
and **fails fast** (exit 2) with clear `[CONFIG]` errors on: negative deadlines,
`support_wait > final_wait`, interim upload/email enabled without interim
generation, empty S3 input prefixes / output bucket in `--auto`, missing email
endpoint/recipients when email is enabled, or a non-writable workspace/log/temp
dir. It then logs a **redacted** effective-settings summary (recipients shown as
counts; no secrets).

**Backward compatibility.** Existing CLI modes (`--auto`, `--once`, `--batch`,
`--local-only`) parse and behave as before; `--local-only` still runs a single
complete-set batch offline. Extraction is **opt-in** via the pipeline source —
with `WAGONEYE_PIPELINE_SOURCE` unset (`trimmed`), `--auto` is the same pure
trimmed-input consumer it always was, so existing two-service deployments are
unaffected; `--source raw` / `--source trimmed` override the env at runtime.
Legacy flat `wagon_states/<feature>/GW_n.json` batches remain readable. New
batches write only the camera-scoped layout.

> ⚠️ **Production gate.** The code is structurally production-ready, but a real
> deployment should still be preceded by one full four-camera run and one
> delayed-camera run using **real models, real videos, live S3, and the real
> email/upload services** — the automated tests use mocks/fixtures and do not
> exercise those live integrations or `.pt` inference quality.
