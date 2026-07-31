# Production auto-pipeline runbook

One command, one service, raw CCTV in → reports + dashboard feed out:

```bash
python -m orchestrator.master_runner --auto --source raw --no-interactive
```

```
RAW CCTV (S3)
    │   biro-wagon-raw-video-copy/camera_CCTV_HZBN_DHN_*/
    ▼
[ExtractionManager]  detect train pass → trim → upload
    │   biro-wagon-pre-processed-video-copy/camera_CCTV_HZBN_DHN_*/
    ▼
[master_runner --auto]  cluster the 4 cameras into one train batch
    ▼
Stage 1  seal GlobalTrainState (RIGHT_UP canonical count + numbering + direction)
Stage 2  materialize wagon_cache
Stage 3  features: load → damage → door → ocr (Rekognition)
Stage 4  fusion → unified wagon states
Stage 4b overlay videos
Stage 5  4 camera PDFs + combined PDF/JSON
Stage 6  S3 archive · ONE email · 4 exact-V4 *_inspection.json → dashboard
```

Both halves are decoupled through S3, so the extractor's trimmed bucket **is** the
consumer's `WAGONEYE_S3_INPUT_BUCKET`. That is why one process can own both
without any code coupling.

---

## 1. Buckets — identical to the V4 engine

Every bucket below is a **built-in default**, taken from the V4 engine's
`configs/cameras/*.yaml` and `configs/combiner.yaml`. An empty env file already
points at the existing production topology — nothing to configure.

| V4 config key | Bucket | Layout | Used by |
|---|---|---|---|
| `raw_video_bucket` | `biro-wagon-raw-video-copy` | `<camera_folder>/` | extraction input (`--source raw`) |
| `trimmed_video_bucket` | `biro-wagon-pre-processed-video-copy` | `<camera_folder>/` | extraction output **and** `--auto` input |
| `detected_video_bucket` | `biro-wagon-processed-video-copy` | `<camera_folder>/` | overlay-video mirror |
| `inspection_output_bucket` | `biro-wagon-report-biro-copy` | `train_batch/<batch_key>/` | reports, evidence, archive, manifests |
| `combined_output_bucket` | `biro-combined-report-copy` | — | combined report |
| *(models)* | `wagon-eye-models` | bucket **root** (flat) | missing-model auto-sync |
| `region` | `ap-south-1` | | everything |

`<camera_folder>` is one of
`camera_CCTV_HZBN_DHN_{2_RIGHT_UP,1_LEFT_UP,5_RIGHT_TOP,6_LEFT_TOP}`, defined once
in `core.constants.CAMERA_S3_FOLDER` and shared by the extraction driver, the
report layout, and the dashboard feed — so a rig rename is a one-line edit and the
producer and consumer can never drift apart.

The one bucket that is **not** from V4 is the dashboard inspection-JSON bucket
(`ankit-version-1-prod`, `WAGONEYE_INSPECTION_JSON_BUCKET`): that is the live V1
dashboard's own location, kept as-is so the existing dashboard keeps working. V4
writes its equivalent under `inspection_output_bucket/<camera_folder>/`.

Overrides: `WAGONEYE_S3_{RAW_VIDEO,TRIMMED_VIDEO,DETECTED_VIDEO,COMBINED_REPORT,
OUTPUT,INPUT}_BUCKET`, plus per-camera
`WAGONEYE_EXTRACTION_<CAMERA>_{RAW,TRIMMED}_BUCKET` (values are
`"<bucket>/<prefix>"`).

> **Model-filename collision.** `side_classification.pt` and
> `top_classification.pt` each exist in **both** `models/reconstruction/` and
> `models/extraction/` with **different weights**. A flat bucket cannot tell them
> apart by key, so these two are never auto-downloaded — startup reports them and
> you place them yourself. Everything else auto-syncs normally.

## 2. Models

```
models/reconstruction/   right_up_gap.pt left_up_gap.pt top_gap.pt
                         side_classification.pt top_classification.pt
models/features/         door_state.pt loaded.pt damage.pt
                         wagon_number_update.pt      ← V4 right_up plate detector
models/extraction/       side_classification.pt top_classification.pt
                                                     ← ONLY for --source raw
```

`models/extraction/side_classification.pt` and
`models/reconstruction/side_classification.pt` are **different weights with the
same filename** — see [models/extraction/README.md](models/extraction/README.md).
Never copy one over the other.

`wagon_number_update.pt` is the canonical OCR detector. If only the older
`wagon_id_counting.pt` is present it is used automatically (and named as such in
the model report and the completion marker), so an existing checkout keeps
running.

## 3. IAM

The instance role needs, beyond the existing S3 permissions:

```json
{ "Effect": "Allow", "Action": "rekognition:DetectText", "Resource": "*" }
```

Verify everything before starting the service:

```bash
python scripts/preflight.py --mode auto
```

It checks dependencies, directories, effective config, every required model
(including the extraction pair when `--source raw`), S3 reachability, and makes a
real `DetectText` probe call.

## 4. Minimum env file

`deploy/wagon-eye.env` — full annotated reference in
[deploy/wagon-eye.env.example](deploy/wagon-eye.env.example):

Because every bucket already defaults to the V4 set, the whole file is one line:

```bash
WAGONEYE_PIPELINE_SOURCE=raw
```

A fuller example, if you want the paths and recipients pinned explicitly:

```bash
WAGONEYE_PIPELINE_SOURCE=raw
WAGONEYE_EXTRACTION_MODELS_DIR=/opt/wagon_eye_v4/models/extraction
WAGONEYE_WORKSPACE_ROOT=/data/wagon_eye/batch_outputs
WAGONEYE_LOG_DIR=/var/log/wagon_eye
WAGONEYE_DEVICE=cpu
WAGONEYE_OCR_ENGINE=rekognition
WAGONEYE_EMAIL_RECEIVER=ops@example.com
```

Then:

```bash
sudo cp deploy/wagon-eye.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now wagon-eye
journalctl -u wagon-eye -f
```

With `--source raw` do **not** also enable `wagon-eye-extraction.service` — two
producers would cut the same raw clips twice.

Startup is fail-fast: a missing extraction model, an empty input-prefix list, an
unwritable workspace, or a missing `boto3` for Rekognition all refuse to start and
name the problem (`core.config.validate_config`).

---

## 5. Wagon-number OCR (Amazon Rekognition)

`WAGONEYE_OCR_ENGINE=rekognition` (default) reproduces the V4 engine's OCR:

1. `wagon_number_update.pt` detects plate boxes on RIGHT_UP frames (OCR authority
   — no other camera reads numbers).
2. Detections are grouped into **bands** by frame gap
   (`WAGONEYE_OCR_GAP_TOLERANCE`, default 8) — one band per plate passing view.
3. Per band, **three** crops are stacked on one vertical white sheet, so each
   plate lands on its own horizontal band and Rekognition returns it as a separate
   `LINE`. Which three depends on load state, because cargo occludes the plate
   from one end depending on travel direction:
   - **loaded** → primary `End-7/-5/-3`, fallback `Start+3/+5/+7`
   - **empty** → primary `Start+3/+5/+7`, fallback `End-7/-5/-3`

   Load state comes from this wagon's own `load` result — the wagon-wise scheduler
   runs LOAD before OCR precisely so that value is final.
4. One `DetectText` call per sheet. The reader picks the single best **valid
   11-digit** `LINE` (it never concatenates the three copies into a 33-digit run),
   and reassembles a plate split across two rows. First valid read wins.

**Cost control.** `WAGONEYE_REKOGNITION_MAX_CALLS_PER_WAGON` (default 4) is a hard
ceiling — worst case 4 billed requests per wagon, and ENGINE / BRAKE_VAN wagons are
skipped entirely. A ~58-wagon train therefore costs well under 232 calls, typically
~1 per wagon since the first sheet usually validates.

**Degradation.** If a client cannot be built (no boto3, no credentials) the run
logs it once and falls back to `easyocr` rather than emitting `NO_DATA` for every
wagon. Throttles and API errors degrade that one wagon to `NO_DATA`; a wrong number
is never reported. Set `WAGONEYE_OCR_ENGINE=easyocr` to run fully offline.

**Evidence.** Per wagon, `evidence/<GW_n>/ocr/RIGHT_UP/` holds `best_frame.jpg`
(annotated), `number_crop.jpg`, and `ocr_sheet.jpg` — the exact image sent to
Rekognition (V4's `wagon_numbers/wagon_number_segment_NNN.jpg`).

## 6. Per-camera inspection JSON

At finalization, one document per camera angle is written to
`delivery/dashboard/<raw_name>_inspection.json`, uploaded, and POSTed to the
ingest API. The schema is the **exact V4 contract** —
[delivery/inspection_json.py](delivery/inspection_json.py) is a port of the V4
engine's `reporting/json_builder.py` (both flavours, same keys, same key order,
same derivations).

| | side (`RIGHT_UP`, `LEFT_UP`) | top (`RIGHT_UP_TOP`, `LEFT_UP_TOP`) |
|---|---|---|
| `segment_type` | `wagon` / `engine` / `brakevan` | `wagon_empty` / `wagon_loaded` / `engine` / `brakevan` |
| `segment_type_map[id]` | `{type, number}` | `{type, number, wagon_count}` |
| per-wagon | `door_status`, `door_close_detected`, `door_partial_detected`, `damage_detected` | `load_status`, 3 damage booleans |
| counts | `doors_open/partially_closed/closed`, `damaged_wagons` | `floor_dmg_wagons`, `inner_wall_dmg_wagons`, `floor_dmg_probable_wagons`, `probable_damage_wagons` |
| `rake_status` | from travel `direction` (L→R = Loaded) | majority vote of loaded vs empty |

All four files describe the **same** wagon sequence (the sealed GlobalTrainState),
but each reports only what its camera is authoritative for: RIGHT_UP the right door
+ OCR, LEFT_UP the left door, each top camera its own load + damage reads.

### Will it show on the dashboard?

Mechanically the pipeline always does all four steps at finalization: build one
document per present camera → write it under `delivery/dashboard/` → upload to
`s3://ankit-version-1-prod/<Folder>/<YYYY-MM-DD>/<raw_name>_inspection.json` →
`POST {camera_id, inspection_s3_uri, version}` to the ingest API. The outcome of
each is recorded per camera in `delivery/finalization.json` under
`dashboard_ingested`, so a run never silently half-delivers.

**Which receiver — this decides which dashboard sees it.** The payload is the same
three fields the V4 engine sends, but the *host* differs:

| | Endpoint | Default |
|---|---|---|
| V1 receiver (`v1`) | `ms-pnr-location-notification-api.suvidhaen.com/cctv-receiver/inspections/ingest` | **yes** |
| V4 receivers (`v4`) | `cctv-wagon-api.suvidhaen.com/inspections/ingest` **and** `cctv-wagon-uat-api.suvidhaen.com/inspections/ingest` (V4 posts to both) | no |

A report delivered to one is **not** visible on the other. The default is the V1
receiver — the endpoint this feed has always used — so an upgrade never silently
repoints live traffic. To deliver to the V4 dashboard instead (or as well):

```bash
WAGONEYE_INSPECTION_INGEST_API_URLS=v4          # both V4 receivers
WAGONEYE_INSPECTION_INGEST_API_URLS=v1,v4       # V1 + both V4
WAGONEYE_INSPECTION_VERSION=v4                  # set this too, for the v4 tab
```

A document counts as ingested when at least one receiver accepts it; every
endpoint's outcome is recorded under `dashboard_ingested[<camera>].endpoints`.

V4 additionally fires a separate ML callback
(`cctv-wagon-api.suvidhaen.com/api/v1/ml`, `X-ML-SECRET` header) carrying video +
PDF ids. global_train does **not** implement that callback.

Whether it then *appears* depends on three identifiers matching what the
dashboard expects:

| Variable | Default | Why it matters |
|---|---|---|
| `WAGONEYE_INSPECTION_VERSION` | `v1` | The dashboard picks its **tab** from this. Left at the pre-existing production value so reports keep landing where they do today. `v4` = byte-identical to the V4 engine, and the v4 tab. |
| `WAGONEYE_INSPECTION_STRIP_CAMERA_PREFIX` | *follows the version* | `v1` → `camera_CCTV_HZBN_DHN_2_RIGHT_UP` (what the live feed has always sent); `v4` → `CCTV_HZBN_DHN_2_RIGHT_UP` (V4-exact). **These two must agree** — a v1 document carrying a v4 `camera_id` cannot be matched to a camera. Pin the variable to override. |
| `WAGONEYE_INSPECTION_FOLDERS` | `Right_up`, `Left_up`, `Right_Top`, `Left_Top` | The S3 folder per camera. **Verified** — each is that camera's own `INSPECTION_JSON_FOLDER` from the old per-camera pipeline that feeds this dashboard. Note the top cameras are `Right_Top` / `Left_Top` (capital T, no "up"). |

The ingest API payload's `camera_id` is always the full prefixed form, unaffected
by the setting above.

Check what actually happened after a run:

```bash
python -c "import json;print(json.dumps(json.load(open('batch_outputs/<key>/delivery/finalization.json'))['dashboard_ingested'],indent=2))"
```

Each camera reports `ingested` (with a `run_id`), `already_ingested`,
`upload_failed`, or `ingest_failed` (with the error and HTTP status).

Disable the whole feed with `WAGONEYE_DASHBOARD_INGEST_ENABLED=false`.

### Loco numbers

Every `ENGINE`-classified Global Wagon is one loco band, numbered from 1 in train
order. `wagon_number_update.pt` detects **both** plate classes — `wagon_id` (the
11-digit wagon plate) and `loco_no` (the 5-digit loco plate) — so the same
detector serves both paths. On an engine, the OCR processor takes the loco path:
`loco_no` boxes → bands → a `Middle-2 / Middle / Middle+2` sheet (a loco's plate is
centred on its face, so there's no cargo occlusion to steer around) → one
DetectText call → best valid **5-digit** line.

That populates `loco_number_results` (keyed by `str(loco_id)`), `loco_frames`, and
`total_loco_frames`. An invalid read is still emitted with its raw digits so it
stays auditable — only `is_valid_5_digit` marks a usable number. The loco path
never guesses a plate class: an unrecognised label is rejected rather than risk
reporting a mis-detected wagon plate as a loco number. Requires the Rekognition
engine; under `easyocr` the engine segment is recorded but no number is read.

### Probable top damage

`damage.pt` emits three classes: `Floor_damage`, `Inner_wall_damage`, and
`Floor__probable_damage` — **note the double underscore**, which is the model's
real class name. Probable damage feeds `floor_dmg_probable_wagons` /
`probable_damage_wagons` and is deliberately **not** counted as confirmed damage
(`damaged_wagons`, `damage_detected`), matching V4.

### Fields with no source in this pipeline

Reported empty, never invented (also echoed under `inspection_data._adapter`):

- `problem_frames[].s3_key` — the already-uploaded `train_batch/…/evidence/…` URL
  is referenced instead of re-uploading into the legacy key layout.

## 7. Travel direction

Stage 1 now persists `travel_direction` (`left-to-right` / `right-to-left` /
`unknown`) in `global_train_state.json`. It is read off the sign of each master
gap's `centre_x` drift — data the tracker already produced — so it costs no extra
compute and no video re-read.

V4 derives the same label from dense optical flow. **Same vocabulary, cheaper
estimator**: both only ever use the *sign* of horizontal motion. Ties and
trajectory-less gaps return `unknown` rather than guessing. A batch sealed before
this field existed reads `unknown`.

## 8. Disk

A batch is roughly **3 GB** on local disk (measured on 960×540 @ 15 fps clips:
158 KB per JPEG × ~4000 frames × 4 cameras ≈ 2.4 GB of `wagon_cache`, plus
`downloads/` and the overlay videos).

Only three subdirectories are ever local-only — `wagon_cache/`, `downloads/`,
`archive/`. Everything else (`global_state`, `wagon_states`, `reports`,
`evidence`, `processed_videos`) is mirrored to S3 at finalization, so deleting
those three is lossless.

`WAGONEYE_PRUNE_INTERMEDIATES` (**on by default**) does exactly that, once a batch
reaches a *successful* terminal state — which drops a finished batch from ~3 GB to
tens of MB. Two deliberate exceptions:

- a **FAILED** batch keeps its intermediates so the failure is still diagnosable
  on the box;
- an **ACTIVE** batch is never touched — a late camera regenerating its report
  reads quartile frames back out of `wagon_cache`, so pruning early would quietly
  degrade those PDFs.

`WAGONEYE_BATCH_RETENTION_DAYS` (default 0 = off) additionally removes whole batch
directories older than N days, skipping active ones regardless of age.

**Why not put `wagon_cache` on S3?** Stage 3 globs the directory and does one
`cv2.imread` per frame — ~22k small random reads per batch. At S3's 10–50 ms per
GET that adds tens of minutes, and `glob` doesn't work against an object store;
`mountpoint-s3`/`s3fs` would mount it but this is their worst-case access pattern.
The cache is a regenerable intermediate, so deleting it beats relocating it.

Provision a dedicated volume for headroom:

```bash
WAGONEYE_WORKSPACE_ROOT=/data/wagon_eye/batch_outputs
```

## 9. Operations

```bash
systemctl status wagon-eye
journalctl -u wagon-eye -f
tail -f logs/wagon_eye.log

sudo systemctl stop wagon-eye        # SIGTERM: finishes the current batch, then exits
sudo systemctl restart wagon-eye     # resumes from manifests + markers, repeats no work
```

Restart safety and duplicate prevention are unchanged — see
[RUNTIME.md](RUNTIME.md) §6–7. Extraction adds one more layer: a per-camera local
ledger of processed raw keys plus the extractor's own S3 state store, so a train
split across two raw clips is cut once.

| Symptom | Cause / action |
|---|---|
| `refusing to start … extraction classify model(s) missing` | `--source raw` without `models/extraction/`. Populate it or use `--source trimmed`. |
| `Rekognition unavailable … falling back to easyocr` | No boto3/credentials. Check the instance role; `preflight --mode auto` shows the exact reason. |
| Every wagon `NO_DATA` with `rekognition_calls=0` | The plate detector found nothing — check `wagon_number_update.pt` is the right model, and `WAGONEYE_OCR_GAP_TOLERANCE`. |
| `direction: unknown` in the JSON | Gap tracks had no `bbox_history`, or rightward/leftward votes tied. Side `rake_status` then reads `Unknown`. |
| Raw clips pile up, no trimmed output | Extraction sweep errors — grep `[EXTRACT-MGR]` in the log. A missing extraction model is non-fatal to the consumer but produces nothing. |
| Dashboard shows no new trains | `WAGONEYE_DASHBOARD_INGEST_ENABLED`, then the per-camera `dashboard_ingested` block in `delivery/finalization.json` for the recorded status/error. |
