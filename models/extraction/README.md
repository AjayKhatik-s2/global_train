# models/extraction/ — raw→trimmed EXTRACTION classifiers

Drop the two **extraction** classify models here:

| File                     | Cameras                    | Classes (typical) |
|--------------------------|----------------------------|-------------------|
| `side_classification.pt` | RIGHT_UP, LEFT_UP          | `empty_track` / `wagon` / `engine` / `second_track` |
| `top_classification.pt`  | RIGHT_UP_TOP, LEFT_UP_TOP  | `empty_track` / `wagon` / `engine` |

These are only needed when this process **produces its own trimmed clips** —
i.e. `WAGONEYE_PIPELINE_SOURCE=raw` or `--source raw`. A pure consumer
(`--source trimmed`, the default) never loads them, and startup does not require
them.

> ⚠️ **`side_classification.pt` here is NOT the same model as
> `models/reconstruction/side_classification.pt`.** Same filename, different
> weights: this one decides "is a train in frame right now" for train extraction;
> the reconstruction one classifies a segment as ENGINE / WAGON / BRAKE_VAN for
> Stage 1. That collision is exactly why they live in separate directories —
> never symlink or copy one over the other.

In the V4 Train-Inspection-Engine these live at the **root** of
`s3://wagon-eye-models/` (`configs/cameras/*.yaml` → `classification_model_path`),
so the simplest route is an explicit copy:

```bash
aws s3 cp s3://wagon-eye-models/side_classification.pt models/extraction/
aws s3 cp s3://wagon-eye-models/top_classification.pt  models/extraction/
```

Startup auto-sync (`WAGONEYE_MODELS_S3_BUCKET`) also covers these, but it expects
the category layout `s3://<bucket>/<WAGONEYE_MODELS_S3_PREFIX>/extraction/<file>`
— **not** V4's flat root layout. Either mirror the models into that layout, or use
the `aws s3 cp` above.

With `--source raw` and either model absent, `master_runner` **refuses to start**
and names the missing file (`core.config.validate_config`) rather than failing a
per-camera sweep once a minute.

Override the directory with `WAGONEYE_EXTRACTION_MODELS_DIR`, and an individual
model with `WAGONEYE_EXTRACTION_<CAMERA>_CLASSIFICATION_MODEL`.
