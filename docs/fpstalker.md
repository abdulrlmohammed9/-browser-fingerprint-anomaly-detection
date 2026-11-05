# FP-Stalker real-data workflow

## Source and scope
Source: https://github.com/Spirals-Team/FPStalker . The source repository supplies extension1.txt.tar.gz and extension2.txt.tar.gz; its README describes a 15k-fingerprint sample. Dataset instructions: https://github.com/tandriamil/BrFAST/blob/main/executables/dataset/readme.md . Check upstream terms before redistributing any raw dataset; a code license is not assumed to independently resolve dataset redistribution rights. This repository publishes its own loader and aggregate outputs, not the source dump or copied upstream algorithms.

This dataset is external to the author's original master's research. It contains no verified bot/anomaly labels used by this implementation. Synthetic benchmark metrics remain in docs/results; real-data artifacts use artifacts/fpstalker. No supervised accuracy metrics or automatic winning model are produced.

## Run in Codespaces
After extracting both archives and combining extension1.txt and extension2.txt into data/raw/fpstalker/tableFingerprints.sql, run from the repository root:

```bash
git pull
source .venv/bin/activate
pip install -r requirements.txt
export MLFLOW_ALLOW_FILE_STORE=true
python -m src.train_fpstalker --sql data/raw/fpstalker/tableFingerprints.sql
pytest -q
```

The SQL is parsed as text, never executed. The parser supports quoted scalar strings, MySQL escapes, NULLs, numeric literals, explicit INSERT columns and schema column order. It fails on unsupported values, malformed tuples and missing required fields. Use an unchanged upstream dump. No database server is required. Only whitelisted fields are retained; addresses, hashes and raw canvas images are discarded.

## Features and quality
Numeric: resolutionJS to screen_width/screen_height; timezoneJS to timezone_offset; pluginsJS markers to plugin_count. Unknown nonempty plugin formats remain missing, not zero. Malformed numeric values become missing and median imputation plus missing-value indicators is fitted on training data only. An entirely missing numeric fit feature stops the run for inspection.

Categories: platformJS, cookiesJS, localJS, sessionJS, adBlock, and coarse browser family derived from userAgentHttp using documented substring rules. Capability values are categories, not assumed Boolean encodings. Raw user-agent strings are discarded after feature preparation. No canvas entropy is invented from hashes. Raw id is used transiently for overlap counts only; creationDate is used for splitting only. Neither is a model input or exported record identifier.

Chronological boundaries use 60%/80% positions among distinct creation timestamps, so identical timestamps never cross splits. Actual row fractions can differ from 60/20/20. Invalid timestamps are excluded and counted. Repeated browsers may occur in later windows: overlap counts are reported without identifiers. This evaluates future observations including returning browsers, not generalization to entirely unseen devices. The public release is a sample rather than a complete chronological production history.

The historical fit period is deterministically subsampled to at most 3,000 records to limit quadratic clustering costs. Calibration and test use all their rows. Override with --max-fit only when adequate memory is available. No duplicate removal is silently applied; repeated observations are retained as part of the longitudinal sample.

All six models calibrate an 8% target alert rate on the calibration period. This is an operational choice, not an estimate of bot prevalence. Score scales differ between algorithms. Temporal alert-rate changes and weak performance do not establish maliciousness.

## Outputs and MLflow
artifacts/fpstalker contains one trusted joblib per algorithm, comparison.csv/json, data_quality.json (source SHA-256, row counts, missing fractions, time periods and browser overlap), reference.json, and drift_report.json. Drift compares numeric test means to fit means; it does not monitor categorical drift or prove concept drift. Constant feature baselines can make standardized shifts large. These files contain aggregates, not example fingerprint rows.

Open the existing MLflow UI on port 5000, switch to Model training, then select **browser-fingerprint-fpstalker-unlabeled**. Inspect test_alert_rate, calibration_alert_rate, score quantiles and thresholds. No real-data results are claimed until the downloaded dump has been run in Codespaces. Automated tests use an invented SQL fixture and validate mechanics, not real bot detection.

## Separate real-data API
Stop the synthetic API with Ctrl+C in its terminal, then run:

```bash
source .venv/bin/activate
export FPSTALKER_ALGORITHM=isolation_forest
uvicorn src.api_fpstalker:app --host 0.0.0.0 --port 8000
```

Open port 8000's actual forwarded URL and append /docs. Isolation Forest is an explicit default, not a selected winner. Choose any of the six algorithm names through FPSTALKER_ALGORITHM, then restart the server. This API expects models at the default artifacts/fpstalker path. Load only trusted artifacts. Example POST /predict:

```json
{"screen_width":1920,"screen_height":1080,"timezone_offset":-60,"plugin_count":5,"platform":"Win32","cookies":"yes","local_storage":"yes","session_storage":"yes","adblock":"no","browser_family":"Chrome"}
```

The API accepts the prepared feature schema, not raw SQL. Use capability spellings matching the dataset for representative requests; unknown categories are tolerated by the encoder. HTTP 200 confirms inference mechanics, not accuracy.

## Preserve aggregate evidence
After a successful run:

```bash
mkdir -p docs/results/fpstalker
cp artifacts/fpstalker/comparison.csv artifacts/fpstalker/comparison.json artifacts/fpstalker/data_quality.json artifacts/fpstalker/drift_report.json docs/results/fpstalker/
git add docs/results/fpstalker
git commit -m "Add FP-Stalker unlabeled evaluation results"
git push
```

Do not add raw data, fitted joblib artifacts or record-level identifiers. Retain synthetic results separately. The original src.api and synthetic workflow remain available.
