# Browser Fingerprint Anomaly Detection — MLOps Extension

A **synthetic-data demonstration** of MLflow tracking, inference APIs, containerization, tests, CI and basic feature-shift monitoring. It is **not** the original master's implementation and **does not** establish detection of real malicious bots.

## Requirements
Python 3.11, optional Docker, and (for later) Azure subscription.

## Windows PowerShell quickstart
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.generate_data
python -m src.train
pytest -q
uvicorn src.api:app --reload
```

Visit http://127.0.0.1:8000/docs and try POST `/predict`.

## Example request
```json
{"screen_width":1920,"screen_height":1080,"plugin_count":5,"canvas_entropy":0.5,"hour":12,"ua_family":"Chrome"}
```

## MLflow local UI
In another terminal from the repository root:
```powershell
mlflow ui --backend-store-uri ./mlruns --port 5000
```
Visit http://127.0.0.1:5000 .

## Monitor a recent batch
```powershell
python -m src.monitor data/synthetic_fingerprints.csv
```
Reports mean feature shift in units of baseline standard deviations; **not** a validated concept-drift or statistical significance test.

## Docker
Train before building (the ignored local `artifacts/` folder must exist):
```powershell
docker build -t fingerprint-api .
docker run --rm -p 8000:8000 fingerprint-api
```

## Adaptation to the actual research
Replace `src/generate_data.py` with a permissioned original dataset loader; preserve the original schema and preprocessing, document provenance and privacy treatment, avoid direct identifiers, reproduce reported results, evaluate temporal splits and bias, and only then consider an Azure deployment.

## Limitations
- Synthetic labels were created for benchmarking and do not represent real bot ground truth.
- `joblib.load` loads Python objects: use only trusted artifacts.
- No authentication, rate limits or production security hardening.
- This is a starter CI workflow, not continuous deployment.
- Cloud deployment/IaC are future extensions, not included as working deployments.
- Unsupervised anomaly != confirmed bot.

## Reconstructed multi-algorithm benchmark (new)

The original research code and dataset were lost. This extension **reimplements**
the approaches named in the master's poster, not their original fitted models,
parameter selections, experimental results or data. Available algorithms:

- Isolation Forest: inverse isolation score
- Local Outlier Factor (LOF): novelty-mode inverse local density score
- K-means: distance to closest centroid
- Agglomerative Clustering: distance to nearest *training-cluster centroid*
- BIRCH: distance to nearest *training-cluster centroid*
- DBSCAN: distance to nearest DBSCAN core sample; if no cores, nearest training point fallback

For fairer anomaly flagging, all algorithms use a **separate unlabeled calibration
subset** and flag scores strictly above the 92nd percentile (8% expected alert
rate). Scores have different units; **do not compare raw scores across models**.
Labels are reserved for synthetic evaluation. The train/calibration/test split
is 60%/20%/20%, stratified using simulated labels for test stability. A real
study should use chronological splits and not rely on synthetic-label
stratification.

```powershell
# Compare all algorithms and write artifacts/comparison.csv and comparison.json
python -m src.train --compare

# Train and serve only a specific algorithm (example: LOF)
python -m src.train --algorithm lof
uvicorn src.api:app --reload
```

`/predict` returns the selected algorithm, its algorithm-specific anomaly score,
threshold and an anomaly indicator. A comparison run selects the highest
**synthetic test-set average precision** for demonstration purposes only; that
winner-selection procedure biases its reported performance. Validate on a
fresh, independent holdout before making any real-world performance claim.

### Important method limitations

Agglomerative Clustering and DBSCAN do not intrinsically support out-of-sample
prediction. The saved centroid/core-neighbor distance strategies are explicit
engineering approximations, **not** their original clustering labels.
Likewise, anomaly scores from K-means and BIRCH measure proximity to clusters,
not maliciousness. DBSCAN's `eps` is a baseline guess and must be tuned to the
feature space and data scale. K-means and BIRCH assume cluster geometry that
may not suit browser fingerprint distributions. PCA and t-SNE were mentioned
in the poster but are not part of this new benchmark; t-SNE particularly
should not be treated as an ordinary out-of-sample production transform.

This package is **not a faithful reproduction of the original master's
experiments** and cannot substantiate its reported 0.5% false-positive rate or
real-world bot-detection performance.
