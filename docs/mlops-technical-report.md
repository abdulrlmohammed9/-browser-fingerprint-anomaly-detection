# MLOps technical report

## Scope and provenance
This repository is a reconstructed extension of Abdul Rahman Mohammed's University of the Cumberlands master's research. The [original poster](original-research/university-of-the-cumberlands-poster.pdf) is preserved separately. The original source code, fitted models and research dataset are unavailable. Current results come from newly generated synthetic data and cannot validate historical academic claims.

## Data and experimental method
The demonstrated command was `python -m src.train --compare --n 5000`, using generator seed 42 and a nominal synthetic anomaly rate of 8%. Features are screen_width, screen_height, plugin_count, canvas_entropy, hour and ua_family; synthetic_anomaly_label is reserved for evaluation and split stratification. Numeric features use StandardScaler, and ua_family uses one-hot encoding, fitted only on the fit split.

The stratified split is 60% fit / 20% calibration / 20% test: 3,000 / 1,000 / 1,000 rows. Labels influence stratification but are excluded from estimator fitting and threshold calibration. Each algorithm flags scores strictly above its calibration-score 92nd percentile. An 8% calibration target does not guarantee an 8% test alert rate.

## Algorithm interpretation
| Method | Out-of-sample anomaly score |
|---|---|
| Isolation Forest | Negative score_samples |
| LOF | Negative novelty-mode score_samples |
| K-means | Distance to nearest fitted centroid |
| Agglomerative | Distance to nearest training-cluster centroid |
| BIRCH | Distance to nearest training-cluster centroid |
| DBSCAN | Distance to nearest core sample; nearest training point fallback if no cores |

Agglomerative and DBSCAN scores are engineering approximations, not native predictions of training cluster labels. Raw scores have different units and should not be compared between methods. PCA and t-SNE are not implemented in this extension.

## Recorded results
Source: [comparison.json](results/comparison.json); [CSV](results/comparison.csv).
| Algorithm | Precision | Recall | F1 | Average precision | ROC AUC | False-positive rate |
|---|---:|---:|---:|---:|---:|---:|
| Isolation Forest | 0.696 | 0.714 | 0.705 | 0.837 | 0.978 | 0.026 |
| LOF | 0.051 | 0.052 | 0.051 | 0.076 | 0.375 | 0.081 |
| K-means | 0.174 | 0.195 | 0.184 | 0.177 | 0.534 | 0.077 |
| Agglomerative | 0.203 | 0.182 | 0.192 | 0.145 | 0.530 | 0.060 |
| BIRCH | 0.141 | 0.169 | 0.154 | 0.095 | 0.476 | 0.086 |
| DBSCAN | 0.200 | 0.208 | 0.204 | 0.222 | 0.400 | 0.069 |

Isolation Forest achieved the highest recorded average precision (0.837), F1 (0.705), and ROC AUC (0.978). Its false-positive rate was approximately 2.60%. LOF's average precision was 0.076 and ROC AUC 0.375, showing weak ranking performance in this configuration; the benchmark does not establish which method is best on real fingerprints.

The program selects the model with highest test-set average precision. Selection on this same test set biases claims about the winner. Future evaluation needs a separate validation set for selection and a fresh independent test set, ideally split chronologically for real event data.

## Operational validation
- The captured Codespaces test session reports **8 passed, 368 warnings**. Warnings shown include dependency deprecations under Python 3.14; they are not failures, but deserve review before production use.
- MLflow 3.18.0 displayed six experiment runs. The file backend required `MLFLOW_ALLOW_FILE_STORE=true`.
- FastAPI returned HTTP 200 using Isolation Forest. The captured example score was approximately 0.411, below the threshold of approximately 0.574, giving `is_anomaly: false`.
- Monitoring completed with no flags across the five numeric features.
- Docker build/run and Azure deployment have not been demonstrated.

Evidence: [tests](screenshots/tests-passed.png), [MLflow](screenshots/mlflow-experiments.png), [API](screenshots/api-prediction.png), [monitoring](screenshots/monitoring-output.png). Installed packages: [environment.txt](results/environment.txt); Python version is visible in the test screenshot, not recorded by pip freeze.

## Monitoring interpretation
[drift_report.json](results/drift_report.json) records numeric mean shifts in baseline standard deviations: screen width 0.003, screen height 0.012, plugin count 0.006, canvas entropy 0.006 and hour 0.010. The threshold is greater than 1.0 baseline standard deviation, and all flags were false.

The monitored batch uses the same synthetic generator and seed as the benchmark and overlaps the population used for fitting. This is an execution smoke check, not independent production monitoring. The check does not inspect categorical shifts, missingness, variance changes or model-performance drift, and is not a statistical significance test.

## Reproducibility and next work
Follow the [README](../README.md) commands and retain the version snapshot. requirements.txt uses version ranges rather than a fully pinned cross-platform lockfile. Different library versions can change results. Test runs operate in temporary directories; preserve actual benchmark outputs from artifacts into docs/results after training.

Next work: validate the supplied Docker image, establish data permissions and provenance for a real dataset, introduce chronological independent holdouts, tune methods on validation data, and strengthen monitoring. Use only trusted joblib artifacts. Authentication, rate limits and production hardening are not implemented in the API; keep development ports private.
