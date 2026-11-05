"""Unlabeled real-data evaluation, isolated from synthetic artifacts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import joblib
import mlflow
import numpy as np
import pandas as pd
from src.fpstalker import NUMERIC, CATEGORICAL, FEATURES, read_sql, prepare, chronological_split
from src.model import ALGORITHMS, AnomalyModel


def train(path, output='artifacts/fpstalker', max_fit=3000):
    if max_fit < 20: raise ValueError('max-fit must be at least 20')
    out = Path(output)
    if out.resolve() == Path('artifacts').resolve(): raise ValueError('Use a separate real-data output directory')
    raw = read_sql(path)
    frame = prepare(raw)
    fit, cal, test = chronological_split(frame)
    # Deterministic subsample only within the historical fit period; clustering is quadratic.
    sampled = fit.sample(n=min(max_fit, len(fit)), random_state=42).sort_values('_time')
    if sampled[NUMERIC].notna().sum().eq(0).any():
        raise ValueError('A numeric feature is entirely missing in the fit sample; inspect the source format')
    out.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): digest.update(block)
    overlap = {}
    for name, x in [('calibration', cal), ('test', test)]:
        old = set(fit._browser) - {''}
        overlap[name] = int(x._browser.isin(old).sum())
    quality = {'source': 'https://github.com/Spirals-Team/FPStalker', 'sha256': digest.hexdigest(),
        'raw_rows': len(raw), 'duplicate_retained_records': int(raw.duplicated().sum()), 'invalid_timestamp_rows': int(frame._time.isna().sum()),
        'features': FEATURES, 'missing_numeric_fraction': frame[NUMERIC].isna().mean().to_dict(),
        'split_policy': '60/20/20 of distinct timestamps; no timestamp shared across splits',
        'rows': {'fit_available':len(fit), 'fit_used':len(sampled), 'calibration':len(cal), 'test':len(test)},
        'periods': {name: [str(x._time.min()), str(x._time.max())] for name,x in [('fit',fit),('calibration',cal),('test',test)]},
        'later_rows_with_browser_seen_in_fit': overlap,
        'interpretation': 'Temporal evaluation can contain returning browsers; not an unseen-browser evaluation. No bot/anomaly labels.'}
    os.environ.setdefault('MLFLOW_ALLOW_FILE_STORE', 'true')
    mlflow.set_tracking_uri((Path.cwd()/'mlruns').as_uri())
    mlflow.set_experiment('browser-fingerprint-fpstalker-unlabeled')
    results = {}
    for name in ALGORITHMS:
        model = AnomalyModel(name, numeric_features=NUMERIC, categorical_features=CATEGORICAL)
        model.fit(sampled[FEATURES]).calibrate(cal[FEATURES])
        scores = model.score_samples(test[FEATURES])
        metrics = {'test_alert_rate':float(np.mean(model.predict(test[FEATURES]) == -1)),
            'calibration_alert_rate':float(np.mean(model.predict(cal[FEATURES]) == -1)),
            'score_p05':float(np.quantile(scores,.05)), 'score_median':float(np.median(scores)),
            'score_p95':float(np.quantile(scores,.95)), 'threshold':model.threshold_}
        if not all(np.isfinite(v) for v in metrics.values()): raise ValueError('Nonfinite model metrics')
        results[name] = metrics
        joblib.dump(model, out/f'{name}.joblib')
        with mlflow.start_run(run_name=name):
            mlflow.log_params({'dataset':'fpstalker', 'labels':'unavailable', 'fit_rows':len(sampled),
                'calibration_rows':len(cal), 'test_rows':len(test), 'fit_subsample_seed':42,
                'calibration_target_alert_rate':.08, 'source_sha256':quality['sha256']})
            mlflow.log_metrics(metrics)
        print(f'{name:19s} test alert rate={metrics["test_alert_rate"]:.3f}')
    reference = {k:{'mean':float(sampled[k].mean()),'std':float(sampled[k].std())} for k in NUMERIC}
    drift = {}
    for k in NUMERIC:
        shift=abs(float(test[k].mean())-reference[k]['mean'])/max(reference[k]['std'],1e-8)
        drift[k]={'mean_shift_in_baseline_sd':float(shift) if np.isfinite(shift) else None,
                  'flag':bool(shift>1) if np.isfinite(shift) else None,
                  'test_missing_fraction':float(test[k].isna().mean())}
    for filename, obj in [('comparison.json',results),('data_quality.json',quality),('reference.json',reference),('drift_report.json',drift)]:
        (out/filename).write_text(json.dumps(obj,indent=2,allow_nan=False))
    pd.DataFrame.from_dict(results,orient='index').rename_axis('algorithm').to_csv(out/'comparison.csv')
    print(f'Saved aggregate outputs to {out}; no best model selected without labels.')
    return results

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--sql',required=True)
    p.add_argument('--output',default='artifacts/fpstalker')
    p.add_argument('--max-fit',type=int,default=3000)
    a=p.parse_args()
    train(a.sql,a.output,a.max_fit)
