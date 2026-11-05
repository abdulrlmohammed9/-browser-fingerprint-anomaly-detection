"""Reconstruct and compare the approaches named in the master's poster.
Synthetic anomaly labels are withheld from fitting and threshold calibration.
"""
import argparse
import json
from pathlib import Path
import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from src.features import FEATURES
from src.generate_data import make_data
from src.model import ALGORITHMS, build_pipeline

ARTIFACT_DIR=Path('artifacts')

def evaluate(actual,pred,score):
    tn,fp,fn,tp=confusion_matrix(actual,pred,labels=[0,1]).ravel()
    metrics={'precision':float(precision_score(actual,pred,zero_division=0)),
             'recall':float(recall_score(actual,pred,zero_division=0)),
             'f1':float(f1_score(actual,pred,zero_division=0)),
             'false_positive_rate':float(fp/max(1,fp+tn)),
             'predicted_anomaly_fraction':float(np.mean(pred))}
    if len(np.unique(actual))==2:
        metrics['average_precision']=float(average_precision_score(actual,score))
        metrics['roc_auc']=float(roc_auc_score(actual,score))
    return metrics

def train(algorithm='isolation_forest',n=2000,compare=False,seed=42):
    if algorithm not in ALGORITHMS and not compare:
        raise ValueError(f'Choose from {ALGORITHMS}')
    df=make_data(n=n,seed=seed)
    development,test=train_test_split(df,test_size=.20,random_state=seed,stratify=df.synthetic_anomaly_label)
    fit_df,cal_df=train_test_split(development,test_size=.25,random_state=seed,stratify=development.synthetic_anomaly_label)
    ARTIFACT_DIR.mkdir(parents=True,exist_ok=True)
    reference={c:{'mean':float(fit_df[c].mean()),'std':float(fit_df[c].std())} for c in FEATURES[:-1]}
    (ARTIFACT_DIR/'reference.json').write_text(json.dumps(reference,indent=2))
    mlflow.set_tracking_uri((Path.cwd()/'mlruns').as_uri())
    mlflow.set_experiment('browser-fingerprint-reconstructed-algorithms')
    results={}
    models={}
    for name in (ALGORITHMS if compare else (algorithm,)):
        model=build_pipeline(name)
        model.fit(fit_df[FEATURES])
        model.calibrate(cal_df[FEATURES])
        score=model.score_samples(test[FEATURES])
        pred=(model.predict(test[FEATURES])==-1).astype(int)
        metrics=evaluate(test.synthetic_anomaly_label.to_numpy(),pred,score)
        results[name]=metrics
        models[name]=model
        with mlflow.start_run(run_name=name):
            mlflow.log_params({'algorithm':name,'seed':seed,'fit_rows':len(fit_df),
                               'calibration_rows':len(cal_df),'test_rows':len(test),
                               'contamination':model.contamination,'dataset':'synthetic',
                               'calibration':'unlabeled holdout quantile'})
            mlflow.log_metrics(metrics)
            mlflow.log_dict(metrics,'metrics.json')
        print(f'{name:19s} F1={metrics["f1"]:.3f} AP={metrics.get("average_precision",0):.3f} FPR={metrics["false_positive_rate"]:.3f}')
    # Choose default model based on held-out synthetic AP ONLY for demo. The
    # ranking on this test set is exploratory, not a valid unbiased winner estimate.
    selected=max(results,key=lambda k:results[k].get('average_precision',0)) if compare else algorithm
    joblib.dump(models[selected],ARTIFACT_DIR/'model.joblib')
    (ARTIFACT_DIR/'selected_model.json').write_text(json.dumps({'algorithm':selected},indent=2))
    (ARTIFACT_DIR/'metrics.json').write_text(json.dumps(results[selected],indent=2))
    (ARTIFACT_DIR/'comparison.json').write_text(json.dumps(results,indent=2))
    pd.DataFrame.from_dict(results,orient='index').rename_axis('algorithm').to_csv(ARTIFACT_DIR/'comparison.csv')
    return results if compare else results[selected]

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--algorithm',choices=ALGORITHMS,default='isolation_forest')
    p.add_argument('--compare',action='store_true',help='Train and compare every algorithm')
    p.add_argument('--n',type=int,default=2000)
    args=p.parse_args()
    train(algorithm=args.algorithm,compare=args.compare,n=args.n)
