import json
import sys
from pathlib import Path
import pandas as pd
from src.features import FEATURES

def monitor(path):
    reference=json.loads(Path('artifacts/reference.json').read_text())
    recent=pd.read_csv(path)
    findings={}
    for col in FEATURES[:-1]:
        if col not in recent.columns: raise ValueError(f'Missing column: {col}')
        baseline=reference[col]
        shift=abs(float(recent[col].mean())-baseline['mean'])/max(baseline['std'],1e-8)
        findings[col]={'mean_shift_in_baseline_sd':round(shift,3),'flag':shift>1.0}
    Path('artifacts/drift_report.json').write_text(json.dumps(findings,indent=2))
    print(json.dumps(findings,indent=2))
    return findings

if __name__=='__main__':
    if len(sys.argv)!=2: raise SystemExit('Usage: python -m src.monitor path/to/recent.csv')
    monitor(sys.argv[1])
