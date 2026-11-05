"""Separate API for the FP-Stalker feature schema. Serve only trusted artifacts."""
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.fpstalker import FEATURES
from src.model import ALGORITHMS

app = FastAPI(title='FP-Stalker Unlabeled Anomaly API', version='1.0')

class Fingerprint(BaseModel):
    screen_width: int | None = Field(default=None,ge=1,le=20000)
    screen_height: int | None = Field(default=None,ge=1,le=20000)
    timezone_offset: float | None = Field(default=None,ge=-1440,le=1440)
    plugin_count: int | None = Field(default=None,ge=0,le=1000)
    platform: str = Field(default='Missing',max_length=80)
    cookies: str = Field(default='Missing',max_length=80)
    local_storage: str = Field(default='Missing',max_length=80)
    session_storage: str = Field(default='Missing',max_length=80)
    adblock: str = Field(default='Missing',max_length=80)
    browser_family: Literal['Chrome','Firefox','Safari','Edge','Opera','Other'] = 'Other'

@lru_cache(maxsize=1)
def get_model():
    algorithm=os.getenv('FPSTALKER_ALGORITHM','isolation_forest')
    if algorithm not in ALGORITHMS: raise ValueError('Invalid FPSTALKER_ALGORITHM')
    return joblib.load(Path('artifacts/fpstalker')/f'{algorithm}.joblib')

@app.get('/health')
def health():
    return {'status':'ok','dataset':'fpstalker','algorithm':os.getenv('FPSTALKER_ALGORITHM','isolation_forest')}

@app.post('/predict')
def predict(fp: Fingerprint):
    try: model=get_model()
    except FileNotFoundError as exc:
        raise HTTPException(503,'Train FP-Stalker first; model artifact is missing') from exc
    data=pd.DataFrame([fp.model_dump()])[FEATURES]
    # Numeric None values must be converted to NaN for the fitted median imputer.
    from src.fpstalker import NUMERIC
    for k in NUMERIC: data[k]=pd.to_numeric(data[k],errors='coerce')
    return {'algorithm':model.algorithm,'is_anomaly':bool(model.predict(data)[0]==-1),
            'anomaly_score':float(model.score_samples(data)[0]),'threshold':model.threshold_,
            'note':'Unlabeled real-data anomaly score; not a verified bot classification.'}
