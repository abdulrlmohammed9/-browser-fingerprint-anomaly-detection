from functools import lru_cache
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal

app=FastAPI(title='Reconstructed Browser Fingerprint Anomaly API',version='0.2.0')
class Fingerprint(BaseModel):
    screen_width: int=Field(ge=200,le=10000)
    screen_height: int=Field(ge=200,le=10000)
    plugin_count: int=Field(ge=0,le=200)
    canvas_entropy: float=Field(ge=0,le=1)
    hour: int=Field(ge=0,le=23)
    ua_family: Literal['Chrome','Firefox','Safari','Edge','Other']

@lru_cache(maxsize=1)
def get_model():
    path=Path('artifacts/model.joblib')
    if not path.exists():
        raise FileNotFoundError('Train first: python -m src.train --compare')
    return joblib.load(path)  # NEVER load untrusted artifacts

@app.get('/health')
def health():
    return {'status':'ok','model_ready':Path('artifacts/model.joblib').exists()}

@app.post('/predict')
def predict(fp:Fingerprint):
    try:
        model=get_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503,detail=str(exc)) from exc
    x=pd.DataFrame([fp.model_dump()])
    score=float(model.score_samples(x)[0])
    return {'algorithm':model.algorithm,'is_anomaly':bool(model.predict(x)[0]==-1),
            'anomaly_score':score,'threshold':model.threshold_,
            'note':'Scores are algorithm-specific; an anomaly is not proof of bot activity.'}
