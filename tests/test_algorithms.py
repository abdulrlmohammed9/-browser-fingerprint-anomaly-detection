import numpy as np
import pytest
from src.features import FEATURES
from src.generate_data import make_data
from src.model import ALGORITHMS, build_pipeline
from src.train import train

@pytest.mark.parametrize('name',ALGORITHMS)
def test_each_algorithm_supports_out_of_sample_prediction(name):
    x=make_data(n=170,seed=19)
    m=build_pipeline(name).fit(x.iloc[:120][FEATURES]).calibrate(x.iloc[120:145][FEATURES])
    scores=m.score_samples(x.iloc[145:][FEATURES])
    assert len(scores)==25
    assert np.all(np.isfinite(scores))
    assert set(np.unique(m.predict(x.iloc[145:][FEATURES]))).issubset({-1,1})
    assert m.threshold_==pytest.approx(m.threshold_)

def test_all_algorithms_export_results(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    results=train(compare=True,n=300)
    assert set(results)==set(ALGORITHMS)
    assert (tmp_path/'artifacts/comparison.csv').exists()
    assert (tmp_path/'artifacts/model.joblib').exists()
