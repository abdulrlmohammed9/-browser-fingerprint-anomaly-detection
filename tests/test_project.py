from fastapi.testclient import TestClient
from src.api import app, get_model
from src.train import train
from src.monitor import monitor
from src.generate_data import make_data

def test_training_and_prediction(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    result=train()
    assert 0 <= result['f1'] <= 1
    get_model.cache_clear()
    client=TestClient(app)
    assert client.get('/health').json()['model_ready']
    res=client.post('/predict',json={'screen_width':1920,'screen_height':1080,
      'plugin_count':5,'canvas_entropy':.5,'hour':12,'ua_family':'Chrome'})
    assert res.status_code==200
    assert isinstance(res.json()['is_anomaly'],bool)
    assert client.post('/predict',json={'screen_width':-1}).status_code==422
    path=tmp_path/'recent.csv'
    make_data(100,seed=11).to_csv(path,index=False)
    assert 'screen_width' in monitor(path)
