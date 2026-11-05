from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from src.fpstalker import tuples, read_sql, prepare, chronological_split, NUMERIC, CATEGORICAL, FEATURES
from src.model import AnomalyModel, ALGORITHMS


def test_sql_literals_and_rejection():
    assert list(tuples("(1,'a,b (c)',NULL,'can\\'t','it''s','line\\nnext');")) == [['1','a,b (c)',None,"can't","it's",'line\nnext']]
    with pytest.raises(ValueError): list(tuples("(1,'unfinished"))
    with pytest.raises(ValueError): list(tuples('(1, evil());'))


def raw_fixture(n=100):
    return pd.DataFrame({'id':[str(i%5) for i in range(n)],
        'creationDate':pd.date_range('2020-01-01',periods=n,freq='h').astype(str),
        'resolutionJS':['1920x1080x24']*n,'pluginsJS':['Plugin 0: PDF; Plugin 1: Flash']*n,
        'timezoneJS':['-60']*n,'platformJS':['Win32']*n,'cookiesJS':['yes']*n,
        'localJS':['yes']*n,'sessionJS':['yes']*n,'adBlock':['no']*n,'userAgentHttp':['Firefox/100']*n})


def write_dump(path,raw):
    names=list(raw.columns)
    def quote(v): return "'"+str(v).replace('\\','\\\\').replace("'","\\'")+"'"
    text='CREATE TABLE `extensionDataScheme` (\n'+',\n'.join('`'+k+'` text' for k in names)+'\n);\n'
    text+='INSERT INTO `extensionDataScheme` ('+','.join('`'+k+'`' for k in names)+') VALUES\n'
    text+=',\n'.join('('+','.join(quote(v) for v in row)+')' for row in raw.itertuples(index=False,name=None))+';\n'
    path.write_text(text)


def test_loader_features_and_time_boundaries(tmp_path):
    raw=raw_fixture()
    raw['addressHttp']='should-never-be-retained'
    path=tmp_path/'sample.sql';write_dump(path,raw)
    loaded=read_sql(path)
    assert 'addressHttp' not in loaded
    frame=prepare(loaded)
    assert frame.screen_width.eq(1920).all()
    assert frame.plugin_count.eq(2).all()
    assert frame.browser_family.eq('Firefox').all()
    fit,cal,test=chronological_split(frame)
    assert len(fit)==60 and len(cal)==20 and len(test)==20
    assert fit._time.max()<cal._time.min()<test._time.min()
    assert '_browser' not in FEATURES
    loaded.loc[0,'pluginsJS']='unrecognized'
    assert np.isnan(prepare(loaded).loc[0,'plugin_count'])


def test_real_training_and_api(tmp_path,monkeypatch):
    from src.train_fpstalker import train
    from src.api_fpstalker import app,get_model
    monkeypatch.chdir(tmp_path)
    path=tmp_path/'sample.sql';write_dump(path,raw_fixture())
    # Existing synthetic artifacts must survive the real-data run.
    Path('artifacts').mkdir();Path('artifacts/model.joblib').write_text('sentinel')
    results=train(path,max_fit=40)
    assert set(results)==set(ALGORITHMS)
    assert Path('artifacts/model.joblib').read_text()=='sentinel'
    assert all('f1' not in metrics for metrics in results.values())
    get_model.cache_clear()
    client=TestClient(app)
    response=client.post('/predict',json={'screen_width':1920,'screen_height':1080,'plugin_count':2,'timezone_offset':-60,'platform':'Win32','cookies':'yes','local_storage':'yes','session_storage':'yes','adblock':'no','browser_family':'Firefox'})
    assert response.status_code==200
    assert isinstance(response.json()['is_anomaly'],bool)
    assert client.post('/predict',json={'screen_width':-1}).status_code==422
    get_model.cache_clear()
