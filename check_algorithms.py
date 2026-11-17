"""Технічні перевірки реалізацій; не результати індивідуального варіанта №4."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'project'))
import common
import numpy as np
from lab02 import SoftmaxRegression,Client,Aggregator
from lab04 import read_annealing,pipeline,nadeau_bengio
from lab05 import MLP
from scipy import stats
import shap

# Стабільний softmax навіть для великих логітів.
m=SoftmaxRegression(2,3);m.w[:]=[[10000,-10000,5000],[2,3,4]];p=m.predict_proba(np.array([[1.,1.],[-1.,3.]]));assert np.isfinite(p).all() and np.allclose(p.sum(1),1)
# Транспорт FedAvg і зважування перевірені на різних розмірах клієнтів.
clients=[Client(np.zeros((2,2)),np.zeros(2,dtype=int),1),Client(np.zeros((6,2)),np.ones(6,dtype=int),2)]
server=Aggregator(2,3,np.zeros((3,2)),np.zeros(3,dtype=int),0)
states=[c.update(server.model.state(),1,.1,0) for c in clients]
expected=tuple((states[0][1][i]*2+states[1][1][i]*6)/8 for i in range(2))
for client in clients:client.rng=np.random.default_rng(1)
server.round(clients,1,.1,0);assert all(np.allclose(a,b) for a,b in zip(server.model.state(),expected))
assert set(server.__dict__)=={'model','xv','yv'}

# SHAP і пайплайн проходять smoke-test без індивідуального N.
raw=read_annealing();x=raw.drop(columns='class')
from sklearn.preprocessing import LabelEncoder
y=LabelEncoder().fit_transform(raw['class'])
shap_checks={}
for name in ['DT','RF','XGB']:
    params={'n_estimators':10} if name!='DT' else {}
    pipe=pipeline(name,123,True,params).fit(x,y);z=pipe[:-1].transform(x.iloc[[1]])
    names=pipe[:-1].get_feature_names_out();assert len(names)==z.shape[1]
    exp=shap.TreeExplainer(pipe[-1]);s=exp.shap_values(z);assert np.asarray(s).shape==(1,len(names),5)
    col=np.argmax(pipe.predict_proba(x.iloc[[1]])[0]);v=float(np.asarray(s)[0,:,col].sum()+exp.expected_value[col]);expected=float(pipe[-1].predict(z,output_margin=True)[0,col]) if name=='XGB' else float(pipe[-1].predict_proba(z)[0,col]);assert abs(v-expected)<1e-4
    shap_checks[name]=float(abs(v-expected))
fixture=nadeau_bengio([.8,.9,.7,.85,.84],[.7,.9,.8,.7,.81]);diff=np.array([.1,0,-.1,.15,.03]);expected=diff.mean()/np.sqrt((.2+.25)*np.var(diff,ddof=1));assert np.isclose(expected,fixture['t'])

# Відновлення моделі та inference без Dropout.
ann=MLP((4,5,4,3),dropout=.5);data=np.random.default_rng(42).normal(size=(12,4)).astype(np.float32)
assert np.array_equal(ann.predict_proba(data),ann.predict_proba(data))

checks={'stable_softmax':True,'weighted_fedavg':True,'server_has_no_client_data':True,'tree_pipelines_and_shap':shap_checks,'nadeau_bengio':True,'dropout_disabled_inference':True}
(ROOT/'analysis'/'algorithm_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print(json.dumps(checks,indent=2))
