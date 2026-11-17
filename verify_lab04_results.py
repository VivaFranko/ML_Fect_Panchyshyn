"""Незалежне звіряння збережених моделей, тестових матриць і журналу Optuna."""
from pathlib import Path
import sys,json,sqlite3
import numpy as np
import cloudpickle
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'project'))
from lab04 import read_annealing
out=ROOT/'lab_04/results';r=json.loads((out/'results.json').read_text(encoding='utf-8'))
raw=read_annealing();indices=np.load(out/'split_indices.npz');x=raw.drop(columns='class').iloc[indices['test']]
y=np.array([r['classes'].index(str(v)) for v in raw['class'].iloc[indices['test']]])
checked={}
for name in ['DT','RF','XGB']:
    with (out/(name+'.pkl')).open('rb') as f:model=cloudpickle.load(f)
    p=model.predict_proba(x);pred=p.argmax(1);cm=np.zeros((len(r['classes']),len(r['classes'])),dtype=int)
    np.add.at(cm,(y,pred),1)
    assert cm.tolist()==r['test'][name]['confusion_matrix']
    diag=np.diag(cm);precision=np.divide(diag,cm.sum(0),out=np.zeros_like(diag,dtype=float),where=cm.sum(0)>0);recall=diag/cm.sum(1)
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros_like(precision),where=precision+recall>0)
    metrics={'accuracy':float(diag.sum()/cm.sum()),'macro_f1':float(f1.mean()),'balanced_accuracy':float(recall.mean())}
    assert all(np.isclose(value,r['test'][name][key],rtol=1e-12) for key,value in metrics.items())
    checked[name]=metrics
db=sqlite3.connect(out/'optuna.sqlite3')
studies=db.execute('SELECT count(*) FROM studies').fetchone()[0];complete=db.execute("SELECT count(*) FROM trials WHERE state='COMPLETE'").fetchone()[0]
assert studies==18 and complete==540,(studies,complete)
assert all(len(item['test_f1m']['folds'])==15 for item in r['baseline'].values())
assert r['seed']==62 and r['number']==20
checks={'number':20,'seed':62,'studies':studies,'completed_trials':complete,'test_rows':len(y),'independently_recomputed_metrics':checked,'shap_additivity_errors':{k:v['additivity_error'] for k,v in r['local_explanation']['models'].items()}}
(ROOT/'analysis/lab04_independent_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(checks,ensure_ascii=False,indent=2))
