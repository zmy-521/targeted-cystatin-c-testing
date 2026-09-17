"""Replay frozen development settings. There is deliberately no hyperparameter search."""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap
import joblib
from .io import (FROZEN,output,read_json,write_json,save_table,save_private,load_private,
                 save_arrays,checkpoint,require,digest)

NAMES=['LR','SVM','KNN','DT','RF','XGBoost','LightGBM']
METRICS=['AUROC','AUPRC','Brier']

def metrics(y,p):
    if set(np.unique(y))!={0,1} or not np.isfinite(p).all():
        raise ValueError('Metrics require both classes and finite probabilities.')
    return dict(zip(METRICS,map(float,[roc_auc_score(y,p),average_precision_score(y,p),brier_score_loss(y,p)])))

def pipeline(name):
    cls={'Baseline':LogisticRegression,'LR':LogisticRegression,'KNN':KNeighborsClassifier,
         'SVM':SVC,'DT':DecisionTreeClassifier,'RF':RandomForestClassifier,
         'XGBoost':XGBClassifier,'LightGBM':LGBMClassifier}[name]
    steps=[('imputer',SimpleImputer(strategy='median',keep_empty_features=True))]
    if name in ['Baseline','LR','KNN','SVM']:steps.append(('scaler',StandardScaler()))
    steps.append(('clf',cls(**FROZEN['model_params'][name])))
    return Pipeline(steps)

def oof(name,X,y,seed=42):
    y=np.asarray(y,dtype=int)
    if np.bincount(y,minlength=2).min()<5:raise ValueError('Too few events/non-events for stratified five-fold CV.')
    p=np.full(len(y),np.nan);fold=np.zeros(len(y),int)
    for k,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(X,y),1):
        m=pipeline(name).fit(X.iloc[tr],y[tr]);p[te]=m.predict_proba(X.iloc[te])[:,1];fold[te]=k
        expected=X.iloc[tr].median().fillna(0).to_numpy()
        if not np.allclose(expected,m.named_steps['imputer'].statistics_,equal_nan=True):raise AssertionError('Training-only imputer check failed.')
        if 'scaler' in m.named_steps:
            train=m.named_steps['imputer'].transform(X.iloc[tr])
            if not np.allclose(train.mean(axis=0),m.named_steps['scaler'].mean_):raise AssertionError('Training-only scaler check failed.')
    if not np.isfinite(p).all():raise AssertionError('Incomplete OOF prediction.')
    return p,fold

def shap_values(model,X,name):
    xt=np.asarray(model[:-1].transform(X));clf=model.named_steps['clf']
    explainer=shap.TreeExplainer(clf,model_output='raw',feature_perturbation='tree_path_dependent')
    values=explainer.shap_values(xt,check_additivity=True)
    if isinstance(values,list):values=values[1]
    values=np.asarray(values)
    if values.ndim==3:values=values[:,:,1]
    base=float(np.asarray(explainer.expected_value).ravel()[-1])
    expected=clf.predict(xt,output_margin=True) if name=='XGBoost' else model.predict_proba(X)[:,1]
    err=float(np.max(np.abs(values.sum(axis=1)+base-expected)))
    if err>=1e-4:raise ValueError('SHAP additivity check failed.')
    return values,xt,base,err

def near_optimal(t):
    ref=t.sort_values(['AUPRC','Feature_Count'],ascending=[False,False],kind='stable').iloc[0]
    ok=(t.AUROC>=ref.AUROC-.01)&(t.AUPRC>=ref.AUPRC-.01)&(t.Brier<=ref.Brier+.01)
    row=t.loc[ok].sort_values('Feature_Count').iloc[0].to_dict()
    row['reference_k']=int(ref.Feature_Count)
    return row

def compare_candidates(a,b):
    # Same Decimal-based hierarchy as the frozen selection module.
    from decimal import Decimal
    difference=lambda x,y:Decimal(str(x))-Decimal(str(y))
    tol=Decimal('.01');ap=difference(a['AUPRC'],b['AUPRC']);au=difference(a['AUROC'],b['AUROC']);br=difference(a['Brier'],b['Brier'])
    if abs(ap)>=tol:return (a if ap>0 else b)['Model'],'AUPRC'
    if abs(au)<tol and abs(br)<tol:
        if a['Feature_Count']!=b['Feature_Count']:return min((a,b),key=lambda x:x['Feature_Count'])['Model'],'parsimony'
        return None,'indistinguishable'
    if au>=-tol and br<0:return a['Model'],'AUROC/Brier'
    if au<=tol and br>0:return b['Model'],'AUROC/Brier'
    return None,'unresolved tradeoff'

def development(cfg):
    require(cfg,'prepare')
    d=load_private(cfg,'development');y=d.D2.to_numpy();full=FROZEN['full']
    rows=[];pred=pd.DataFrame({'D2':y})
    for name in ['Baseline']+NAMES:
        features=FROZEN['core'] if name=='Baseline' else full
        p,f=oof(name,d[features],y);pred[name]=p;pred['fold']=f
        rows.append({'Model':name,'Feature_Count':len(features),**metrics(y,p)})
        print('Fixed-parameter CV:',name,flush=True)
    save_table(cfg,'Full_model_comparison',pd.DataFrame(rows))
    save_private(cfg,'full_oof',pred)
    ranked=pd.DataFrame(rows[1:]).sort_values(['AUPRC','AUROC','Brier'],ascending=[False,False,True],kind='stable')
    observed=ranked.Model.iloc[:2].tolist()
    write_json(output(cfg)/'candidate_check.json',{'observed_top2':observed,'frozen_candidates':FROZEN['candidates']})
    if cfg['mode']=='paper_reproduction' and set(observed)!=set(FROZEN['candidates']):
        raise ValueError('Candidate algorithms do not match frozen development. Investigate inputs/environment, not external outcomes.')
    checkpoint(cfg,'development')

def reduction(cfg):
    require(cfg,'development');d=load_private(cfg,'development');full=FROZEN['full'];core=FROZEN['core'];y=d.D2.to_numpy()
    rows=[];candidates=[]
    for name in FROZEN['candidates']:
        model=pipeline(name).fit(d[full],y)
        sv,xt,base,err=shap_values(model,d[full],name)
        order=np.argsort(-np.abs(sv).mean(axis=0),kind='stable');ranking=[full[j] for j in order]
        save_arrays(cfg,name+'_full_shap',values=sv,inputs=xt,features=np.array(full),base=np.array(base))
        save_table(cfg,name+'_full_SHAP_ranking',pd.DataFrame({'feature':ranking,'mean_abs_shap':np.abs(sv).mean(axis=0)[order]}))
        extra=[f for f in ranking if f not in core];pred=pd.DataFrame({'D2':y});local=[]
        for k in range(4,29):
            chosen=set(core+extra[:k-3]);features=[f for f in full if f in chosen]
            p,f=oof(name,d[features],y);pred[f'k_{k}']=p;pred['fold']=f
            row={'Model':name,'Feature_Count':k,**metrics(y,p),'features':' | '.join(features)}
            rows.append(row);local.append(row)
        candidates.append(near_optimal(pd.DataFrame(local)))
        save_private(cfg,name+'_reduction_oof',pred)
        print('Complete SHAP and k=4..28 reduction:',name,flush=True)
    save_table(cfg,'Feature_reduction',pd.DataFrame(rows))
    winner,basis=compare_candidates(*candidates)
    write_json(output(cfg)/'reduction_candidates.json',{'candidates':candidates,'provisional_champion':winner,'basis':basis,'final_panel_reselected':False})
    # Stability, not mechanical minimum k, supported the already approved nine-variable panel.
    if cfg['mode']=='paper_reproduction':
        x=pd.DataFrame(rows).query("Model == 'XGBoost' and Feature_Count == 9").iloc[0]
        if winner!='XGBoost' or x.features.split(' | ')!=FROZEN['final']:
            raise ValueError('Frozen champion/panel mismatch; no automatic replacement allowed.')
    checkpoint(cfg,'reduction')

def stability(cfg):
    require(cfg,'reduction');d=load_private(cfg,'development');rows=[]
    for repeat,seed in enumerate(FROZEN['repeat_seeds'],1):
        row={'repeat':repeat,'split_seed':seed}
        for k in [8,9]:
            features=[f for f in FROZEN['final'] if k==9 or f!='ALP']
            p,fold=oof('XGBoost',d[features],d.D2,seed=seed)
            for m,v in metrics(d.D2,p).items():row[f'k{k}_{m}']=v
        for m in METRICS:row['delta_'+m]=row['k9_'+m]-row['k8_'+m]
        rows.append(row);print('Panel stability repeat',repeat,flush=True)
    t=pd.DataFrame(rows);summary=[]
    for group in ['k8','k9','delta']:
        for m in METRICS:
            v=t[f'{group}_{m}'];q=v.quantile([.25,.75])
            summary.append({'Group':group,'Metric':m,'Mean':v.mean(),'SD':v.std(ddof=1),'Median':v.median(),'Q1':q.loc[.25],'Q3':q.loc[.75],'IQR':q.loc[.75]-q.loc[.25],'Min':v.min(),'Max':v.max()})
    save_table(cfg,'Panel_stability_repeats',t);save_table(cfg,'Panel_stability_summary',pd.DataFrame(summary))
    write_json(output(cfg)/'panel_stability.json',{'k9_better_repeats':{m:int((t['delta_'+m]<0 if m=='Brier' else t['delta_'+m]>0).sum()) for m in METRICS},'final_panel':FROZEN['final'],'selection':'Previously approved; this replay does not change the panel.'})
    checkpoint(cfg,'stability')

def final(cfg):
    require(cfg,'stability');d=load_private(cfg,'development');pdir=output(cfg)/'private';pred=pd.DataFrame({'D2':d.D2})
    for name,features in [('XGBoost',FROZEN['final']),('Baseline',FROZEN['core'])]:
        p,f=oof(name,d[features],d.D2);pred[name]=p;pred['fold']=f
        model=pipeline(name).fit(d[features],d.D2)
        joblib.dump(model,pdir/(name+'.joblib'))
    save_private(cfg,'final_development_oof',pred)
    write_json(output(cfg)/'model_lock.json',{'version':FROZEN['model_version'],'features':FROZEN['final'],'threshold':FROZEN['threshold'],
               'params':FROZEN['model_params']['XGBoost'],'models':{n:digest(pdir/(n+'.joblib')) for n in ['XGBoost','Baseline']},'external_used_for_development':False})
    checkpoint(cfg,'final')

def global_shap(cfg):
    require(cfg,'final');d=load_private(cfg,'development');features=FROZEN['final'];path=output(cfg)/'private/XGBoost.joblib'
    if digest(path)!=read_json(output(cfg)/'model_lock.json')['models']['XGBoost']:raise ValueError('Model changed after lock.')
    model=joblib.load(path)
    sv,xt,base,error=shap_values(model,d[features],'XGBoost')
    # Native XGBoost contribution cross-check; no retraining or outcome-guided selection.
    from xgboost import DMatrix
    native=model.named_steps['clf'].get_booster().predict(DMatrix(xt),pred_contribs=True)
    if not np.allclose(native[:,:9],sv,atol=1e-5):raise AssertionError('Native SHAP mismatch.')
    save_arrays(cfg,'final_shap',values=sv,inputs=xt,features=np.array(features),base=np.array(base))
    rank=pd.DataFrame({'feature':features,'mean_abs_SHAP':np.abs(sv).mean(axis=0)}).sort_values('mean_abs_SHAP',ascending=False)
    save_table(cfg,'Final_SHAP_ranking',rank)
    write_json(output(cfg)/'shap_qc.json',{'scale':'raw margin / log-odds','max_additivity_error':error,'native_contributions_match':True})
    checkpoint(cfg,'shap')
