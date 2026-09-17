"""Locked external prediction and descriptive evaluation; no model selection."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import joblib
from scipy.special import logit,expit
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.proportion import proportion_confint,confint_proportions_2indep
from .models import metrics,METRICS
from .io import (FROZEN,output,read_json,write_json,digest,save_table,save_private,
                 load_private,checkpoint,require)

def calibration(y,p):
    y=np.asarray(y);p=np.asarray(p);z=logit(np.clip(p,1e-8,1-1e-8))
    fit=sm.GLM(y,sm.add_constant(z),family=sm.families.Binomial()).fit()
    citl=sm.GLM(y,np.ones((len(y),1)),offset=z,family=sm.families.Binomial()).fit()
    if not fit.converged or not citl.converged:raise ValueError('Calibration estimation did not converge.')
    return {'Calibration_intercept':float(fit.params[0]),'Calibration_slope':float(fit.params[1]),
            'CITL':float(citl.params[0]),'Brier':float(np.mean((p-y)**2)),
            'Mean_prediction':float(p.mean()),'Event_prevalence':float(y.mean())}

def calibration_bins(y,p):
    t=pd.DataFrame({'y':y,'p':p,'bin':pd.qcut(p,10,labels=False,duplicates='drop')})
    return pd.DataFrame([{'Bin':int(k)+1,'N':len(g),'Events':int(g.y.sum()),'Mean_predicted':g.p.mean(),'Observed':g.y.mean()} for k,g in t.groupby('bin',observed=True)])

def crossfit(y,p):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float)
    if (p<=0).any() or (p>=1).any():raise ValueError('Cross-fit original probabilities must lie strictly in (0,1).')
    if np.bincount(y,minlength=2).min()<5:raise ValueError('Too few outcomes for five-fold recalibration.')
    z=logit(p);out=np.full(len(p),np.nan);seen=np.zeros(len(p),int);rows=[]
    for i,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(z,y),1):
        f=sm.GLM(y[tr],sm.add_constant(z[tr]),family=sm.families.Binomial()).fit()
        a,b=map(float,f.params)
        if not f.converged or b<=0:raise ValueError('Recalibration unstable or non-positive slope; stop for review.')
        out[te]=expit(a+b*z[te]);seen[te]+=1
        if not np.array_equal(rankdata(p[te]),rankdata(out[te])):raise ValueError('Within-fold ranking changed numerically.')
        rows.append({'Fold':i,'Train_n':len(tr),'Train_events':int(y[tr].sum()),'Heldout_n':len(te),'Heldout_events':int(y[te].sum()),'alpha':a,'beta':b})
    if not (seen==1).all():raise AssertionError('Each participant must be held out exactly once.')
    return out,pd.DataFrame(rows)

def paired_bootstrap(y,p,seed,reps=2000):
    y=np.asarray(y);p=np.asarray(p);rng=np.random.default_rng(seed);b=np.empty((reps,2,3))
    for i in range(reps):
        while True:
            idx=rng.integers(0,len(y),len(y))
            if len(np.unique(y[idx]))==2:break
        for j in range(2):b[i,j]=list(metrics(y[idx],p[idx,j]).values())
    return b

def dca(y,p):
    y=np.asarray(y);p=np.asarray(p);rows=[];n=len(y)
    for t in np.arange(0,1,.001):
        selected=p>=t;tp=int(y[selected].sum());fp=int(selected.sum()-tp)
        rows.append({'threshold':t,'XGBoost':tp/n-fp/n*t/(1-t),'Treat_all':y.mean()-(1-y.mean())*t/(1-t),'Treat_none':0.})
    return pd.DataFrame(rows)

def external(cfg):
    require(cfg,'final');lock=read_json(output(cfg)/'model_lock.json');d=load_private(cfg,'external')
    pred=pd.DataFrame({'D2':d.D2})
    for name,features in [('Baseline',FROZEN['core']),('XGBoost',FROZEN['final'])]:
        f=output(cfg)/'private'/(name+'.joblib')
        if digest(f)!=lock['models'][name]:raise ValueError('Model changed after lock.')
        model=joblib.load(f)
        if list(model.feature_names_in_)!=features:raise ValueError('Predictor order mismatch.')
        pred[name]=model.predict_proba(d[features])[:,1]
        if digest(f)!=lock['models'][name]:raise AssertionError('Locked pipeline changed.')
    save_private(cfg,'external_predictions',pred)
    estimates=[];diffs=[];cal=[];boots=[]
    for co,file in [('development','final_development_oof'),('external','external_predictions')]:
        t=load_private(cfg,file);y=t.D2.to_numpy();p=t[['Baseline','XGBoost']].to_numpy()
        b=paired_bootstrap(y,p,FROZEN['bootstrap_seeds'][co],FROZEN['bootstrap_replicates'])
        for j,name in enumerate(['Baseline','XGBoost']):
            point=metrics(y,p[:,j])
            for k,m in enumerate(METRICS):
                lo,hi=np.quantile(b[:,j,k],[.025,.975])
                estimates.append({'Cohort':co,'Model':name,'N':len(y),'D2_n':int(y.sum()),'Metric':m,'Estimate':point[m],'CI_lower':lo,'CI_upper':hi})
        for k,m in enumerate(METRICS):
            lo,hi=np.quantile(b[:,1,k]-b[:,0,k],[.025,.975])
            diffs.append({'Cohort':co,'Metric':m,'Estimate':metrics(y,p[:,1])[m]-metrics(y,p[:,0])[m],'CI_lower':lo,'CI_upper':hi,'Contrast':'XGBoost minus clinical core'})
        for rep in range(len(b)):
            boots.append({'Cohort':co,'replicate':rep+1,**{name+'_'+m:b[rep,j,k] for j,name in enumerate(['Baseline','XGBoost']) for k,m in enumerate(METRICS)}})
        cal.append({'Cohort':co,'Stage':'Raw',**calibration(y,p[:,1])})
        save_table(cfg,co+'_raw_calibration_bins',calibration_bins(y,p[:,1]))
    cp,pars=crossfit(pred.D2,pred.XGBoost)
    save_private(cfg,'crossfit_probabilities',pd.DataFrame({'D2':pred.D2,'probability':cp}))
    save_table(cfg,'Crossfit_parameters',pars)
    cal.append({'Cohort':'external','Stage':'Cross-fitted local recalibration',**calibration(pred.D2,cp)})
    save_table(cfg,'external_crossfit_calibration_bins',calibration_bins(pred.D2,cp))
    for name,rows in [('Performance_CI',estimates),('Paired_differences',diffs),('Calibration',cal),('Bootstrap_performance',boots)]:save_table(cfg,name,pd.DataFrame(rows))
    save_table(cfg,'DCA_raw_full_range',dca(pred.D2,pred.XGBoost))
    write_json(output(cfg)/'external_receipt.json',{'model_refit':False,'imputer_refit':False,'raw_for_ROC_PR_DCA_targeting':True,
          'crossfit_for_calibration_only':True,'crossfit_global_rank_identical':bool(np.array_equal(rankdata(pred.XGBoost),rankdata(cp)))})
    checkpoint(cfg,'external')

def testing_measures(y,order,counts):
    n=len(y);events=y.sum();cases=np.cumsum(y[order])[counts-1]
    if events==0:raise ValueError('No events in testing-utility sample.')
    yield_=cases/counts
    tests=np.divide(counts,cases,out=np.full(len(counts),np.inf),where=cases!=0)
    return cases,np.column_stack([cases/events,yield_,tests,yield_/(events/n)])

def targeted(cfg):
    require(cfg,'external');allrows=[];bootrows=[];ties=[]
    budget=FROZEN['budgets'];names=['D2_recall','Screening_yield','Tests_per_D2_detected','Enrichment_vs_random']
    for co,file in [('development','final_development_oof'),('external','external_predictions')]:
        d=load_private(cfg,co);t=load_private(cfg,file);y=t.D2.to_numpy();p=t.XGBoost.to_numpy();n=len(y);events=int(y.sum())
        if not np.array_equal(d.D2,y):raise ValueError('Prediction alignment failure.')
        key=d._tie_key.to_numpy(dtype=np.uint64);order=np.lexsort((key,-p));counts=np.array([n*b//100 for b in budget])
        if counts.min()<1:raise ValueError('Cohort too small for fixed testing budgets.')
        cases,point=testing_measures(y,order,counts)
        rank=np.empty(n,int);rank[order]=np.arange(1,n+1)
        save_private(cfg,co+'_selection',pd.DataFrame({'rank':rank,'selected_20pct':rank<=n*20//100}))
        reps=FROZEN['bootstrap_replicates'];boot=np.empty((reps,5,4));rng=np.random.default_rng(FROZEN['bootstrap_seeds'][co])
        for rep in range(reps):
            idx=rng.integers(0,n,n)
            while y[idx].sum()==0:idx=rng.integers(0,n,n)
            oo=np.lexsort((np.arange(n),key[idx],-p[idx]));cc,mm=testing_measures(y[idx],oo,counts);boot[rep]=mm
            for j,b in enumerate(budget):bootrows.append({'Cohort':co,'Replicate':rep+1,'Budget_percent':b,'N_tested':counts[j],'D2_detected':cc[j],**dict(zip(names,mm[j]))})
        if not np.isfinite(boot).all():
            raise ValueError('Infinite tests-per-case bootstrap result (zero detected events); review instead of hiding it.')
        for j,b in enumerate(budget):
            k=counts[j];boundary=p[order[k-1]];eq=p==boundary;above=p>boundary;selected_ties=int(k-above.sum())
            ties.append({'Cohort':co,'Budget_percent':b,'Boundary_tie_n':int(eq.sum()),'Boundary_tie_selected_n':selected_ties,
                         'Cross_boundary_tie':bool(selected_ties<eq.sum())})
            row={'Cohort':co,'Total_N':n,'Total_D2':events,'Budget_percent':b,'Actual_number_tested':k,'Actual_proportion_tested':k/n,
                 'D2_detected':cases[j],'Cohort_prevalence':events/n,'Random_expected_cases':k*events/n,'Random_expected_recall':k/n,
                 'Random_yield':events/n,'Random_tests_per_case':n/events,'Random_enrichment':1.}
            for k2,m in enumerate(names):
                lo,hi=np.quantile(boot[:,j,k2],[.025,.975]);row.update({m:point[j,k2],m+'_CI_lower':lo,m+'_CI_upper':hi})
            allrows.append(row)
    save_table(cfg,'Targeted_testing',pd.DataFrame(allrows));save_table(cfg,'Targeted_bootstrap',pd.DataFrame(bootrows));save_table(cfg,'Targeted_boundary_ties',pd.DataFrame(ties))
    checkpoint(cfg,'targeted')

GROUPS=['D2− / Model−','D2− / Model+','D2+ / Model−','D2+ / Model+']
PROFILE_VARS=['eGFRcys_2012','eGFR_difference','CysC_mg_L','eGFR_ratio','eGFRcr_2021']

def profiles(cfg):
    require(cfg,'targeted');rows=[];binary=[];continuous=[];boots=[]
    for co in ['development','external']:
        d=load_private(cfg,co).copy();selected=load_private(cfg,co+'_selection').selected_20pct
        d['Group_index']=d.D2.astype(int)*2+selected.astype(int)
        # D3 frozen joint indicator becomes NA outside the eligible denominator for prevalence only.
        d.loc[d.D3_denominator==0,'D3']=np.nan
        d['eGFRcys_lt60']=(d.eGFRcys_2012<60).astype(int)
        for i in range(4):
            g=d[d.Group_index==i]
            if len(g)==0:raise ValueError('Empty phenotype group; cannot create the specified four-group comparison.')
            row={'Cohort':co,'Group':GROUPS[i],'Group_index':i,'N':len(g),'Cohort_fraction':len(g)/len(d)}
            for f in ['D2','D1','D3','eGFRcys_lt60']:
                valid=g[f].dropna();n=len(valid);e=int(valid.sum())
                lo,hi=proportion_confint(e,n,method='wilson') if n else (np.nan,np.nan)
                row.update({f+'_n':e,f+'_denominator':n,f+'_prevalence':e/n if n else np.nan,f+'_CI_lower':lo,f+'_CI_upper':hi})
            for f in PROFILE_VARS:
                q=g[f].quantile([.25,.5,.75]);row.update({f+'_median':q.loc[.5],f+'_Q1':q.loc[.25],f+'_Q3':q.loc[.75]})
            rows.append(row)
        pos=d[d.Group_index==1];neg=d[d.Group_index==0]
        for f in ['D1','D3','eGFRcys_lt60']:
            x=pos[f].dropna();y=neg[f].dropna();a=int(x.sum());c=int(y.sum());n1=len(x);n0=len(y)
            if min(n1,n0)==0:raise ValueError('No eligible D3 comparator denominator.')
            lo,hi=confint_proportions_2indep(a,n1,c,n0,method='newcomb',compare='diff')
            rl,rh=confint_proportions_2indep(a,n1,c,n0,method='log',compare='ratio') if a and c else (np.nan,np.nan)
            binary.append({'Cohort':co,'Phenotype':f,'Positive_events':a,'Positive_denominator':n1,'Negative_events':c,'Negative_denominator':n0,
                           'Risk_difference':a/n1-c/n0,'RD_CI_lower':lo,'RD_CI_upper':hi,'Risk_ratio':(a/n1)/(c/n0) if c else np.inf,'RR_CI_lower':rl,'RR_CI_upper':rh})
        x=pos[PROFILE_VARS].to_numpy();y=neg[PROFILE_VARS].to_numpy();rng=np.random.default_rng(FROZEN['bootstrap_seeds'][co])
        delta=np.array([np.median(x[rng.integers(0,len(x),len(x))],axis=0)-np.median(y[rng.integers(0,len(y),len(y))],axis=0) for _ in range(FROZEN['bootstrap_replicates'])])
        for j,f in enumerate(PROFILE_VARS):
            lo,hi=np.quantile(delta[:,j],[.025,.975]);continuous.append({'Cohort':co,'Phenotype':f,'Positive_median':np.median(x[:,j]),'Negative_median':np.median(y[:,j]),'Median_difference':np.median(x[:,j])-np.median(y[:,j]),'CI_lower':lo,'CI_upper':hi})
        for i,vals in enumerate(delta):boots.append({'Cohort':co,'Replicate':i+1,**dict(zip(PROFILE_VARS,vals))})
    for name,t in [('Four_group_profile',rows),('Phenotype_binary_comparisons',binary),('Phenotype_continuous_comparisons',continuous),('Phenotype_bootstrap',boots)]:save_table(cfg,name,pd.DataFrame(t))
    checkpoint(cfg,'profiles')
