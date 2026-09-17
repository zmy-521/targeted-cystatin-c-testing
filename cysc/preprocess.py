"""CKD-EPI equations, field harmonization and frozen cohort/variable governance."""
import hashlib
import itertools
import numpy as np
import pandas as pd
from .io import (FROZEN, DICTIONARY, read_table, resolve_path, save_private,
                 save_table, write_json, output, checkpoint, digest)

CORE = FROZEN['core']
CBC = ['WBC','RBC','HGB','HCT','MCV','MCH','MCHC','RDW','PLT',
       'NEUT#','LYMPH#','MONO#','EO#','BASO#','NEUT%','LYMPH%','MONO%','EO%','BASO%','MPV']
REFERENCE = ['SCr_mg_dL','CysC_mg_L','eGFRcys_2012','eGFR_difference','eGFR_ratio']

def egfr(age, sex, scr, cys):
    age, sex, scr, cys = (np.asarray(x, dtype=float) for x in (age,sex,scr,cys))
    if not (np.isfinite(age).all() and np.isin(sex,[0,1]).all() and
            np.isfinite(scr).all() and np.isfinite(cys).all() and (scr>0).all() and (cys>0).all()):
        raise ValueError('eGFR requires finite age, female=0/male=1, positive SCr and CysC.')
    female = sex == 0
    ratio = scr / np.where(female,.7,.9)
    cr = 142 * np.minimum(ratio,1)**np.where(female,-.241,-.302) * np.maximum(ratio,1)**-1.2 * .9938**age * np.where(female,1.012,1)
    ratio = cys / .8
    cy = 133 * np.minimum(ratio,1)**-.499 * np.maximum(ratio,1)**-1.328 * .996**age * np.where(female,.932,1)
    return cr, cy

def phenotype(d):
    d = d.copy()
    cr, cy = egfr(d.age,d.sex,d.SCr_mg_dL,d.CysC_mg_L)
    d['eGFRcr_2021'],d['eGFRcys_2012'] = cr,cy
    d['eGFR_difference'],d['eGFR_ratio'] = cy-cr,cy/cr
    d['D1'] = (d.eGFR_difference<=-15).astype(int)
    d['D2'] = (d.eGFR_ratio<.70).astype(int)
    d['D3_denominator'] = (cr>=60).astype(int)
    d['D3'] = ((cr>=60)&(cy<60)).astype(int)
    return d

def numeric_map(raw, spec):
    """Explicit column names and unit factors; never infer units from magnitudes."""
    fields = set(FROZEN['initial']) | {'CysC_mg_L'}
    d = pd.DataFrame(index=raw.index)
    mapping = spec.get('columns', {})
    factors = spec.get('factors', {})
    for f in fields:
        if f in ('eGFRcr_2021','A/G'):
            continue
        src = mapping.get(f, f)
        if src in raw:
            values=raw[src]
            value_map=spec.get('value_maps',{}).get(f)
            if value_map is not None:
                translated=values.map(value_map)
                if (values.notna()&translated.isna()).any():raise ValueError('Unmapped category in '+f)
                values=translated
            d[f]=pd.to_numeric(values,errors='raise')
        else:d[f]=np.nan
        d[f] *= factors.get(f,1)
        d[f] /= spec.get('divisors',{}).get(f,1)
    if np.isinf(d.to_numpy(dtype=float)).any():
        raise ValueError('Infinite measurement; resolve source data before running.')
    # Current frozen definition: ALB/GLB, no TP-ALB backfill or non-HDL derivation.
    d['A/G'] = (d.ALB / d.GLB).where(d.GLB>0)
    return d

def _index_input(raw, spec):
    checks = spec.get('attestations',{})
    required = ['one_index_per_patient','adult_diabetes','same_specimen_scr_cysc',
                'frozen_time_window','cbc_window_minus7_to0','source_qc_resolved',
                'frozen_row_order']
    if not all(checks.get(k) is True for k in required):
        raise ValueError('Index-level input requires all source preparation attestations; see DATA_PREPARATION.md.')
    return raw.copy()

def specimen_index(raw, spec, cfg):
    """Optional site-neutral wide-specimen adapter; identifiers/dates remain local only.

    Identity resolution must be completed by the data custodian, not guessed here.
    Each chemistry row is a verified single specimen, with both renal markers.
    Auxiliary panels are linked as whole records; values are never stitched across panels.
    """
    link = spec['linkage']
    if link.get('identity_resolution_verified') is not True or link.get('same_specimen_verified') is not True:
        raise ValueError('Resolve identity conflicts and verify paired renal measurements against specimen sources first.')
    pid, dt, sid = [link[k] for k in ('patient_column','date_column','specimen_column')]
    z = raw.copy()
    z['_day'] = pd.to_datetime(z[dt],errors='raise').dt.normalize()
    if not link.get('period_start') or not link.get('period_end_exclusive'):
        raise ValueError('Supply approved source-period bounds locally.')
    lo, hi = pd.Timestamp(link['period_start']), pd.Timestamp(link['period_end_exclusive'])
    if hi<=lo:raise ValueError('Invalid source-period bounds.')
    m = numeric_map(z,spec)
    eligible = z[link['diabetes_column']].eq(1) & m.age.ge(18) & m.sex.isin([0,1])
    eligible &= m.SCr_mg_dL.gt(0) & m.CysC_mg_L.gt(0) & z[pid].notna() & z[sid].notna()
    eligible &= z['_day'].ge(lo) & z['_day'].lt(hi)
    z['_specimen_sort'] = z[sid].astype(str)
    z = z.loc[eligible].sort_values(['_day','_specimen_sort'],kind='stable').drop_duplicates(pid)
    # The index is selected BEFORE applying the CBC modeling inclusion rule.
    for group, fields in [('cbc',CBC),('hba1c',['HbA1c'])]:
        panel = link.get(group)
        if panel is None:
            if group=='cbc':
                raise ValueError('Specimen mode requires a CBC panel source.')
            continue
        a = read_table(resolve_path(cfg,panel['path']))
        ad = pd.to_datetime(a[panel['date_column']],errors='raise').dt.normalize()
        a = a.assign(_day=ad)
        a = a[a._day.ge(lo)&a._day.lt(hi)]
        dcol = spec.get('columns',{})
        columns = [dcol.get(f,f) for f in fields if dcol.get(f,f) in a]
        a = a[a[columns].notna().any(axis=1)]
        invalid = panel.get('invalid_record_column')
        if invalid:
            if a[invalid].isna().any():raise ValueError('Unresolved auxiliary-panel QC.')
            a = a[~a[invalid].astype(bool)]
        # Zero/extreme values require source adjudication, never automatic plausible replacement.
        if group=='cbc':
            for f in ['WBC','HGB','PLT']:
                col=dcol.get(f,f)
                if col in a and (pd.to_numeric(a[col],errors='raise').dropna()<=0).any():
                    raise ValueError('CBC sentinel-zero found; trace source, mark confirmed invalid panels, and retry.')
        a['_specimen_sort'] = a[panel['specimen_column']].astype(str)
        for idx,row in z.iterrows():
            matches=a[a[panel['patient_column']].eq(row[pid]) & a._day.le(row._day) & a._day.ge(row._day-pd.Timedelta(days=7))]
            matches=matches.sort_values(['_day','_specimen_sort'],ascending=[False,True],kind='stable')
            for col in columns:z.loc[idx,col]=matches.iloc[0][col] if len(matches) else np.nan
    # Reproduce custodian-supplied frozen order; do not sort on outcomes or predictions.
    order=link['frozen_order_column']
    if z[order].isna().any() or z[order].duplicated().any():raise ValueError('A unique frozen order is required.')
    return z.sort_values(order,kind='stable').reset_index(drop=True)

def prepare_cohort(cfg, cohort):
    spec=cfg['cohorts'][cohort]
    raw=read_table(resolve_path(cfg,spec['path']))
    raw=_index_input(raw,spec) if spec['input_mode']=='index' else specimen_index(raw,spec,cfg)
    raw=raw.reset_index(drop=True)
    d=numeric_map(raw,spec)
    if not (d.age.ge(18)&d.sex.isin([0,1])&d.SCr_mg_dL.gt(0)&d.CysC_mg_L.gt(0)).all():
        raise ValueError('Input violates adult/sex/positive renal-measurement eligibility.')
    invalid_col=spec.get('invalid_cbc_column')
    if invalid_col:
        if raw[invalid_col].isna().any() or not raw[invalid_col].isin([0,1,False,True]).all():
            raise ValueError('Invalid-CBC flag must be explicitly adjudicated 0/1.')
        d.loc[raw[invalid_col].astype(bool),CBC]=np.nan
    for f in ['WBC','RBC','HGB','HCT','MCV','MCH','MCHC','PLT','MPV']:
        if (d[f].dropna()<=0).any():
            raise ValueError('Unresolved physiologic/sentinel-zero CBC measurement. Trace source before imputation.')
    d=phenotype(d)
    # Frozen basic-CBC rule, not a raw missing-proportion filter.
    mask=d[['WBC','HGB','PLT']].notna().all(axis=1)
    tie_col=spec.get('tie_key_column')
    if tie_col:
        keys=raw[tie_col].map(lambda s:int(str(s))).to_numpy(dtype=np.uint64)
    elif spec.get('private_study_key_column'):
        names={'development':'Wanbei','external':'Anyi'}
        keys=np.array([int(hashlib.sha256(f'42|{names[cohort]}|{v}'.encode()).hexdigest()[:16],16)
                       for v in raw[spec['private_study_key_column']]],dtype=np.uint64)
    elif cfg['mode']=='synthetic_test':
        keys=np.random.default_rng(42).permutation(len(d)).astype(np.uint64)
    else:
        raise ValueError('Exact rank-boundary replay requires a local precomputed tie key or private study key.')
    if len(np.unique(keys))!=len(keys):raise ValueError('Tie keys must be unique before resampling.')
    d['_tie_key']=keys
    g=d.loc[mask].reset_index(drop=True)
    if cfg['mode']=='paper_reproduction':
        expected=FROZEN['expected'][cohort]
        if (len(g),int(g.D2.sum()))!=(expected['n'],expected['events']):
            raise ValueError('Cohort fingerprint differs from frozen manuscript; do not change values to force a match.')
    save_private(cfg,cohort,g)
    return g,{'Cohort':cohort,'phenotype_n':len(d),'excluded_basic_CBC_n':int((~mask).sum()),'modeling_n':len(g),'D2_n':int(g.D2.sum())}

def run(cfg):
    data={};flow=[]
    for co in ['development','external']:
        data[co],row=prepare_cohort(cfg,co);flow.append(row)
    save_table(cfg,'COHORT_FLOW',pd.DataFrame(flow))
    rows=[];passing=[]
    for meta in DICTIONARY:
        f=meta['name'];r=dict(meta)
        for co,d in data.items():
            missing=int(d[f].isna().sum())
            r.update({co+'_available_n':len(d)-missing,co+'_missing_n':missing,co+'_missing_fraction':missing/len(d)})
        eligible=all(r[c+'_missing_fraction']<=.25 for c in data)
        if eligible:passing.append(f)
        r['recomputed_missingness_eligible']=eligible
        r['Final_Full_pool']=f in FROZEN['full'];rows.append(r)
    pre=[f for f in passing if f not in ['SCr_mg_dL','CO2']]
    if cfg['mode']=='paper_reproduction' and pre!=FROZEN['pre_redundancy']:
        raise ValueError('49 -> 41 -> 40 -> 39 governance changed; manual review required, no new pool chosen.')
    save_table(cfg,'Table_S1_variable_eligibility',pd.DataFrame(rows))
    dev=data['development'];cols=FROZEN['pre_redundancy']
    corr=dev[cols].corr(method='spearman')
    pair_n=dev[cols].notna().astype(int).T.dot(dev[cols].notna().astype(int))
    save_table(cfg,'S1_correlation_matrix',corr.reset_index(names='feature'))
    pairs=[{'feature_1':a,'feature_2':b,'rho':corr.loc[a,b],'pairwise_n':pair_n.loc[a,b]} for a,b in itertools.combinations(cols,2)]
    save_table(cfg,'correlation_pairs_all',pd.DataFrame(pairs))
    for t in [.7,.8]:save_table(cfg,f'correlation_pairs_{t}',pd.DataFrame([r for r in pairs if abs(r['rho'])>=t]))
    # This is evidence only; fixed clinical exclusions are never reselected by correlations.
    write_json(output(cfg)/'governance.json',{'recomputed_chain':[49,len(passing),len([f for f in passing if f!='SCr_mg_dL']),len(pre)],'frozen_full':FROZEN['full'],'frozen_pre_redundancy':cols,'automatic_pruning':False})
    write_json(output(cfg)/'cohort_receipt.json',{co:digest(output(cfg)/'private'/(co+'.parquet')) for co in data})
    checkpoint(cfg,'prepare')
