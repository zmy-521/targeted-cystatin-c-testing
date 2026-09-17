"""Deterministic tests with invented arrays only; no clinical fixture is distributed."""
import unittest
import numpy as np
import pandas as pd
from cysc.io import FROZEN,outside_repo,ROOT
from cysc.preprocess import egfr,phenotype,numeric_map,specimen_index
from cysc.models import pipeline,oof,near_optimal
from cysc.evaluation import dca,testing_measures,crossfit

class Contract(unittest.TestCase):
    def test_frozen_pools(self):
        self.assertEqual([len(FROZEN[k]) for k in ['initial','pre_redundancy','full','final']],[49,39,28,9])
        self.assertEqual(FROZEN['final'],['age','sex','eGFRcr_2021','HGB','PLT','BUN','ALB','ALP','GLU'])
        self.assertEqual(FROZEN['repeat_seeds'],list(range(42,52)))
        self.assertNotIn('CysC_mg_L',FROZEN['full'])

    def test_equations_and_boundaries(self):
        cr,cy=egfr([50,50],[0,1],[.7,.9],[.8,.8])
        np.testing.assert_allclose(cr,[142*.9938**50*1.012,142*.9938**50])
        np.testing.assert_allclose(cy,[133*.996**50*.932,133*.996**50])
        d=phenotype(pd.DataFrame({'age':[60,40],'sex':[0,1],'SCr_mg_dL':[1,1],'CysC_mg_L':[1.8,.9]}))
        np.testing.assert_array_equal(d.D2,(d.eGFRcys_2012/d.eGFRcr_2021<.7).astype(int))
        np.testing.assert_array_equal(d.D3,((d.eGFRcr_2021>=60)&(d.eGFRcys_2012<60)).astype(int))
        with self.assertRaises(ValueError):egfr([50],[1],[0],[1])

    def test_training_fold_imputation(self):
        rng=np.random.default_rng(1);x=pd.DataFrame(rng.normal(size=(120,3)),columns=['a','b','c']);x.loc[::4,'b']=np.nan
        y=np.tile([0,1],60);p,f=oof('Baseline',x,y)
        self.assertTrue(np.isfinite(p).all());self.assertEqual(set(f),set(range(1,6)))
        self.assertEqual(pipeline('XGBoost').named_steps['clf'].get_params()['random_state'],42)

    def test_dca_and_budget(self):
        y=np.array([0,1,1,0]);p=np.array([.1,.8,.6,.2]);t=dca(y,p)
        self.assertAlmostEqual(t.loc[np.isclose(t.threshold,.5),'XGBoost'].iloc[0],.5)
        cases,m=testing_measures(y,np.argsort(-p),np.array([1,2]))
        np.testing.assert_array_equal(cases,[1,2]);np.testing.assert_allclose(m[:,0],[.5,1])

    def test_near_optimal(self):
        t=pd.DataFrame({'Model':['XGBoost']*3,'Feature_Count':[8,9,10],'AUROC':[.86,.865,.864],'AUPRC':[.60,.606,.605],'Brier':[.10,.101,.102]})
        self.assertEqual(near_optimal(t)['Feature_Count'],8)

    def test_crossfit(self):
        rng=np.random.default_rng(11);p=rng.uniform(.05,.8,1000);y=rng.binomial(1,p)
        q,pars=crossfit(y,p)
        self.assertEqual(len(q),len(p));self.assertEqual(len(pars),5)
        self.assertEqual(pars.Heldout_n.sum(),len(p));self.assertTrue(np.isfinite(q).all())

    def test_repository_blocks_private_io(self):
        with self.assertRaises(ValueError):outside_repo(ROOT/'outputs')

    def test_harmonization(self):
        raw=pd.DataFrame({'creat':[88.4,176.8],'ALB':[40,42],'GLB':[20,0]})
        d=numeric_map(raw,{'columns':{'SCr_mg_dL':'creat'},'divisors':{'SCr_mg_dL':88.4}})
        np.testing.assert_allclose(d.SCr_mg_dL,[1,2]);self.assertEqual(d['A/G'].iloc[0],2)
        self.assertTrue(np.isnan(d['A/G'].iloc[1]))
        self.assertTrue(d['Non-HDL-C'].isna().all())

    def test_specimen_index_before_completeness(self):
        from unittest.mock import patch
        # Invented epoch offsets only, no real dates or identifiers.
        origin=pd.Timestamp(0)
        raw=pd.DataFrame({'private_link_key':['invented_a','invented_a'],
            'private_specimen_key':['s1','s2'],'collection_date':[origin+pd.Timedelta(days=10),origin+pd.Timedelta(days=12)],
            'diabetes_eligible':[1,1],'frozen_order':[0,1],'age':[50,50],'sex':[0,0],'SCr_mg_dL':[1,1],'CysC_mg_L':[1,1]})
        aux=pd.DataFrame({'private_link_key':['invented_a']*3,'private_specimen_key':['p0','p1','p2'],
            'collection_date':[origin+pd.Timedelta(days=2),origin+pd.Timedelta(days=9),origin+pd.Timedelta(days=11)],
            'WBC':[4,5,6],'HGB':[110,120,130],'PLT':[180,190,200],'invalid_panel':[0,0,0]})
        link={'identity_resolution_verified':True,'same_specimen_verified':True,'patient_column':'private_link_key',
            'date_column':'collection_date','specimen_column':'private_specimen_key','diabetes_column':'diabetes_eligible',
            'frozen_order_column':'frozen_order','period_start':origin.isoformat(),'period_end_exclusive':(origin+pd.Timedelta(days=30)).isoformat(),
            'cbc':{'path':'aux.parquet','patient_column':'private_link_key','date_column':'collection_date','specimen_column':'private_specimen_key','invalid_record_column':'invalid_panel'}}
        with patch('cysc.preprocess.read_table',return_value=aux):
            z=specimen_index(raw,{'linkage':link},{'_base':str(ROOT.parent)})
        self.assertEqual(len(z),1);self.assertEqual(z.HGB.iloc[0],120)
        # A future CBC cannot rescue the earliest index when the in-window record is invalid.
        aux.loc[1,'invalid_panel']=1
        with patch('cysc.preprocess.read_table',return_value=aux):
            z=specimen_index(raw,{'linkage':link},{'_base':str(ROOT.parent)})
        self.assertTrue(z.HGB.isna().all())

if __name__=='__main__':unittest.main()
