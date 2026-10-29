import json
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
from pipeline.jk9_prediction import baseline, raw_input, splits, summary, EVIDENCE
from pipeline.jk9_fit import training_data
from pipeline.prediction.posterior import predictions, scalar_check
from pipeline.prediction.native import FullPath
from pipeline.prediction import baselines as b


class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.row=splits(2445,'primary_hour')[0]
        cls.raw=raw_input(cls.row)
        cls.columns={k:np.asarray([v,v*1.01]) for k,v in dict(c_i=10.,c_j=12.,sigma=1.,
            **{'lambda':4.},mu_0=.1,log_r=math.log(100.)).items()}

    def test_primary_and_complete_date_freeze_all_inputs(self):
        rows=splits()
        self.assertEqual(len(rows),146)
        self.assertEqual({s:sum(r['forecast_hours'] for r in rows if r['split']==s)
            for s in ('primary_hour','complete_day')},dict(primary_hour=18936,complete_day=19705))
        for r in rows:
            raw=raw_input(r)
            data=training_data(raw,'early',r['cutoff_hour'])
            self.assertEqual(data['N_Delta'],r['cutoff_hour'])
            self.assertEqual(data['T_contest'],raw['N_Delta']*raw['Delta2f'])
            self.assertEqual(len(data['hat_y']),r['cutoff_hour'])
            self.assertTrue(all(t<=r['cutoff_hour'] for t in data['hat_t_i']+data['hat_t_j']))

    def test_scalar_nested_and_future_readout_timing(self):
        raw=self.raw;col=self.columns;cut=self.row['cutoff_hour']
        fixed=predictions(raw,col,cut)[0]
        free=predictions(raw,dict(col,deadline_power=np.full(2,.5),gap_log_scale=np.zeros(2)),cut,free=True)[0]
        np.testing.assert_allclose(fixed,free,atol=1e-10,rtol=1e-10)
        np.testing.assert_allclose(fixed[0],scalar_check(raw,{k:v[:1] for k,v in col.items()})[cut:],atol=1e-9)
        changed=dict(raw,hat_y=list(raw['hat_y']));changed['hat_y'][cut]+=10.
        after=predictions(changed,col,cut)[0]
        np.testing.assert_array_equal(after[:,0],fixed[:,0])
        self.assertGreater(float(np.max(abs(after[:,1:]-fixed[:,1:]))),0.)

    def test_original_cpp_fullpath_matches_independent_python_for_both_kernels(self):
        with tempfile.TemporaryDirectory() as folder:
            engine=FullPath(folder)
            for free in (False,True):
                cols=dict(self.columns,deadline_power=np.asarray([.4,.7]),gap_log_scale=np.asarray([-.2,.3]))
                train,native=engine.evaluate(self.raw,cols,self.row['cutoff_hour'],free)
                python=predictions(self.raw,cols,self.row['cutoff_hour'],free=free)[0]
                self.assertTrue(np.isfinite(train).all())
                np.testing.assert_allclose(native,python,atol=1e-8,rtol=1e-10)

    def test_baseline_rerun_matches_recovered_scores(self):
        reference=json.loads((EVIDENCE/'recovered_scores.json').read_text())['records']
        for row in splits(2445):
            result=baseline(row)
            self.assertTrue(result['numerical_score_check_passed'])
            for m in 'AB':
                ref=next(r for r in reference if r['contest_id']==2445 and r['split']==row['split'] and r['model']==m)
                self.assertAlmostEqual(result[m+'_mean_joint_logscore'],ref['mean_joint_logscore'],places=6)

    def test_zero_count_gamma_predictive_joint_normalizes(self):
        k=np.arange(100)
        counts=np.array(np.meshgrid(k,k)).reshape(2,-1)
        score,_,_=b.forecast(np.array([0.]),np.array([math.log(2.)]),np.array([0.]),
                            .5,.5,np.zeros(counts.shape[1]),counts,1/24)
        self.assertAlmostEqual(float(np.exp(score).sum()),1.,places=12)

    def test_paired_bootstrap_rejects_missing_and_duplicate_records(self):
        records=json.loads((EVIDENCE/'recovered_scores.json').read_text())['records']
        for broken in (records[:-1],records+[records[0]]):
            with self.assertRaises(ValueError):summary(broken,replicates=50)
        result=summary(records,replicates=500)
        self.assertAlmostEqual(result['folds']['complete_day']['contrasts']['D-B']['mean'],.1179006392755,places=12)

    def test_importance_proposal_matches_independent_student_density(self):
        from pipeline.prediction.importance import Proposal
        from scipy.stats import multivariate_t
        from scipy.special import logsumexp
        rng=np.random.default_rng(42);points=rng.normal(size=(2000,6))
        proposal=Proposal(points,43)
        samples=proposal.sample(9,44)
        expected=[logsumexp([math.log(w)+multivariate_t.logpdf(x,loc=m,shape=s,df=7.)
            for w,m,s in zip(proposal.weights,proposal.means,proposal.scales)]) for x in samples]
        np.testing.assert_allclose(proposal.logq(samples),expected,atol=1e-10)
        self.assertAlmostEqual(float(proposal.weights[-1]),.05)

    def test_importance_target_nesting_and_hard_support(self):
        from pipeline.prediction.importance import Target
        cols=self.columns
        points=np.column_stack([np.log(cols[k]) if j<4 else cols[k]
            for j,k in enumerate(('c_i','c_j','sigma','lambda','mu_0','log_r'))])
        fixed=Target(self.raw,self.row['cutoff_hour'],False)
        free=Target(self.raw,self.row['cutoff_hour'],True)
        ld,path=fixed.evaluate(points)
        lc,other=free.evaluate(np.column_stack((points,np.full(2,.5),np.zeros(2))))
        np.testing.assert_allclose(path,other,atol=1e-8)
        np.testing.assert_allclose(lc-ld,-2*math.log(.5*math.sqrt(2*math.pi)),atol=1e-8)
        invalid=points.copy();invalid[:,0]=math.log(.001)
        a,b_=fixed.evaluate(invalid)
        self.assertTrue(np.isneginf(a).all() and np.isneginf(b_).all())

    @unittest.skipUnless(__import__('os').environ.get('JK9_STAN_INTEGRATION')=='1',
                         'Set JK9_STAN_INTEGRATION=1 for independent C/D Stan target checks')
    def test_importance_targets_match_independent_stan(self):
        from pipeline.prediction.parity import check
        self.assertTrue(all(r['passed'] for r in check(self.raw,self.row['cutoff_hour'])))
