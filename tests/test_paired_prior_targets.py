"""Target parity checks, including macOS-spawned numerical grid workers."""
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import numpy as np
from pipeline.empirical import nonlinear_reestimate as n
from pipeline.empirical import paired_prior_improvement as p
from pipeline.empirical import paired_prior_recovery as r
from tests.fixtures import count_data, parameter_point


class PriorTargetTests(unittest.TestCase):
    def setUp(self):
        self.originals={key:getattr(n,key) for key in ('SOURCE','OUT','PRIOR_SD','Likelihood','kernel','load_baseline','_grid_worker')}

    def tearDown(self):
        for key,value in self.originals.items():setattr(n,key,value)

    def test_spawn_grid_worker_uses_common_new_prior(self):
        p.configure('scale_weak')
        data=count_data()
        z=parameter_point()
        expected=n.Likelihood(data,.025,64).posterior(z)
        old=n.PRIOR_SD.copy()
        n.PRIOR_SD=np.array([.75,.75,.5,.75,1.,.5])
        wrong=n.Likelihood(data,.025,64).posterior(z)
        n.PRIOR_SD=old
        assert abs(expected-wrong)>1
        payload=(str(n.SOURCE),data,.025,64,7,0,z)
        with ProcessPoolExecutor(max_workers=1,mp_context=get_context('spawn')) as pool:
            _,actual=pool.submit(p.paired_grid_worker,payload).result()
        assert abs(actual-expected)<1e-9


    @unittest.skipUnless(os.environ.get('JK9_STAN_INTEGRATION') == '1',
                         'Set JK9_STAN_INTEGRATION=1 to compile the independent Stan reference')
    def test_simulator_feedback_target_matches_stan(self):
        from cmdstanpy import CmdStanModel
        r.configure('nonlinear','scale_weak')
        data=count_data()
        data['feedback_gap']=[.1]+data['hat_y'][:-1]
        data.update(Ni=len(data['hat_t_i']),Nj=len(data['hat_t_j']))
        folder=tempfile.TemporaryDirectory();self.addCleanup(folder.cleanup)
        source=Path(folder.name)/'feedback_model.stan'
        shutil.copy2(n.ROOT/'pipeline/models/references/feedback_model.stan',source)
        model=CmdStanModel(stan_file=str(source),cpp_options={'PRECOMPILED_HEADERS':'false'})
        z=parameter_point()
        evaluator=r.RecoveryLikelihood(data,.025,64)
        delta=[]
        for point in (z,z+np.array([.1,-.1,.05,.1,.02,.1])):
            reference=float(model.log_prob(r.a.physical(point),data=dict(data,**r.a.settings(data,'scale_weak')),
              jacobian=False,sig_figs=18).iloc[0,0])
            delta.append(evaluator.loglik(point,False)+evaluator.prior(point)-point[:4].sum()-reference)
        assert abs(delta[1]-delta[0])<1e-7


if __name__=="__main__":
    unittest.main()
