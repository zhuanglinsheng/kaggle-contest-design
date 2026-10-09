import unittest
import numpy as np
from scipy.stats import multivariate_normal
from pipeline.synthetic.model_comparison import (
    to_unconstrained, from_unconstrained, log_likelihood, log_prior, bridge_iteration)
from pipeline.synthetic.table1 import TRUTH, likelihood_check


class ModelComparisonTests(unittest.TestCase):
    def test_transform_roundtrip_and_jacobian(self):
        x=np.array([[1.2,1.5,2.,1.,0.,np.log(15),4.]])
        for free in (0,1):
            u=to_unconstrained(x,free); back,jac=from_unconstrained(u,free)
            np.testing.assert_allclose(back[:,:6+free],x[:,:6+free],atol=1e-12)
            if not free: self.assertEqual(back[0,6],2.)
            steps=[]
            for j in range(u.shape[1]):
                d=np.zeros_like(u); d[0,j]=1e-5
                hi,_=from_unconstrained(u+d,free); lo,_=from_unconstrained(u-d,free)
                steps.append((hi[0,j]-lo[0,j])/2e-5)
            self.assertAlmostEqual(float(jac[0]),sum(np.log(steps)),places=8)

    def test_vector_likelihood_equals_dense_both_models(self):
        data=dict(N=6,dt=1/24,theta=1.,ratio=15.,estimate_r=1,
            public_gap=[0.,0.,1.,1.,-.5,-.5,.7],count_i=[0,2,0,1,0,1],count_j=[0,1,0,0,0,1])
        for S0 in (2.,4.):
            p=dict(TRUTH,S0=S0); check=likelihood_check(data,p)
            x=np.array([[1.2,1.5,2.,1.,0.,np.log(15),S0]])
            self.assertAlmostEqual(float(log_likelihood(x,data)[0]),
                check['innovations_logpdf']+check['point_process_loglik'],places=10)

    def test_bridge_known_normal_integral_and_constant_shift(self):
        rng=np.random.default_rng(181)
        posterior=multivariate_normal([0.,1.],[[1.,.3],[.3,2.]])
        proposal=multivariate_normal([.1,.9],[[1.2,.2],[.2,1.9]])
        p=posterior.rvs(size=40000,random_state=rng)
        q=proposal.rvs(size=40000,random_state=rng)
        a=posterior.logpdf(p)-proposal.logpdf(p)
        b=posterior.logpdf(q)-proposal.logpdf(q)
        base=bridge_iteration(a,b)['log_evidence']
        self.assertLess(abs(base),.008)
        shifted=bridge_iteration(a-450,b-450)['log_evidence']
        self.assertAlmostEqual(shifted,base-450,places=8)

    def test_prior_support(self):
        data=dict(free_S0=1,ratio=15.,mu_prior_sd=1.,S0_prior_logmean=np.log(2),S0_prior_logsd=.75)
        x=np.array([[1.2,1.5,2.,1.,0.,np.log(15),4.]])
        self.assertTrue(np.isfinite(log_prior(x,data)[0]))
        x[0,0]=8.
        self.assertEqual(log_prior(x,data)[0],-np.inf)


if __name__=='__main__': unittest.main()
