import unittest
import numpy as np
from pipeline.synthetic.richer_recovery import pack_contests, independent_joint
from pipeline.synthetic.model_comparison import log_likelihood, log_prior
from pipeline.synthetic.table1 import TRUTH


class RicherRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.d=dict(N=4,dt=1/24,theta=1.,ratio=15.,mu_prior_sd=1.,estimate_r=1,
                    public_gap=[0.,0.,1.,1.,-.5],count_i=[0,2,0,1],count_j=[0,1,0,0])

    def test_pack_preserves_contest_boundaries_and_counts(self):
        other=dict(self.d,public_gap=[0.,.4,.4,.4,.7],count_i=[1,0,0,1])
        packed=pack_contests([self.d,other],15.)
        self.assertEqual(packed['public_gap'].shape,(5,2))
        np.testing.assert_array_equal(packed['public_gap'][:,1],other['public_gap'])
        self.assertEqual(packed['count_i'].sum(),5)

    def test_pool_adds_likelihoods_but_prior_only_once(self):
        x=np.array([[1.2,1.5,2.,1.,0.,np.log(15),2.]])
        ll=log_likelihood(x,self.d)[0]
        prior=log_prior(x,dict(self.d,free_S0=0))[0]
        joint=independent_joint(TRUTH,[self.d,self.d],15.)
        self.assertAlmostEqual(joint,2*ll+prior,places=10)

    def test_incompatible_horizon_rejected(self):
        with self.assertRaises(ValueError):
            pack_contests([self.d,dict(self.d,N=5)],15.)


if __name__=='__main__': unittest.main()
