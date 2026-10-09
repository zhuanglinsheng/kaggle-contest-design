"""Likelihood checks for Table 1, with no posterior fitting."""
import unittest
from pipeline.synthetic.table1 import likelihood_check, TRUTH


class HourlyEstimationTests(unittest.TestCase):
    def test_innovations_equal_dense_with_shared_hourly_readouts(self):
        data=dict(N=6,dt=1/24,theta=1.,ratio=15.,estimate_r=0,
                  public_gap=[0.,0.,1.,1.,-.5,-.5,.7],
                  count_i=[0,2,0,1,0,1],count_j=[0,1,0,0,0,1])
        check=likelihood_check(data,TRUTH)
        self.assertAlmostEqual(check['difference'],0.,places=9)

    def test_multiple_submissions_do_not_repeat_the_readout_density(self):
        data=dict(N=2,dt=1/24,theta=1.,ratio=15.,estimate_r=0,
                  public_gap=[0.,0.,1.],count_i=[0,1],count_j=[0,0])
        a=likelihood_check(data,TRUTH)
        b=likelihood_check(dict(data,count_i=[0,3]),TRUTH)
        self.assertEqual(a['innovations_logpdf'],b['innovations_logpdf'])
        self.assertNotEqual(a['point_process_loglik'],b['point_process_loglik'])
