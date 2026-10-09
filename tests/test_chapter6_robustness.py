import unittest
import numpy as np
from scipy.special import ndtri
from pipeline.robustness.chapter6_data import norm_scales, weighted_quantile
from pipeline.empirical.counterfactual import expected_best_output, precision_best_response


class NormalReferenceScaleTests(unittest.TestCase):
    def test_gaussian_scale(self):
        z = ndtri((np.arange(999)+.5)/999)
        scales,_ = norm_scales(2*z,np.ones(999))
        for a in scales.values():
            self.assertAlmostEqual(a,.5,places=6)

    def test_own_norm_not_worse_than_baseline(self):
        z = ndtri((np.arange(999)+.5)/999)
        _,loss = norm_scales(np.sinh(z),np.ones(999))
        for name in ("L1","L2","Linf"):
            self.assertLessEqual(loss[name][name],loss["baseline"][name]+1e-9)

    def test_equal_contest_weights(self):
        q = weighted_quantile(np.array([0.,10.,10.]),np.array([1.,.5,.5]),np.array([.5]))
        self.assertAlmostEqual(q[0],20/3)

    def test_calibrated_output_scale_covariance(self):
        baseline=expected_best_output(mu0=2,theta=.2,duration=3,precision=.7,sigma=.5,cost=1.2)
        a=.45
        scaled=expected_best_output(mu0=a*2,theta=.2,duration=3,precision=.7/a**2,sigma=a*.5,cost=1.2/a**2)
        self.assertAlmostEqual(scaled,a*baseline)
        self.assertAlmostEqual(precision_best_response(a*2,a*.5,3),precision_best_response(2,.5,3)/a**2)


if __name__ == "__main__":
    unittest.main()
