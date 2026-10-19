import math
import unittest
import numpy as np
from scipy.special import ndtr
from pipeline.empirical.unified_prior_moments import conditional_moments


class ConditionalMomentTests(unittest.TestCase):
    def test_flat_likelihood_recovers_truncated_lognormal(self):
        mu=math.log(100);sd=2.25;lo=math.log(.01)
        expected=[math.exp(k*mu+.5*k*k*sd*sd)*ndtr((mu+k*sd*sd-lo)/sd)/ndtr((mu-lo)/sd) for k in (1,2)]
        np.testing.assert_allclose(conditional_moments(lambda z:0.,192),expected,rtol=1e-10)

    def test_normal_log_likelihood_has_analytic_conditional_moments(self):
        center=8.;noise=.4;mu=math.log(100);prior_sd=2.25
        variance=1/(1/prior_sd**2+1/noise**2)
        mean=variance*(mu/prior_sd**2+center/noise**2)
        expected=[math.exp(k*mean+.5*k*k*variance) for k in (1,2)]
        np.testing.assert_allclose(conditional_moments(lambda z:-.5*((z-center)/noise)**2,384),expected,rtol=1e-9)


if __name__=='__main__':unittest.main()
