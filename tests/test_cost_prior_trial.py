import unittest
import numpy as np
from scipy.stats import multivariate_normal, norm
from pipeline.empirical import cost_prior_trial as t


class CostPriorTests(unittest.TestCase):
    def test_transformed_density_matches_correlated_gaussian(self):
        z=np.random.default_rng(1381).normal(size=(50,6))
        for config in t.CANDIDATES.values():
            u,d=config['common_sd']**2,config['relative_sd']**2
            covariance=np.array([[u+d/4,u-d/4],[u-d/4,u+d/4]])
            expected=multivariate_normal.logpdf(z[:,:2],cov=covariance)-multivariate_normal.logpdf(z[:,:2],cov=np.eye(2)*1.25**2)
            expected+=norm.logpdf(z[:,2],np.log(2),config['sigma_sd'])-norm.logpdf(z[:,2],np.log(2),.5)
            expected+=norm.logpdf(z[:,3],0,config['lambda_sd'])-norm.logpdf(z[:,3],0,.75)
            np.testing.assert_allclose(t.log_ratio(z,config),expected,atol=1e-12)

    def test_exchange_of_contestants_leaves_prior_unchanged(self):
        z=np.random.default_rng(3291).normal(size=(50,6));swapped=z.copy();swapped[:,[0,1]]=z[:,[1,0]]
        for config in t.CANDIDATES.values():np.testing.assert_allclose(t.log_ratio(z,config),t.log_ratio(swapped,config),atol=1e-12)


if __name__=='__main__':unittest.main()
