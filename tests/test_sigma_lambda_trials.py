import unittest
import numpy as np
from pipeline.empirical import nonlinear_reestimate as n
from pipeline.empirical import sigma_lambda_trials as s


class SigmaLambdaTests(unittest.TestCase):
 def test_original_kernel_and_likelihood_decomposition(self):
  original_source=n.SOURCE
  try:
   n.SOURCE=n.ROOT/'pipeline/empirical/nonlinear_fast_kernel.cpp'
   data,_,z,_=n.load_baseline(4657);old=n.Likelihood(data)
   point=n.representative_point(z)
   for point in (point,point+np.array([.1,-.1,.05,.1,.02,.1])):
    full,parts=s.Target(data,'baseline').components(point)
    self.assertAlmostEqual(full,old.loglik(point,False),places=8)
    self.assertAlmostEqual(full,float(parts.sum()),places=8)
    weighted,_=s.Target(data,'count_quarter').components(point)
    self.assertAlmostEqual(weighted,float(parts[0]+.25*parts[1]),places=8)
  finally:n.SOURCE=original_source

 def test_lambda_only_prior_leaves_other_coordinates_unchanged(self):
  data={'hat_y':[.15]};mean,sd=s.prior_config(data,'baseline');newmean,newsd=s.prior_config(data,'lambda_external')
  np.testing.assert_array_equal(np.delete(mean,3),np.delete(newmean,3))
  np.testing.assert_array_equal(np.delete(sd,3),np.delete(newsd,3))
  self.assertAlmostEqual(np.exp(newmean[3]),100.);self.assertEqual(newsd[3],2.25)


if __name__=='__main__':unittest.main()
