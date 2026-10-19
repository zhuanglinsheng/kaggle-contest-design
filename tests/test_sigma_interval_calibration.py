import math
import unittest
from pipeline.empirical.sigma_interval_calibration import normalized_error,calibrated_interval


class CalibrationTests(unittest.TestCase):
 def test_scale_invariance(self):
  self.assertAlmostEqual(normalized_error(1.5,1.,.8,1.2),normalized_error(15.,10.,8.,12.))
 def test_factor_one_preserves_original_interval(self):
  lower,upper=calibrated_interval(2.,1.,3.,1.)
  self.assertAlmostEqual(lower,1.);self.assertAlmostEqual(upper,3.)
 def test_training_score_is_sufficient_in_each_direction(self):
  for truth in (.4,3.):
   score=max(1.,normalized_error(truth,1.,.8,1.2))
   lower,upper=calibrated_interval(1.,.8,1.2,score*(1+1e-12))
   self.assertLessEqual(lower,truth);self.assertGreaterEqual(upper,truth)
 def test_rejects_shrinking_interval(self):
  with self.assertRaises(ValueError):calibrated_interval(1.,.8,1.2,.9)


if __name__=='__main__':unittest.main()
