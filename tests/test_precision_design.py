import unittest
import numpy as np
from pipeline.empirical.precision_design import optimal_noise


class PrecisionPolicyTests(unittest.TestCase):
    def test_full_disclosure(self):
        x, _, _ = optimal_noise(np.array([2.,3.]), np.array([1.,2.]), np.ones(2), np.ones(2))
        self.assertEqual(x, 0.)

    def test_single_draw_finite(self):
        x, _, _ = optimal_noise(np.array([1.]), np.array([4.]), np.array([2.]), np.ones(1))
        self.assertAlmostEqual(x, 1.5)

    def test_mixture_beats_endpoints(self):
        q, m, s, w = map(np.array, ([1.,2.], [4.,8.], [2.,1.], [1.,2.]))
        x, value, _ = optimal_noise(q,m,s,w)
        grid = np.linspace(0,6,10001)
        vals = np.mean(w[:,None]*np.exp(-m[:,None]/(2*(q[:,None]+s[:,None]*grid)))/np.sqrt(q[:,None]+s[:,None]*grid), axis=0)
        self.assertGreaterEqual(value + 1e-10, vals.max())
        self.assertGreater(x, 0)


if __name__ == "__main__":
    unittest.main()
