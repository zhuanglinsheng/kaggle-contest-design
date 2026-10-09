import unittest
import numpy as np
from scipy.stats import norm
from pipeline.empirical.mode_mass import bridge


class ModeMassTests(unittest.TestCase):
    def test_bridge_recovers_known_normalizer(self):
        rng = np.random.default_rng(14)
        p, g = rng.normal(size=600), rng.normal(size=1000)
        lp, lg = norm.logpdf(p), norm.logpdf(g)
        value, _ = bridge(lp + 3.2, lp, lg + 3.2, lg)
        self.assertAlmostEqual(value, 3.2, places=8)

    def test_bridge_respects_region_indicator(self):
        rng = np.random.default_rng(94)
        p, g = np.abs(rng.normal(size=10000)), rng.normal(size=10000)
        lp, lqg = norm.logpdf(p), norm.logpdf(g)
        lg = np.where(g > 0, lqg, -np.inf)
        value, _ = bridge(lp, lp, lg, lqg)
        self.assertLess(abs(value - np.log(.5)), .03)
