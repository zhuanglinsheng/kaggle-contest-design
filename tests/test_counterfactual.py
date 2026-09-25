import math
import unittest

from pipeline.empirical.counterfactual import (
    expected_best_output,
    precision_best_response,
    required_prize,
)


class RevisedCounterfactualTests(unittest.TestCase):
    def test_precision_rule(self):
        self.assertTrue(math.isinf(precision_best_response(1.0, 1.0, 2.0)))
        self.assertAlmostEqual(precision_best_response(2.0, 1.0, 1.0), 1.0 / 9.0)

    def test_observed_design_requires_observed_prize(self):
        values = dict(mu0=0.3, duration=20.0, precision=1.2, sigma=1.1, cost=1.4)
        theta0 = 2.0
        baseline = expected_best_output(theta=theta0, **values)
        theta = required_prize(
            baseline,
            theta_min=0.01,
            theta_max=theta0,
            **values,
        )
        self.assertAlmostEqual(theta, theta0)


if __name__ == "__main__":
    unittest.main()
