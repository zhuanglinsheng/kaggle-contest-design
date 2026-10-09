import unittest

import numpy as np

from pipeline.validation.chapter5_robust import huber_fit


class SignalRobustTests(unittest.TestCase):
    def test_response_outlier_is_downweighted_without_deletion(self):
        x = np.linspace(0, 5, 100)
        y = 1 - 0.3 * x + 0.03 * np.sin(11 * x)
        y[50] += 20
        fit = huber_fit(x, y)
        self.assertEqual(len(fit.weights), 100)
        self.assertGreater(fit.weights[50], 0)
        self.assertLess(fit.weights[50], 0.05)
        self.assertAlmostEqual(fit.params[1], -0.3, places=2)

    def test_exactly_same_fit_on_repeated_input(self):
        x = np.linspace(0, 5, 100)
        y = 1 - 0.3 * x + 0.03 * np.sin(11 * x)
        np.testing.assert_array_equal(huber_fit(x, y).params, huber_fit(x, y).params)


if __name__ == "__main__":
    unittest.main()
