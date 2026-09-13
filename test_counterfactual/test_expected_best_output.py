"""Numerical checks for best-output-preserving counterfactuals."""

from __future__ import annotations

import math
import unittest

from all_contests_joint_optimize import ContestParams, total_expected_effort
from duration_multiplier_substitution import (
    expected_best_output,
    find_min_theta_fast,
    sampling_component,
)


def _params(*, sigma: float = 1.0) -> ContestParams:
    return ContestParams(
        contest_id=1,
        theta0=1.0,
        T0_days=20.0,
        lamb=1.0,
        sigma=sigma,
        mu0=0.0,
        c_i=1.0,
        c_j=1.0,
        r=1.0,
    )


MODEL_KWARGS = {"z_boundary": 10.0, "dz_target": 0.01}


class ExpectedBestOutputTests(unittest.TestCase):
    def test_sampling_component_formula(self) -> None:
        params = _params(sigma=1.7)
        duration = 18.0
        expected = params.sigma * math.sqrt(duration) / math.sqrt(2.0 * math.pi)
        self.assertAlmostEqual(sampling_component(duration, params), expected)

    def test_best_output_decomposition(self) -> None:
        params = _params()
        theta = 0.8
        duration = 20.0
        effort = total_expected_effort(theta, duration, params, **MODEL_KWARGS)
        output = expected_best_output(theta, duration, params, **MODEL_KWARGS)
        self.assertAlmostEqual(
            output,
            0.5 * effort + sampling_component(duration, params),
        )

    def test_matches_symmetric_leading_order_for_small_prize(self) -> None:
        params = _params()
        theta = 1e-2
        duration = 20.0
        actual = expected_best_output(theta, duration, params, **MODEL_KWARGS)
        leading = math.sqrt(duration) / math.sqrt(2.0 * math.pi) * (
            params.sigma + theta / (params.sigma * params.c_i)
        )
        self.assertAlmostEqual(actual / leading, 1.0, delta=2e-6)

    def test_bisection_preserves_best_output(self) -> None:
        params = _params()
        baseline = expected_best_output(
            params.theta0,
            params.T0_days,
            params,
            **MODEL_KWARGS,
        )
        longer_duration = 1.1 * params.T0_days
        theta_new = find_min_theta_fast(
            longer_duration,
            baseline,
            params,
            theta_min=0.01,
            theta0=params.theta0,
            **MODEL_KWARGS,
        )
        self.assertIsNotNone(theta_new)
        assert theta_new is not None
        self.assertLess(theta_new, params.theta0)
        counterfactual = expected_best_output(
            theta_new,
            longer_duration,
            params,
            **MODEL_KWARGS,
        )
        self.assertGreaterEqual(counterfactual + 1e-10, baseline)


if __name__ == "__main__":
    unittest.main()
