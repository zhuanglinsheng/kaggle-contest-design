"""Numerical checks for the theorem-consistent total-effort solver."""

from __future__ import annotations

import math
import unittest

from all_contests_joint_optimize import ContestParams, total_expected_effort


def _params(*, mu0: float = 0.0, c_i: float = 1.0, c_j: float = 1.0) -> ContestParams:
    return ContestParams(
        contest_id=1,
        theta0=1.0,
        T0_days=15.0,
        lamb=1.0,
        sigma=1.0,
        mu0=mu0,
        c_i=c_i,
        c_j=c_j,
        r=1.0,
    )


class ExpectedTotalEffortTests(unittest.TestCase):
    def test_matches_symmetric_small_w_limit(self) -> None:
        theta = 1e-2
        duration = 15.0
        actual = total_expected_effort(theta, duration, _params())
        leading_order = (
            2.0 * theta * math.sqrt(duration) / math.sqrt(2.0 * math.pi)
        )
        self.assertAlmostEqual(actual / leading_order, 1.0, delta=2e-4)

    def test_symmetric_solution_is_even_in_initial_state(self) -> None:
        positive = total_expected_effort(0.8, 20.0, _params(mu0=1.25))
        negative = total_expected_effort(0.8, 20.0, _params(mu0=-1.25))
        self.assertAlmostEqual(positive, negative, delta=1e-10)

    def test_general_asymmetric_solution_is_positive(self) -> None:
        actual = total_expected_effort(
            1.5,
            30.0,
            _params(mu0=-0.4, c_i=1.2, c_j=2.1),
        )
        self.assertGreater(actual, 0.0)

    def test_truncation_boundary_is_stable(self) -> None:
        params = _params(mu0=0.5, c_i=1.3, c_j=1.8)
        q10 = total_expected_effort(1.0, 25.0, params, z_boundary=10.0)
        q12 = total_expected_effort(1.0, 25.0, params, z_boundary=12.0)
        self.assertAlmostEqual(q10 / q12, 1.0, delta=2e-5)


if __name__ == "__main__":
    unittest.main()
