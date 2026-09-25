"""Consistency checks for the revised PaperJK7 synthetic model."""

from __future__ import annotations

import math
import unittest
from datetime import datetime, timedelta

import numpy as np

from synthetic_data.synthetic_data import (
    leading_order_efforts,
    steady_state_variance,
    synthetic_data_simulation,
)


class RevisedSyntheticModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.start = datetime(2026, 1, 1)
        self.end = self.start + timedelta(days=90)

    def test_effort_matches_theorem_one(self) -> None:
        y = 1.3
        t = self.start + timedelta(days=12)
        theta, c_i, c_j, sigma, lamb = 1.0, 1.2, 1.5, 2.0, 0.8
        actual_i, actual_j = leading_order_efforts(
            y, t, self.end, prize=theta, c_i=c_i, c_j=c_j, sigma=sigma, lamb=lamb
        )
        q_t = sigma**2 * 78 + sigma / math.sqrt(lamb)
        kernel = math.exp(-y**2 / (2 * q_t)) / math.sqrt(2 * math.pi * q_t)
        self.assertAlmostEqual(actual_i, theta * kernel / c_i)
        self.assertAlmostEqual(actual_j, theta * kernel / c_j)

    def test_precision_has_state_dependent_effort_effect(self) -> None:
        common = dict(
            t=self.end - timedelta(days=1),
            T=self.end,
            prize=1.0,
            c_i=1.0,
            c_j=1.0,
            sigma=1.0,
        )
        tie_low, _ = leading_order_efforts(0.0, lamb=0.1, **common)
        tie_high, _ = leading_order_efforts(0.0, lamb=100.0, **common)
        gap_low, _ = leading_order_efforts(3.0, lamb=0.1, **common)
        gap_high, _ = leading_order_efforts(3.0, lamb=100.0, **common)
        self.assertGreater(tie_high, tie_low)
        self.assertGreater(gap_low, gap_high)

    def test_simulator_draws_initial_gap_from_common_prior(self) -> None:
        mu_0, sigma, lamb, seed = 0.7, 2.0, 0.5, 91
        result = synthetic_data_simulation(
            theta=1.0,
            c_i=1.2,
            c_j=1.5,
            sigma=sigma,
            lamb=lamb,
            intensity_effort_ratio=15.0,
            hour_arrival_ub=1.0,
            start_time=self.start,
            end_time=self.start + timedelta(hours=2),
            mu_0=mu_0,
            seed_initial=seed,
        )
        real_gap, perceived_gap, observed_gap = result[3], result[4], result[5]
        expected = np.random.default_rng(seed).normal(
            mu_0, math.sqrt(steady_state_variance(sigma, lamb))
        )
        self.assertAlmostEqual(real_gap[0], expected)
        self.assertEqual(perceived_gap[0], mu_0)
        self.assertEqual(observed_gap[0], mu_0)

    def test_public_gap_is_carried_forward_without_submission(self) -> None:
        result = synthetic_data_simulation(
            theta=1.0,
            c_i=1.2,
            c_j=1.5,
            sigma=2.0,
            lamb=1.0,
            intensity_effort_ratio=100.0,
            hour_arrival_ub=1.0,
            start_time=self.start,
            end_time=self.start + timedelta(days=2),
            seed_brownian=100,
            seed_poisson=101,
            seed_uniform=102,
            seed_initial=103,
        )
        observed_gap = result[5]
        submissions = result[6] + result[7]
        update_steps = {
            math.ceil((event - self.start).total_seconds() / 3600)
            for event in submissions
        }
        self.assertTrue(update_steps)
        for step in range(1, len(observed_gap)):
            if step not in update_steps:
                self.assertEqual(observed_gap[step], observed_gap[step - 1])


if __name__ == "__main__":
    unittest.main()
