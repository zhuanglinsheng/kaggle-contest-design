"""The observation window and strategic contest deadline are distinct."""
import unittest
from pipeline.empirical.estimate import early_cutoff, stan_data


class EmpiricalDeadlineTests(unittest.TestCase):
    def setUp(self):
        self.raw = dict(N_Delta=100, Delta2f=1 / 24, theta=100, count_start_hour=0,
                        hat_t_i=[10, 75, 80], hat_t_j=[20, 90],
                        hat_y=[0.0] * 100)

    def test_early_fit_preserves_full_deadline(self):
        full = stan_data(self.raw, "full")
        early = stan_data(self.raw, "early")
        self.assertEqual(early["N_Delta"], 81)
        self.assertEqual(early["T_contest"], full["T_contest"])
        self.assertAlmostEqual(early["T_contest"], 100 / 24)
        self.assertEqual(early["hat_t_i"], [10, 75, 80])
        self.assertEqual(early["hat_t_j"], [20])
        self.assertEqual(len(early["hat_y"]), 81)

    def test_full_fit_retains_original_strategy_horizon(self):
        full = stan_data(self.raw, "full")
        self.assertEqual(full["T_contest"], full["N_Delta"] * full["Delta2f"])

    def test_backfill_does_not_supply_zero_count_exposure(self):
        raw = dict(self.raw, count_start_hour=23,
                   hat_t_i=[23.5, 75], hat_t_j=[24.2, 90])
        fitted = stan_data(raw, "full")
        self.assertEqual(fitted["count_start_idx"], 24)
        self.assertEqual(fitted["hat_t_i"][0], 23.5)
        self.assertEqual(fitted["N_Delta"], 100)

    def test_count_window_rejects_discarded_prefix_events(self):
        raw = dict(self.raw, count_start_hour=23)
        with self.assertRaisesRegex(ValueError, "Submission precedes"):
            stan_data(raw, "full")

    def test_cutoff_excludes_initialization_prefix(self):
        # 24 initialization hours + 100 genuine contest hours.
        raw = dict(self.raw, N_Delta=124, hat_y=[0.0] * 124)
        self.assertEqual(early_cutoff(raw), 24 + 75)
        early = stan_data(raw, "early")
        self.assertAlmostEqual(early["T_contest"] - early["N_Delta"] / 24,
                               25 / 24)

    def test_validation_uses_same_cutoff_and_boundary_increment(self):
        import numpy as np
        import pandas as pd
        from pipeline.validation.chapter5 import benchmark_row

        private = np.arange(124, dtype=float) ** 2
        raw = dict(self.raw, N_Delta=124, hat_y=[0.0] * 124,
                   real_y=private.tolist())
        params = pd.Series(dict(sigma=2.0, **{"lambda": 1.0}, c_i=1.2,
                                c_j=1.5, mu_0=0.0))
        diagnostics = benchmark_row(raw, params)
        self.assertAlmostEqual(diagnostics["D_path"], np.mean(private[24:]))
        self.assertAlmostEqual(diagnostics["V_path"],
                               np.std(np.diff(private[24:]), ddof=1))
        self.assertAlmostEqual(diagnostics["D_late"], np.mean(private[99:]))
        self.assertAlmostEqual(diagnostics["V_late"],
                               np.std(np.diff(private)[98:], ddof=1))
