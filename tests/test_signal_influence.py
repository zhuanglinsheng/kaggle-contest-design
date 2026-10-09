import json
import unittest

import numpy as np
import pandas as pd
import statsmodels.api as sm

from pipeline.validation.chapter5_influence import diagnostic


class SignalInfluenceTests(unittest.TestCase):
    def test_original_sample_threshold_and_single_refit(self):
        x = np.linspace(0, 5, 100)
        y = 1 - 0.3 * x + 0.03 * np.sin(11 * x)
        y[50] += 20
        frame = pd.DataFrame({"contest_id": np.arange(100), "x": x, "y": y})
        before = frame.copy(deep=True)
        rows, flags = diagnostic(frame, "x", "y", "test")
        pd.testing.assert_frame_equal(frame, before)
        self.assertTrue((flags.threshold == 4 / 100).all())
        self.assertTrue(flags.loc[50, "excluded"])
        fit = sm.OLS(y, sm.add_constant(x)).fit()
        influence = fit.get_influence()
        h = influence.hat_matrix_diag
        direct = (fit.resid**2 / (2 * fit.mse_resid)) * h / (1 - h)**2
        np.testing.assert_allclose(flags.cook_distance, direct)
        keep = ~flags.excluded.to_numpy()
        refit = sm.OLS(y[keep], sm.add_constant(x[keep])).fit(cov_type="HC1")
        self.assertAlmostEqual(rows.iloc[1].coefficient, refit.params[1])
        self.assertEqual(rows.iloc[1].n, int(keep.sum()))
        self.assertEqual(json.loads(rows.iloc[1].excluded_contest_ids),
                         flags.loc[~keep, "contest_id"].tolist())

    def test_duplicate_contests_rejected(self):
        frame = pd.DataFrame({"contest_id": [1, 1, 2, 3, 4],
                              "x": [0, 1, 2, 3, 4], "y": [1, 2, 2, 4, 4]})
        with self.assertRaises(ValueError):
            diagnostic(frame, "x", "y", "test")
