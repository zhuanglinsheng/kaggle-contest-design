import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, norm
from pipeline.empirical.rebuild_main import assign_times, logits, rebuild

class EmpiricalRebuildTests(unittest.TestCase):
    def test_same_day_records_are_not_deduplicated(self):
        source = pd.DataFrame(dict(Id=[3, 1, 2], TeamId=[1, 1, 2],
                         SubmissionDate=pd.to_datetime(["2020-01-01"] * 3)))
        timed = assign_times(source, 1234)
        self.assertEqual(len(timed), 3)
        self.assertEqual(len(set(timed.imputed_time)), 3)
        self.assertEqual(timed.Id.tolist(), [1, 2, 3])
        np.testing.assert_array_equal(timed.imputed_time, assign_times(source, 1234).imputed_time)

    def test_signed_scores_are_normalized_before_logit(self):
        f = pd.DataFrame(dict(PrivateScoreFullPrecision=[-.5, 0, .5],
                              PublicScoreLeaderboardDisplay=[-.5, 0, .5]))
        np.testing.assert_allclose(logits(f, True)[:, 0], [-np.log(3), 0, np.log(3)])

    def test_scores_include_events_at_the_hourly_endpoint(self):
        first = [22, 24, 100, 200, 300, 380]
        second = [23, 24.5, 100.5, 200.5, 300.5, 380.5]
        records = pd.DataFrame(dict(
            Id=np.arange(12), TeamId=[1] * 6 + [2] * 6,
            imputed_time=np.array(first + second) * 3600,
            SubmissionDate=pd.to_datetime(np.array(first + second) * 3600, unit="s").normalize(),
            PrivateScoreFullPrecision=[.6, .7, .71, .72, .73, .74] + [.5] * 6,
            PublicScoreLeaderboardDisplay=[.6, .7, .71, .72, .73, .74] + [.5] * 6))
        raw = dict(start_time=0, N_Delta=400, theta=1, Delta2f=1 / 24)
        with patch("pipeline.empirical.rebuild_main.assign_times", return_value=records):
            data, _ = rebuild(records, raw, (1, 2), 1234, "Accuracy")
        self.assertEqual(data["count_start_hour"], 23)
        self.assertEqual(data["Ni"], 5)
        self.assertEqual(data["Nj"], 5)
        self.assertAlmostEqual(data["hat_y"][23], np.log(.7 / .3))
        self.assertAlmostEqual(data["hat_y"][22], data["hat_y"][23])
        self.assertEqual(data["hat_t_i"][0], 24)

    def test_scalar_innovations_equal_dense_covariance(self):
        rng = np.random.default_rng(22)
        h = np.array([1, .3, 2, .1, 4.])
        t = np.cumsum(h)
        sigma, precision, initial = 1.2, 2.4, .7
        residual = rng.normal(size=len(h))
        noise = 1 / (precision * h) + 1e-9
        cov = initial + sigma**2 * np.minimum.outer(t, t) + np.diag(noise)
        dense = multivariate_normal.logpdf(residual, mean=np.zeros(len(h)), cov=cov)
        mean, variance, ll = 0., initial, 0.
        for value, elapsed, measurement in zip(residual, h, noise):
            prediction = variance + sigma**2 * elapsed
            innovation_variance = prediction + measurement
            innovation = value - mean
            ll += norm.logpdf(innovation, scale=np.sqrt(innovation_variance))
            mean += prediction / innovation_variance * innovation
            variance = prediction * measurement / innovation_variance
        self.assertAlmostEqual(dense, ll, places=10)
