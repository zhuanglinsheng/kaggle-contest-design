"""Regression checks for the fresh prediction and variance producer interfaces."""
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from pipeline.figures.variance import variance
from pipeline.jk9_prediction import fresh_records, splits
from pipeline.prediction.importance import integrate
from pipeline.publication import ROOT, digest, write_json


class ReproductionProducerTests(unittest.TestCase):
    def test_variance_converges_from_both_initial_conditions(self):
        times = np.array([0., 1., 10.])
        lower = variance(times, .8, 1., 1.)
        upper = variance(times, 1.08330975, 1., 1.)
        np.testing.assert_allclose([lower[0], upper[0]], [.8, 1.08330975])
        self.assertTrue(np.all(np.diff(lower) > 0) and np.all(np.diff(upper) < 0))
        np.testing.assert_allclose([lower[-1], upper[-1]], [1., 1.], atol=1e-8)

    def test_rare_hour_tail_proposal_and_two_budget_acceptance(self):
        class GaussianTarget:
            cutoff, dim = 1, 6
            names = ('c_i','c_j','sigma','lambda','mu_0','log_r')
            means = np.array([1., 1., 1., 1., 0., 1.])
            prior = {'fixture': True}

            def __init__(self, raw, cutoff, free):
                pass

            def evaluate(self, points):
                log_training = -.5*np.sum(((points-self.means)/.2)**2, axis=1)
                return log_training, (-21.-.1*points[:,4]**2)[:,None]

        row = dict(contest_id=1, cutoff_hour=1, input_sha256='fixture')
        raw = dict(N_Delta=2)
        rng = np.random.default_rng(431)
        points = rng.normal(GaussianTarget.means, .2, size=(4,2048,6))
        with TemporaryDirectory() as work, patch('pipeline.prediction.importance.Target', GaussianTarget), \
                patch('pipeline.jk9_prediction.raw_input', return_value=raw):
            root = Path(work)
            paths = []
            for j, values in enumerate(points):
                values[:, :4] = np.exp(values[:, :4])
                path = root/f'chain{j}.csv'
                np.savetxt(path, values, delimiter=',', comments='',
                           header=','.join(GaussianTarget.names))
                paths.append(path)
            accepted = root/'accepted.json'
            write_json(accepted, dict(diagnostics=dict(passed=True),
                configuration=dict(model='D', training_cutoff=1, raw_input_sha256='fixture',
                    prior=GaussianTarget.prior,
                    source_model_sha256=digest(ROOT/'pipeline/models/leading_order.stan')),
                chain_paths=[p.name for p in paths], chain_sha256=[digest(p) for p in paths]))
            ordinary = integrate(row, accepted, 'D', root/'ordinary', (4096,8192))
            self.assertFalse(ordinary['passed'])
            self.assertEqual(ordinary['records'][-1]['unrepaired_rare_probability_hours'], [2])
            repaired = integrate(row, accepted, 'D', root/'repaired', (4096,8192), 100000000,
                                 repair_status=root/'ordinary/status.json')
            self.assertTrue(repaired['passed'])
            self.assertEqual(repaired['tail_hours'], [2])
            hour = repaired['records'][-1]['tail_diagnostics']['2']
            self.assertTrue(hour['screen_passed'])
            self.assertAlmostEqual(hour['joint_logscore'], -21.-.5*math.log(1+.2*.2**2), places=3)

    def test_fresh_registration_requires_matching_accepted_fit_and_passing_integration(self):
        row = splits(2445,'primary_hour')[0]
        cid, split = row['contest_id'], row['split']
        with TemporaryDirectory() as work:
            root = Path(work)
            write_json(root/f'baselines_AB_{cid}_{split}.json', dict(**row,
                numerical_score_check_passed=True, A_mean_joint_logscore=-1., B_mean_joint_logscore=-2.))
            for model in 'CD':
                accepted = root/model/split/str(cid)/'accepted.json'
                write_json(accepted, dict(model=model, diagnostics=dict(passed=True)))
                state = dict(**row, model=model, accepted_sha256=digest(accepted), passed=True,
                    records=[dict(screen_passed=True),
                        dict(screen_passed=True, consistent_with_previous=True, mean_joint_logscore=-3.)])
                status = root/f'importance_{model}_{cid}_{split}/status.json'
                write_json(status, state)
            self.assertEqual([r['model'] for r in fresh_records([row], root, [root], root)], list('ABCD'))
            state['accepted_sha256'] = 'other posterior'
            write_json(status, state)
            with self.assertRaisesRegex(ValueError, 'Integration identity differs'):
                list(fresh_records([row], root, [root], root))
            status.unlink()
            with self.assertRaisesRegex(ValueError, 'No passed integration'):
                list(fresh_records([row], root, [root], root))
