"""Check nonlinear counterfactual semantics, endpoints and failure handling."""
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from pipeline.empirical.nonlinear_design import (
    Draw, Design, Resolution, no_effort, nonlinear_output, checked_output,
    paired_quality, search, load_draws, digest,
)
from pipeline.theory.asymmetric_full_output import full_output


class NonlinearDesignTests(unittest.TestCase):
    def draw(self):
        return Draw(.8, 1.3, 1., 4., .3)

    def test_zero_prize_keeps_original_initial_distribution(self):
        draw = self.draw()
        for precision in (.001, 1000., math.inf):
            with patch('pipeline.theory.joint_design_exact.solve_equilibrium_adaptive') as solve:
                result = nonlinear_output(draw, Design(0., 2., precision), Resolution())
                self.assertAlmostEqual(result['B'], no_effort(draw, 2.))
                solve.assert_not_called()
        self.assertEqual(draw.S0, .5)
        with self.assertRaises(ValueError):
            Design(1., 1., 0.)

    def test_full_disclosure_preserves_player_relabeling(self):
        a = full_output(.4, 2., 1., .8, 1.3, .5, .3)
        b = full_output(.4, 2., 1., 1.3, .8, .5, -.3)
        self.assertAlmostEqual(a['B'], b['B'], delta=1e-8)
        self.assertGreater(a['gain'], 0.)

    def test_full_disclosure_recovers_weak_prize_limit(self):
        from pipeline.empirical.fixed_initial_design import Output
        draw = self.draw()
        values = dict(c_i=np.array([draw.c_i]), c_j=np.array([draw.c_j]),
                      sigma=np.array([draw.sigma]), mu_0=np.array([draw.mu0]),
                      **{'lambda': np.array([draw.observed_precision])})
        first = Output(values, 2., 192)
        theta = 1e-4
        original = full_output(theta, 2., 1., .8, 1.3, .5, .3)
        expected = first.no_effort+theta*first.value(math.inf)
        self.assertAlmostEqual(original['B'], expected, delta=3e-8)

    def test_original_game_does_not_impose_monotone_prize_response(self):
        # Independent value-HJB evidence from the original empirical point.
        d = Draw(.31081771597093727, .5799797624223358,
                 .10699364151553235, 36.23842043820827, -.031688542676287235)
        low = nonlinear_output(d, Design(.015*.9, 83.20833333333333), Resolution())
        high = nonlinear_output(d, Design(.015*1.1, 83.20833333333333), Resolution())
        self.assertGreater(low['B']-high['B'], .03)

    def test_numerical_failure_never_becomes_a_success_subset_mean(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('pipeline.empirical.nonlinear_design.nonlinear_output',
                       side_effect=RuntimeError('failed nonlinear solve')):
                failed = checked_output(self.draw(), Design(.1, 1.),
                                        cache_dir=directory, resolution=Resolution())
            good = dict(resolved=True, B=1., diagnostic_radius=0.)
            result = paired_quality([self.draw(), self.draw()], [good, good], [good, failed])
            self.assertEqual(result['status'], 'unresolved')
            self.assertEqual(result['unresolved_draws'], [1])
            self.assertNotIn('mean_gap', result)

    def test_search_retains_disconnected_feasible_prize_and_cheaper_failure(self):
        draws = [Draw(.8, 1.3, 1., 4., .3, c, i) for c in range(4) for i in range(8)]
        observed = Design(1., 1.)
        candidates = [Design(r, 1.) for r in (.1, .25, .5, .75)]
        def evaluate(draw, design, **kwargs):
            if design.prize == .1:
                return dict(resolved=False)
            # A monotone root assumption would miss the .25 feasible island.
            value = {1.: 1., .25: 1.2, .5: .8, .75: 1.1}[design.prize]
            return dict(resolved=True, B=value, diagnostic_radius=0.)
        with patch('pipeline.empirical.nonlinear_design.checked_output', side_effect=evaluate):
            result = search(draws, observed, candidates, [2.], cache_dir='unused')
        selected = result['selections'][0]
        self.assertEqual(selected['design']['prize'], .25)
        self.assertEqual(selected['cheaper_unresolved_candidates'][0]['prize'], .1)
        self.assertFalse(result['global_continuous_optimum_certified'])
        self.assertFalse(result['publication_ready'])

    def test_baseline_failure_stops_optimization(self):
        with patch('pipeline.empirical.nonlinear_design.checked_output',
                   return_value=dict(resolved=False)):
            result = search([self.draw()], Design(1., 1.), [Design(.5, 1.)],
                            [2.], cache_dir='unused')
        self.assertEqual(result['status'], 'baseline_unresolved')
        self.assertNotIn('selections', result)

    def test_loader_uses_original_joint_rows_and_rejects_changed_chains(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root/'input.json'
            raw.write_text('{}')
            paths = []
            for c in range(4):
                path = root/f'chain{c}.csv'
                path.write_text('c_i,c_j,sigma,lambda,mu_0\n.8,1.3,1,4,.3\n.9,1.4,1.1,5,.4\n')
                paths.append(path)
            accepted = root/'accepted.json'
            accepted.write_text(json.dumps(dict(diagnostics=dict(passed=True),
                configuration=dict(raw_input_sha256=digest(raw)),
                chain_paths=[p.name for p in paths], chain_sha256=[digest(p) for p in paths])))
            draws, source = load_draws(accepted, raw, per_chain=1)
            self.assertEqual(len(draws), 4)
            self.assertTrue(all(d.index == 1 and d.c_i == .9 and d.observed_precision == 5. for d in draws))
            self.assertFalse(source['all_retained_draws_used'])
            paths[0].write_text(paths[0].read_text()+'\n')
            with self.assertRaises(ValueError):
                load_draws(accepted, raw)


if __name__ == '__main__':
    unittest.main()
