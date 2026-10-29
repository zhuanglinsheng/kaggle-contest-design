"""Focused prior sensitivity of fixed original-game redesign directions.

Reuse audited direct refits, not first-order counterfactuals or reweighting.
Test the published design first, then a finite, declared same-direction
prize menu if needed. No joint-optimum or whole-sample robustness claim.
"""
from pathlib import Path
import concurrent.futures as futures
import hashlib
import importlib.util
import json
import math
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from pipeline.config import MAIN_DATA_DIR
from pipeline.empirical import estimate as estimator
from pipeline.empirical.counterfactual import PRIZE_DIVISOR
from pipeline.empirical.nonlinear_design import Draw, Design, save
from pipeline.empirical.nonlinear_design_similarity import SimilarityOutput

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
TABLE = REPORT/'nonlinear_table_rebuild_20261025'
spec = importlib.util.spec_from_file_location('sensitivity_existing_refiner', TABLE/'refine_prizes.py')
refiner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(refiner)
CASES = {8219: (2., 'very_small_saving'),
         2464: (100., 'large_prize_increase'),
         5144: (1., 'weaker_feedback_at_same_duration'),
         2551: (.1, 'longer_duration_lower_prize_higher_precision')}
SCENARIOS = ('common_sd_100', 'cost_floor_0001')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def design(record):
    p = record['precision']
    return Design(record['prize'], record['horizon'],
                  None if p == 'observed' else math.inf if p == 'full' else float(p))


class Evaluator(refiner.RefinedCase):
    # Retain the table's point solvers, recovery and paired multilevel formula;
    # initialize separately because alternative fits have different metadata.
    def __init__(self, cid, scenario, accepted_path):
        self.cid = cid
        self.folder = HERE/'calculations'/scenario/str(cid)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.source = json.loads(accepted_path.read_text())
        assert self.source['diagnostics']['passed']
        raw_path = MAIN_DATA_DIR/f'contest_{cid}.json'
        raw = json.loads(raw_path.read_text())
        inputs = self.source['configuration']['inputs']
        actual_data = estimator.stan_data(raw, 'full')
        assert all(inputs[k] == v for k, v in actual_data.items())
        assert self.source['configuration']['input_sha256'] == sha(raw_path)
        snapshot = REPORT/'prior_reconstruction_20261025/model_snapshot/common_cost'
        assert sha(snapshot.with_suffix('.stan')) == self.source['configuration']['model_sha256']
        assert sha(snapshot) == self.source['configuration']['executable_sha256']
        expected_sd = 1. if scenario == 'common_sd_100' else .75/math.sqrt(2)
        expected_floor = .01 if scenario == 'common_sd_100' else .001
        assert inputs['common_log_sd'] == expected_sd and inputs['cost_floor'] == expected_floor
        assert inputs['common_log_mean'] == 0. and inputs['relative_log_sd'] == .75*math.sqrt(2)
        self.observed = Design(raw['theta']/PRIZE_DIVISOR, raw['N_Delta']*raw['Delta2f'])
        self.engine = SimilarityOutput(TABLE/'build')
        self.groups = dict(coarse=[], fine=[])
        self.validated = {}
        rng = np.random.default_rng(202610260000+cid)
        for chain, (path, expected) in enumerate(zip(self.source['chain_paths'], self.source['chain_sha256'])):
            assert sha(path) == expected
            df = pd.read_csv(path, comment='#', usecols=['c_i', 'c_j', 'sigma', 'lambda', 'mu_0'])
            for name, count in [('coarse', 32), ('fine', 4)]:
                group = []
                for index in rng.integers(0, len(df), size=count):
                    row = df.iloc[int(index)]
                    group.append(Draw(float(row.c_i), float(row.c_j), float(row.sigma),
                                      float(row['lambda']), float(row.mu_0), chain, int(index)))
                self.groups[name].append(group)
        self.provenance = dict(accepted=str(accepted_path), accepted_sha256=sha(accepted_path),
                               chain_sha256=self.source['chain_sha256'],
                               raw_input_sha256=sha(raw_path), seed=202610260000+cid,
                               script_sha256=sha(__file__), refiner_sha256=sha(TABLE/'refine_prizes.py'),
                               batch_sha256=sha(TABLE/'design_batch.py'))
        save(self.folder/'quadrature.json', dict(provenance=self.provenance,
             inputs=inputs, draws={k: [[vars(d) for d in g] for g in v]
                                  for k, v in self.groups.items()}))

    def compare(self, candidate, reference):
        identity = dict(candidate=candidate.record(), reference=reference.record(),
                        provenance=self.provenance)
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        path = self.folder/'comparisons'/f'{key}.json'
        if path.exists():
            return json.loads(path.read_text())
        coarse, correction, radii, failures = [], [], [], []
        for chain in range(4):
            c, f = [], []
            for draw in self.groups['coarse'][chain]:
                a, b = self.native(draw, candidate), self.native(draw, reference)
                if not (a['resolved'] and b['resolved']):
                    failures.append(dict(level='coarse', chain=chain, index=draw.index))
                else:
                    c.append(a['B']-b['B'])
            for draw in self.groups['fine'][chain]:
                a, b = self.native(draw, candidate), self.native(draw, reference)
                fa, fb = self.fine_output(draw, candidate), self.fine_output(draw, reference)
                if not all(r['resolved'] for r in (a, b, fa, fb)):
                    failures.append(dict(level='fine', chain=chain, index=draw.index,
                                         candidate_resolved=fa['resolved'], reference_resolved=fb['resolved']))
                else:
                    f.append(fa['B']-fb['B']-a['B']+b['B'])
                    radii.append(fa['diagnostic_radius']+fb['diagnostic_radius'])
            coarse.append(c)
            correction.append(f)
            print('PAIR PROGRESS', self.cid, candidate.record(), chain+1, '/4', flush=True)
        result = dict(identity=identity, resolved=not failures, failed_draws=failures,
                      posterior_draws_coarse=128, posterior_draws_fine=16)
        if not failures:
            arrays = coarse+correction
            terms = [float(np.var(g, ddof=1)/len(g)/16) for g in arrays]
            var = sum(terms)
            den = sum(v*v/(len(g)-1) for v, g in zip(terms, arrays))
            df = var*var/den if den else 1e9
            mean = float(np.mean(coarse)+np.mean(correction))
            radius = float(np.mean(radii))
            error = float(student_t.ppf(.975, df))*math.sqrt(var)+radius
            result.update(mean_gap=mean, mcse=math.sqrt(var), numerical_radius=radius,
                          lower_diagnostic=mean-error, upper_diagnostic=mean+error,
                          direction='positive' if mean-error > 0 else 'negative' if mean+error < 0 else 'uncertain',
                          scope='Conditional posterior integration and grid diagnostics, not a posterior credible interval or rigorous continuous PDE bound')
        save(path, result)
        return result


def run(cid, scenario):
    target = HERE/'results'/f'{cid}_{scenario}.json'
    if target.exists():
        return json.loads(target.read_text())
    delta, reason = CASES[cid]
    original_path = TABLE/f'contests/{cid}/refined_selections.json'
    original = json.loads(original_path.read_text())
    selected = next(s for s in original['selections'] if s['delta'] == delta)
    candidate = design(selected['selected'])
    accepted_path = REPORT/f'prior_reconstruction_20261025/{scenario}/contest_{cid}/accepted.json'
    evaluator = Evaluator(cid, scenario, accepted_path)
    observed = evaluator.observed
    result = dict(contest_id=cid, delta=delta, scenario=scenario, selection_reason=reason,
                  provenance=evaluator.provenance, original_selection=selected,
                  original_selection_sha256=sha(original_path), trials=[])
    q0 = candidate.prize/observed.prize
    t = candidate.horizon/observed.horizon
    cost_neutral_q = 1+delta-delta*t*t
    upper_q = min(1., cost_neutral_q) if q0 < 1. else cost_neutral_q
    # These are finite existence witnesses along the original direction,
    # not an optimization or a monotonicity assertion.
    probes = [q0]+[q0+(upper_q-q0)*fraction for fraction in (.5, .75, .875, .9375)]
    supported = None
    for q in probes:
        trial_design = Design(q*observed.prize, candidate.horizon, candidate.precision)
        comparison = evaluator.compare(trial_design, observed)
        saving = 100*(1-(q+delta*t*t)/(1+delta))
        result['trials'].append(dict(design=trial_design.record(), prize_ratio=q,
                                     cost_saving_pct=saving, comparison=comparison))
        save(evaluator.folder/'progress.json', result)
        if comparison.get('direction') == 'positive' and saving > 0:
            supported = trial_design
            break
    result['fixed_published_design_direction'] = result['trials'][0]['comparison'].get('direction', 'unresolved')
    result['same_direction_cost_saving_supported'] = supported is not None
    result['supported_design'] = None if supported is None else supported.record()
    result['supported_saving_pct'] = None if supported is None else result['trials'][-1]['cost_saving_pct']
    # Isolate the feedback change at the tested prize and duration. A joint
    # improvement alone does not show that the feedback component helps.
    feedback_test = supported or candidate
    result['matched_observed_feedback_contrast'] = evaluator.compare(
        feedback_test, Design(feedback_test.prize, feedback_test.horizon, None))
    draws = [d for group in evaluator.groups['coarse'] for d in group]
    result['precision_higher_than_observed_fraction'] = float(np.mean([
        feedback_test.precision > d.observed_precision for d in draws]))
    result['status'] = 'completed'
    result['scope'] = ('Four selected cases; two direct-refit prior/support changes; '
                       'fixed absolute feedback and duration; finite prize adjustment menu; '
                       'directional existence test, not a new optimum or whole-sample robustness test')
    save(target, result)
    print('SENSITIVITY COMPLETED', cid, scenario,
          result['fixed_published_design_direction'], supported is not None, flush=True)
    return result


if __name__ == '__main__':
    import argparse
    import fcntl
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    with (HERE/'controller.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        state = dict(status='running', pid=os.getpid(), completed=[], failed=[],
                     total=8, workers=args.workers, script_sha256=sha(__file__))
        save(HERE/'status.json', state)
        with futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs = {pool.submit(run, cid, sc): (cid, sc) for cid in CASES for sc in SCENARIOS}
            for job in futures.as_completed(jobs):
                cid, sc = jobs[job]
                try:
                    r = job.result()
                    state['completed'].append(dict(contest_id=cid, scenario=sc,
                         fixed_direction=r['fixed_published_design_direction'],
                         same_direction_supported=r['same_direction_cost_saving_supported']))
                except Exception as error:
                    import traceback
                    state['failed'].append(dict(contest_id=cid, scenario=sc,
                         error=repr(error), traceback=traceback.format_exc()))
                state['updated_utc'] = __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()
                save(HERE/'status.json', state)
        state['status'] = 'completed' if len(state['completed']) == 8 else 'completed_with_failures'
        save(HERE/'status.json', state)
