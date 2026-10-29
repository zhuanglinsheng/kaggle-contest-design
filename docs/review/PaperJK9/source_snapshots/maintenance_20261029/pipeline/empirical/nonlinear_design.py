"""Design optimization in the original game, conditional on the original posterior.

Only posterior estimation uses the leading-order policy. Baselines and all
candidate quality constraints here use nonlinear HJB output. Prize response
is not assumed monotone. Search is explicitly over the supplied finite grid;
unresolved cheaper candidates prevent certification even of that grid.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from functools import partial
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.config import MAIN_DATA_DIR, resolve_manifest_path
from pipeline.empirical.counterfactual import PRIZE_DIVISOR
from pipeline.empirical.unified_prior_reestimate import OUT
from pipeline.synthetic.nonlinear import Primitives, variance
from pipeline.theory import asymmetric_full_output, joint_design_exact


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


@dataclass(frozen=True)
class Draw:
    c_i: float
    c_j: float
    sigma: float
    observed_precision: float
    mu0: float
    chain: int = 0
    index: int = 0

    def __post_init__(self):
        if not all(math.isfinite(x) and x > 0 for x in
                   (self.c_i, self.c_j, self.sigma, self.observed_precision)):
            raise ValueError('Positive finite posterior primitives required')
        if not math.isfinite(self.mu0):
            raise ValueError('Finite initial mean required')

    @property
    def S0(self):
        return self.sigma/math.sqrt(self.observed_precision)


@dataclass(frozen=True)
class Design:
    prize: float
    horizon: float
    # None retains the observed feedback rule. A positive number specifies
    # ONE absolute precision shared across posterior draws. Infinity = full.
    precision: float | None = None

    def __post_init__(self):
        if not math.isfinite(self.prize) or self.prize < 0:
            raise ValueError('Prize must be finite and nonnegative')
        if not math.isfinite(self.horizon) or self.horizon <= 0:
            raise ValueError('Horizon must be positive and finite')
        if self.precision is not None and not self.precision > 0:
            raise ValueError('Zero feedback is not an attained positive-precision design')

    def record(self):
        return dict(prize=self.prize, horizon=self.horizon,
                    precision=('observed' if self.precision is None else
                               'full' if math.isinf(self.precision) else self.precision))


@dataclass(frozen=True)
class Resolution:
    spacing: float = .01
    steps: int = 2000
    extent: float = 8.
    full_nodes: int = 8001
    full_cutoff: float = 26.
    full_quadrature: int = 160
    max_policy_cells: int = 30_000_000

    def __post_init__(self):
        if (self.spacing <= 0 or self.steps < 1 or self.extent <= 0
                or self.full_nodes < 5 or self.full_cutoff <= 0
                or self.full_quadrature < 1 or self.max_policy_cells < 1):
            raise ValueError('Invalid numerical resolution or memory budget')

    def refined(self):
        return Resolution(self.spacing/2, self.steps*2, self.extent,
                          2*self.full_nodes-1, self.full_cutoff+2,
                          self.full_quadrature*2, self.max_policy_cells)


def no_effort(draw, horizon):
    q = draw.S0+draw.sigma**2*horizon
    return (math.sqrt(q/(2*math.pi))*math.exp(-draw.mu0**2/(2*q))
            + draw.mu0*math.erf(draw.mu0/math.sqrt(2*q))/2)


def nonlinear_output(draw, design, resolution):
    """Numerical original-game output; never substitutes a leading-order policy."""
    baseline = no_effort(draw, design.horizon)
    if design.prize == 0:
        return dict(B=baseline, boundary_budget=0., tail_budget=0.,
                    method='analytic_zero_effort')
    precision = draw.observed_precision if design.precision is None else design.precision
    if math.isinf(precision):
        result = asymmetric_full_output.full_output(
            design.prize, design.horizon, draw.sigma, draw.c_i, draw.c_j,
            draw.S0, draw.mu0, nodes=resolution.full_nodes,
            cutoff=resolution.full_cutoff, quadrature=resolution.full_quadrature)
        return dict(result, method='nonlinear_analytic_policy_transformed_output')
    p = Primitives(T=design.horizon, theta=design.prize, c_i=draw.c_i,
                   c_j=draw.c_j, sigma=draw.sigma, lamb=precision,
                   S0=draw.S0, mu0=draw.mu0)
    scale = math.sqrt(draw.S0+draw.sigma**2*design.horizon)
    spacing = min(resolution.spacing*scale, .2*math.sqrt(float(variance(p.T, p))))
    # Reserve the extra adaptive layer's time nodes as well as the base grid.
    nodes = 2*math.ceil(resolution.extent*scale/spacing)+1
    if (resolution.steps+242)*nodes > resolution.max_policy_cells:
        raise RuntimeError('Policy grid exceeds explicit memory budget')
    if abs(draw.mu0)+6*math.sqrt(draw.S0) >= resolution.extent*scale:
        raise RuntimeError('Initial state too close to the finite state boundary')
    eq = joint_design_exact.solve_equilibrium_adaptive(
        p, dy=spacing, base_steps=resolution.steps, extent=resolution.extent)
    if eq.diagnostics['value_min'] < -1e-6 or eq.diagnostics['value_max'] > 1+1e-6:
        raise RuntimeError('Nonlinear value outside probability bounds')
    value = float(joint_design_exact.expected_best_output(eq))
    if not math.isfinite(value) or value < baseline-1e-8*max(1., baseline):
        raise RuntimeError('Invalid original-model expected output')
    return dict(B=value, method='nonlinear_value_HJB_and_output_PDE',
                diagnostics=eq.diagnostics, boundary_budget=0., tail_budget=0.)


def checked_output(draw, design, *, cache_dir, resolution):
    """Keep failures and both levels; no success-subset posterior averaging."""
    sources = {str(Path(f).resolve()): digest(f) for f in
               (__file__, joint_design_exact.__file__, asymmetric_full_output.__file__,
                Path(joint_design_exact.__file__).parents[1]/'synthetic/nonlinear.py')}
    signature = dict(draw=asdict(draw), design=design.record(),
                     resolution=asdict(resolution), sources=sources)
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    path = Path(cache_dir)/(key+'.json')
    if path.exists():
        record = json.loads(path.read_text())
        if record['signature'] != signature:
            raise RuntimeError('Changed nonlinear output cache signature')
        return record
    trials = []
    for level in (resolution, resolution.refined()):
        try:
            trials.append(dict(resolution=asdict(level), **nonlinear_output(draw, design, level)))
        except (RuntimeError, ValueError, FloatingPointError, OverflowError) as error:
            trials.append(dict(resolution=asdict(level), error=str(error)))
    record = dict(signature=signature, trials=trials,
                  resolved=all('B' in t for t in trials))
    if record['resolved']:
        record.update(B=trials[-1]['B'], diagnostic_radius=(
            3*abs(trials[-1]['B']-trials[0]['B'])
            +trials[-1]['boundary_budget']+trials[-1]['tail_budget']))
    save(path, record)
    return record


def paired_quality(draws, baseline, candidate):
    if len(baseline) != len(draws) or len(candidate) != len(draws):
        raise ValueError('All posterior draws must be retained in paired order')
    if not all(r['resolved'] for r in baseline+candidate):
        return dict(status='unresolved', unresolved_draws=[i for i, (a, b) in
                    enumerate(zip(baseline, candidate)) if not a['resolved'] or not b['resolved']])
    changes = np.array([b['B']-a['B'] for a, b in zip(baseline, candidate)])
    radius = float(np.mean([a['diagnostic_radius']+b['diagnostic_radius']
                            for a, b in zip(baseline, candidate)]))
    chains = sorted({d.chain for d in draws})
    groups = [[v for d, v in zip(draws, changes) if d.chain == c] for c in chains]
    mcse = None
    if len(groups) >= 2 and len({len(g) for g in groups}) == 1 and len(groups[0]) >= 8:
        # Reuse only the existing statistical diagnostic, not its estimator.
        from pipeline.empirical.nonlinear_reestimate import effective_sample_size
        ess = min(len(draws), max(1., effective_sample_size(np.array(groups))))
        mcse = float(changes.std(ddof=1)/math.sqrt(ess))
    margin = float(changes.mean())-radius
    if mcse is not None:
        margin -= 1.96*mcse
    return dict(status='evaluated', mean_gap=float(changes.mean()), mcse=mcse,
                diagnostic_radius=radius, lower_diagnostic_margin=margin,
                sample_quality_preserved=margin >= 0,
                posterior_mean_screen_passed=mcse is not None and margin >= 0)


def search(draws, observed, candidates, deltas, *, cache_dir, resolution=Resolution(), workers=1):
    """Cost-ordered exhaustive candidate-grid comparison, without monotone roots."""
    if not draws or observed.precision is not None:
        raise ValueError('Nonempty original posterior and observed baseline required')
    if not deltas or any(not math.isfinite(d) or d < 0 for d in deltas):
        raise ValueError('Finite nonnegative delay costs required')
    evaluation = partial(checked_output, cache_dir=cache_dir, resolution=resolution)
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    def evaluate(design):
        if executor:
            return list(executor.map(evaluation, draws, itertools.repeat(design)))
        return [evaluation(d, design) for d in draws]
    def cost(design, delta):
        return design.prize+delta*observed.prize*(design.horizon/observed.horizon)**2
    result = dict(output_model='original_nonlinear_game', posterior='unchanged_leading_order_estimation',
                  observed=observed.record(), posterior_draws=len(draws), candidates=[],
                  global_continuous_optimum_certified=False, publication_ready=False)
    try:
        base = evaluate(observed)
        result['baseline'] = base
        if not all(v['resolved'] for v in base):
            result['status'] = 'baseline_unresolved'
            return result
        best = {delta: observed for delta in deltas}
        unresolved = {delta: [] for delta in deltas}
        choices = sorted(set(candidates)-{observed},
                         key=lambda d: min(cost(d, delta) for delta in deltas))
        for design in choices:
            relevant = [delta for delta in deltas if cost(design, delta) < cost(best[delta], delta)]
            if not relevant:
                result['candidates'].append(dict(design=design.record(), status='cost_dominated'))
                continue
            outputs = evaluate(design)
            quality = paired_quality(draws, base, outputs)
            result['candidates'].append(dict(design=design.record(), quality=quality, outputs=outputs))
            for delta in relevant:
                if quality['status'] == 'unresolved':
                    unresolved[delta].append(design)
                elif quality['posterior_mean_screen_passed']:
                    best[delta] = design
            print('NONLINEAR DESIGN', design.record(), quality, flush=True)
        result['selections'] = []
        for delta in deltas:
            design = best[delta]
            threats = [d.record() for d in unresolved[delta] if cost(d, delta) < cost(design, delta)]
            result['selections'].append(dict(delta=delta, design=design.record(),
                cost_saving_pct=100*(1-cost(design, delta)/cost(observed, delta)),
                cheaper_unresolved_candidates=threats,
                scope='best supported by posterior-mean screen on supplied candidate grid; numerical diagnostics are not rigorous error bounds'))
        result['status'] = 'completed_candidate_grid'
        return result
    finally:
        if executor:
            executor.shutdown()


def load_draws(accepted_path, raw_path, per_chain=0):
    accepted_path, raw_path = Path(accepted_path), Path(raw_path)
    accepted = json.loads(accepted_path.read_text())
    if not accepted['diagnostics']['passed']:
        raise ValueError('Original posterior has not passed its sampling checks')
    if accepted['configuration']['raw_input_sha256'] != digest(raw_path):
        raise ValueError('Original posterior does not match contest input')
    if per_chain < 0:
        raise ValueError('Draw count must be nonnegative; zero uses every retained draw')
    names = ['c_i', 'c_j', 'sigma', 'lambda', 'mu_0']
    draws, retained_counts = [], []
    for chain, (path, sha) in enumerate(zip(accepted['chain_paths'], accepted['chain_sha256'])):
        path = resolve_manifest_path(path, accepted_path)
        if digest(path) != sha:
            raise ValueError(f'Original posterior chain changed: {path}')
        frame = pd.read_csv(path, comment='#', usecols=names)
        count = len(frame)
        if not count:
            raise ValueError('Empty posterior chain')
        retained_counts.append(count)
        indices = (np.arange(count) if per_chain == 0 else
                   np.floor((np.arange(min(per_chain, count))+.5)*count/min(per_chain, count)).astype(int))
        for index in indices:
            r = frame.iloc[int(index)]
            draws.append(Draw(*(float(r[k]) for k in names), chain=chain, index=int(index)))
    if len(retained_counts) != 4:
        raise ValueError('Expected the four original accepted posterior chains')
    return draws, dict(accepted_sha256=digest(accepted_path), raw_input_sha256=digest(raw_path),
                       chain_sha256=accepted['chain_sha256'], original_draws=sum(retained_counts),
                       selected_draws=len(draws), all_retained_draws_used=len(draws) == sum(retained_counts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contest', type=int, required=True)
    parser.add_argument('--accepted', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--draws-per-chain', type=int, default=8,
                        help='Stratified screening draws; 0 uses every original draw')
    parser.add_argument('--times', default='.1,.25,.5,.75,1,1.25,1.5,2')
    parser.add_argument('--prize-ratios', default='0,.25,.5,.75,.9,1,1.1,1.5,2,3,4')
    parser.add_argument('--precisions', default='observed,1,3,10,100,1000,full')
    parser.add_argument('--deltas', default='.1,.25,.5,1,2,5,10,25,100')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    raw_path = MAIN_DATA_DIR/f'contest_{args.contest}.json'
    accepted = args.accepted or OUT/'estimation/full'/f'contest_{args.contest}'/'accepted.json'
    draws, provenance = load_draws(accepted, raw_path, args.draws_per_chain)
    raw = json.loads(raw_path.read_text())
    observed = Design(float(raw['theta'])/PRIZE_DIVISOR, float(raw['N_Delta'])*float(raw['Delta2f']))
    numbers = lambda text: [float(v) for v in text.split(',')]
    precisions = [None if v == 'observed' else math.inf if v == 'full' else float(v)
                  for v in args.precisions.split(',')]
    candidates = [Design(observed.prize*r, observed.horizon*t, precision)
                  for t, r, precision in itertools.product(numbers(args.times), numbers(args.prize_ratios), precisions)]
    destination = args.out or OUT/'nonlinear_design'/f'contest_{args.contest}'
    result = search(draws, observed, candidates, numbers(args.deltas),
                    cache_dir=destination/'evaluations', workers=args.workers)
    result['provenance'] = provenance
    save(destination/'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('baseline', 'candidates')}, indent=2))


if __name__ == '__main__':
    main()
