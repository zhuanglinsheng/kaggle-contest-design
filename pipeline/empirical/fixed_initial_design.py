"""Fixed-initial-distribution, asymmetric first-order design comparisons.

The posterior is unchanged.  A candidate precision changes the Riccati path,
not the initial latent distribution.  Zero precision denotes a limiting
benchmark; it is not an attained design when positive feedback is required.
Run with ``python -m pipeline.empirical.fixed_initial_design``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize_scalar
from scipy.special import ndtr

from pipeline.empirical import counterfactual as cf
from pipeline.empirical.unified_prior_reestimate import OUT

DEST = OUT / "fixed_initial_design"
TIMES = (.1, .25, .5, .75, 1., 1.25, 1.5, 2.)
DELTAS = (.1, .25, .5, 1., 2., 5., 10., 25., 100.)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def variance(t, sigma, initial, precision):
    """Stable Riccati solution, including both precision endpoints."""
    t, sigma, initial = np.broadcast_arrays(t, sigma, initial)
    precision = np.broadcast_to(precision, t.shape)
    if np.all(np.isinf(precision)):
        return np.zeros_like(t)
    rate = sigma * np.sqrt(precision)
    z = rate * t
    tanhc = np.ones_like(z)
    np.divide(np.tanh(z), z, out=tanhc, where=z != 0)
    return (initial + sigma**2 * t * tanhc) / (1 + precision * initial * t * tanhc)


def beta(mu, q):
    return (np.sqrt(q / (2 * np.pi)) * np.exp(-mu**2 / (2*q))
            + mu * (ndtr(mu / np.sqrt(q)) - .5))


class Output:
    def __init__(self, draws, horizon, order=24):
        self.draws, self.T = draws, horizon
        self.mu = np.asarray(draws['mu_0'])
        self.sigma = np.asarray(draws['sigma'])
        self.initial = self.sigma / np.sqrt(draws['lambda'])
        self.q = self.initial + self.sigma**2 * horizon
        density = np.exp(-self.mu**2 / (2*self.q)) / np.sqrt(2*np.pi*self.q)
        ii, jj = 1 / draws['c_i'], 1 / draws['c_j']
        self.base = density * horizon * (ii + jj) / 2
        self.weight = density * horizon * (ii - jj)
        self.no_effort = float(np.mean(beta(self.mu, self.q)))
        # t/T=1-u^2 removes the square-root terminal singularity at infinity.
        nodes, weights = leggauss(order)
        u, weights = (nodes + 1) / 2, weights / 2
        self.t = horizon * (1-u*u)
        self.weights = weights * 2*u

    def coefficients(self, precision):
        sig, initial = self.sigma[:, None], self.initial[:, None]
        p = np.asarray(precision)
        if p.ndim:
            p = p[:, None]
        qt = variance(self.t[None, :], sig, initial, p)
        qt += sig**2 * (self.T-self.t[None, :])
        q = self.q[:, None]
        argument = self.mu[:, None] * np.sqrt(qt / (q * (2*q-qt)))
        expectation = (ndtr(argument)-.5) @ self.weights
        return self.base + self.weight * expectation

    def value(self, precision):
        return float(np.mean(self.coefficients(precision)))


def optimize(model, grid_size=65):
    """Search the whole closed precision range, verifying on a doubled grid.

    The search uses log precision and includes exact zero/infinity limits.
    Endpoints win numerical ties; tiny gains never create an interior label.
    """
    T, s, sig = model.T, model.initial, model.sigma
    low = float(np.min(1/(T*(s+sig**2*T)))) * 1e-9
    high = float(np.max(np.maximum(1/(T*s), 1/(sig*T)**2))) * 1e9
    logs = np.linspace(math.log(low), math.log(high), grid_size)
    cache = {}

    def f(z):
        if float(z) not in cache:
            cache[float(z)] = model.value(math.exp(z))
        return cache[float(z)]

    values = np.array([f(z) for z in logs])
    candidates = [(0., model.value(0.)), (math.inf, model.value(math.inf))]
    for k in range(1, len(logs)-1):
        if values[k] > values[k-1] and values[k] > values[k+1]:
            result = minimize_scalar(lambda z: -f(z), bounds=(logs[k-1], logs[k+1]),
                                     method='bounded', options={'xatol': 1e-7})
            candidates.append((math.exp(result.x), -result.fun))
    # Include every finite grid point so no sampled value can be missed.
    k = int(np.argmax(values))
    candidates.append((math.exp(logs[k]), float(values[k])))
    tolerance = 1e-10 * max(abs(candidates[0][1]), abs(candidates[1][1]), 1e-12)
    best = candidates[0]
    for candidate in candidates[1:]:
        if candidate[1] > best[1] + tolerance:
            best = candidate
    return best, dict(log_low=float(logs[0]), log_high=float(logs[-1]),
                      candidates=len(candidates), grid_size=grid_size)


def run_contest(cid, order=24, grid_size=65):
    destination = DEST / 'contests' / f'{cid}.json'
    accepted = OUT/'estimation/full'/f'contest_{cid}'/'accepted.json'
    record = json.loads(accepted.read_text())
    for path, sha in zip(record['chain_paths'], record['chain_sha256']):
        if digest(path) != sha:
            raise RuntimeError(f'Changed chain: {path}')
    rawpath = cf.DATA_DIR/f'contest_{cid}.json'
    signature = dict(source_sha256=digest(__file__), accepted_sha256=digest(accepted),
                     input_sha256=digest(rawpath), quadrature_order=order, grid_size=grid_size)
    if destination.exists():
        saved = json.loads(destination.read_text())
        if saved['signature'] == signature:
            return saved
    cf.FIT_ROOT = OUT/'estimation'
    draws = cf.read_draws(cid)
    raw = json.loads(rawpath.read_text())
    theta = float(raw['theta']) / cf.PRIZE_DIVISOR
    T0 = float(raw['N_Delta'])*float(raw['Delta2f'])
    current = Output(draws, T0, order*2)
    target = current.no_effort + theta*current.value(draws['lambda'])
    sign = draws['mu_0']*(1/draws['c_i']-1/draws['c_j'])
    decrease, increase = float(np.mean(sign>0)), float(np.mean(sign<0))
    rows = []
    for multiple in TIMES:
        model = Output(draws, T0*multiple, order)
        (precision, A), audit = optimize(model, grid_size)
        # Independently double time quadrature and the global search grid.
        check = Output(draws, T0*multiple, order*2)
        (p2, A2), audit2 = optimize(check, 2*grid_size-1)
        A_at_first = check.value(precision)
        quadrature_error = abs(A_at_first-A)
        grid_error = A2-A_at_first
        if quadrature_error > 2e-5*max(abs(A2), 1e-12):
            raise RuntimeError(f'Quadrature refinement failed {cid} {multiple}: {quadrature_error/A2}')
        if grid_error > 1e-7*max(abs(A2), 1e-12):
            raise RuntimeError(f'Precision search refinement failed {cid} {multiple}')
        # Publish the refined calculation, including its full-posterior average.
        precision, A = p2, A2
        A_current, A_full = check.value(draws['lambda']), check.value(math.inf)
        B0 = check.no_effort
        required = max(0., (target-B0)/A)
        full_required = max(0., (target-B0)/A_full)
        if required > full_required + 1e-9*theta:
            raise RuntimeError('Optimized precision worse than full disclosure')
        kind = ('no_feedback_limit' if precision == 0 else
                'full_disclosure' if math.isinf(precision) else 'finite_precision')
        rows.append(dict(contest_id=cid, time_multiplier=multiple, duration=T0*multiple,
            observed_duration=T0, observed_prize=theta, posterior_draws=len(sign),
            precision=None if math.isinf(precision) else precision, precision_kind=kind,
            no_effort_output=B0, marginal_prize_output=A,
            full_marginal_prize_output=A_full, current_marginal_prize_output=A_current,
            target_current_output=target, required_prize=required,
            required_prize_ratio=required/theta, full_required_prize_ratio=full_required/theta,
            output_gain_pct=100*((B0+theta*A)/target-1),
            decrease_probability=decrease, increase_probability=increase,
            quadrature_relative_error=quadrature_error/max(abs(A),1e-30),
            grid_relative_error=max(0.,grid_error)/max(abs(A),1e-30),
            achieved_quality=B0+required*A, audit=audit2))
    saved = dict(signature=signature, rows=rows)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(saved, indent=2, allow_nan=False)+'\n')
    return saved


def summarize(rows):
    frame = pd.DataFrame([{k:v for k,v in row.items() if k!='audit'} for row in rows])
    frame.to_csv(DEST/'contest_frontiers.csv', index=False)
    fixed = frame[frame.time_multiplier==1].copy()
    fixed['direction'] = np.where(fixed.increase_probability>.75, 'Increase precision',
                         np.where(fixed.decrease_probability>.75, 'Decrease precision', 'Direction uncertain'))
    fixed.to_csv(DEST/'fixed_precision.csv', index=False)
    policies, comparisons, summary = [], [], []
    for delta in DELTAS:
        selected = []
        for cid, group in frame.groupby('contest_id'):
            costs = group.required_prize_ratio + delta*group.time_multiplier**2
            best = group.loc[costs.idxmin()]
            best_cost = float(costs.min())
            full_costs = group.full_required_prize_ratio + delta*group.time_multiplier**2
            full_cost = float(full_costs.min())
            choice = 'zero_prize' if best.required_prize_ratio==0 else best.precision_kind
            retained = best_cost >= 1+delta-1e-10
            if retained:
                best_cost, choice = 1+delta, 'observed'
            row = dict(delta=delta, contest_id=int(cid), choice=choice,
                time_multiplier=1. if retained else float(best.time_multiplier),
                prize_ratio=1. if retained else float(best.required_prize_ratio),
                cost_saving_pct=100*(1-best_cost/(1+delta)),
                decrease_supported=bool(best.decrease_probability>.75 and choice not in ('zero_prize','observed')),
                limiting=choice=='no_feedback_limit',
                additional_saving_pct=100*(1-best_cost/full_cost) if full_cost>0 else 0.,
                achieved_quality=float(best.achieved_quality), target_quality=float(best.target_current_output))
            if not retained and row['achieved_quality'] < row['target_quality']-1e-10:
                raise RuntimeError('Quality constraint failed')
            if best_cost > full_cost+1e-9:
                raise RuntimeError('Full disclosure comparator failed')
            selected.append(row); policies.append(row)
            comparisons.append(dict(delta=delta, contest_id=int(cid), full_cost=full_cost,
                                    selected_cost=best_cost, full_time_multiplier=float(group.loc[full_costs.idxmin()].time_multiplier)))
        work = pd.DataFrame(selected)
        lower = work[work.choice.isin(['finite_precision','no_feedback_limit'])]
        summary.append(dict(delta=delta, **{kind:int((work.choice==kind).sum()) for kind in
            ['full_disclosure','finite_precision','no_feedback_limit','zero_prize','observed']},
            median_time=float(work.time_multiplier.median()), median_prize=float(work.prize_ratio.median()),
            median_saving=float(work.cost_saving_pct.median()),
            median_additional_saving=float(lower.additional_saving_pct.median()) if len(lower) else None,
            decrease_supported=int(work.decrease_supported.sum())))
    pd.DataFrame(policies).to_csv(DEST/'joint_policies.csv', index=False)
    pd.DataFrame(comparisons).to_csv(DEST/'full_disclosure_comparison.csv', index=False)
    pd.DataFrame(summary).to_csv(DEST/'joint_summary.csv', index=False)
    report = dict(contests=len(fixed), quadrature_max_relative_error=float(frame.quadrature_relative_error.max()),
        grid_max_relative_error=float(frame.grid_relative_error.max()),
        fixed_classification=fixed.precision_kind.value_counts().to_dict(),
        fixed_direction=fixed.direction.value_counts().to_dict(),
        fixed_gain_quantiles=fixed.output_gain_pct.quantile([.25,.5,.75]).tolist(),
        joint_summary=summary,
        scope='Asymmetric first-order output; initial latent distribution fixed per posterior draw; unchanged accepted posterior.',
        boundary='Zero precision is the no-feedback limit, not an attained positive-feedback design. Savings at this endpoint are limiting bounds.',
        source_sha256=digest(__file__))
    (DEST/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--contest', type=int)
    parser.add_argument('--order', type=int, default=24)
    parser.add_argument('--grid-size', type=int, default=65)
    args = parser.parse_args()
    manifest = OUT/'estimation/accepted_chain_sources.csv'
    ids = sorted(int(r['contest_id']) for r in csv.DictReader(manifest.open()) if r['mode']=='full')
    rows = []
    for k, cid in enumerate(ids):
        if args.contest and cid!=args.contest:
            continue
        result = run_contest(cid, args.order, args.grid_size)
        rows.extend(result['rows'])
        print(f'{k+1}/73 contest {cid}: '+','.join(r['precision_kind'] for r in result['rows']), flush=True)
    if not args.contest:
        print(json.dumps(summarize(rows), indent=2), flush=True)


if __name__ == '__main__':
    main()
