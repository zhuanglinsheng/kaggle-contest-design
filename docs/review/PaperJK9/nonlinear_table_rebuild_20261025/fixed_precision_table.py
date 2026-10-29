"""Recompute the existing fixed-prize/fixed-duration appendix table.

Use the original posterior and nonlinear game, with the same finite feedback
grid as the main table. Grid refinement is an accuracy diagnostic, not a
continuous-error bound or a certificate of the best unsampled precision.
"""
from pathlib import Path
from pipeline import solver_cache
import argparse
import concurrent.futures as futures
import fcntl
import hashlib
import json
import math
import os
import time

import numpy as np

from pipeline.empirical.nonlinear_design import Draw, Design, Resolution, save
from pipeline.empirical.nonlinear_design_similarity import SimilarityOutput
from pipeline.empirical.nonlinear_value_stream import checked_output

HERE = Path(__file__).resolve().parent
DEST = HERE/'fixed_precision'
LEVELS = ((.05, 128, 4001, 80), (.025, 256, 8001, 160),
          (.0125, 512, 16001, 320))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def point(draw, design, engine, folder):
    identity = dict(draw=vars(draw), design=design.record(), levels=LEVELS,
                    similarity_source=engine.source_sha256,
                    full_source=sha('pipeline/theory/asymmetric_full_output.py'),
                    script_source=sha(__file__))
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = folder/'point_cache'/f'{key}.json'
    cached = solver_cache.get(path)
    if cached is not None:
        return cached
    trials = []
    for dx, nt, nodes, gh in LEVELS:
        try:
            trials.append(engine.evaluate(draw, design, spacing=dx, steps=nt,
                                          full_nodes=nodes, quadrature=gh))
        except Exception as error:
            trials.append(dict(error=repr(error)))
    resolved = all('B' in r for r in trials)
    if resolved:
        changes = [abs(b['B']-a['B']) for a, b in zip(trials, trials[1:])]
        resolved = bool(changes[-1] <= .8*changes[0]+1e-8*max(1., abs(trials[-1]['B'])))
    result = dict(identity=identity, resolved=resolved, trials=trials)
    if resolved:
        result.update(B=trials[-1]['B'], radius=3*changes[-1]+trials[-1]['budget'])
    else:
        # Only repair points needed for this table, reusing the established
        # original value-HJB solver and existing main-table caches.
        fallback = checked_output(draw, design, cache_dir=folder/'fine_cache',
                                  resolution=Resolution())
        result['value_fallback'] = fallback
        if fallback['resolved']:
            result.update(resolved=True, B=fallback['B'],
                          radius=fallback['diagnostic_radius'])
    save(path, result)
    return result


def compute(cid):
    folder = HERE/f'contests/{cid}'
    quadrature_path = folder/'quadrature.json'
    quad = json.loads(quadrature_path.read_text())
    accepted_path = Path(quad['provenance']['accepted'])
    assert sha(accepted_path) == quad['provenance']['accepted_sha256']
    accepted = json.loads(accepted_path.read_text())
    for path, expected in zip(accepted['chain_paths'], accepted['chain_sha256']):
        assert sha(path) == expected
    from pipeline.config import MAIN_DATA_DIR
    from pipeline.empirical.counterfactual import PRIZE_DIVISOR
    raw_path = MAIN_DATA_DIR/f'contest_{cid}.json'
    assert sha(raw_path) == accepted['configuration']['raw_input_sha256']
    raw = json.loads(raw_path.read_text())
    observed = Design(float(raw['theta'])/PRIZE_DIVISOR,
                      float(raw['N_Delta'])*float(raw['Delta2f']))
    draws = [Draw(**d) for group in quad['draws']['coarse'] for d in group]
    assert len(draws) == 128 and {d.chain for d in draws} == {0, 1, 2, 3}
    reference = float(np.median([d.observed_precision for d in draws]))
    designs = [observed, Design(observed.prize, observed.horizon, math.inf)]
    designs += [Design(observed.prize, observed.horizon, reference*r)
                for r in (.001, .03, .3, 3.)]
    engine = SimilarityOutput(HERE/'build')
    columns = []
    for design in designs:
        column = [point(d, design, engine, folder) for d in draws]
        columns.append(column)
        print('FIXED PRECISION', cid, design.record()['precision'],
              'resolved', sum(r['resolved'] for r in column), '/128', flush=True)
        save(DEST/f'{cid}_progress.json', dict(contest_id=cid,
             completed_precisions=len(columns), total_precisions=len(designs)))
    if not all(r['resolved'] for r in columns[0]):
        raise RuntimeError('Observed-policy quality unresolved; do not drop posterior draws')
    base = np.array([r['B'] for r in columns[0]])
    base_radius = np.array([r['radius'] for r in columns[0]])
    comparisons = []
    for design, column in zip(designs, columns):
        complete = all(r['resolved'] for r in column)
        row = dict(design=design.record(), complete=complete,
                   failed_draws=[i for i, r in enumerate(column) if not r['resolved']])
        if complete:
            gaps = np.array([r['B'] for r in column])-base
            radius = (np.array([r['radius'] for r in column])+base_radius
                      if design != observed else np.zeros(len(base)))
            row.update(mean_quality=float(np.mean(base+gaps)),
                       mean_gain=float(np.mean(gaps)),
                       numerical_radius=float(np.mean(radius)),
                       mcse=float(np.std(gaps, ddof=1)/math.sqrt(len(gaps))),
                       supported=design == observed or float(np.mean(gaps-radius)) > 0)
        comparisons.append(row)
    eligible = [i for i, row in enumerate(comparisons) if row.get('supported')]
    chosen = max(eligible, key=lambda i: (comparisons[i]['mean_quality'], -i))
    selected = designs[chosen]
    # Preserve the appendix's posterior direction summary. A draw contributes
    # only if its preferred direction survives the numerical intervals. An
    # unresolved candidate makes that draw's direction uncertain.
    counts = dict(increase=0, decrease=0, uncertain=0)
    for index, draw in enumerate(draws):
        rows = [column[index] for column in columns]
        if not all(r['resolved'] for r in rows):
            counts['uncertain'] += 1
            continue
        lower = np.array([r['B']-r['radius'] for r in rows])
        upper = np.array([r['B']+r['radius'] for r in rows])
        possible = [i for i in range(len(rows)) if upper[i] >= max(lower)]
        directions = set()
        for i in possible:
            precision = designs[i].precision
            if precision is None:
                directions.add('uncertain')
            elif precision > draw.observed_precision:
                directions.add('increase')
            elif precision < draw.observed_precision:
                directions.add('decrease')
            else:
                directions.add('uncertain')
        counts[next(iter(directions)) if len(directions) == 1 else 'uncertain'] += 1
    group = ('increase' if counts['increase']/len(draws) > .75 else
             'decrease' if counts['decrease']/len(draws) > .75 else 'uncertain')
    # Retaining observed feedback is a finite interior option. The smallest
    # searched positive precision is NOT called an attained no-feedback limit.
    regime = ('full_disclosure' if selected.precision == math.inf else
              'lowest_searched_precision' if selected == designs[2] else 'interior_precision')
    result = dict(contest_id=cid, status='completed', observed=observed.record(),
                  selected=selected.record(), regime=regime, direction_group=group,
                  direction_draw_counts=counts, comparisons=comparisons,
                  quality_gain_pct=100*comparisons[chosen]['mean_gain']/float(np.mean(base)),
                  posterior_draws=len(draws), precision_reference=reference,
                  accepted_sha256=sha(accepted_path), raw_input_sha256=sha(raw_path),
                  script_source_sha256=sha(__file__),
                  scope='Original nonlinear game; best numerically supported searched precision; '
                        'original approximate Bayesian posterior and initial distribution fixed; '
                        'unresolved candidates retained and not called infeasible')
    save(DEST/f'{cid}.json', result)
    return result


def aggregate():
    results = [json.loads(p.read_text()) for p in DEST.glob('[0-9]*.json')
               if not p.name.endswith('_progress.json')]
    from csv import DictReader
    expected = {int(r['contest_id']) for r in DictReader(Path(
        'results/reestimation/lambda100_sd225/estimation/accepted_chain_sources.csv').open())
        if r['mode'] == 'full'}
    if len(results) != 73 or {r['contest_id'] for r in results} != expected:
        raise ValueError(f'Fixed-precision table incomplete: {len(results)}/73')
    assert all(r['status'] == 'completed' and r['posterior_draws'] == 128 for r in results)
    counts = {k: sum(r['regime'] == k for r in results) for k in
              ['full_disclosure', 'interior_precision', 'lowest_searched_precision']}
    directions = {k: sum(r['direction_group'] == k for r in results)
                  for k in ['increase', 'decrease', 'uncertain']}
    percentiles = {}
    for k in ['all', 'increase', 'decrease', 'uncertain']:
        values = [r['quality_gain_pct'] for r in results
                  if k == 'all' or r['direction_group'] == k]
        percentiles[k] = np.percentile(values, [25, 50, 75]).tolist() if values else None
    report = dict(ready=True, contests=73, counts=counts, directions=directions,
                  percentiles=percentiles,
                  contests_with_unresolved_candidates=[r['contest_id'] for r in results
                      if any(not c['complete'] for c in r['comparisons'])])
    save(DEST/'table_numbers.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--contest', type=int)
    parser.add_argument('--workers', type=int, default=1, choices=range(1, 7))
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--aggregate', action='store_true')
    args = parser.parse_args()
    DEST.mkdir(exist_ok=True)
    if args.aggregate:
        print(json.dumps(aggregate(), indent=2))
        raise SystemExit
    with (DEST/'controller.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = dict(pid=os.getpid(), status='running', completed={}, active={}, failed={})
        jobs = {}
        with futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            while True:
                ready = []
                for path in sorted(HERE.glob('contests/*/result.json'),
                                   key=lambda p: int(p.parent.name)):
                    cid = int(path.parent.name)
                    if args.contest is not None and cid != args.contest:
                        continue
                    dest = DEST/f'{cid}.json'
                    if dest.exists() and json.loads(dest.read_text()).get('status') == 'completed':
                        state['completed'][str(cid)] = str(dest)
                    elif str(cid) not in state['active'] and str(cid) not in state['failed']:
                        ready.append(cid)
                while ready and len(jobs) < args.workers:
                    cid = ready.pop(0)
                    jobs[pool.submit(compute, cid)] = cid
                    state['active'][str(cid)] = True
                save(DEST/'status.json', state)
                if jobs:
                    done, _ = futures.wait(jobs, timeout=5, return_when=futures.FIRST_COMPLETED)
                    for job in done:
                        cid = jobs.pop(job)
                        state['active'].pop(str(cid), None)
                        try:
                            job.result()
                            state['completed'][str(cid)] = str(DEST/f'{cid}.json')
                        except Exception as error:
                            state['failed'][str(cid)] = repr(error)
                    save(DEST/'status.json', state)
                    continue
                main = json.loads((HERE/'design_status.json').read_text())
                if not args.watch or main['status'].startswith('completed'):
                    break
                time.sleep(5)
        state['status'] = 'completed' if len(state['completed']) == 73 else 'incomplete'
        save(DEST/'status.json', state)
        if len(state['completed']) == 73:
            aggregate()
