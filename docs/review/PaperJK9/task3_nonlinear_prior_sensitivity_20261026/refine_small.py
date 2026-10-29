"""Resolve the tiny-saving case with direct fine-grid paired integration.

Keep the same 128 joint posterior draws, replacing noisy 16-draw multilevel
correction with direct fine-grid outputs. No new posterior, design search,
or successful-subset average. All original results remain available.
"""
from pathlib import Path
import concurrent.futures as futures
import json
import math
import numpy as np
from scipy.stats import t as student_t
from run import HERE, SCENARIOS, Evaluator, design, save, sha


def compute(scenario):
    cid = 8219
    result_path = HERE/'results'/f'{cid}_{scenario}.json'
    old = json.loads(result_path.read_text())
    e = Evaluator(cid, scenario, Path(old['provenance']['accepted']))
    selected = design(old['original_selection']['selected'])
    draws = [d for group in e.groups['coarse'] for d in group]
    baselines = []
    for index, draw in enumerate(draws):
        baselines.append(e.fine_output(draw, e.observed))
        if (index+1) % 16 == 0:
            print('DIRECT BASE', scenario, index+1, '/128', flush=True)
    result = dict(contest_id=cid, scenario=scenario, original_result_sha256=sha(result_path),
                  provenance=e.provenance, fine_joint_draws=128, trials=[],
                  protocol='Direct fine-grid paired posterior integration; equal chain weights; no sample deletion')
    # The initial published design, plus the already declared prize probes.
    # Test exact original prize only as a feedback-effect diagnostic, not
    # as evidence of a strictly positive cost saving.
    q0 = old['original_selection']['prize_ratio']
    qs = [q0]+[q0+(1-q0)*f for f in (.5, .75, .875, .9375)]+[1.]
    for q in qs:
        candidate = type(selected)(q*e.observed.prize, selected.horizon, selected.precision)
        values = [e.fine_output(draw, candidate) for draw in draws]
        failures = [i for i, (a, b) in enumerate(zip(values, baselines))
                    if not (a['resolved'] and b['resolved'])]
        comparison = dict(resolved=not failures, failed_draws=failures)
        if not failures:
            gaps = np.array([a['B']-b['B'] for a, b in zip(values, baselines)]).reshape(4, 32)
            radii = np.array([a['diagnostic_radius']+b['diagnostic_radius']
                              for a, b in zip(values, baselines)])
            terms = [float(np.var(g, ddof=1)/len(g)/16) for g in gaps]
            var = sum(terms)
            den = sum(t*t/31 for t in terms)
            df = var*var/den if den else 1e9
            mean = float(np.mean(gaps))
            radius = float(np.mean(radii))
            error = float(student_t.ppf(.975, df))*math.sqrt(var)+radius
            comparison.update(mean_gap=mean, mcse=math.sqrt(var), numerical_radius=radius,
                              lower_diagnostic=mean-error, upper_diagnostic=mean+error,
                              direction='positive' if mean-error > 0 else 'negative' if mean+error < 0 else 'uncertain')
        row = dict(prize_ratio=q, design=candidate.record(),
                   cost_saving_pct=100*(1-(q+2)/3), comparison=comparison)
        result['trials'].append(row)
        save(HERE/'small_refinement'/f'{cid}_{scenario}.json', result)
        print('DIRECT TRIAL', scenario, q, comparison.get('direction', 'unresolved'),
              comparison.get('mean_gap'), flush=True)
        if comparison.get('direction') == 'positive' and q < 1:
            result['same_direction_cost_saving_supported'] = True
            result['supported_saving_pct'] = row['cost_saving_pct']
            break
    result['status'] = 'completed'
    result.setdefault('same_direction_cost_saving_supported', False)
    save(HERE/'small_refinement'/f'{cid}_{scenario}.json', result)
    return result


if __name__ == '__main__':
    with futures.ProcessPoolExecutor(max_workers=2) as pool:
        list(pool.map(compute, SCENARIOS))
