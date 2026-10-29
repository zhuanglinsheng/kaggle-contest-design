"""Resolve the missing table comparator using already computed baseline draws.

This is a direct fine-grid contrast on all 128 original coarse-group draws,
not a new posterior or an independent validation campaign. Baseline solves
are forbidden; candidate solves reuse the unchanged appendix point method.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math

import numpy as np
from scipy.stats import t

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('fixed', HERE/'fixed_precision_table.py')
fixed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixed)


def evaluate(prize):
    folder = HERE/'contests/6322'
    quad_path = folder/'quadrature.json'
    quad = json.loads(quad_path.read_text())
    accepted_path = Path(quad['provenance']['accepted'])
    assert fixed.sha(accepted_path) == quad['provenance']['accepted_sha256']
    accepted = json.loads(accepted_path.read_text())
    for path, expected in zip(accepted['chain_paths'], accepted['chain_sha256']):
        assert fixed.sha(path) == expected
    source = HERE/'fixed_precision/6322.json'
    appendix = json.loads(source.read_text())
    assert appendix['status'] == 'completed' and appendix['posterior_draws'] == 128
    assert appendix['accepted_sha256'] == fixed.sha(accepted_path)
    observed = fixed.Design(**{k: v for k, v in appendix['observed'].items()
                              if k != 'precision'})
    candidate = fixed.Design(prize, 2*observed.horizon, math.inf)
    engine = fixed.SimilarityOutput(HERE/'build')
    groups = [[fixed.Draw(**row) for row in group] for group in quad['draws']['coarse']]
    assert len(groups) == 4 and all(len(g) == 32 for g in groups)
    baseline = []
    baseline_caches = []
    for group in groups:
        for draw in group:
            identity = dict(draw=vars(draw), design=observed.record(), levels=fixed.LEVELS,
                            similarity_source=engine.source_sha256,
                            full_source=fixed.sha('pipeline/theory/asymmetric_full_output.py'),
                            script_source=fixed.sha(fixed.__file__))
            key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            path = folder/'point_cache'/f'{key}.json'
            result = json.loads(path.read_text())  # Never solve a baseline.
            assert result['resolved']
            baseline.append(result)
            baseline_caches.append(dict(path=str(path), sha256=fixed.sha(path)))
    columns = [fixed.point(draw, candidate, engine, folder) for group in groups for draw in group]
    evidence = dict(contest_id=6322, design=candidate.record(), posterior_draws=128,
                    baseline_caches=baseline_caches, candidate_results=columns,
                    quadrature_sha256=fixed.sha(quad_path),
                    accepted_sha256=fixed.sha(accepted_path),
                    script_sha256=fixed.sha(__file__),
                    fixed_script_sha256=fixed.sha(fixed.__file__),
                    source_appendix_sha256=fixed.sha(source),
                    scope='Required table comparator only; direct fine-grid paired contrast; '
                          'all128 original joint draws; existing baseline caches reused')
    if not all(row['resolved'] for row in columns):
        result = dict(status='unresolved', passed=False,
                      failed_draw_positions=[i for i, row in enumerate(columns) if not row['resolved']])
    else:
        gaps = np.array([a['B']-b['B'] for a, b in zip(columns, baseline)]).reshape(4, 32)
        terms = np.var(gaps, axis=1, ddof=1)/32/16
        variance = float(np.sum(terms))
        denominator = float(np.sum(terms**2/31))
        df = variance**2/denominator if denominator else 1e9
        se = math.sqrt(variance)
        radius = float(np.mean([a['radius']+b['radius'] for a, b in zip(columns, baseline)]))
        mean = float(np.mean(gaps))
        critical = float(t.ppf(.975, df))
        margin = mean-critical*se-radius
        result = dict(status='evaluated', passed=margin >= 0, mean_gap=mean,
                      mcse=se, degrees_freedom=df, critical_value=critical,
                      numerical_radius=radius, lower_diagnostic_margin=margin,
                      evaluation_method='direct128_original_game_paired_fine_grid',
                      scope='Conditional original empirical posterior integration; '
                            'same95percent diagnostic and numerical-radius subtraction; '
                            'not a strict continuous error bound')
    evidence['result'] = result
    tag = hashlib.sha256(json.dumps(candidate.record(), sort_keys=True).encode()).hexdigest()
    path = HERE/'required_baseline_repairs'/f'6322_direct_comparator_{tag}.json'
    fixed.save(path, evidence)
    print(json.dumps(dict(evidence=str(path), result=result)), flush=True)
    return result, path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prize', type=float, required=True)
    args = parser.parse_args()
    evaluate(args.prize)
