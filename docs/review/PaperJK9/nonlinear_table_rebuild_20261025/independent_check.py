"""Independent posterior quadrature for a frozen original-game redesign.

Compare it with the original policy and with full disclosure at the identical
prize and duration. No design fitting or nonlinear parameter reestimation is
performed with these independent draws. Unresolved draws prevent a comparison.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from pipeline.config import MAIN_DATA_DIR
from pipeline.empirical.counterfactual import PRIZE_DIVISOR
from pipeline.empirical.nonlinear_design import Draw, Design, Resolution, checked_output, save
from pipeline.empirical.nonlinear_value_stream import checked_output
from pipeline.empirical.nonlinear_design_recovery import recover
from pipeline.empirical.nonlinear_design_similarity import SimilarityOutput

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(groups, names, values):
    coarse, correction, radius, failures = [], [], [], []
    for chain in range(4):
        c, f = [], []
        for d in groups['coarse'][chain]:
            a, b = (values[('coarse', d.chain, d.index, name)] for name in names)
            if a['resolved'] and b['resolved']:
                c.append(a['B']-b['B'])
            else:
                failures.append(dict(level='coarse', chain=d.chain, index=d.index))
        for d in groups['fine'][chain]:
            ca, cb = (values[('coarse', d.chain, d.index, name)] for name in names)
            fa, fb = (values[('fine', d.chain, d.index, name)] for name in names)
            if all(v['resolved'] for v in [ca, cb, fa, fb]):
                f.append(fa['B']-fb['B']-(ca['B']-cb['B']))
                radius.append(fa['diagnostic_radius']+fb['diagnostic_radius'])
            else:
                failures.append(dict(level='fine', chain=d.chain, index=d.index))
        coarse.append(c)
        correction.append(f)
    if failures:
        return dict(resolved=False, failed_draws=failures)
    samples = coarse+correction
    terms = [float(np.var(g, ddof=1)/len(g)/16) for g in samples]
    variance = sum(terms)
    denominator = sum(v*v/(len(g)-1) for v, g in zip(terms, samples))
    degrees = variance*variance/denominator if denominator > 0 else 1e9
    critical = float(student_t.ppf(.975, degrees))
    mean = float(np.mean([np.mean(g) for g in coarse])+np.mean([np.mean(g) for g in correction]))
    error = critical*math.sqrt(variance)+float(np.mean(radius))
    return dict(resolved=True, mean_gap=mean, mcse=math.sqrt(variance),
                numerical_radius=float(np.mean(radius)), degrees_freedom=degrees,
                lower_diagnostic=mean-error, upper_diagnostic=mean+error,
                direction='positive' if mean-error > 0 else 'negative' if mean+error < 0 else 'uncertain',
                scope='Independent conditional posterior quadrature; grid diagnostic is not a strict PDE bound')


def run(cid, candidate, *, coarse_count=32, fine_count=4):
    sources = [HERE.parent/f'reconstruction_20261025/estimation/full/contest_{cid}/accepted.json',
               HERE/f'posterior/contest_{cid}/accepted.json']
    accepted_path = next(p for p in sources if p.exists())
    accepted = json.loads(accepted_path.read_text())
    raw_path = MAIN_DATA_DIR/f'contest_{cid}.json'
    assert accepted['diagnostics']['passed'] and accepted['configuration']['raw_input_sha256'] == sha(raw_path)
    raw = json.loads(raw_path.read_text())
    observed = Design(float(raw['theta'])/PRIZE_DIVISOR, float(raw['N_Delta'])*float(raw['Delta2f']))
    designs = dict(candidate=candidate, original=observed,
                   matched_full=Design(candidate.prize, candidate.horizon, math.inf))
    seed = 202610250000+cid+999999
    rng = np.random.default_rng(seed)
    groups = dict(coarse=[], fine=[])
    for chain, (path, expected_sha) in enumerate(zip(accepted['chain_paths'], accepted['chain_sha256'])):
        assert sha(path) == expected_sha
        data = pd.read_csv(path, comment='#', usecols=['c_i', 'c_j', 'sigma', 'lambda', 'mu_0'])
        for name, count in [('coarse', coarse_count), ('fine', fine_count)]:
            draws = []
            for index in rng.integers(0, len(data), size=count):
                row = data.iloc[int(index)]
                draws.append(Draw(float(row.c_i), float(row.c_j), float(row.sigma),
                                  float(row['lambda']), float(row.mu_0), chain, int(index)))
            groups[name].append(draws)
    identity = dict(contest_id=cid, candidate=candidate.record(), seed=seed,
                    accepted_sha256=sha(accepted_path), coarse_per_chain=coarse_count,
                    fine_per_chain=fine_count, source_sha256=sha(__file__))
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    folder = HERE/f'independent_validation/{cid}/{key}'
    save(folder/'draws.json', dict(identity=identity,
                                 groups={k: [[vars(d) for d in g] for g in v] for k, v in groups.items()}))
    engine = SimilarityOutput(HERE/'build')
    values = {}
    for level, chains in groups.items():
        for chain, draws in enumerate(chains):
            for draw in draws:
                for name, design in designs.items():
                    cache_identity = dict(draw=vars(draw), design=design.record(), engine=engine.source_sha256)
                    cache_key = hashlib.sha256(json.dumps(cache_identity, sort_keys=True).encode()).hexdigest()
                    cache_path = folder/'coarse_cache'/f'{cache_key}.json'
                    if cache_path.exists():
                        coarse = json.loads(cache_path.read_text())
                    else:
                        try:
                            coarse = dict(resolved=True, **engine.evaluate(draw, design, spacing=.15, steps=32,
                                                                          full_nodes=2001, quadrature=64))
                        except Exception as error:
                            coarse = dict(resolved=False, error=repr(error))
                        save(cache_path, coarse)
                    values[('coarse', draw.chain, draw.index, name)] = coarse
                    if level == 'fine':
                        fine = checked_output(draw, design, cache_dir=folder/'fine_cache', resolution=Resolution())
                        if not fine['resolved']:
                            fine = recover(draw, design, cache_dir=folder/'recovery', build_dir=HERE/'build')
                        values[('fine', draw.chain, draw.index, name)] = fine
            print('INDEPENDENT', cid, level, chain, 'completed', flush=True)
            save(folder/'progress.json', dict(identity=identity, stage=level, completed_chain=chain))
    results = {f'candidate_minus_{name}': compare(groups, ('candidate', name), values)
               for name in ['original', 'matched_full']}
    record = dict(identity=identity, comparisons=results,
                  designs={k: v.record() for k, v in designs.items()},
                  all_resolved=all(v['resolved'] for v in results.values()),
                  nonlinear_reestimation=False, fixed_before_independent_draws=True)
    save(folder/'result.json', record)
    print(json.dumps(record, indent=2), flush=True)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--contest', type=int, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--coarse-per-chain', type=int, default=32)
    parser.add_argument('--fine-per-chain', type=int, default=4)
    args = parser.parse_args()
    if args.coarse_per_chain < 2 or args.fine_per_chain < 2:
        parser.error('At least two draws per chain and level required for variance estimation')
    record = json.loads(args.candidate.read_text())
    precision = record['precision']
    precision = math.inf if precision == 'full' else None if precision == 'observed' else precision
    run(args.contest, Design(record['prize'], record['horizon'], precision),
        coarse_count=args.coarse_per_chain, fine_count=args.fine_per_chain)
