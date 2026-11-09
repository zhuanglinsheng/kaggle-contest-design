"""Regenerate the forty fixed nonlinear-policy recovery experiments."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import tomllib

import numpy as np

from pipeline.empirical import sigma_lambda_posterior as posterior
from pipeline.empirical import sigma_lambda_trials as trials
from pipeline.empirical.unified_prior_interval_report import calibrated_intervals, recovery_report
from pipeline.jk9_fit import plan, training_data
from pipeline.publication import ROOT, digest, write_json

DESIGN = Path(__file__).with_name('recovery_design.toml')
REPLICATES = (2, 3, 4, 5)


def generate(cid, entry, replicates, output):
    row = plan('main', 'full', cid)[0]
    if row['input_sha256'] != entry['input_sha256']:
        raise ValueError(f'Frozen contest input differs for {cid}')
    data = training_data(json.loads((ROOT / row['input_snapshot']).read_text()), 'full')
    truth = {k: entry[k] for k in ('c_i', 'c_j', 'sigma', 'lambda', 'mu_0', 'log_r', 'r')}
    z = np.r_[np.log([truth[k] for k in ('c_i', 'c_j', 'sigma', 'lambda')]),
              truth['mu_0'], truth['log_r']]
    provenance = dict(simulation_design_sha256=digest(DESIGN), input_sha256=row['input_sha256'])
    for replicate in replicates:
        folder = output/'recovery'/f'{cid}_{replicate}'
        if (folder/'data.json').exists():
            saved = json.loads((folder/'truth.json').read_text())
            if saved['truth']!=truth or any(saved.get(k)!=v for k,v in provenance.items()):
                raise ValueError(f'Existing simulation differs from the frozen design: {folder}')
    trials.simulate(cid, data, z, truth, provenance, replicates, output)
    for replicate in replicates:
        folder = output/'recovery'/f'{cid}_{replicate}'
        data_hash = digest(folder/'data.json')
        expected = entry['historical_data_sha256'][replicate-2]
        write_json(folder/'generation_audit.json', dict(data_sha256=data_hash,
                   historical_data_sha256=expected, matches_historical=data_hash==expected))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('generate', 'fit', 'report'))
    parser.add_argument('--output', type=Path, default=Path('tmp/runs/nonlinear_recovery'))
    parser.add_argument('--contests', nargs='+', type=int, choices=trials.IDS, default=trials.IDS)
    parser.add_argument('--replicates', nargs='+', type=int, choices=REPLICATES, default=REPLICATES)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--samples', type=int, default=30000)
    parser.add_argument('--posterior-summary', type=Path,
                        help='Optional full empirical posterior summary CSV for calibrated intervals')
    args = parser.parse_args(argv)
    if args.workers < 1 or args.samples < 400:
        parser.error('Use at least one worker and 400 draws per chain')
    design = tomllib.loads(DESIGN.read_text())
    output = args.output.resolve()
    contests, replicates = dict.fromkeys(args.contests), dict.fromkeys(args.replicates)
    if args.action == 'generate':
        for cid in contests:
            generate(cid, design['contests'][str(cid)], replicates, output)
    elif args.action == 'fit':
        jobs = [('recovery', f'{cid}_{replicate}', 'lambda_external', output)
                for cid in contests for replicate in replicates]
        for job in jobs:
            posterior.prepare(*job)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(posterior.chain, [(*job, j, args.samples)
                                          for job in jobs for j in range(1, 5)]))
        for job in jobs:
            posterior.summarize(*job)
        rejected = [case for _, case, variant, _ in jobs if not json.loads(
            (output / 'posterior/recovery' / case / variant / 'summary.json').read_text())['passed']]
        if rejected:
            raise SystemExit(f'Recovery posterior diagnostics failed for {", ".join(rejected)}')
    else:
        if set(contests) != set(trials.IDS) or set(replicates) != set(REPLICATES):
            parser.error('The recovery report requires the complete forty-case design')
        report = recovery_report(output, output / 'validation', design['calibration'], trials.IDS)
        if args.posterior_summary:
            calibrated_intervals(args.posterior_summary, output/'validation', design['calibration'])
        write_json(output / 'validation/interval_and_recovery_report.json',
                   dict(parameters=report, simulation_design_sha256=digest(DESIGN)))


if __name__ == '__main__':
    main()
