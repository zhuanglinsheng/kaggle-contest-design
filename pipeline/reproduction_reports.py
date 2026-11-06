"""Connect fresh portable fits to the retained empirical reporting implementation.

No sampling is performed. Historical inputs and publication results are read only.
Run from the repository root; see docs/REPRODUCIBILITY.md for the stage order.
"""
import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from pipeline.publication import ROOT, digest, write_json
from pipeline.config import resolve_manifest_path
from pipeline.jk9_fit import plan, training_data


def stage_fit(row, accepted_root, output):
    accepted = accepted_root / row['scenario'] / row['mode'] / str(row['contest_id']) / 'accepted.json'
    record = json.loads(accepted.read_text())
    config = record['configuration']
    raw = ROOT / row['input_snapshot']
    if not (record['diagnostics']['passed'] and config['model'] == 'D'
            and config['raw_input_sha256'] == row['input_sha256'] == digest(raw)
            and config['mode'] == row['mode'] and config['scenario'] == row['scenario']
            and config['source_model_sha256'] == digest(ROOT / 'pipeline/models/leading_order.stan')
            and config['training_cutoff'] == (json.loads(raw.read_text())['N_Delta'] if row['mode'] == 'full'
                else training_data(json.loads(raw.read_text()), 'early')['N_Delta'])):
        raise ValueError(f'Fit does not match the declared target: {accepted}')
    paths = [resolve_manifest_path(p, accepted) for p in record['chain_paths']]
    if len(paths) != 4 or len(record['chain_sha256']) != 4:
        raise ValueError(f'Four chains required: {accepted}')
    folder = output / 'adapters' / row['scenario'] / row['mode'] / str(row['contest_id'])
    folder.mkdir(parents=True, exist_ok=True)
    for j, (path, expected) in enumerate(zip(paths, record['chain_sha256'], strict=True), 1):
        if digest(path) != expected:
            raise ValueError(f'Chain hash changed: {path}')
        target = folder / f'chain_{j}.csv'
        if target.is_symlink():
            if target.resolve() != path.resolve():
                raise ValueError(f'Adapter already refers to another run: {target}')
        elif target.exists():
            raise FileExistsError(target)
        else:
            target.symlink_to(path.resolve())
    data = json.loads((paths[0].parent / 'data.json').read_text())
    write_json(folder / 'data.json', data)
    return dict(record['summary'], scenario=row['scenario'], accepted_source='fresh_portable_fit', deadline_preserved=1,
                input_sha256=row['input_sha256'], run_dir=str(folder.resolve()))


def empirical(args):
    from pipeline.empirical import unified_prior_outputs as outputs
    from pipeline.empirical import unified_prior_reestimate as original
    from pipeline.empirical import unified_prior_aux_summary as auxiliary
    from pipeline.empirical import precision_design
    from pipeline.jk9 import main as jk9
    jk9(['prepare-inputs'])
    out = args.output.resolve()
    if ROOT / 'tmp' not in out.parents:
        raise ValueError('Reporting output must be below tmp/')
    source = args.accepted_root.resolve() if args.accepted_root else None
    summary_root = args.summary_root.resolve() if args.summary_root else None
    expected = [row for scenario in ('main', 'pair', 'L1', 'L2', 'seed_20261008', 'seed_20261009')
                for mode in (('full',) if scenario in ('pair', 'L1', 'L2') else ('full', 'early'))
                for row in plan(scenario, mode)]
    main_rows = [r for r in expected if r['scenario'] == 'main']
    aux_rows = [r for r in expected if r['scenario'] != 'main']
    assert len(main_rows) == 146 and len(aux_rows) == 251
    fit_dir = out / 'estimation'
    fit_dir.mkdir(parents=True, exist_ok=True)
    if source:
        main = pd.DataFrame([stage_fit(r, source, out) for r in main_rows])
        aux = pd.DataFrame([stage_fit(r, source, out) for r in aux_rows])
        main.to_csv(fit_dir / 'posterior_summary_final.csv', index=False)
        main[['contest_id', 'mode', 'accepted_source', 'run_dir', 'input_sha256']].to_csv(fit_dir / 'accepted_chain_sources.csv', index=False)
    else:
        main = pd.read_csv(summary_root / 'estimation/posterior_summary_final.csv')
        aux = pd.read_csv(summary_root / 'auxiliary/posterior_summary.csv')
        main.to_csv(fit_dir / 'posterior_summary_final.csv', index=False)
    assert len(main) == 146 and len(aux) == 251
    original.OUT = out
    outputs.OUT = out
    ch, report, influence, audit, precision = outputs.configure()
    ch.main()
    influence.main()
    if source:
        report.main()
        audit.main()
    # Materialize the scenario-specific frozen data for the auxiliary report.
    inputs = out / 'inputs'
    for row in aux_rows:
        target = inputs / row['scenario'] / f"contest_{row['contest_id']}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        raw = ROOT / row['input_snapshot']
        if digest(raw) != row['input_sha256']:
            raise ValueError(f'Frozen input hash changed: {raw}')
        shutil.copyfile(raw, target)
    if not source:
        # Only non-posterior auxiliary quantities are needed by the manuscript.
        # Compute these directly with the same benchmark/regression functions.
        from scipy.stats import spearmanr
        details = []
        regressions = []
        base = main[main['mode'] == 'full'].set_index('contest_id')
        scales = []
        for scenario in ('pair', 'L1', 'L2'):
            selected = aux[aux.scenario == scenario]
            rows = []
            for row in selected.itertuples():
                raw = json.loads((inputs / scenario / f'contest_{row.contest_id}.json').read_text())
                params = pd.Series({p: getattr(row, p + '_median') for p in original.NAMES})
                rows.append(dict(scenario=scenario, contest_id=row.contest_id, **params.to_dict(), **ch.benchmark_row(raw, params)))
            frame = pd.DataFrame(rows)
            details.extend(rows)
            for p in ('lambda', 'sigma'):
                import numpy as np
                frame['x'] = np.log1p(frame[p]); frame['y'] = np.log1p(frame[p + '_bench'])
                result, fit = ch.fit_row(frame, 'x', 'y', scenario + '_' + p + '_benchmark')
                regressions.append(result)
                if scenario == 'pair':
                    dest = out / 'auxiliary/reports'
                    dest.mkdir(parents=True, exist_ok=True)
                    outputs.blue_plot(frame, 'x', 'y', fit, dest / f'validate_{p}_pair.pdf',
                        rf'$\log(1+\hat{{\{p}}})$', rf'$\log(1+\{p}^{{bench}})$', show_statistics=True, color='black')
            if scenario != 'pair':
                frame = frame.set_index('contest_id')
                scale = float(raw.get('score_scale_a', 1))
                scales.append(dict(scenario=scenario, scale_a=scale,
                    lambda_rank_correlation=float(spearmanr(scale**2 * frame['lambda'], base.loc[frame.index, 'lambda_median']).statistic),
                    sigma_rank_correlation=float(spearmanr(frame['sigma'] / scale, base.loc[frame.index, 'sigma_median']).statistic)))
        dest = out / 'auxiliary/reports'; dest.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(details).to_csv(dest / 'contest_diagnostics.csv', index=False)
        pd.DataFrame(regressions).to_csv(dest / 'regressions.csv', index=False)
        write_json(dest / 'report.json', dict(scales=scales, scope='Recomputed manuscript regressions and parameter rank correlations from retained posterior summaries; no chain-only policy calculations'))
    else:
        precision.main()
        auxiliary.BASE_OUT = out
        auxiliary.assemble = lambda: aux
        auxiliary.input_directory = lambda scenario: inputs / scenario
        auxiliary.report()
    write_json(out / 'reproduction_scope.json', dict(main_fits=146, auxiliary_fits=251,
        input=str(source or summary_root), mode='fresh_posteriors' if source else 'retained_posterior_summaries',
        historical_results_overwritten=False))
    print(f'Empirical reports written to {out}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['empirical'])
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--accepted-root', type=Path, help='Fresh jk9_fit output root; all 397 accepted fits required')
    source.add_argument('--summary-root', type=Path, help='Retained lambda100_sd225 root; regenerate regressions without draws')
    p.add_argument('--output', type=Path, required=True, help='Separate report output below tmp/')
    args = p.parse_args()
    empirical(args)


if __name__ == '__main__':
    main()
