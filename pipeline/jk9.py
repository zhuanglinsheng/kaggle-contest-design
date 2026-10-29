"""Portable JK9 publication snapshot and table checks (no sampling or HJB solve).

Frozen research controllers stay in docs/review to preserve their source hashes.
The maintained publication depends on compact evidence, not point caches.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
from pipeline.storage import historical_path, HISTORY, compile_stan

from pipeline.publication import (ROOT, assert_assets, check_files, check_evidence,
                                  digest, load, local, tex_dependencies, write_json)

RELEASE = ROOT / 'results/releases/paperjk9-20261029'
REVIEW = ROOT / 'docs/review/PaperJK9'
TABLE = REVIEW / 'nonlinear_table_rebuild_20261025'


def relocated(value):
    """Resolve historical absolute paths without changing frozen record bytes."""
    path = Path(value)
    if not path.is_absolute():
        return historical_path(local(ROOT, value))
    try:
        path.relative_to(ROOT)
        return historical_path(path)
    except ValueError:
        for folder in ('docs', 'results', 'metakaggle', 'pipeline', 'paper'):
            marker = f'/{folder}/'
            if marker in value:
                return historical_path(local(ROOT, folder + '/' + value.split(marker, 1)[1]))
    raise ValueError(f'Cannot relocate historical path: {value}')


def snapshot():
    """Copy only completed outputs; never replace or recompute scientific results."""
    sources = {}

    def copy(source, target):
        source, target = Path(source), RELEASE / target
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        sources[str(target.relative_to(RELEASE))] = dict(
            source=str(source.relative_to(ROOT)), sha256=digest(source))

    working_table = HISTORY / TABLE.relative_to(ROOT)
    if not working_table.exists():
        working_table = TABLE
    for p in sorted(working_table.glob('contests/*/refined_selections.json')):
        copy(p, f'evidence/main_design/{p.parent.name}.json')
    for p in sorted((working_table / 'fixed_precision').glob('[0-9]*.json')):
        if p.stem.isdigit():
            copy(p, f'evidence/fixed_precision/{p.name}')
    for name in ('table_numbers.json', 'table_readiness.json', 'final_main_audit.json'):
        copy(working_table / name, f'evidence/{name}')
    copy(working_table / 'fixed_precision/table_numbers.json', 'evidence/fixed_precision_summary.json')
    sensitivity = REVIEW / 'task3_nonlinear_prior_sensitivity_20261026'
    for p in sorted(historical_path(sensitivity / 'results').glob('*.json')):
        copy(p, f'evidence/cost_sensitivity/{p.name}')
    for name in ('final_audit.json', 'manifest.json'):
        copy(historical_path(sensitivity / name), f'evidence/cost_sensitivity/{name}')
    copy(historical_path(REVIEW / 'task2_four_class_jointfit_20261025/numerical_regions_four_class_jointfit_bw.pdf'),
         'assets/numerical_regions_four_class_bw.pdf')
    copy(historical_path(REVIEW / 'task2_four_class_jointfit_20261025/metadata.json'), 'evidence/theory_metadata.json')
    copy(historical_path(REVIEW / 'task2_cell_uncertainty_inventory_20261024.csv'), 'evidence/theory_unresolved.csv')
    fits = []
    status = json.loads((working_table / 'posterior_status.json').read_text())
    if status['status'] != 'completed' or len(status['completed']) != 73:
        raise ValueError('73 completed original-posterior records required')
    for cid, value in sorted(status['completed'].items(), key=lambda x: int(x[0])):
        accepted = relocated(value)
        record = json.loads(accepted.read_text())
        if not record['diagnostics']['passed'] or len(record['chain_paths']) != 4:
            raise ValueError(f'Unaccepted posterior: {cid}')
        chains = [relocated(p) for p in record['chain_paths']]
        for path, expected in zip(chains, record['chain_sha256'], strict=True):
            if digest(path) != expected:
                raise ValueError(f'Changed chain: {path}')
        copy(accepted, f'evidence/main_posteriors/{cid}.json')
        fits.append(dict(contest_id=int(cid), accepted_path=str(accepted.relative_to(ROOT)),
                         accepted_sha256=digest(accepted),
                         chain_paths=[str(p.relative_to(ROOT)) for p in chains],
                         chain_sha256=record['chain_sha256']))
    write_json(RELEASE / 'accepted_fits.json', dict(schema_version=1, fits=fits,
               scope='73 posteriors used by JK9 nonlinear tables; paths relative to checkout'))
    write_json(RELEASE / 'evidence_sources.json', dict(schema_version=1, files=sources,
               scope='Byte-preserving copies of completed scientific evidence; no new acceptance audit'))
    print(f'Exported {len(sources)} compact evidence files and {len(fits)} portable fit records.')


def rebuild_tables():
    """Reaggregate final per-contest results; no cache or posterior draws needed."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('jk9_frozen_table', TABLE / 'build_table.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The original aggregation checks original-path provenance. Validate the
    # archived byte lock here, and retain its exact scope/content conditions.
    def delimitation(case, selection):
        record = case.get('unresolved_full_comparator_delimitation')
        if not record:
            return False
        path = RELEASE / 'evidence/6322_full_comparator_delimitation.json'
        if digest(path) != record['sha256']:
            raise ValueError('Changed unresolved-comparison evidence')
        evidence = json.loads(path.read_text())
        return (case['contest_id'] == evidence['contest_id'] == 6322
                and evidence['reporting_rule'] == 'median_bounds_with_arbitrary_unknown'
                and evidence['status'] == 'unresolved_full_disclosure_benchmark'
                and not evidence['quality_criterion_relaxed'] and not evidence['draws_dropped']
                and selection['delta'] in evidence['affected_deltas']
                and selection['retained_observed'] and selection['selected'] == case['observed']
                and selection['full_comparator'] is None
                and selection['additional_reduction_pct'] is None)
    module.delimited_benchmark = delimitation
    paths = sorted((RELEASE / 'evidence/main_design').glob('*.json'))
    rows = module.aggregate(paths)
    if rows != json.loads((RELEASE / 'evidence/table_numbers.json').read_text()):
        raise ValueError('Main table differs from recorded numbers')
    fixed = [json.loads(p.read_text()) for p in (RELEASE / 'evidence/fixed_precision').glob('*.json')]
    if len(fixed) != 73 or {r['contest_id'] for r in fixed} != {
            int(p.stem) for p in paths}:
        raise ValueError('Fixed-precision contest coverage differs')
    if not all(r['status'] == 'completed' and r['posterior_draws'] == 128 for r in fixed):
        raise ValueError('Incomplete fixed-precision result')
    summary = json.loads((RELEASE / 'evidence/fixed_precision_summary.json').read_text())
    counts = {k: sum(r['regime'] == k for r in fixed) for k in summary['counts']}
    directions = {k: sum(r['direction_group'] == k for r in fixed) for k in summary['directions']}
    import numpy as np
    percentiles = {k: np.percentile([r['quality_gain_pct'] for r in fixed
                   if k == 'all' or r['direction_group'] == k], [25, 50, 75]).tolist()
                   for k in summary['percentiles']}
    if (counts != summary['counts'] or directions != summary['directions']
            or percentiles != summary['percentiles']):
        raise ValueError('Fixed-precision table differs')
    source = (ROOT / 'paper/PaperJK9.tex').read_text()
    appendix = (ROOT / 'paper/appendix_joint_design_jk9.tex').read_text()
    if module.replace_numbers(source, rows) != source:
        raise ValueError('Main manuscript numerical rows differ')
    if module.replace_fixed_numbers(appendix, summary) != appendix:
        raise ValueError('Appendix numerical rows differ')
    print('PASS: main 73 x 9 scenarios and fixed-precision 73 contests reproduce manuscript rows.')


def files(config):
    selected = tex_dependencies(local(ROOT, config['main_tex']), ROOT)
    for destination, source in config['assets'].items():
        selected.update((local(ROOT, destination), local(ROOT, source)))
    rows = check_evidence(ROOT, config, selected)
    for row in rows:
        selected.update((local(ROOT, row['evidence']), local(ROOT, row['producer'])))
    for folder in ('pipeline', 'tests'):
        selected.update(p for p in (ROOT / folder).rglob('*')
                        if p.is_file() and p.suffix in {'.py', '.cpp', '.stan'})
    for folder in ('metakaggle', 'synthetic_data'):
        selected.update((ROOT / folder).glob('*.py'))
        selected.update((ROOT / folder).glob('*.stan'))
    # Frozen controllers are source/protocol, never bulk caches or executables.
    selected.update(REVIEW.rglob('*.py'))
    selected.update(REVIEW.rglob('*.stan'))
    selected.update(p for p in (ROOT / 'docs/review/source_snapshots').rglob('*')
                    if p.suffix in {'.py', '.md', '.stan', '.cpp'})
    selected.update(REVIEW.glob('*method*.md'))
    selected.update((REVIEW / 'recovered_prediction_sources_20261029').glob('*.md'))
    selected.update((ROOT / 'archive/research/precision_conditions').rglob('*.txt'))
    selected.update((REVIEW / 'cleanup_20261027').glob('*.json'))
    selected.update((RELEASE / 'evidence').rglob('*'))
    selected.update((RELEASE / 'assets').glob('*'))
    for group in ('richer_steady', 'initialization_comparison'):
        selected.update(p for p in (ROOT / 'results/synthetic' / group).rglob('*.json')
                        if p.name.endswith('summary.json') or p.name in {'design.json', 'comparison.json'})
    selected.update((ROOT / config['empirical_root'] / 'input_snapshots').rglob('*.json'))
    for suffix in ('.bib', '.bst', '.cls', '.sty'):
        selected.update((ROOT / 'paper').glob('*' + suffix))
    for name in ('current_release.toml', 'Makefile', '.gitignore', 'README.md',
                 'requirements.txt', 'requirements-lock.txt', 'requirements-py314.txt', 'utils.py', 'docs/PAPER_EVIDENCE.csv',
                 'docs/REPRODUCIBILITY.md', 'pipeline/README.md', 'paper/README.md',
                 'docs/REPOSITORY_MAINTENANCE.md', 'docs/STORAGE_LAYOUT.md', 'docs/storage_layout.json',
                 'results/README.md', 'docs/README.md', 'docs/review/PaperJK9/INDEX.md',
                 'docs/JK9_MAINTENANCE_20261029.md', 'pipeline/jk9.py',
                 'results/releases/paperjk7-review-20261016/accepted_fits.json'):
        selected.add(ROOT / name)
    selected.update((RELEASE / 'accepted_fits.json', RELEASE / 'evidence_sources.json',
                     RELEASE / 'artifact_inventory.json'))
    empirical = ROOT / config['empirical_root']
    for folder in ('estimation', 'validation', 'auxiliary/reports'):
        selected.update(p for p in (empirical / folder).glob('*')
                        if p.suffix in {'.csv', '.json', '.txt'})
    selected.add(empirical / 'auxiliary/posterior_summary.csv')
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(selected) if p.is_file()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['snapshot', 'lock', 'verify', 'tables', 'compile-models', 'prepare-inputs'])
    parser.add_argument('--check-chains', action='store_true')
    args = parser.parse_args(argv)
    if args.action == 'prepare-inputs':
        from pipeline.jk9_fit import plan
        from pipeline.config import MAIN_DATA_DIR
        target = MAIN_DATA_DIR
        target.mkdir(parents=True, exist_ok=True)
        for row in plan('main', 'full'):
            source = local(ROOT, row['input_snapshot'])
            destination = target / f'contest_{row["contest_id"]}.json'
            if destination.exists() and digest(destination) != digest(source):
                raise ValueError(f'Refusing to overwrite different input: {destination}')
            if not destination.exists():
                shutil.copy2(source, destination)
        print('73 frozen main inputs available; no raw-data download or focal-pair reselection.')
        return
    if args.action == 'snapshot':
        snapshot()
        # Keep the original unresolved-comparison certificate available portably.
        case = json.loads((RELEASE / 'evidence/main_design/6322.json').read_text())
        source = relocated(case['unresolved_full_comparator_delimitation']['path'])
        shutil.copy2(source, RELEASE / 'evidence/6322_full_comparator_delimitation.json')
        return
    if args.action == 'tables':
        rebuild_tables()
        return
    if args.action == 'compile-models':
        for source in sorted((ROOT / 'pipeline/models').glob('*.stan')):
            compile_stan(source)
        return
    config = load()
    assert_assets(ROOT, config)
    current = files(config)
    manifest = local(ROOT, config['manifest'])
    if args.action == 'lock':
        write_json(manifest, dict(schema_version=1, release_id=config['release_id'],
                   scope='Compact JK9 byte lock, not an end-to-end scientific rerun', files=current))
        print(f'Locked {len(current)} JK9 source/input/evidence files.')
        return
    saved = json.loads(manifest.read_text())
    if saved['release_id'] != config['release_id'] or saved['files'].keys() != current.keys():
        raise ValueError('JK9 release file set changed; review before explicitly refreshing lock')
    check_files(ROOT, saved['files'])
    if args.check_chains:
        index = json.loads((RELEASE / 'accepted_fits.json').read_text())
        for fit in index['fits']:
            check_files(ROOT, {fit['accepted_path']: fit['accepted_sha256']})
            check_files(ROOT, dict(zip(fit['chain_paths'], fit['chain_sha256'], strict=True)))
    print(f'PASS: {len(current)} compact JK9 files' + (' and 292 main-chain hashes.' if args.check_chains else '.'))


if __name__ == '__main__':
    main()
