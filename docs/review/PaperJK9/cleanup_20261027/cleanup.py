"""User-authorized removal of experiments outside the Task 1--3 response.

Run --plan first. --apply checks the exact plan, all protected artifact
hashes, accepted-chain provenance, and absence of target/protection overlap.
Never writes Git metadata, moves data outside the repository, or modifies
the manuscript/scientific implementation. Deletions are intentional.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import time

ROOT = Path(__file__).resolve().parents[4]
REVIEW = ROOT/'docs/review/PaperJK9'
HERE = Path(__file__).resolve().parent
TABLE = REVIEW/'nonlinear_table_rebuild_20261025'
SHARED_METHODS = {
    'task4_fast_density_evaluator_method_20261025.md',
    'task4_predictive_density_parity_method_20261025.md',
    'task4_final_protocol_training_method_20261025.md',
}
SHARED_BACKEND = {'build.json', 'density_components.so', 'parity.json'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    tmp.replace(path)


def protected():
    # Entire scientific evidence trees stay in place, including failures.
    trees = [ROOT/'pipeline', ROOT/'tests', ROOT/'synthetic_data',
             ROOT/'metakaggle', ROOT/'formalization', ROOT/'ryvkin',
             ROOT/'results/reestimation', ROOT/'results/theory',
             ROOT/'results/synthetic', ROOT/'results/validation',
             ROOT/'archive/research', TABLE,
             REVIEW/'reconstruction_20261025', REVIEW/'prior_reconstruction_20261025',
             REVIEW/'task3_nonlinear_prior_sensitivity_20261026']
    files = {p for p in (ROOT/'paper').iterdir()
             if p.is_file() and p.suffix in {'.tex', '.pdf', '.png', '.bib', '.cls', '.sty', '.bst', '.drawio'}}
    for name in ['task1_implementation_response_20261024.md',
                 'task2_response_and_delimitation_20261025.md',
                 'task2_theory_numerics_progress.md',
                 'task2_cell_uncertainty_inventory_20261024.csv',
                 'input_diagnostics_audit_20261024.py',
                 'input_diagnostics_audit_20261024.json',
                 'numerical_audit_20261024.py', 'numerical_audit_20261024.json']:
        files.add(REVIEW/name)
    for p in REVIEW.iterdir():
        if p.name.startswith(('task1', 'task2', 'task3')):
            if p.is_dir():
                trees.append(p)
            else:
                files.add(p)
    for name in SHARED_METHODS:
        files.add(REVIEW/name)
    for name in SHARED_BACKEND:
        files.add(REVIEW/'predictive_integration_20261025'/name)
    # Compact source and final summaries, without walking millions of caches.
    for parent in [ROOT/'pipeline', ROOT/'tests', ROOT/'synthetic_data']:
        files.update(p for p in parent.rglob('*') if p.is_file() and p.suffix in {'.py', '.stan', '.cpp'})
    for cid_dir in (TABLE/'contests').iterdir():
        if not cid_dir.is_dir() or not cid_dir.name.isdigit():
            continue
        for name in ['refined_selections.json', 'quadrature.json', 'prize_refinement.json']:
            files.add(cid_dir/name)
        files.add(TABLE/'fixed_precision'/f'{cid_dir.name}.json')
    for name in ['table_numbers.json', 'final_main_audit.json', 'table_readiness.json',
                 'fixed_precision/table_numbers.json', 'fixed_precision/final_audit.json',
                 'manuscript_stage/completion_audit.json']:
        files.add(TABLE/name)
    sensitivity = REVIEW/'task3_nonlinear_prior_sensitivity_20261026'
    files.add(sensitivity/'final_audit.json')
    files.add(sensitivity/'manifest.json')
    for parent in [ROOT/'metakaggle/__jsondata__',
                   ROOT/'results/reestimation/lambda100_sd225/input_snapshots/main']:
        files.update(parent.glob('*.json'))
    accepted_paths = set()
    for cid_dir in (TABLE/'contests').iterdir():
        if cid_dir.is_dir() and cid_dir.name.isdigit():
            q = json.loads((cid_dir/'quadrature.json').read_text())
            path = Path(q['provenance']['accepted'])
            assert sha(path) == q['provenance']['accepted_sha256']
            accepted_paths.add(path)
    for entry in json.loads((sensitivity/'manifest.json').read_text())['records']:
        path = Path(entry['accepted'])
        assert sha(path) == entry['accepted_sha256']
        accepted_paths.add(path)
    chains = []
    for path in sorted(accepted_paths):
        accepted = json.loads(path.read_text())
        assert accepted['diagnostics']['passed'] and len(accepted['chain_paths']) == 4
        files.add(path)
        for raw_path, expected in zip(accepted['chain_paths'], accepted['chain_sha256']):
            chain = Path(raw_path)
            assert sha(chain) == expected
            files.add(chain)
            chains.append(dict(path=str(chain.relative_to(ROOT)), sha256=expected))
    return trees, files, chains


def candidates():
    selected = {}
    for p in REVIEW.glob('task4*'):
        if p.name not in SHARED_METHODS:
            selected[p] = 'Task4-only prediction experiment, result or method'
    for name in ['prediction_reconstruction_20261025', 'prediction_precision_20261025',
                 'prediction_score_snapshots_20261025', 'global_path_importance_20261025']:
        selected[REVIEW/name] = 'Task4-only posterior/predictive integration artifacts'
    for p in (REVIEW/'predictive_integration_20261025').iterdir():
        if p.name not in SHARED_BACKEND:
            selected[p] = 'Task4-only integration cache; shared Task3 backend excluded'
    for p in (ROOT/'archive').iterdir():
        if p.name not in {'research', 'README.md'}:
            selected[p] = 'Superseded experiment/draft archive outside current Task1--3 evidence'
    for name in ['prediction', 'example', 'tmp', '.cache', '__pycache__']:
        selected[ROOT/name] = 'Independent old example/prediction exercise or regenerable scratch/cache'
    for name in ['early', 'full', 'reruns', 'reruns_long']:
        selected[ROOT/'results/estimation/main'/name] = 'Superseded pre-unified-prior raw fits; compact summaries and indexes retained'
    selected[ROOT/'results/estimation/robustness/full'] = 'Superseded raw robustness fits; current unified-prior evidence retained'
    selected[ROOT/'results/counterfactual/main/counterfactual_draws.csv'] = 'Superseded first-order bulk draws; current nonlinear Task3 calculations retained'
    for p in (ROOT/'paper').iterdir():
        if p.suffix in {'.aux', '.blg', '.log', '.out', '.gz', '.fdb_latexmk', '.fls'}:
            selected[p] = 'Regenerable LaTeX compilation artifact; sources, bibliography and PDFs retained'
    selected[ROOT/'.DS_Store'] = 'Finder metadata'
    return {p: reason for p, reason in selected.items() if p.exists() or p.is_symlink()}


def inventory(path):
    count = logical = allocated = 0
    def add(p):
        nonlocal count, logical, allocated
        stat = p.lstat()
        count += 1
        logical += stat.st_size
        allocated += stat.st_blocks*512
    if path.is_file() or path.is_symlink():
        add(path)
    else:
        for current, dirs, files in os.walk(path, followlinks=False):
            for name in files:
                add(Path(current)/name)
            for name in list(dirs):
                p = Path(current)/name
                if p.is_symlink():
                    add(p)
                    dirs.remove(name)
    return dict(files=count, logical_bytes=logical, allocated_bytes=allocated)


def assert_safe(targets, trees, files):
    for target in targets:
        assert target != ROOT and target.is_relative_to(ROOT)
        assert not target.is_relative_to(ROOT/'.git')
        assert not target.is_relative_to(HERE)
        for protected in trees:
            assert not (target.is_relative_to(protected) or protected.is_relative_to(target)), (target, protected)
        for protected in files:
            assert not (protected == target or protected.is_relative_to(target)), (target, protected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    assert args.plan != args.apply
    assert ROOT.name == 'kaggle-contest-design'
    trees, files, chains = protected()
    assert len(chains) == 324, len(chains)
    if args.plan:
        targets = candidates()
        assert_safe(targets, trees, files)
        records = []
        for target, reason in sorted(targets.items()):
            records.append(dict(path=str(target.relative_to(ROOT)), reason=reason, **inventory(target)))
        missing = [str(p) for p in files if not p.is_file()]
        assert not missing, missing
        manifest = dict(status='planned', date='2026-10-27',
             user_authorization='Delete contents unrelated to responding to Tasks1,2,3',
             deletion_targets=records, protected_trees=[str(p.relative_to(ROOT)) for p in trees],
             protected_files={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)},
             accepted_posteriors=81, accepted_chain_records=chains,
             shared_task3_dependencies_retained=sorted(SHARED_METHODS|SHARED_BACKEND),
             raw_data_archive_retained=True, git_metadata_untouched=True,
             total_files=sum(r['files'] for r in records),
             total_logical_bytes=sum(r['logical_bytes'] for r in records),
             total_allocated_bytes=sum(r['allocated_bytes'] for r in records))
        save(HERE/'manifest.json', manifest)
        print(json.dumps({k:manifest[k] for k in ['status','total_files','total_logical_bytes','total_allocated_bytes']}), flush=True)
    else:
        manifest = json.loads((HERE/'manifest.json').read_text())
        assert manifest['status'] == 'planned'
        targets = [ROOT/r['path'] for r in manifest['deletion_targets']]
        assert_safe(targets, trees, files)
        for path, expected in manifest['protected_files'].items():
            assert sha(ROOT/path) == expected, path
        manifest['status'] = 'deleting'
        save(HERE/'manifest.json', manifest)
        deleted = []
        for serial, target in enumerate(targets, 1):
            if target.is_symlink() or target.is_file():
                target.unlink()
            elif target.is_dir():
                shutil.rmtree(target)
            assert not target.exists()
            deleted.append(str(target.relative_to(ROOT)))
            if serial % 25 == 0:
                print('DELETED', serial, '/', len(targets), flush=True)
                save(HERE/'progress.json', dict(deleted=deleted, total=len(targets)))
        for path, expected in manifest['protected_files'].items():
            assert sha(ROOT/path) == expected, path
        manifest.update(status='deleted_and_protected_hashes_verified', deleted_targets=len(deleted),
                        protected_hashes_unchanged=True, verified_at_unix=time.time())
        save(HERE/'manifest.json', manifest)
        save(HERE/'progress.json', dict(status='completed', deleted=deleted, total=len(targets)))
        print('COMPLETE', len(deleted), 'targets; all protected hashes unchanged', flush=True)


if __name__ == '__main__':
    main()
