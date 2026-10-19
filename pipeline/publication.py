"""Read-only publication checks; historical scientific runners keep their paths.

The lock records bytes, not statistical validity. Scientific diagnostics remain
in the original verification reports. No estimation or manuscript integration
runner is imported here, and checks never refresh a lock implicitly.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def local(root, name):
    if Path(name).is_absolute():
        raise ValueError(f'Expected repository-relative path: {name}')
    path = (root / name).resolve()
    path.relative_to(root.resolve())
    return path


def load(root=ROOT):
    return tomllib.loads((root / 'current_release.toml').read_text())


def uncomment(text):
    return re.sub(r'(?<!\\)%[^\n]*', '', text)


def tex_dependencies(main, root):
    """Follow local inputs and graphics, including appendix fragments."""
    found = set()

    def visit(path):
        path = path.resolve()
        path.relative_to(root.resolve())
        if path in found:
            return
        if not path.is_file():
            raise FileNotFoundError(path)
        found.add(path)
        if path.suffix != '.tex':
            return
        text = uncomment(path.read_text())
        for kind, name in re.findall(
                r'\\(input|include|includegraphics)(?:\[[^\]]*\])?\{([^}]+)\}', text):
            base = path.parent / name
            suffixes = ['.tex'] if kind != 'includegraphics' else ['.pdf', '.png', '.jpg']
            candidates = [base] if base.suffix else [base.with_suffix(s) for s in suffixes]
            child = next((p for p in candidates if p.is_file()), candidates[0])
            visit(child)
    visit(main)
    return found


def assert_assets(root, config):
    for destination, source in config['assets'].items():
        if digest(local(root, destination)) != digest(local(root, source)):
            raise ValueError(f'Paper asset differs from selected source: {destination}')


def check_files(root, files):
    for name, expected in files.items():
        if digest(local(root, name)) != expected:
            raise ValueError(f'Changed release file: {name}')


def check_evidence(root, config, dependencies):
    with local(root, config['evidence_index']).open() as stream:
        rows = list(csv.DictReader(stream))
    labels = set()
    for path in dependencies:
        if path.suffix == '.tex':
            text = uncomment(path.read_text())
            for body in re.findall(
                    r'\\begin\{(?:figure|table|longtable)\}\*?(.*?)'
                    r'\\end\{(?:figure|table|longtable)\}', text, re.S):
                labels.update(re.findall(r'\\label\{([^}]+)\}', body))
    indexed = {row['label'] for row in rows}
    if len(indexed) != len(rows) or labels != indexed:
        raise ValueError(f'Evidence labels differ: missing={labels-indexed}, extra={indexed-labels}')
    for row in rows:
        for field in ('evidence', 'producer'):
            if not local(root, row[field]).exists():
                raise FileNotFoundError(row[field])
    return rows


def publication_files(root, config):
    empirical = local(root, config['empirical_root'])
    files = tex_dependencies(local(root, config['main_tex']), root)
    rows = check_evidence(root, config, files)
    for row in rows:
        p = local(root, row['evidence'])
        if p.is_file():
            files.add(p)
        files.add(local(root, row['producer']))
    for folder in ('pipeline', 'tests'):
        files.update(p for p in (root / folder).rglob('*')
                     if p.is_file() and p.suffix in ('.py', '.cpp', '.stan'))
    for folder in ('metakaggle', 'synthetic_data'):
        files.update((root / folder).glob('*.stan'))
    for name in ('verified_fits.json', 'release_verification.json', 'completion.json',
                 'report.json', 'report.txt', 'median_protocol.json',
                 'auxiliary_median_protocol.json', 'release_summary_zh.txt',
                 'target_parity.json', 'target_parity_all146.json',
                 'target_parity_joint_all146.json', 'auxiliary/design.json'):
        files.add(empirical / name)
    for folder in ('estimation', 'counterfactual', 'validation',
                   'auxiliary/reports', 'design_sensitivity', 'design_sensitivity/paper_update',
                   'fixed_initial_design', 'fixed_initial_design/contests'):
        files.update(p for p in (empirical / folder).glob('*')
                     if p.is_file() and p.suffix in ('.json', '.csv', '.txt'))
    files.add(empirical / 'auxiliary/posterior_summary.csv')
    files.update(empirical.glob('input_snapshots/**/*.json'))
    # Recovery source summaries carry the generating truths and original intervals.
    recovery = empirical / 'validation/simulation_recovery_reference.csv'
    with recovery.open() as stream:
        recovery_rows = list(csv.DictReader(stream))
    for row in recovery_rows:
        source = root / 'results/validation/sigma_lambda_trials/posterior/recovery' / row['case'] / 'lambda_external/summary.json'
        if digest(source) != row['source_summary_sha256']:
            raise ValueError(f'Recovery source differs: {source}')
        files.add(source)
    # All retained Chapter 4 recovery configurations, not just the table's first row.
    for name in ('initialization_comparison', 'richer_steady'):
        files.update((root / 'results/synthetic' / name).glob('**/summary.json'))
        files.update((root / 'results/synthetic' / name).glob('*.json'))
    files.update((root / 'paper').glob('*.bib'))
    files.update((root / 'paper').glob('*.bst'))
    files.update((root / 'paper').glob('*.cls'))
    files.update((root / 'paper').glob('*.sty'))
    for name in ('current_release.toml', 'Makefile', 'requirements.txt',
                 'requirements-lock.txt', 'README.md', '.gitignore', 'pipeline/README.md',
                 'paper/README.md', 'results/README.md', 'docs/REPRODUCIBILITY.md',
                 'docs/REPOSITORY_MAINTENANCE.md', config['evidence_index'], config['chain_index']):
        files.add(local(root, name))
    for destination, source in config['assets'].items():
        files.update((local(root, destination), local(root, source)))
    return {str(p.relative_to(root)): digest(p) for p in sorted(files)}


def create_chain_index(root, config):
    """Export relative paths without editing frozen, absolute-path run records."""
    empirical = local(root, config['empirical_root'])
    verified = json.loads((empirical / 'verified_fits.json').read_text())
    if not verified['complete'] or verified['fits'] != 397 or len(verified['checks']) != 397:
        raise ValueError('Expected completed 397-fit scientific verification')
    rows = []
    for check in verified['checks']:
        scenario, cid, mode = (check[k] for k in ('scenario', 'contest_id', 'mode'))
        base = empirical if scenario == 'main' else empirical / 'auxiliary' / scenario
        accepted = base / 'estimation' / mode / f'contest_{cid}' / 'accepted.json'
        if digest(accepted) != check['accepted_sha256']:
            raise ValueError(f'Accepted fit changed: {accepted}')
        record = json.loads(accepted.read_text())
        if len(record['chain_paths']) != 4 or record['chain_sha256'] != check['chain_sha256']:
            raise ValueError(f'Chain records disagree: {accepted}')
        # Current export is made at the original checkout; subsequent verification
        # uses only these relative paths and supports relocated artifact bundles.
        chains = [str(Path(p).resolve().relative_to(root)) for p in record['chain_paths']]
        rows.append(dict(check, accepted_path=str(accepted.relative_to(root)), chain_paths=chains,
                         input_snapshot=str((empirical / check['input_snapshot']).relative_to(root))))
    return {'schema_version': 1, 'fits': rows,
            'scope': 'Portable paths and saved hashes; original accepted.json records are immutable.'}


def check_chain_index(root, config, deep=False):
    empirical = local(root, config['empirical_root'])
    protocol = json.loads((empirical / 'median_protocol.json').read_text())
    check_files(root, {
        'pipeline/empirical/unified_prior_medians.py': protocol['source_sha256'],
        'pipeline/empirical/unified_prior_reestimate.py': protocol['target_source_sha256'],
        'pipeline/empirical/unified_prior_auxiliary.py': protocol['auxiliary_source_sha256'],
    })
    index = json.loads(local(root, config['chain_index']).read_text())
    rows = index['fits']
    keys = {(r['scenario'], r['contest_id'], r['mode']) for r in rows}
    if len(rows) != 397 or len(keys) != 397:
        raise ValueError('Missing or duplicated accepted fits')
    for row in rows:
        check_files(root, {row['input_snapshot']: row['input_sha256']})
        if deep:
            check_files(root, {row['accepted_path']: row['accepted_sha256']})
            if len(row['chain_paths']) != 4 or len(row['chain_sha256']) != 4:
                raise ValueError('Expected four chains per fit')
            check_files(root, dict(zip(row['chain_paths'], row['chain_sha256'], strict=True)))
    return len(rows)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check', action='store_true')
    group.add_argument('--write-manifest', action='store_true')
    group.add_argument('--export-chain-index', action='store_true')
    group.add_argument('--sync-assets', action='store_true')
    parser.add_argument('--check-chains', action='store_true')
    args = parser.parse_args()
    config = load()
    if args.export_chain_index:
        write_json(local(ROOT, config['chain_index']), create_chain_index(ROOT, config))
        print('Exported portable provenance for 397 accepted fits; original records unchanged.')
        return
    if args.sync_assets:
        # Check every source against the saved lock before copying any file.
        lock = json.loads(local(ROOT, config['manifest']).read_text())
        sources = {p: lock['files'][p] for p in config['assets'].values()}
        check_files(ROOT, sources)
        for destination, source in config['assets'].items():
            target, origin = local(ROOT, destination), local(ROOT, source)
            if not target.exists() or digest(target) != digest(origin):
                shutil.copy2(origin, target)
        print('Paper figures match current release sources.')
        return
    assert_assets(ROOT, config)
    count = check_chain_index(ROOT, config, args.check_chains)
    files = publication_files(ROOT, config)
    manifest_path = local(ROOT, config['manifest'])
    if args.write_manifest:
        write_json(manifest_path, {'schema_version': 1, 'release_id': config['release_id'],
                                  'scope': 'Byte-level publication snapshot; not a new statistical audit.',
                                  'files': files})
        print(f'Recorded {len(files)} files for {config["release_id"]}.')
    else:
        saved = json.loads(manifest_path.read_text())
        if saved['release_id'] != config['release_id']:
            raise ValueError('Release ID differs from saved manifest')
        if saved['files'].keys() != files.keys():
            raise ValueError('Publication file set changed; review additions/removals before refreshing the lock')
        check_files(ROOT, saved['files'])
        level = 'including accepted records and all chain bytes' if args.check_chains else 'compact artifacts and frozen inputs'
        print(f'PASS: {len(files)} files; {count} accepted-fit references; {level}.')


if __name__ == '__main__':
    main()
