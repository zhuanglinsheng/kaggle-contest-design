"""Audit retained reproduction inputs separately from optional historical draws.

Absent generated chains do not invalidate a compact publication. Present chains
must match their recorded hashes. This audit never certifies historical integrals.
"""
import argparse
import json
from pathlib import Path
from pipeline.publication import ROOT, load, local, digest, check_files, write_json
from pipeline.jk9 import RELEASE, files
from pipeline.jk9_prediction import splits, raw_input, verify_recovered_sources


def audit():
    config = load()
    manifest = json.loads(local(ROOT, config['manifest']).read_text())
    current = files(config)
    if manifest['release_id'] != config['release_id'] or current != manifest['files']:
        raise ValueError('Publication lock differs; review changes before refreshing it')
    check_files(ROOT, manifest['files'])
    frozen = json.loads((ROOT / 'results/releases/paperjk7-review-20261016/accepted_fits.json').read_text())['fits']
    for row in frozen:
        if digest(local(ROOT, row['input_snapshot'])) != row['input_sha256']:
            raise ValueError(f"Changed frozen input: {row['input_snapshot']}")
    index = json.loads((RELEASE / 'accepted_fits.json').read_text())
    absent = []; present = 0
    for fit in index['fits']:
        check_files(ROOT, {fit['accepted_path']: fit['accepted_sha256']})
        for name, expected in zip(fit['chain_paths'], fit['chain_sha256'], strict=True):
            path = local(ROOT, name)
            if path.exists():
                if digest(path) != expected:
                    raise ValueError(f'Changed historical chain: {name}')
                present += 1
            else:
                absent.append(name)
    for row in splits():
        raw_input(row)
    sources = verify_recovered_sources()
    return dict(schema_version=1, release_id=config['release_id'],
        retained_materials_passed=True, frozen_fit_input_references=len(frozen),
        prediction_splits=len(splits()), recovered_prediction_sources=sources,
        historical_main_chains=dict(present_and_hash_verified=present,
            absent_regenerable=len(absent), missing_paths=absent,
            required_for_compact_release=False),
        regeneration=dict(posterior='python -m pipeline.jk9_fit --scenario main --mode full --all',
            design='python -m pipeline.jk9_design all --all --accepted-root tmp/runs',
            prediction='See docs/REPRODUCIBILITY.md: fit, score and integrate for each C/D target'),
        historical_prediction_integration='not_certified_by_this_audit',
        scope='Retained source/input/result completeness; generated historical bytes are optional. '
              'Fresh runs need their own diagnostics and need not reproduce old Monte Carlo bytes.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'tmp/checks/reproducibility_audit.json')
    args = parser.parse_args()
    result = audit()
    write_json(args.output, result)
    print(f"PASS: {result['frozen_fit_input_references']} frozen fit input references, "
          f"{result['prediction_splits']} prediction splits, retained source/result lock. "
          f"Optional historical chains: {result['historical_main_chains']['absent_regenerable']} absent. "
          'Historical integration certification is separate.')


if __name__ == '__main__':
    main()
