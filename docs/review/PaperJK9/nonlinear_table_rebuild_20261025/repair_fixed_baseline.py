"""Recover only unresolved baseline draws required by the existing appendix.

Keep the live table script and its cache identity unchanged. Extra original-
model, cross-coordinate diagnostics are fully recorded as cache augmentation.
Activation replaces an already cached failure atomically; it never changes a
live worker's in-memory column or progress. Retry the case after that worker
has finished, using all 128 original draws and all existing candidate caches.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json

from pipeline.config import MAIN_DATA_DIR
from pipeline.empirical.counterfactual import PRIZE_DIVISOR
from pipeline.empirical.nonlinear_design_recovery import recover

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('fixed_table', HERE/'fixed_precision_table.py')
fixed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixed)


def repair(cid, activate=False):
    folder = HERE/f'contests/{cid}'
    quad = json.loads((folder/'quadrature.json').read_text())
    accepted_path = Path(quad['provenance']['accepted'])
    assert fixed.sha(accepted_path) == quad['provenance']['accepted_sha256']
    accepted = json.loads(accepted_path.read_text())
    for path, expected in zip(accepted['chain_paths'], accepted['chain_sha256']):
        assert fixed.sha(path) == expected
    raw_path = MAIN_DATA_DIR/f'contest_{cid}.json'
    assert fixed.sha(raw_path) == accepted['configuration']['raw_input_sha256']
    raw = json.loads(raw_path.read_text())
    observed = fixed.Design(float(raw['theta'])/PRIZE_DIVISOR,
                            float(raw['N_Delta'])*float(raw['Delta2f']))
    engine = fixed.SimilarityOutput(HERE/'build')
    root = HERE/'required_baseline_repairs'
    root.mkdir(exist_ok=True)
    records = []
    draws = [fixed.Draw(**row) for group in quad['draws']['coarse'] for row in group]
    assert len(draws) == 128
    for position, draw in enumerate(draws):
        identity = dict(draw=vars(draw), design=observed.record(), levels=fixed.LEVELS,
                        similarity_source=engine.source_sha256,
                        full_source=fixed.sha('pipeline/theory/asymmetric_full_output.py'),
                        script_source=fixed.sha(fixed.__file__))
        encoded = json.dumps(identity, sort_keys=True)
        key = hashlib.sha256(encoded.encode()).hexdigest()
        cache = folder/'point_cache'/f'{key}.json'
        original = json.loads(cache.read_text())
        assert json.dumps(original['identity'], sort_keys=True) == encoded
        if original['resolved']:
            continue
        stem = f'{cid}_fixed_chain{draw.chain}_index{draw.index}'
        staged = root/f'{stem}_staged.json'
        if activate:
            result = json.loads(staged.read_text())
            extension = result['cache_augmentation']
            assert result['resolved']
            assert json.dumps(result['identity'], sort_keys=True) == encoded
            assert extension['accepted_sha256'] == fixed.sha(accepted_path)
            assert extension['repair_script_sha256'] == fixed.sha(__file__)
            assert extension['original_cache_sha256'] == fixed.sha(cache)
            fixed.save(root/f'{stem}_original.json', original)
            # point() reads existing caches without rewriting them. The live
            # case will retain its old column and require a later full retry.
            fixed.save(cache, result)
            print('Activated required appendix baseline', cid, position, flush=True)
        else:
            print('Recover required appendix baseline', cid, position,
                  draw.chain, draw.index, flush=True)
            recovered = recover(draw, observed,
                                cache_dir=folder/'cross_coordinate_recovery',
                                build_dir=HERE/'build')
            result = dict(original, resolved=recovered['resolved'],
                          cache_augmentation=dict(
                              repair_script_sha256=fixed.sha(__file__),
                              original_cache=str(cache), original_cache_sha256=fixed.sha(cache),
                              accepted_sha256=fixed.sha(accepted_path), position=position,
                              recovery=recovered,
                              scope='Required observed-policy draw; established original-model '
                                    'cross-coordinate recovery diagnostic; no dropped draws'))
            if recovered['resolved']:
                result.update(B=recovered['B'], radius=recovered['diagnostic_radius'])
            fixed.save(staged, result)
            print('Staged', cid, position, 'resolved', result['resolved'], flush=True)
        records.append(dict(position=position, chain=draw.chain, index=draw.index,
                            resolved=result['resolved'], evidence=str(staged)))
    fixed.save(root/f'{cid}_fixed_baseline_repair_status.json', dict(
        contest_id=cid, activated=activate, records=records,
        all_requested_resolved=all(row['resolved'] for row in records),
        scope='Only cached, unresolved observed-policy draws; retry the finished case afterwards'))
    return all(row['resolved'] for row in records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--contest', type=int, required=True)
    parser.add_argument('--activate', action='store_true')
    args = parser.parse_args()
    raise SystemExit(0 if repair(args.contest, args.activate) else 2)
