"""Refine a required baseline point without relaxing the existing diagnostics.

Stage extra nested grids first. Activation preserves the original failed
record and invalidates only quality failures that depend on this point.
Run activation while the prize-refinement controller is stopped.
"""
from pathlib import Path
import argparse
import hashlib
import json

from pipeline.config import MAIN_DATA_DIR
from pipeline.empirical.counterfactual import PRIZE_DIVISOR
from pipeline.empirical.nonlinear_design import Draw, Design, save
from pipeline.empirical.nonlinear_design_native import NativeOutput
from pipeline.empirical.nonlinear_design_similarity import SimilarityOutput
from pipeline.empirical import nonlinear_design_recovery as recovery

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def repair(cid, chain, index, *, activate=False):
    folder = HERE/f'contests/{cid}'
    quad = json.loads((folder/'quadrature.json').read_text())
    accepted_path = Path(quad['provenance']['accepted'])
    assert sha(accepted_path) == quad['provenance']['accepted_sha256']
    accepted = json.loads(accepted_path.read_text())
    raw_path = MAIN_DATA_DIR/f'contest_{cid}.json'
    assert sha(raw_path) == accepted['configuration']['raw_input_sha256']
    raw = json.loads(raw_path.read_text())
    design = Design(float(raw['theta'])/PRIZE_DIVISOR,
                    float(raw['N_Delta'])*float(raw['Delta2f']))
    draw = next(Draw(**d) for group in quad['draws']['fine'] for d in group
                if d['chain'] == chain and d['index'] == index)
    engines = dict(similarity=SimilarityOutput(HERE/'build'),
                   physical_gradient=NativeOutput(HERE/'build'))
    identity = dict(draw=vars(draw), design=design.record(),
                    similarity_source=engines['similarity'].source_sha256,
                    physical_source=engines['physical_gradient'].source_sha256,
                    protocol_source=sha(recovery.__file__))
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = folder/'cross_coordinate_recovery'/f'{key}.json'
    original = json.loads(cache.read_text())
    assert original['identity'] == identity
    root = HERE/'required_baseline_repairs'
    root.mkdir(exist_ok=True)
    stem = f'{cid}_chain{chain}_index{index}'
    staged = root/f'{stem}_staged.json'
    archive = root/f'{stem}_original.json'

    if activate:
        result = json.loads(staged.read_text())
        assert result['resolved'] and result['identity'] == identity
        extension = result['cache_augmentation']
        assert extension['script_sha256'] == sha(__file__)
        if original.get('cache_augmentation') == extension:
            print('Required baseline repair already active.')
            return result
        assert sha(cache) == extension['original_cache_sha256']
        save(archive, original)
        # Extra grids strengthen the same assess() criterion. Preserve their
        # full provenance rather than pretending the default six grids passed.
        save(cache, result)
        progress_path = folder/'validation_progress.json'
        progress = json.loads(progress_path.read_text())
        assert progress['accepted_sha256'] == quad['provenance']['accepted_sha256']
        dependent = {k: v for k, v in progress['results'].items()
                     if v.get('status') == 'unresolved' and
                     dict(stage='fine', chain=chain, index=index) in v.get('failed_draws', [])}
        save(root/f'{stem}_dependent_failures.json', dict(
            accepted_sha256=progress['accepted_sha256'],
            protocol_sha256=progress['protocol_sha256'], results=dependent))
        for k in dependent:
            del progress['results'][k]
        progress.setdefault('required_baseline_repairs', []).append(dict(
            evidence=str(staged), evidence_sha256=sha(staged),
            chain=chain, index=index, invalidated_quality_failures=len(dependent)))
        save(progress_path, progress)
        print('Activated baseline repair; stale dependent failures removed:', len(dependent))
        return result

    if original['resolved']:
        print('Baseline is already resolved; no additional work required.')
        return original
    trials = list(original['trials'])
    assessment = recovery.assess(trials)
    for _ in range(2):
        if assessment['resolved']:
            break
        groups = {name: [t for t in trials if t['engine'] == name] for name in engines}
        refine = []
        for name, group in groups.items():
            tail = group[-3:]
            if len(tail) != 3 or any('B' not in t for t in tail):
                raise RuntimeError('This targeted extension requires three computed baseline grids')
            previous = abs(tail[1]['B']-tail[0]['B'])
            last = abs(tail[2]['B']-tail[1]['B'])
            if last > .8*previous+1e-8*max(1., abs(tail[-1]['B'])):
                refine.append(name)
        if not refine:
            refine = list(engines)  # Contracting sequences but cross-coordinate disagreement.
        for name in refine:
            last = groups[name][-1]
            dx, steps = last['spacing']/2, last['requested_steps']*2
            signature = dict(identity=identity, engine=name, spacing=dx, steps=steps,
                             max_cells=80_000_000,
                             wrapper_sha256=sha(__import__(engines[name].__module__,
                                 fromlist=['__file__']).__file__))
            trial_key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
            trial_path = root/'trial_cache'/f'{trial_key}.json'
            if trial_path.exists():
                trial = json.loads(trial_path.read_text())['trial']
            else:
                print('Refining required baseline', cid, name, dx, steps, flush=True)
                value = engines[name].evaluate(draw, design, spacing=dx, steps=steps,
                                               max_cells=80_000_000)
                trial = dict(value, engine=name, spacing=dx, requested_steps=steps)
                save(trial_path, dict(signature=signature, trial=trial))
            trials.append(trial)
        tails = [t for name in engines for t in [x for x in trials if x['engine'] == name][-3:]]
        assessment = recovery.assess(tails)
    result = dict(identity=identity, **assessment, cache_augmentation=dict(
        script_sha256=sha(__file__), original_cache=str(cache),
        original_cache_sha256=sha(cache), accepted_sha256=sha(accepted_path),
        all_grid_trials=trials,
        scope='Additional nested baseline grids; unchanged contraction and cross-coordinate criteria'))
    save(staged, result)
    print(json.dumps(dict(resolved=result['resolved'], B=result.get('B'),
                          diagnostic_radius=result.get('diagnostic_radius'),
                          error=result.get('error'), staged=str(staged)), indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--contest', type=int, required=True)
    parser.add_argument('--chain', type=int, required=True)
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--activate', action='store_true')
    args = parser.parse_args()
    result = repair(args.contest, args.chain, args.index, activate=args.activate)
    raise SystemExit(0 if result['resolved'] else 2)
