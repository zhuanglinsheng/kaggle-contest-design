"""Cross-coordinate recovery of unresolved original-game quality evaluations.

Two conservative gradient-HJB solvers must converge on three nested grids
and agree within their refinement diagnostics. This is numerical evidence,
not a rigorous continuous-PDE error bound or a proof of global optimality.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pipeline.empirical.nonlinear_design import save
from pipeline import solver_cache
from pipeline.empirical.nonlinear_design_native import NativeOutput
from pipeline.empirical.nonlinear_design_similarity import SimilarityOutput


def assess(trials):
    """Require shrinking final grid changes and cross-coordinate agreement."""
    groups = {name: [t for t in trials if t['engine'] == name]
              for name in ['similarity', 'physical_gradient']}
    if any(len(g) != 3 or any('error' in t for t in g) for g in groups.values()):
        return dict(resolved=False, error='Incomplete cross-coordinate grid sequence', trials=trials)
    differences = {}
    for name, group in groups.items():
        previous = abs(group[1]['B'] - group[0]['B'])
        last = abs(group[2]['B'] - group[1]['B'])
        tolerance = 1e-8 * max(1., abs(group[-1]['B']))
        if last > .8 * previous + tolerance:
            return dict(resolved=False, error='Grid changes not contracting', trials=trials)
        differences[name] = last
    first, second = (groups[name][-1] for name in ['similarity', 'physical_gradient'])
    disagreement = abs(first['B'] - second['B'])
    tolerance = 1e-4 * max(1., abs(first['B']), abs(second['B']))
    if disagreement > max(3 * sum(differences.values()), tolerance):
        return dict(resolved=False, error='Cross-coordinate disagreement', trials=trials)
    radius = 3 * max(differences.values()) + disagreement + first['budget'] + second['budget']
    return dict(resolved=True, B=(first['B'] + second['B']) / 2,
                diagnostic_radius=radius, trials=trials,
                method='original_nonlinear_three_grid_cross_coordinate_recovery',
                scope='Refinement and cross-coordinate diagnostics; not a strict PDE bound')


def recover(draw, design, *, cache_dir, build_dir):
    similarity = SimilarityOutput(build_dir)
    physical = NativeOutput(build_dir)
    identity = dict(draw=vars(draw), design=design.record(),
                    similarity_source=similarity.source_sha256,
                    physical_source=physical.source_sha256,
                    protocol_source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = Path(cache_dir) / f'{key}.json'
    cached = solver_cache.get(path)
    if cached is not None:
        return cached
    trials = []
    for engine, name, settings in [
        (similarity, 'similarity', [(.025, 512), (.0125, 1024), (.00625, 2048)]),
        (physical, 'physical_gradient', [(.01, 512), (.005, 1024), (.0025, 2048)]),
    ]:
        for dx, steps in settings:
            try:
                result = engine.evaluate(draw, design, spacing=dx, steps=steps)
            except Exception as error:
                result = dict(error=repr(error))
            trials.append(dict(result, engine=name, spacing=dx, requested_steps=steps))
            if 'error' in result:
                break
    result = dict(identity=identity, **assess(trials))
    save(path, result)
    return result
