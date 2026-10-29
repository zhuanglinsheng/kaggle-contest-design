"""Baseline controls for interpreting the isolated feedback contrasts.

Reuse the original table's exact draws and numerical caches. No fit or
new design selection; distinguish a prior-induced reversal from a feedback
component that was not separately beneficial in the original joint design.
"""
import concurrent.futures as futures
import json
from pathlib import Path
from run import HERE, TABLE, CASES, Evaluator, Draw, Design, SimilarityOutput, design, save, sha


def compute(cid):
    folder = TABLE/f'contests/{cid}'
    original = json.loads((folder/'refined_selections.json').read_text())
    q = json.loads((folder/'quadrature.json').read_text())
    selected = next(s for s in original['selections'] if s['delta'] == CASES[cid][0])
    candidate = design(selected['selected'])
    e = object.__new__(Evaluator)
    e.cid = cid
    # Original validated caches are read through the same point evaluators;
    # comparisons are additions, not modifications of prior selection files.
    e.folder = folder
    e.engine = SimilarityOutput(TABLE/'build')
    e.groups = {k: [[Draw(**d) for d in g] for g in v] for k, v in q['draws'].items()
                if k in ('coarse', 'fine')}
    e.observed = design(original['observed'])
    e.provenance = dict(q['provenance'], baseline_control=True,
                        original_selection_sha256=sha(folder/'refined_selections.json'))
    result = dict(contest_id=cid, delta=CASES[cid][0], selected=candidate.record(),
                  comparison=e.compare(candidate, Design(candidate.prize, candidate.horizon, None)),
                  scope='Original posterior isolated feedback contrast at original selected prize and duration')
    save(HERE/'baseline_controls'/f'{cid}.json', result)
    print('BASELINE FEEDBACK', cid, result['comparison'].get('direction'), flush=True)
    return result


if __name__ == '__main__':
    with futures.ProcessPoolExecutor(max_workers=2) as pool:
        list(pool.map(compute, [2551, 5144]))
