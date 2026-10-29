"""Assemble deterministic, source-checked selections after prize refinement.

Checks the inputs needed for the existing table. The table writer separately
requires the complete sample. This script never edits the manuscript.
"""
from pathlib import Path
import hashlib
import json
import math
from pipeline.config import MAIN_DATA_DIR
from pipeline.empirical.counterfactual import PRIZE_DIVISOR
from pipeline.empirical.nonlinear_design import Design, save
from pipeline.empirical.nonlinear_design_selection import select_designs

HERE = Path(__file__).resolve().parent
DELTAS = (.1, .25, .5, 1., 2., 5., 10., 25., 100.)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decode(record):
    precision = record['precision']
    return Design(record['prize'], record['horizon'], math.inf if precision == 'full'
                  else None if precision == 'observed' else precision)


def assemble(folder):
    refined_path = folder/'prize_refinement.json'
    refined = json.loads(refined_path.read_text())
    if refined['status'] != 'completed_local_prize_refinement':
        raise ValueError('Prize refinement still running')
    if refined.get('original_prize_probe_version') != 1:
        raise ValueError('Original-policy nonmonotone prize probes still required')
    first_path = folder/'result.json'
    first = json.loads(first_path.read_text())
    assert sha(first_path) == refined['original_result_sha256']
    progress_path = folder/'validation_progress.json'
    progress = json.loads(progress_path.read_text())
    assert progress['accepted_sha256'] == refined['provenance']['accepted_sha256']
    assert progress['protocol_sha256'] == refined['provenance']['validation_protocol_sha256']
    accepted_path = Path(refined['provenance']['accepted'])
    assert sha(accepted_path) == progress['accepted_sha256']
    accepted = json.loads(accepted_path.read_text())
    cid = first['contest_id']
    raw_path = MAIN_DATA_DIR/f'contest_{cid}.json'
    assert sha(raw_path) == accepted['configuration']['raw_input_sha256']
    raw = json.loads(raw_path.read_text())
    observed = Design(float(raw['theta'])/PRIZE_DIVISOR,
                      float(raw['N_Delta'])*float(raw['Delta2f']))
    validation = progress['results']
    candidates = [decode(json.loads(k)) for k in validation]
    selections = select_designs(cid, observed, candidates, validation, DELTAS)
    required_refinement = []

    def is_refined(record):
        if record is None:
            return False
        for policy in refined['policies']:
            if (not policy.get('supported') or policy.get('prize') is None
                    or policy.get('local_width', math.inf) > 1e-4*observed.prize):
                continue
            same_policy = (policy['horizon'] == record['horizon']
                           and (record['prize'] == policy['prize'] == 0
                                or policy['precision'] == record['precision']))
            if same_policy and abs(policy['prize']-record['prize']) <= 1e-12*max(1., observed.prize):
                return True
        return False

    uncertainty = []
    for selection in selections:
        for name in ['selected', 'full_comparator']:
            if not is_refined(selection[name]):
                required_refinement.append(dict(delta=selection['delta'], role=name,
                                                design=selection[name]))
        selected_cost = selection['prize_ratio'] + selection['delta']*selection['time_multiplier']**2
        full = selection['full_comparator']
        full_cost = (full['prize']/observed.prize + selection['delta']*(full['horizon']/observed.horizon)**2
                     if full is not None else math.inf)
        threats = []
        reviewed = dict(validation)
        for record in first.get('screen_unresolved', []):
            reviewed.setdefault(json.dumps(record, sort_keys=True), dict(status='screen_unresolved', passed=False))
        for encoded, result in reviewed.items():
            if result.get('passed'):
                continue
            design = decode(json.loads(encoded))
            cost = design.prize/observed.prize + selection['delta']*(design.horizon/observed.horizon)**2
            roles = []
            if cost < selected_cost:
                roles.append('selected_design')
            if (design.prize == 0 or design.precision == math.inf) and cost < full_cost:
                roles.append('full_disclosure_comparator')
            if not roles:
                continue
            if result.get('status') in ['unresolved', 'screen_unresolved']:
                threats.append(dict(design=design.record(), cost=cost, affects=roles,
                                    reason=result['status']))
            elif result.get('status') == 'evaluated':
                upper = result['mean_gap']+result['critical_value']*result['mcse']+result['numerical_radius']
                if upper >= 0:
                    threats.append(dict(design=design.record(), cost=cost, affects=roles,
                                        reason='quality_diagnostic_includes_zero'))
        uncertainty.append(dict(delta=selection['delta'], cheaper_ambiguous_evaluated_candidates=threats,
                                unsampled_designs_excluded=False))
    independent = []
    for path in (HERE/f'independent_validation/{cid}').glob('*/result.json'):
        evidence = json.loads(path.read_text())
        if evidence['identity']['accepted_sha256'] == progress['accepted_sha256']:
            covered = [s['delta'] for s in selections if s['selected'] == evidence['identity']['candidate']]
            independent.append(dict(path=str(path), sha256=sha(path), selected_deltas=covered,
                                    comparisons=evidence['comparisons']))
    missing_comparators = [s['delta'] for s in selections
                           if s['choice'] == 'less_disclosure'
                           and s['prize_ratio'] > 0 and s['full_comparator'] is None]
    ready = not required_refinement and not missing_comparators
    baseline_repairs = []
    for path in (HERE/'required_baseline_repairs').glob(f'{cid}_chain*_staged.json'):
        evidence = json.loads(path.read_text())
        if not evidence.get('resolved'):
            continue
        extension = evidence['cache_augmentation']
        cache_path = Path(extension['original_cache'])
        active = json.loads(cache_path.read_text())
        if active.get('cache_augmentation') != extension:
            continue
        assert extension['accepted_sha256'] == progress['accepted_sha256']
        baseline_repairs.append(dict(path=str(path), sha256=sha(path),
                                     active_cache=str(cache_path), active_cache_sha256=sha(cache_path)))
    record = dict(contest_id=cid, status='completed_verified_incumbents',
                  publication_ready=ready, provenance=refined['provenance'],
                  observed=observed.record(),
                  selections=selections, validation=validation,
                  required_further_prize_refinement=required_refinement,
                  uncertainty=uncertainty, independent_checks=independent,
                  required_baseline_repairs=baseline_repairs,
                  input_files_sha256={str(p): sha(p) for p in [first_path, refined_path, progress_path]},
                  assembly_source_sha256=sha(__file__),
                  selection_source_sha256=sha(Path('pipeline/empirical/nonlinear_design_selection.py')),
                  scope='Best quality-supported searched candidates; equal-cost full disclosure preferred; '
                        'continuous global optimality not certified',
                  missing_full_comparators=missing_comparators,
                  remaining_checks=['Full73 coverage and table arithmetic',
                                    'Manuscript notes delimit the finite search and unresolved competitors'])
    save(folder/'refined_selections.json', record)
    return record


if __name__ == '__main__':
    summary = dict(assembled=[], failed={}, formal_table_replaced=False, publication_ready=False)
    for path in sorted(HERE.glob('contests/*/prize_refinement.json')):
        if json.loads(path.read_text()).get('status') != 'completed_local_prize_refinement':
            continue
        try:
            result = assemble(path.parent)
            summary['assembled'].append(dict(contest_id=result['contest_id'],
                prize_refinement_gaps=len(result['required_further_prize_refinement']),
                case_table_ready=result['publication_ready'],
                missing_full_comparators=result['missing_full_comparators'],
                independent_checks=len(result['independent_checks'])))
        except Exception as error:
            summary['failed'][path.parent.name] = repr(error)
    save(HERE/'refined_assembly_status.json', summary)
    print(json.dumps(summary, indent=2))
