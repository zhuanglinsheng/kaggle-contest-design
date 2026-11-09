"""Refit and evaluate the four frozen nonlinear redesign directions."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

from pipeline.jk9 import RELEASE, REVIEW
from pipeline.jk9_fit import COST_SCENARIOS, fit, plan
from pipeline.publication import ROOT, digest, write_json

CONTROLLER = REVIEW / 'task3_nonlinear_prior_sensitivity_20261026'
CASES = (8219, 2464, 5144, 2551)
BASELINE_CONTROLS = (2551, 5144)


def controller(name):
    spec = importlib.util.spec_from_file_location(name, CONTROLLER / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summarize(output):
    results = []
    for cid in CASES:
        for scenario in COST_SCENARIOS:
            source = output / 'results' / f'{cid}_{scenario}.json'
            row = json.loads(source.read_text())
            fixed = row['fixed_published_design_direction']
            if cid == 8219:
                refined = output / 'small_refinement' / source.name
                detail = json.loads(refined.read_text())
                if detail['original_result_sha256'] != digest(source):
                    raise ValueError(f'Fine refinement belongs to another run: {refined}')
                row = detail
                fixed = row['trials'][0]['comparison'].get('direction','unresolved')
            if row['status']!='completed' or row['contest_id']!=cid or row['scenario']!=scenario:
                raise ValueError(f'Unexpected sensitivity record: {source}')
            results.append(dict(contest_id=cid, scenario=scenario, fixed_direction=fixed,
                fixed_prize_adjustment_required=row['same_direction_cost_saving_supported'] and fixed!='positive',
                same_direction_supported=row['same_direction_cost_saving_supported'],
                supported_saving_pct=row.get('supported_saving_pct'), result_sha256=digest(source)))
    controls = {str(cid):json.loads((output/'baseline_controls'/f'{cid}.json').read_text())
                for cid in BASELINE_CONTROLS}
    write_json(output/'direction_summary.json', dict(records=results, baseline_feedback_controls=controls,
        fixed_designs_supported=sum(r['fixed_direction']=='positive' for r in results),
        directional_existence_tests_supported=sum(r['same_direction_supported'] for r in results)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'fit', 'evaluate', 'report'))
    parser.add_argument('--contests', type=int, nargs='+', choices=CASES, default=CASES)
    parser.add_argument('--scenario', choices=COST_SCENARIOS, action='append')
    parser.add_argument('--output', type=Path, default=Path('tmp/runs/cost_sensitivity'))
    parser.add_argument('--baseline-root', type=Path, default=Path('tmp/runs'))
    parser.add_argument('--design-root', type=Path, default=RELEASE/'evidence/main_design',
                        help='Frozen release selections as <cid>.json')
    args = parser.parse_args(argv)
    output = args.output.resolve()
    scenarios = tuple(dict.fromkeys(args.scenario or COST_SCENARIOS))
    if args.action == 'report':
        if set(args.contests)!=set(CASES) or set(scenarios)!=set(COST_SCENARIOS):
            parser.error('report requires all four contests and both scenarios')
        summarize(output)
        return
    if args.action == 'evaluate':
        run = controller('run')
        # The historical standalone refinement imports its companion as "run".
        previous = sys.modules.get('run')
        sys.modules['run'] = run
        try:
            small = controller('refine_small')
        finally:
            if previous is None:
                sys.modules.pop('run')
            else:
                sys.modules['run'] = previous
        run.refiner.HERE = output
    for cid in dict.fromkeys(args.contests):
        row = plan('main','full',cid)[0]
        baseline = args.baseline_root.resolve()/'main/full'/str(cid)/'accepted.json'
        root = args.design_root.resolve()
        selection = root/f'{cid}.json'
        if not selection.is_file():
            selection = root/str(cid)/'refined_selections.json'
        for scenario in scenarios:
            if args.action == 'plan':
                print(cid, scenario, row['input_snapshot'], selection, baseline)
            elif args.action == 'fit':
                fit(row, output, cost_scenario=scenario, baseline=baseline)
            else:
                accepted = output/'fits'/scenario/str(cid)/'accepted.json'
                evaluator = run.Evaluator(cid, scenario, accepted, output=output,
                    raw_path=ROOT/row['input_snapshot'], model_source=ROOT/'pipeline/models/common_cost.stan')
                run.run(cid, scenario, output=output, original_path=selection, evaluator=evaluator)
                if cid == 8219:
                    small.compute(scenario, output=output, evaluator=evaluator)
        if args.action == 'evaluate' and cid in BASELINE_CONTROLS:
            evaluator = run.Evaluator(cid, 'baseline', baseline, output=output, baseline=True,
                raw_path=ROOT/row['input_snapshot'], model_source=ROOT/'pipeline/models/leading_order.stan')
            original = json.loads(selection.read_text())
            selected = next(s for s in original['selections'] if s['delta']==run.CASES[cid][0])
            candidate = run.design(selected['selected'])
            comparison = evaluator.compare(candidate, run.Design(candidate.prize,candidate.horizon,None))
            write_json(output/'baseline_controls'/f'{cid}.json', dict(contest_id=cid,
                comparison=comparison, original_selection_sha256=digest(selection)))


if __name__ == '__main__':
    main()
