"""Write the two existing tables only after complete original-game coverage.

Use the reviewed staged prose, check frozen sources, and leave PDF compilation
and visual inspection explicitly pending. This command performs no model solve.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import subprocess
import sys

from build_table import aggregate, replace_numbers, replace_fixed_numbers

HERE = Path(__file__).resolve().parent
STAGE = HERE/'manuscript_stage'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(apply=False):
    expected = {int(row['contest_id']) for row in csv.DictReader(Path(
        'results/reestimation/lambda100_sd225/estimation/accepted_chain_sources.csv').open())
        if row['mode'] == 'full'}
    state = json.loads((HERE/'refinement_status.json').read_text())
    if len(expected) != 73 or {int(x) for x in state['completed']} != expected or state['active']:
        raise ValueError('All73 completed cases required before authoring numerical results')
    subprocess.run([sys.executable, str(HERE/'assemble_refined.py')], check=True,
                   stdout=subprocess.DEVNULL)
    paths = list(HERE.glob('contests/*/refined_selections.json'))
    cases = [json.loads(p.read_text()) for p in paths]
    assert {c['contest_id'] for c in cases} == expected and len(cases) == 73
    for case in cases:
        for path, expected_sha in case['input_files_sha256'].items():
            assert sha(path) == expected_sha
        assert sha(case['provenance']['accepted']) == case['provenance']['accepted_sha256']
    rows = aggregate(paths)
    assert all(r['additional_unknown_count'] == 1 for r in rows)
    fixed = json.loads((HERE/'fixed_precision/table_numbers.json').read_text())
    fixed_audit = json.loads((HERE/'fixed_precision/final_audit.json').read_text())
    assert fixed_audit['ready'] and set(fixed_audit['contest_ids']) == expected
    assert sha(HERE/'fixed_precision/table_numbers.json') == fixed_audit['table_numbers_sha256']
    for path, expected_sha in fixed_audit['sources_sha256'].items():
        assert sha(path) == expected_sha
    manifest = json.loads((STAGE/'manifest.json').read_text())
    assert manifest['pdf_skill_marker_completed']
    for path, expected_sha in manifest['original_sha256'].items():
        assert sha(path) == expected_sha, f'Original changed: {path}'
    for path, expected_sha in manifest['staged_sha256'].items():
        assert sha(path) == expected_sha, f'Stage changed: {path}'
    assert sha('paper/numerical_regions_four_class_bw.pdf') == manifest['protected_task2_figure_sha256']
    source = replace_numbers((STAGE/'PaperJK9.tex').read_text(), rows)
    label = source.index(r'\label{tab-counterfactual-precision}')
    begin = source.index(r'\end{table}', label)+len(r'\end{table}')
    end = source.index(r'\section{Robustness}', begin)
    assert ('reports first-order, quality-matched design comparisons' in source[begin:end]
            or 'reports nonlinear, quality-matched comparisons' in source[begin:end])
    one = next(row for row in rows if row['delta'] == 1)
    hundred = next(row for row in rows if row['delta'] == 100)
    paragraph = rf'''

{{Table~\ref{{tab-counterfactual-precision}} reports nonlinear, quality-matched comparisons among numerically supported searched designs. At $\delta=1$, the selected designs use full disclosure in {one['full_disclosure']} contests, finite feedback in {one['less_disclosure']}, and zero prizes in {one['zero_prize']}. Across all 73 contests, the median cost reduction is {one['cost_reduction_pct']:.2f}\%. At $\delta=100$, the median prize change is ${hundred['delta_prize_pct']:.2f}\%$; large prize changes remain extrapolations. These comparisons are conditional on the approximate parameter posterior and the finite search, with unresolved alternatives left open.}}


'''
    source = source[:begin]+paragraph+source[end:]
    appendix = (STAGE/'appendix_joint_design_jk9.tex').read_text()
    assert replace_fixed_numbers(appendix, fixed) == appendix
    for stale in ['reports first-order, quality-matched', 'These calculations have not been checked against exact nonlinear',
                  'Quality-Constrained First-Order Design Comparisons']:
        assert stale not in source
    assert 'including limiting gains' not in appendix and 'First-Order Precision Comparisons' not in appendix
    (STAGE/'PaperJK9.tex').write_text(source)
    (HERE/'table_numbers.json').write_text(json.dumps(rows, indent=2)+'\n')
    audit = dict(ready_for_source_sync=True, contests=73, contest_ids=sorted(expected),
                 scenarios_per_contest=9, unresolved_benchmark_contests=[6322],
                 unknown_comparisons_retained_in_median_bounds=True,
                 case_sources_sha256={str(p): sha(p) for p in paths},
                 table_writer_sha256=sha(HERE/'build_table.py'),
                 assembly_source_sha256=sha(HERE/'assemble_refined.py'),
                 authoring_script_sha256=sha(__file__),
                 fixed_audit_sha256=sha(HERE/'fixed_precision/final_audit.json'),
                 scope='All 73 original-posterior nonlinear cases over the evaluated design menu; '
                       'one unresolved benchmark included in the reported comparison bounds')
    (HERE/'final_main_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    if apply:
        Path('paper/PaperJK9.tex').write_text(source)
        Path('paper/appendix_joint_design_jk9.tex').write_text(appendix)
        manifest['applied_source_sha256'] = {str(Path('paper', name).resolve()): sha(Path('paper', name))
                                           for name in ['PaperJK9.tex', 'appendix_joint_design_jk9.tex']}
    manifest['staged_sha256'][str((STAGE/'PaperJK9.tex').resolve())] = sha(STAGE/'PaperJK9.tex')
    manifest['status'] = 'sources_applied_pdf_pending' if apply else 'all_numbers_staged_pdf_pending'
    manifest['pending'] = ['Compile actual final PDF', 'Visual and reference checks', 'Final audit and records']
    (STAGE/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('All73 numerical results synchronized; PDF compilation and visual checks remain pending.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    try:
        main(args.apply)
    except ValueError as error:
        print(str(error))
        raise SystemExit(2)
