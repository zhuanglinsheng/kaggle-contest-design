"""Static audit of retained result families and manuscript source mappings.

Run from the repository root: python -m pipeline.result_code_audit.
This inventories result files, implementation sources and their workflow roles.
"""
from pathlib import Path
import json, csv, collections, re, fnmatch
root=Path(__file__).resolve().parents[1]; review='docs/review/PaperJK9/'; table=review+'nonlinear_table_rebuild_20261025/'; pred=review+'recovered_prediction_sources_20261029/'
rules=[]
def rule(pattern,category,*sources):rules.append((pattern,category,sources))
rule('results/README.md','documentation','results/README.md')
rule('results/*.cpp','source','{self}')
rule('results/*/input_snapshots/*.json','frozen_input','pipeline/empirical/rebuild_main.py','pipeline/empirical/unified_prior_reestimate.py','pipeline/jk9_fit.py')
rule('results/synthetic/effort_comparison/*','calculation','pipeline/synthetic/effort_comparison.py')
rule('results/synthetic/initialization_comparison/*','calculation','pipeline/synthetic/model_comparison.py')
rule('results/synthetic/richer_steady/*','calculation','pipeline/synthetic/richer_recovery.py')
rule('results/synthetic/optimal_hourly/*','calculation','pipeline/synthetic/optimal_generate.py','pipeline/synthetic/generate.py','pipeline/synthetic/figure3.py')
rule('results/validation/main/*','calculation','pipeline/validation/rank_agreement.py')
rule('results/reestimation/*/estimation/*','calculation','pipeline/empirical/unified_prior_reestimate.py','pipeline/empirical/unified_prior_medians.py')
rule('results/reestimation/*/paper_assets/*','calculation','pipeline/empirical/unified_prior_outputs.py','pipeline/validation/chapter5.py')
rule('results/reestimation/*/auxiliary/*','calculation','pipeline/empirical/unified_prior_auxiliary.py','pipeline/empirical/unified_prior_aux_summary.py')
for names,sources in [
 (['chapter5_contest_level.csv','chapter5_regressions.csv'],['pipeline/empirical/unified_prior_outputs.py','pipeline/validation/chapter5.py']),
 (['chapter5_paper_numbers.json'],['pipeline/empirical/unified_prior_outputs.py','pipeline/validation/chapter5_report.py']),
 (['chapter5_signal_cook_regressions.csv','chapter5_signal_cook_flags.csv'],['pipeline/validation/chapter5_influence.py']),
 (['chapter5_final_audit.json','chapter5_accepted_count_floor_audit.csv'],['pipeline/validation/chapter5_final_audit.py']),
 (['regression_comparison.csv','parameter_comparison.csv'],['pipeline/empirical/unified_prior_outputs.py']),
 (['same_prior_sampler_comparison.csv','same_prior_sampler_report.json'],['pipeline/empirical/unified_prior_sampler_check.py']),
 (['empirical_calibrated_intervals.csv','interval_and_recovery_report.json','simulation_recovery_reference.csv'],['pipeline/empirical/unified_prior_interval_report.py'])]:
 for name in names:rule('results/reestimation/*/validation/'+name,'calculation',*sources)
release='results/releases/paperjk9-20261029/'
rule(release+'evidence/main_design/*','calculation',table+'design_batch.py',table+'refine_prizes.py',table+'assemble_refined.py','pipeline/empirical/nonlinear_design.py','pipeline/jk9_design.py')
rule(release+'evidence/fixed_precision/*','calculation',table+'fixed_precision_table.py','pipeline/empirical/nonlinear_design.py','pipeline/jk9_design.py')
rule(release+'evidence/main_posteriors/*','calculation',table+'rebuild_posterior.py',table+'audit_posteriors.py','pipeline/empirical/unified_prior_medians.py')
rule(release+'evidence/cost_sensitivity/*','calculation_and_summary',review+'task3_nonlinear_prior_sensitivity_20261026/run.py',review+'task3_nonlinear_prior_sensitivity_20261026/refine_small.py',review+'task3_nonlinear_prior_sensitivity_20261026/baseline_feedback.py','pipeline/jk9.py')
rule(release+'evidence/theory_source/*','calculation','pipeline/theory/joint_design_moderate_dense.py','pipeline/theory/joint_design_three_class.py')
for n in ['table_numbers.json','table_readiness.json','final_main_audit.json']:rule(release+'evidence/'+n,'calculation',table+'assemble_refined.py',table+'build_table.py','pipeline/jk9.py')
rule(release+'evidence/fixed_precision_summary.json','calculation',table+'fixed_precision_table.py','pipeline/jk9.py')
rule(release+'evidence/6322_full_comparator_delimitation.json','calculation_and_summary',table+'required_baseline_repairs/6322_direct_comparator.py','pipeline/jk9.py')
rule(release+'evidence/theory_metadata.json','calculation','pipeline/figures/joint_design.py',review+'task2_four_class_jointfit_method_20261025.md')
rule(release+'evidence/theory_unresolved.csv','calculation_and_summary','pipeline/theory/joint_design_three_class.py','pipeline/jk9.py')
rule(release+'assets/*','calculation','pipeline/figures/joint_design.py',review+'task2_four_class_jointfit_method_20261025.md')
predmap={
 'recovered_scores.json':(['pipeline/jk9_prediction.py','pipeline/prediction/posterior.py','pipeline/prediction/importance.py',pred+'task4_calibrated_posterior_prediction_method_20261025.md',pred+'task4_complete_day_prediction_method_20261025.md',pred+'task4_day_verified_score_refresh_method_20261025.md'],'calculation_with_recovered_export'),
 'historical_summary.json':(['pipeline/jk9_prediction.py',pred+'task4_cluster_bootstrap_precision_method_20261025.md'],'calculation'),
 'baseline_rerun_audit.json':(['pipeline/jk9_prediction.py','pipeline/prediction/baselines.py'],'calculation'),
 'native_stan_parity.json':(['pipeline/prediction/parity.py'],'calculation'),
 'splits.json':([pred+'task4_complete_day_training_method_20261025.md',pred+'task4_final_protocol_training_method_20261025.md','pipeline/jk9_prediction.py'],'configuration_and_recovered_export'),
 'artifact_recovery_manifest.json':(['pipeline/git_recovery.py'],'provenance'),
 'source_recovery_manifest.json':(['pipeline/jk9_prediction.py'],'provenance_verifier'),
 'recovered_chain_coverage.json':(['pipeline/git_recovery.py','pipeline/jk9_prediction.py'],'provenance_inventory')}
for n,(src,cat) in predmap.items():rule(release+'evidence/prediction/'+n,cat,*src)
rule('results/releases/*/accepted_fits.json','provenance','pipeline/publication.py','pipeline/jk9.py')
rule('results/releases/*/manifest.json','publication_metadata','pipeline/publication.py','pipeline/jk9.py')
rule(release+'evidence_sources.json','publication_metadata','pipeline/jk9.py')
rule(release+'artifact_inventory.json','historical_storage_metadata','pipeline/cache_store.py')
rows=[]
for path in sorted(Path('results').rglob('*')):
 if not path.is_file():continue
 matched=next(((cat,src) for pattern,cat,src in rules if fnmatch.fnmatch(str(path),pattern)),None)
 if not matched:rows.append(dict(result=str(path),category='unmapped',sources=[]));continue
 cat,src=matched;src=[str(path) if s=='{self}' else s for s in src]
 assert all(Path(s).is_file() for s in src),(path,src)
 rows.append(dict(result=str(path),category=cat,sources=src))
figures=[]
for r in csv.DictReader(Path('docs/PAPER_EVIDENCE.csv').open()):
 src=[r['producer']];status='implementation_located'
 if r['label'].startswith('kaggledata-diagram'):src=[r['evidence'].replace('.pdf','.drawio')];status='editable_diagram_source'
 if r['label']=='fig-S-evol':src=[];status='original_plot_script_not_located'
 if r['label']=='tab-submission-prediction':src=predmap['recovered_scores.json'][0]
 if r['label'] in ['tab-counterfactual-precision','tab-fixed-precision']:src += [table+('assemble_refined.py' if r['label']=='tab-counterfactual-precision' else 'fixed_precision_table.py'),'pipeline/empirical/nonlinear_design.py']
 assert all(Path(s).is_file() for s in src)
 figures.append(dict(label=r['label'],evidence=r['evidence'],status=status,sources=src))
out=dict(date='2026-11-06',scope='All 606 retained files under results/ plus all 23 entries in PAPER_EVIDENCE.csv; ignored historical caches are outside this inventory.',method='Static result-family mapping of writers, controllers and executable method sources. Source lists identify generating calculations, export steps and content verifiers; metadata inventories have their own categories.',counts=dict(collections.Counter(r['category'] for r in rows)),files=rows,paper_items=figures,limitations=['Original plotting script for paper/figure_S_evolve.pdf not located; model variance equations remain in manuscript and solver code.','Recovery/export inventories have retained verifiers and upstream calculation code; exact one-off collector scripts were not located for every inventory.','Generated-run files are produced by the declared calculation stages; result/source mapping is maintained independently of cache retention.'])
Path('docs/RESULT_CODE_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out['counts'],indent=2));print('Paper items',len(figures));print('Unmapped',[r['result'] for r in rows if r['category']=='unmapped'])
