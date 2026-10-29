# Task 3：48项成本先验/边界敏感性统一汇总

原42项中29项通过重要性诊断，13项改为直接抽样；再加六项floor=.001，总计48项。直接抽样19项由18项NUTS及1项同目标冻结混合MH提供，全部四链哈希与适用诊断重新核对。通过统一接受索引追溯，不把最初NUTS失败诊断改成成功，不混淆HMC与MH。

固定原选定反馈和期限，分别保留重校准原设计目标及固定基准质量目标；候选成本节省按原delta情景计算，仅在重校准目标口径下另列与保留原设计比较后的非负节省。没有重优化反馈/期限，也没有将新先验的一阶结果称作非线性质量保证。截断为零附近MCSE仍对应未截断比率，不能当作截断比率的对称置信区间。

[48项表](task3_cost_prior_combined_48_20261025.csv)、[汇总](task3_cost_prior_combined_summary_20261025.json)、[19项新接受索引](prior_reconstruction_20261025/accepted_index.json)。

汇总口径核查：固定原质量目标下，新先验的原设计未必可行。因此固定目标节省仅相对原实际预算计算，不对负节省作保留原设计兜底的零截断；该兜底只适用于以原设计重新定义质量目标的口径。

```python
import csv,json,hashlib,math
from pathlib import Path
p=Path('docs/review/PaperJK9');old=json.loads((p/'prior_reconstruction_20261025/status.json').read_text());lower=json.loads((p/'prior_reconstruction_20261025/lower_floor_status.json').read_text());assert lower['all_six_lower_floor_accepted'];entries={(r['contest_id'],r['scenario']):r for r in old['finished']+lower['finished']}
for r in json.loads((p/'prior_reconstruction_20261025/supplemental_acceptances.json').read_text())['entries']:entries[(r['contest_id'],r['scenario'])]=r
assert len(entries)==19;accepted=[]
for key,r in sorted(entries.items()):
 ap=Path(r['accepted']);a=json.loads(ap.read_text());assert a['diagnostics']['passed'] and len(a['chain_paths'])==4
 for f,h in zip(a['chain_paths'],a['chain_sha256']):assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h
 accepted.append(dict(contest_id=key[0],scenario=key[1],accepted=str(ap.resolve()),accepted_sha256=hashlib.sha256(ap.read_bytes()).hexdigest(),sampler_family=a['diagnostics'].get('sampler_family','nuts'),chain_sha256=a['chain_sha256']))
(p/'prior_reconstruction_20261025/accepted_index.json').write_text(json.dumps(dict(complete=True,scenarios=19,nuts_scenarios=sum(r['sampler_family']=='nuts' for r in accepted),MH_scenarios=sum(r['sampler_family']=='fixed_mixture_metropolis_hastings' for r in accepted),entries=accepted),indent=2))
screen=json.loads((p/'task3_common_cost_prior_screen_20261025.json').read_text());fixed=json.loads((p/'task3_fixed_target_prior_comparison_20261025.json').read_text());fm={(r['contest_id'],r['scenario']):r for r in fixed['records']};direct=json.loads((p/'task3_direct_prior_results_20261025.json').read_text());assert direct['all_19_scored'];dm={(r['contest_id'],r['scenario']):r for r in direct['records']};selected={int(r['contest_id']):r for r in csv.DictReader((p/'task3_representative_input_inventory_20261025.csv').open())};rows=[]
for r in screen['records']:
 cid=r['contest_id'];key=(cid,r['scenario']);s=selected[cid]
 if r['diagnostics']['passed']:
  q=r['results'];f=fm[key]['results'];row=dict(contest_id=cid,scenario=r['scenario'],method='base_posterior_importance',cost_floor=r['cost_floor'],common_log_mean=r['common_log_mean'],common_log_sd=r['common_log_sd'],recalibrated_required_prize_ratio=q['required_prize_ratio_at_same_precision_duration'],recalibrated_ratio_mcse=q['required_prize_ratio_mcse'],fixed_target_required_prize_ratio=f['fixed_target_required_prize_ratio'],fixed_target_ratio_mcse=f['fixed_target_required_prize_ratio_mcse'],recalibrated_observed_target=f['recalibrated_target_quality'],fixed_baseline_target=fm[key]['fixed_baseline_target'],saved_design_quality_gap=q['saved_design_quality_gap'],saved_design_gap_mcse=q['saved_design_quality_gap_mcse'],numerical_screen_passed=True)
 else:
  d=dm[key];a=json.loads(Path(entries[key]['accepted']).read_text());row=dict(contest_id=cid,scenario=r['scenario'],method=a['diagnostics'].get('sampler_family','nuts'),cost_floor=d['cost_floor'],common_log_mean=r['common_log_mean'],common_log_sd=r['common_log_sd'],recalibrated_required_prize_ratio=d['recalibrated_required_prize_ratio'],recalibrated_ratio_mcse=d['recalibrated_ratio_mcse'],fixed_target_required_prize_ratio=d['fixed_target_required_prize_ratio'],fixed_target_ratio_mcse=d['fixed_target_ratio_mcse'],recalibrated_observed_target=d['observed_recalibrated_target'],fixed_baseline_target=d['baseline_target']['target'],saved_design_quality_gap=d['saved_design_quality_gap'],saved_design_gap_mcse=d['saved_design_quality_gap_mcse'],numerical_screen_passed=d['policy_functional_ess_screen_passed'])
 rows.append(row)
for key,d in dm.items():
 if key[1]!='cost_floor_0001':continue
 row=dict(contest_id=d['contest_id'],scenario=d['scenario'],method='nuts',cost_floor=.001,common_log_mean=0.,common_log_sd=.75/math.sqrt(2),recalibrated_required_prize_ratio=d['recalibrated_required_prize_ratio'],recalibrated_ratio_mcse=d['recalibrated_ratio_mcse'],fixed_target_required_prize_ratio=d['fixed_target_required_prize_ratio'],fixed_target_ratio_mcse=d['fixed_target_ratio_mcse'],recalibrated_observed_target=d['observed_recalibrated_target'],fixed_baseline_target=d['baseline_target']['target'],saved_design_quality_gap=d['saved_design_quality_gap'],saved_design_gap_mcse=d['saved_design_quality_gap_mcse'],numerical_screen_passed=d['policy_functional_ess_screen_passed']);rows.append(row)
assert len(rows)==48 and len({(r['contest_id'],r['scenario']) for r in rows})==48 and all(r['numerical_screen_passed'] for r in rows)
for r in rows:
 s=selected[r['contest_id']];delta=float(s['delta']);t=float(s['time_multiplier']);r['time_multiplier']=t;r['selected_feedback']=s['precision_kind'];r['delay_cost_scenario']=delta;r['recalibrated_candidate_saving_pct']=100*(1-(r['recalibrated_required_prize_ratio']+delta*t*t)/(1+delta));r['fixed_target_candidate_saving_pct']=100*(1-(r['fixed_target_required_prize_ratio']+delta*t*t)/(1+delta));r['retention_adjusted_recalibrated_saving_pct']=max(0.,r['recalibrated_candidate_saving_pct'])
rows.sort(key=lambda r:(r['contest_id'],r['scenario']))
with (p/'task3_cost_prior_combined_48_20261025.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
summary=dict(status='completed',scenarios=48,representative_contests=6,original_importance_passed=29,direct_new_scenarios=19,direct_MH_scenarios=1,all_predeclared_scenarios_numerically_screened=True,scope='First-order sensitivity at fixed originally selected feedback and duration; two target-calibration conventions; no joint reoptimization or nonlinear-prior-robustness claim',point_savings_use_nonnegative_prize=True,MCSEs_refer_to_unclipped_ratio_near_zero_boundary=True,source_sha256={name:hashlib.sha256((p/name).read_bytes()).hexdigest() for name in ['task3_common_cost_prior_screen_20261025.json','task3_fixed_target_prior_comparison_20261025.json','task3_direct_prior_results_20261025.json']});(p/'task3_cost_prior_combined_summary_20261025.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

```
