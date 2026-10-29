# Task 4：四模型的联合评分与概率校准汇总

所有模型采用每小时/参赛者相同的随机数 U，PIT=F(k-1)+U*P(k)。观察计数只用于评分分布，不用于测试期参数拟合。十等分校准与零事件误差先在各竞赛内平均，再对竞赛等权；区间按竞赛整体 bootstrap，不把小时当作独立样本。随机化敏感性尚待多种固定种子检查，PIT图形不作小时独立KS检验。

只有某个模型全部73例时才汇总校准，只有比较双方均73例时才算整体分数差，不以当前成功子集排名。主要增量比较 D−C，D−A/B为活动率基准对照，B−A保留已有结果；有积分精度初筛失败时排名保持未核验，不能靠竞赛 bootstrap消除积分误差。

```python
import csv,json,math,hashlib
from pathlib import Path
import numpy as np
p=Path('docs/review/PaperJK9');basecheck=json.loads((p/'task4_baseline_calibration_input_check_20261025.json').read_text());assert basecheck['all_numerical_checks_passed'];baseline=list(csv.DictReader((p/'task4_baseline_calibrated_predictions_20261025.csv').open()));sources={k:baseline for k in ['A','B']}
for model in ['C','D']:sources[model]=list(csv.DictReader((p/f'task4_{model}_calibrated_forecast_hours_20261025.csv').open()))
posterior={m:json.loads((p/f'task4_{m}_calibrated_forecast_20261025.json').read_text()) for m in ['C','D']};mc={(m,r['contest_id']):r for m in ['C','D'] for r in posterior[m]['records']};uniforms={};records=[];counts_seen={}
for model,rows in sources.items():
 groups={}
 for r in rows:groups.setdefault((int(r['contest_id']),r['split']),[]).append(r)
 for (cid,split),rr in groups.items():
  rr.sort(key=lambda r:int(r['hour_index']));raw=json.loads(next(Path('results/reestimation/lambda100_sd225/input_snapshots/main').glob(f'contest_{cid}_*.json')).read_text());N=raw['N_Delta']
  if cid not in uniforms:uniforms[cid]=np.random.default_rng(20261025+1009*cid).uniform(size=(N,2))
  pit=[];zero_residual=[];zero_observed=[];zero_predicted=[]
  for r in rr:
   h=int(r['hour_index']);signature=(int(r['count_i']),int(r['count_j']));key=(cid,split,h)
   if key in counts_seen:assert counts_seen[key]==signature
   else:counts_seen[key]=signature
   for k,who in enumerate(['i','j']):
    lower=float(r[f'{model}_cdf_below_{who}']);mass=float(r[f'{model}_observed_pmf_{who}']);assert lower>=-1e-12 and mass>=-1e-12 and lower+mass<=1+1e-10;v=lower+uniforms[cid][h-1,k]*mass;assert -1e-12<=v<=1+1e-10;pit.append(min(1.,max(0.,v)));obs=int(r[f'count_{who}'])==0;pr=float(r[f'{model}_zero_{who}']);zero_observed.append(float(obs));zero_predicted.append(pr);zero_residual.append(float(obs)-pr)
  bins=np.histogram(pit,bins=np.linspace(0,1,11))[0]/len(pit);assert abs(bins.sum()-1)<1e-12;score=float(np.mean([float(r[f'{model}_joint_logscore']) for r in rr]));q=mc.get((model,cid));record=dict(contest_id=cid,split=split,model=model,pair_hours=len(rr),marginal_hours=len(pit),mean_joint_logscore=score,zero_observed_fraction=float(np.mean(zero_observed)),zero_predicted_fraction=float(np.mean(zero_predicted)),zero_observed_minus_predicted=float(np.mean(zero_residual)),randomized_PIT_decile_fractions=bins.tolist(),score_mcse=0. if q is None else q['mean_logscore_batch_influence_mcse'],score_numerical_screen_passed=True if q is None else q['mean_score_numerical_screen_passed']);records.append(record)
  if q is not None:assert abs(score-q['mean_joint_logscore'])<1e-12
summaries=[];comparisons=[];rng=np.random.default_rng(20261025+904)
for split in ['primary_hour','complete_day']:
 for m in ['A','B','C','D']:
  rr=[r for r in records if r['split']==split and r['model']==m];item=dict(model=m,split=split,contests=len(rr),all_73=len(rr)==73)
  if len(rr)==73:
   z=np.array([r['zero_observed_minus_predicted'] for r in rr]);bb=np.array([r['randomized_PIT_decile_fractions'] for r in rr]);index=rng.integers(0,73,size=(2000,73));item.update(zero_observed_minus_predicted_equal_contest_mean=float(z.mean()),zero_error_contest_bootstrap_95_interval=np.quantile(z[index].mean(axis=1),[.025,.975]).tolist(),PIT_deciles_equal_contest_mean=bb.mean(axis=0).tolist(),PIT_decile_contest_bootstrap_95_intervals=np.quantile(bb[index].mean(axis=1),[.025,.975],axis=0).tolist())
  summaries.append(item)
 for a,b in [('D','C'),('D','A'),('D','B'),('B','A')]:
  ra={r['contest_id']:r for r in records if r['split']==split and r['model']==a};rb={r['contest_id']:r for r in records if r['split']==split and r['model']==b};common=sorted(ra.keys()&rb.keys());item=dict(split=split,model_difference=a+'-'+b,matched_contests=len(common),all_73=len(common)==73,ranking_reportable=False)
  if len(common)==73:
   diff=np.array([ra[c]['mean_joint_logscore']-rb[c]['mean_joint_logscore'] for c in common]);se=np.array([math.hypot(ra[c]['score_mcse'],rb[c]['score_mcse']) for c in common]);index=rng.integers(0,73,size=(2000,73));item.update(mean_joint_logscore_difference=float(diff.mean()),contest_bootstrap_95_interval=np.quantile(diff[index].mean(axis=1),[.025,.975]).tolist(),aggregate_independent_integration_mcse=float(np.sqrt(np.sum(se*se))/73),ranking_reportable=all(ra[c]['score_numerical_screen_passed'] and rb[c]['score_numerical_screen_passed'] for c in common))
  comparisons.append(item)
result=dict(status='completed_current_scored_sets',scope='Equal-contest calibration and paired scores; no incomplete-case aggregate comparison, no hour-iid p values',PIT_randomization='Same per-hour/per-player U across models, rng seed=20261025+1009*contest_id; full constructed timeline uniforms indexed by hour',records=records,calibration_summaries=summaries,comparisons=comparisons,all_four_primary_models_73=all(sum(r['model']==m and r['split']=='primary_hour' for r in records)==73 for m in ['A','B','C','D']),randomization_sensitivity_not_yet_checked=True,source_sha256={name:hashlib.sha256((p/name).read_bytes()).hexdigest() for name in ['task4_baseline_calibrated_predictions_20261025.csv','task4_C_calibrated_forecast_hours_20261025.csv','task4_D_calibrated_forecast_hours_20261025.csv']})
(p/'task4_calibration_comparison_20261025.json').write_text(json.dumps(result,indent=2,allow_nan=False));print('Calibration inputs checked; model sizes',[(s['model'],s['split'],s['contests']) for s in summaries],flush=True)

```
