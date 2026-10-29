# Task 4：两种切分的四模型评分及校准

按竞赛和切分联合匹配，不将日期后验误用于主切分。仅双方73例齐备才作整体比较，所有积分门槛通过前不发布排名。基准先验求积已经独立加密核验。模型差及跨竞赛计算误差使用MCSE之和作为保守计算诊断，允许数值随机数相关；不把它称为确定误差界。竞赛bootstrap不替代积分精度核验。

PIT使用五个事先固定种子，每个种子下所有模型共享同小时/参赛者U；每竞赛平均后73例等权。原始基准种子给出逐竞赛bootstrap区间，另四种给出描述性敏感性，不作小时独立KS检验。所有计数、CDF、PMF合法性和逐竞赛分数与原评分逐项核对。

```python
import csv,json,math,hashlib
from pathlib import Path
import numpy as np
p=Path('docs/review/PaperJK9');basecheck=json.loads((p/'task4_baseline_calibration_input_check_20261025.json').read_text());assert basecheck['all_numerical_checks_passed'];baseline=list(csv.DictReader((p/'task4_baseline_calibrated_predictions_20261025.csv').open()));sources={k:baseline for k in ['A','B']}
posterior={};source_names=['task4_baseline_calibrated_predictions_20261025.csv']
for model in ['C','D']:
 sources[model]=[];posterior[model]=dict(records=[])
 for suffix in ['', '_complete_day']:
  hf=p/f'task4_{model}{suffix}_calibrated_forecast_hours_20261025.csv';jf=p/f'task4_{model}{suffix}_calibrated_forecast_20261025.json'
  if hf.exists() and jf.exists():sources[model].extend(list(csv.DictReader(hf.open())));posterior[model]['records'].extend(json.loads(jf.read_text())['records']);source_names.extend([hf.name,jf.name])
mc={(m,r['contest_id'],r['split']):r for m in ['C','D'] for r in posterior[m]['records']};uniforms={};records=[];counts_seen={}
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
  bins=np.histogram(pit,bins=np.linspace(0,1,11))[0]/len(pit);assert abs(bins.sum()-1)<1e-12;score=float(np.mean([float(r[f'{model}_joint_logscore']) for r in rr]));q=mc.get((model,cid,split));record=dict(contest_id=cid,split=split,model=model,pair_hours=len(rr),marginal_hours=len(pit),mean_joint_logscore=score,zero_observed_fraction=float(np.mean(zero_observed)),zero_predicted_fraction=float(np.mean(zero_predicted)),zero_observed_minus_predicted=float(np.mean(zero_residual)),randomized_PIT_decile_fractions=bins.tolist(),score_mcse=0. if q is None else q['mean_logscore_batch_influence_mcse'],score_numerical_screen_passed=True if q is None else q['mean_score_numerical_screen_passed']);records.append(record)
  if q is not None:assert abs(score-q['mean_joint_logscore'])<1e-12
hour_signatures={}
for model,rows in sources.items():
 for r in rows:
  key=(model,int(r['contest_id']),r['split']);h=int(r['hour_index']);sig=(int(r['count_i']),int(r['count_j']));d=hour_signatures.setdefault(key,{});assert h not in d;d[h]=sig
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
   assert all(hour_signatures[(a,c,split)]==hour_signatures[(b,c,split)] for c in common),'Compared forecast hours/counts differ'
   diff=np.array([ra[c]['mean_joint_logscore']-rb[c]['mean_joint_logscore'] for c in common]);se=np.array([ra[c]['score_mcse']+rb[c]['score_mcse'] for c in common]);index=rng.integers(0,73,size=(2000,73));item.update(mean_joint_logscore_difference=float(diff.mean()),contest_bootstrap_95_interval=np.quantile(diff[index].mean(axis=1),[.025,.975]).tolist(),aggregate_integration_mcse_conservative_sum=float(se.sum()/73),ranking_reportable=all(ra[c]['score_numerical_screen_passed'] and rb[c]['score_numerical_screen_passed'] for c in common))
  comparisons.append(item)
result=dict(status='completed_current_scored_sets',scope='Equal-contest calibration and paired scores; no incomplete-case aggregate comparison, no hour-iid p values',PIT_randomization='Same per-hour/per-player U across models, rng seed=20261025+1009*contest_id; full constructed timeline uniforms indexed by hour',records=records,calibration_summaries=summaries,comparisons=comparisons,all_four_primary_models_73=all(sum(r['model']==m and r['split']=='primary_hour' for r in records)==73 for m in ['A','B','C','D']),randomization_sensitivity_not_yet_checked=False,source_sha256={name:hashlib.sha256((p/name).read_bytes()).hexdigest() for name in source_names})
# Five paired randomizations, keeping every already-scored distribution fixed.
seed_records=[]
for seed in [20261025,20261125,20261225,20261325,20261425]:
 for model,rows in sources.items():
  groups={};us={}
  for r in rows:groups.setdefault((int(r['contest_id']),r['split']),[]).append(r)
  per_split={}
  for (cid,split),rr in groups.items():
   raw=json.loads(next(Path('results/reestimation/lambda100_sd225/input_snapshots/main').glob(f'contest_{cid}_*.json')).read_text());N=raw['N_Delta']
   if cid not in us:us[cid]=np.random.default_rng(seed+1009*cid).uniform(size=(N,2))
   vals=[float(r[f'{model}_cdf_below_{who}'])+us[cid][int(r['hour_index'])-1,k]*float(r[f'{model}_observed_pmf_{who}']) for r in rr for k,who in enumerate(['i','j'])];bins=np.histogram(vals,bins=np.linspace(0,1,11))[0]/len(vals);assert abs(bins.sum()-1)<1e-12;per_split.setdefault(split,[]).append(bins)
  for split,bb in per_split.items():
   if len(bb)==73:
    mean=np.mean(bb,axis=0);seed_records.append(dict(seed=seed,model=model,split=split,contests=73,PIT_deciles_equal_contest_mean=mean.tolist(),descriptive_L1_distance_from_uniform=float(np.sum(abs(mean-.1)))))
result['all_reported_comparisons_match_exact_hour_and_count_signatures']=True;result['paired_randomization_sensitivity']=seed_records;result['numerical_uncertainty_scope']='Within each model, residual chain MCSE plus tail-hour MCSE sums; model-difference and contest-aggregate numerical diagnostics use sums, allowing correlated numerical estimates. These are Monte Carlo diagnostics, not certified deterministic bounds.'
snapshot_root=p/'prediction_score_snapshots_20261025';snapshot_root.mkdir(exist_ok=True);snapshot_records=[]
for name in source_names:
 source=p/name;sha=hashlib.sha256(source.read_bytes()).hexdigest();assert sha==result['source_sha256'][name];snapshot=snapshot_root/(sha+source.suffix)
 if not snapshot.exists():snapshot.write_bytes(source.read_bytes())
 assert hashlib.sha256(snapshot.read_bytes()).hexdigest()==sha;snapshot_records.append(dict(source_name=name,sha256=sha,snapshot=str(snapshot.resolve())))
result['source_snapshots']=snapshot_records
(p/'task4_calibration_comparison_20261025.json').write_text(json.dumps(result,indent=2,allow_nan=False));print('Calibration inputs checked; model sizes',[(s['model'],s['split'],s['contests']) for s in summaries],flush=True)

```
