# Task 3：固定原质量目标与先验重校准目标的区别

基于同一六例接受链，补算此前明确待办的固定原目标比较。原初筛对每个新先验重算原设计质量；本次另固定基准先验下的原设计平均质量，比较同一选定反馈/期限达到该目标所需奖金。保留初筛失败场景，未通过重要性门槛的不填目标后验结果。

固定目标来自基准抽样估计，其 Monte Carlo 不确定性没有当成零；奖金比影响函数同时包含该未加权目标和同链加权候选的协方差。仍是一阶、固定反馈/期限分析，不重新优化政策，不保证非线性质量可行。新增比率的影响函数 ESS/MCSE 单独保存，微小差异不只凭原参数诊断下结论。

[比较结果](task3_fixed_target_prior_comparison_20261025.json)。29个原通过场景的新增比率 ESS 诊断最小约723.5，均超过400；13个原失败场景继续不填重要性结果。

```python
import csv,json,math,hashlib
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from pipeline.empirical import estimate as e,nonlinear_reestimate as diag
from pipeline.empirical.fixed_initial_design import Output,beta
p=Path('docs/review/PaperJK9');screen=json.loads((p/'task3_common_cost_prior_screen_20261025.json').read_text());selected={int(r['contest_id']):r for r in csv.DictReader((p/'task3_representative_input_inventory_20261025.csv').open())};records=[]
for cid in selected:
 s=selected[cid];ap=p/'reconstruction_20261025/estimation/full'/f'contest_{cid}/accepted.json';a=json.loads(ap.read_text());assert a['diagnostics']['passed']
 for f,h in zip(a['chain_paths'],a['chain_sha256']):assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h
 chains=[e.read_columns(Path(f)) for f in a['chain_paths']];n=len(chains[0]['sigma']);draws={k:np.concatenate([c[k] for c in chains]) for k in ['c_i','c_j','sigma','lambda','mu_0']};data=json.loads((Path(a['chain_paths'][0]).parent/'data.json').read_text());T0=data['T_contest'];theta0=data['theta'];T1=T0*float(s['time_multiplier']);theta1=theta0*float(s['prize_ratio']);delta=float(s['delta']);cur=Output(draws,T0,48);cand=Output(draws,T1,48);B0=beta(draws['mu_0'],cur.q)+theta0*cur.coefficients(draws['lambda']);fixed_target=float(B0.mean());b=beta(draws['mu_0'],cand.q);precision=0. if s['precision_kind']=='no_feedback_limit' else (math.inf if s['precision_kind']=='full_disclosure' else float(s['precision']));A=cand.coefficients(precision);u=(np.log(draws['c_i'])+np.log(draws['c_j']))/2
 for r in [r for r in screen['records'] if r['contest_id']==cid]:
  item=dict(contest_id=cid,scenario=r['scenario'],screen_passed=r['diagnostics']['passed'],fixed_baseline_target=fixed_target,scope='First-order fixed precision/duration, no reoptimization')
  if not item['screen_passed']:item['reason']='Original importance diagnostics failed; no target/prize result reported';records.append(item);continue
  lw=-.5*((u-r['common_log_mean'])**2/r['common_log_sd']**2-u*u/screen['baseline_common_log_sd']**2);lw[np.minimum(draws['c_i'],draws['c_j'])<r['cost_floor']]=-np.inf;w=np.exp(lw-logsumexp(lw));bmean=float(w@b);Amean=float(w@A);G=fixed_target-bmean;req=G/Amean;influence=((B0-fixed_target)-len(B0)*w*((b-bmean)+req*(A-Amean)))/Amean;ess=diag.effective_sample_size(influence.reshape(4,n));mcse=float(influence.std(ddof=1)/math.sqrt(max(ess,1))/theta0)
  # Additional diagnostic for the new reported ratio, beyond the original screen gate.
  variance_target=float(w@((b+req*A-bmean-req*Amean)**2))/Amean**2;variance_if=float(influence.var(ddof=1));ratio_ess=float(min(len(B0),ess*variance_target/max(variance_if,1e-30)));item['ratio_influence_mcmc_ess']=float(ess);item['ratio_mean_ess_diagnostic']=ratio_ess
  item['results']=dict(fixed_target_required_prize_ratio_unclipped=req/theta0,fixed_target_required_prize_ratio=max(0.,req)/theta0,fixed_target_required_prize_ratio_mcse=mcse,fixed_target_saved_design_quality_gap=bmean+theta1*Amean-fixed_target,recalibrated_target_quality=float(w@B0),recalibrated_target_required_prize_ratio=r['results']['required_prize_ratio_at_same_precision_duration'],recalibrated_target_required_prize_ratio_mcse=r['results']['required_prize_ratio_mcse'],target_change_from_baseline=float(w@B0-fixed_target),fixed_target_candidate_saving_pct=100*(1-(max(0.,req)/theta0+delta*float(s['time_multiplier'])**2)/(1+delta)))
  records.append(item)
 print('fixed baseline target checked',cid,flush=True)
result=dict(status='completed',scope='First-order prior sensitivity at fixed selected precision/duration; compares baseline-fixed and prior-recalibrated targets; does not establish nonlinear feasibility',records=records,baseline_target_estimation_uncertainty_included=True,mcse_method='Paired influence includes unweighted baseline target plus weighted candidate coefficient and baseline quality from the same base chains',failed_scenarios_not_filled=True)
(p/'task3_fixed_target_prior_comparison_20261025.json').write_text(json.dumps(result,indent=2,allow_nan=False))

```
