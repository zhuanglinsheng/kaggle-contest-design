# Task 3：共同成本尺度先验与下界的受控初筛

日期：2026-10-25。仓库已有 cost_prior_trial 的共同/相对成本分解及依赖调整的重要性 ESS 方法；原实验使用不同的源先验和数据，不能直接当作当前统一先验的敏感性结果。这里针对 6 个新重建后验重新计算。

当前独立 log 成本 SD=.75 等价于共同尺度 u=(log ci+log cj)/2 的 SD=.75/sqrt(2)，相对成本差 d=log ci-log cj 的 SD=.75*sqrt(2)。仅将共同尺度中位数减半/翻倍或将其 SD 放宽为 .75/1，保持相对成本的高斯密度因子及其他先验不变；另检查提高双方成本下界至 .02/.05。降低原 .01 下界不能用原抽样完成，需新拟合。

共同硬下界仍会耦合 u 与 d；保持相对成本高斯因子不变，不等于保证截断后的相对成本边际分布完全不变。密度比保留相同支持，截断归一常数在自归一后验权重中消去。

采用已有工作流的门槛：权重 ESS 与各报告量的依赖调整均值 ESS 至少 400、最大归一权重小于 .02。均值 ESS 由自归一权重影响函数的 MCMC ESS 与目标方差计算。失败场景只报告诊断，不报告目标后验结论。

同时重算每个先验下的原设计质量目标与固定所选精度、期限的所需奖金；不混用原后验目标。奖金比的 MCSE 使用分子/分母的配对影响函数。精度和期限尚未重优化，非线性稳健性也未认证；微小节省需结合配对 MCSE，不能只看一般参数 Rhat/ESS。

[逐场景记录](task3_common_cost_prior_screen_20261025.json)。

```python
import csv,json,math,hashlib
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from pipeline.empirical import estimate as estimator
from pipeline.empirical import nonlinear_reestimate as diagnostics
from pipeline.empirical.fixed_initial_design import Output,beta
report=Path('docs/review/PaperJK9');selection=list(csv.DictReader((report/'task3_representative_input_inventory_20261025.csv').open()));out=report/'task3_common_cost_prior_screen_20261025.json'
if out.exists():raise RuntimeError('Existing prior screen: inspect before resume')
su0=.75/math.sqrt(2)
scenarios=[('baseline',0.,su0,.01),('common_median_half',-math.log(2),su0,.01),('common_median_double',math.log(2),su0,.01),('common_sd_075',0.,.75,.01),('common_sd_100',0.,1.,.01),('cost_floor_02',0.,su0,.02),('cost_floor_05',0.,su0,.05)]
state=dict(status='running',scope='First-order fixed-design common-cost-prior sensitivity using accepted reconstructed posterior; no nonlinear robustness or joint reoptimization claim',baseline_common_log_sd=su0,relative_log_cost_sd_preserved=.75*math.sqrt(2),diagnostic_rule='Weight ESS and dependence-adjusted mean ESS >=400, max normalized weight <.02. Failing scenarios retain diagnostics only.',records=[])
def save():
 tmp=out.with_suffix('.json.tmp');tmp.write_text(json.dumps(state,indent=2,allow_nan=False));tmp.replace(out)
save()
for selected in selection:
 cid=int(selected['contest_id']);ap=report/'reconstruction_20261025/estimation/full'/f'contest_{cid}/accepted.json';a=json.loads(ap.read_text());assert a['diagnostics']['passed']
 for f,h in zip(a['chain_paths'],a['chain_sha256']):assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h
 cc=[estimator.read_columns(Path(f)) for f in a['chain_paths']];nchain=len(cc[0]['c_i']);draws={k:np.concatenate([c[k] for c in cc]) for k in ('c_i','c_j','sigma','lambda','mu_0')};rr=np.concatenate([np.exp(c['log_r']) for c in cc]);n=len(rr)
 data=json.loads((Path(a['chain_paths'][0]).parent/'data.json').read_text());T0=data['T_contest'];theta0=data['theta'];T1=T0*float(selected['time_multiplier']);theta1=theta0*float(selected['prize_ratio']);delta=float(selected['delta']);current=Output(draws,T0,48);candidate=Output(draws,T1,48)
 obsA=current.coefficients(draws['lambda']);precision=0. if selected['precision_kind']=='no_feedback_limit' else (math.inf if selected['precision_kind']=='full_disclosure' else float(selected['precision']))
 A=candidate.coefficients(precision);b0obs=beta(draws['mu_0'],current.q);b0sel=beta(draws['mu_0'],candidate.q);Bobs=b0obs+theta0*obsA;G=Bobs-b0sel;gap=b0sel+theta1*A-Bobs;u=(np.log(draws['c_i'])+np.log(draws['c_j']))/2
 quantities={'c_i':draws['c_i'],'c_j':draws['c_j'],'r':rr,'sigma':draws['sigma'],'lambda':draws['lambda'],'cost_ratio':draws['c_i']/draws['c_j'],'target_quality':Bobs,'required_quality_increment':G,'selected_coefficient':A,'saved_design_quality_gap':gap}
 for label,mean,sd,floor in scenarios:
  lw=-.5*((u-mean)**2/sd**2-u*u/su0**2);support=np.minimum(draws['c_i'],draws['c_j'])>=floor;lw[~support]=-np.inf
  item=dict(contest_id=cid,scenario=label,common_log_mean=mean,common_log_sd=sd,cost_floor=floor,relative_log_cost_sd=.75*math.sqrt(2),posterior_draws=n,chain_sha256=a['chain_sha256'],retained_support_fraction=float(support.mean()))
  if not support.any():item['diagnostics']=dict(passed=False,reason='No overlap');state['records'].append(item);continue
  w=np.exp(lw-logsumexp(lw));means={k:float(w@v) for k,v in quantities.items()};esss={};mcses={}
  for name,v in quantities.items():
   influence=(n*w*(v-means[name])).reshape(4,nchain);eff=float(diagnostics.effective_sample_size(influence));var=float(influence.var(ddof=1));targetvar=float(w@((v-means[name])**2));esss[name]=float(min(n,eff*targetvar/max(var,1e-30)));mcses[name]=math.sqrt(var/max(eff,1))
  gate=dict(weight_ess=float(1/(w@w)),minimum_dependence_adjusted_mean_ess=min(esss.values()),maximum_normalized_weight=float(w.max()),quantity_mean_ess=esss)
  gate['passed']=bool(gate['weight_ess']>=400 and gate['minimum_dependence_adjusted_mean_ess']>=400 and gate['maximum_normalized_weight']<.02);item['diagnostics']=gate
  if gate['passed']:
   req_unclipped=means['required_quality_increment']/means['selected_coefficient'];req=max(0.,req_unclipped);influence=(n*w*(G-req_unclipped*A)/means['selected_coefficient']).reshape(4,nchain);eff=diagnostics.effective_sample_size(influence);ratio_mcse=float(np.std(influence,ddof=1)/math.sqrt(max(eff,1))/theta0)
   item['results']=dict(target_first_order_quality=means['target_quality'],saved_design_quality_gap=means['saved_design_quality_gap'],saved_design_quality_gap_mcse=mcses['saved_design_quality_gap'],required_prize_ratio_at_same_precision_duration=req/theta0,unclipped_required_prize_ratio=req_unclipped/theta0,required_prize_ratio_mcse=ratio_mcse,retained_original_design_if_candidate_cost_higher=bool(req/theta0+delta*float(selected['time_multiplier'])**2>=1+delta),candidate_saving_pct_before_retention=100*(1-(req/theta0+delta*float(selected['time_multiplier'])**2)/(1+delta)),posterior_cost_ratio_mean=means['cost_ratio'],posterior_r_mean=means['r'])
  state['records'].append(item);save()
 print('common-cost prior screen',cid,'done',flush=True)
state['status']='completed';state['relax_lower_bound_below_001_requires_fresh_fit']=True;save()
```
