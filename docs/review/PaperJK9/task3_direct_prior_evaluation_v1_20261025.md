# Task 3：直接先验后验的政策函数核算

仅对19项批次中已通过诊断的接受链计算，核对全部链 SHA。两种目标分别为新先验下重算的原设计质量、固定原基准后验的平均质量。后者显式合并原目标的积分 MCSE与新后验独立抽样误差；前者用配对影响函数。原零奖金/精度/期限设计保留，不重新选择有利设计。

每个抽样的初始方差 sigma/sqrt(lambda) 在不同反馈/期限下保持相同；48→96点求积独立核对，保存原设计质量、选定基准、选定系数的逐抽样最大差。成本下界附近质量与低于原 .01 的概率单列。政策函数 ESS 不足400时明确标记，微小收益需看自身MCSE，不能用参数 Rhat/ESS 替代。负的未截断奖金和其 MCSE保留，但经济奖金下界截断为零，边界附近不把截断值的对称区间当成准确置信区间。

结果仍为一阶、固定选定设计分析；论文原一阶质量约束不能据此升级成非线性质量保证。当前接受子集不替代全部19项。

```python
import csv,json,math,hashlib,time
from pathlib import Path
import numpy as np
from pipeline.empirical import estimate as e,nonlinear_reestimate as nd
from pipeline.empirical.fixed_initial_design import Output,beta
p=Path('docs/review/PaperJK9');state=json.loads((p/'prior_reconstruction_20261025/status.json').read_text());selection={int(r['contest_id']):r for r in csv.DictReader((p/'task3_representative_input_inventory_20261025.csv').open())};basecache={};records=[]
def digest(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def load(a):
 assert a['diagnostics']['passed'];assert all(digest(f)==h for f,h in zip(a['chain_paths'],a['chain_sha256']));chains=[e.read_columns(Path(f)) for f in a['chain_paths']];return {k:np.concatenate([c[k] for c in chains]) for k in ['c_i','c_j','sigma','lambda','mu_0']},len(chains[0]['sigma'])
def mcse(v,n):
 ess=float(nd.effective_sample_size(np.asarray(v).reshape(4,n)));return float(np.std(v,ddof=1)/math.sqrt(max(ess,1))),ess
for entry in state['finished']:
 ap=Path(entry['accepted']);a=json.loads(ap.read_text());cid=entry['contest_id'];s=selection[cid];draws,n=load(a);data=a['configuration']['inputs'];theta=data['theta'];T0=data['T_contest'];T1=T0*float(s['time_multiplier']);theta_saved=theta*float(s['prize_ratio']);precision=0. if s['precision_kind']=='no_feedback_limit' else (math.inf if s['precision_kind']=='full_disclosure' else float(s['precision']))
 if cid not in basecache:
  ba=json.loads((p/'reconstruction_20261025/estimation/full'/f'contest_{cid}/accepted.json').read_text());bd,bn=load(ba);bmod=Output(bd,T0,96);B=beta(bd['mu_0'],bmod.q)+theta*bmod.coefficients(bd['lambda']);bse,bess=mcse(B,bn);basecache[cid]=dict(target=float(B.mean()),target_mcse=bse,target_mcmc_ess=bess,chain_sha256=ba['chain_sha256'])
 base=basecache[cid];coeff={};errors={}
 for order in [48,96]:
  obs=Output(draws,T0,order);cand=Output(draws,T1,order);Bobs=beta(draws['mu_0'],obs.q)+theta*obs.coefficients(draws['lambda']);b=beta(draws['mu_0'],cand.q);A=cand.coefficients(precision);coeff[order]=(Bobs,b,A)
 Bobs,b,A=coeff[96];quaderr={k:float(np.max(abs(x-y))) for k,x,y in zip(['observed_quality','selected_baseline','selected_coefficient'],coeff[96],coeff[48])};Amean=float(A.mean());bmean=float(b.mean());recalibrated=float((Bobs-b).mean()/Amean);rse,ress=mcse((Bobs-b-recalibrated*A)/Amean,n);fixed=float((base['target']-bmean)/Amean);fi=-(b-bmean+fixed*(A-Amean))/Amean;fse,fess=mcse(fi,n);fse=math.sqrt(fse*fse+(base['target_mcse']/Amean)**2);gap=b+theta_saved*A-Bobs;gse,gess=mcse(gap,n);bs,be=mcse(Bobs,n)
 row=dict(contest_id=cid,scenario=entry['scenario'],posterior_draws=4*n,accepted_sha256=digest(ap),chain_sha256=a['chain_sha256'],baseline_target=base,observed_recalibrated_target=float(Bobs.mean()),observed_target_mcse=bs,observed_target_mcmc_ess=be,recalibrated_required_prize_ratio_unclipped=recalibrated/theta,recalibrated_required_prize_ratio=max(0.,recalibrated)/theta,recalibrated_ratio_mcse=rse/theta,recalibrated_ratio_influence_ess=ress,fixed_target_required_prize_ratio_unclipped=fixed/theta,fixed_target_required_prize_ratio=max(0.,fixed)/theta,fixed_target_ratio_mcse=fse/theta,fixed_target_ratio_influence_ess=fess,saved_design_quality_gap=float(gap.mean()),saved_design_quality_gap_mcse=gse,saved_design_quality_gap_ess=gess,selected_feedback=s['precision_kind'],time_multiplier=float(s['time_multiplier']),saved_prize_ratio=float(s['prize_ratio']),cost_floor=data['cost_floor'],posterior_any_cost_below_old_001_fraction=float(np.mean(np.minimum(draws['c_i'],draws['c_j'])<.01)),posterior_any_cost_within_10pct_floor_fraction=float(np.mean(np.minimum(draws['c_i'],draws['c_j'])<1.1*data['cost_floor'])),quadrature_48_vs_96_max_draw_difference=quaderr,parameter_diagnostics_passed=True,policy_functionals_minimum_mcmc_ess=min(be,ress,fess,gess),scope='First-order fixed selected feedback and duration; posterior target calibration is separated from fixed-original target; no joint reoptimization or nonlinear feasibility claim')
 row['policy_functional_ess_screen_passed']=bool(row['policy_functionals_minimum_mcmc_ess']>=400);records.append(row);print('PRIOR POLICY',cid,entry['scenario'],'recalibrated ratio',row['recalibrated_required_prize_ratio'],'MCSE',row['recalibrated_ratio_mcse'],'ESS screen',row['policy_functional_ess_screen_passed'],flush=True)
result=dict(status='completed_current_accepted_subset',required_scenarios=19,scored_scenarios=len(records),all_19_scored=len(records)==19,records=records,baseline_target_mcse_combined_for_independent_new_fits=True,nonlinear_prior_robustness_established=False)
(p/'task3_direct_prior_results_20261025.json').write_text(json.dumps(result,indent=2,allow_nan=False))

```
