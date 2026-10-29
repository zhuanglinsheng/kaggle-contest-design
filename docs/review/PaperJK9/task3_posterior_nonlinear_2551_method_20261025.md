# Task 3：2551 的后验平均非线性压力测试

日期：2026-10-25。代表点的一阶质量约束在非线性评价下翻转后，继续在 128 个实际联合抽样上评价同一保存设计，避免仅凭单点推断论文的后验平均结果。

从四条已接受链各分为 32 个时间段，每段按预设随机种子选一个样本；索引在输出计算前固定。每样本同时算原有限精度设计与所选全披露设计，固定该样本原初始方差和均值；同时保存同样本的一阶差，以区分点选择与近似误差。

原设计使用现有 HJB 与线性产出方程，全披露使用解析策略与自相似线性输出边值问题。逐样本保留失败项，禁止按成功结果静默删样本。失败项另存解析无努力下界与能量上界，供限定均值范围；这些界不替代成功求解的误差诊断。随后还需子集加密、配对量 MCSE 与权重/先验核查，完成前不认证质量约束。

这是在一阶估计后验下进行非线性设计验证，不是非线性模型的重新估计。原论文和实现均不修改。

[逐抽样记录](task3_posterior_nonlinear_2551_20261025.json)。

```python
import csv,json,math,time,hashlib
from pathlib import Path
import numpy as np
from scipy.special import roots_hermitenorm
from pipeline.empirical import estimate as estimator
from pipeline.synthetic.nonlinear import Primitives,variance
from pipeline.theory.joint_design_exact import solve_equilibrium_adaptive,expected_best_output
from pipeline.empirical.fixed_initial_design import Output,beta
report=Path('docs/review/PaperJK9');method=(report/'task3_full_disclosure_asymmetric_method_20261025.md').read_text().split('```python\n',1)[1].split('```',1)[0];exec(compile(method,'<reported analytic full-disclosure strategy>','exec'))
accepted=report/'reconstruction_20261025/estimation/full/contest_2551/accepted.json';a=json.loads(accepted.read_text());assert a['diagnostics']['passed']
for path,h in zip(a['chain_paths'],a['chain_sha256']):assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
chains=[estimator.read_columns(Path(f)) for f in a['chain_paths']];data=json.loads((Path(a['chain_paths'][0]).parent/'data.json').read_text());selected=next(r for r in csv.DictReader((report/'task3_representative_input_inventory_20261025.csv').open()) if r['contest_id']=='2551')
rng=np.random.default_rng(202610255512);selection=[]
for chain,c in enumerate(chains):
 bins=np.linspace(0,len(c['c_i']),33,dtype=int)
 for j in range(32):selection.append(dict(chain=chain+1,index=int(rng.integers(bins[j],bins[j+1]))))
out=report/'task3_posterior_nonlinear_2551_20261025.json'
if out.exists():raise RuntimeError('Existing posterior batch: inspect before resume')
state=dict(date='2026-10-25',status='running',scope='128 stratified actual draws from accepted first-order posterior, paired nonlinear observed and saved selected full-disclosure design; no nonlinear reestimation',chain_sha256=a['chain_sha256'],selected_policy=selected,selection_seed=202610255512,selection=selection,full_method_sha256=hashlib.sha256(method.encode()).hexdigest(),records=[],posterior_average_quality_verified=False)
def save():
 tmp=out.with_suffix('.json.tmp');tmp.write_text(json.dumps(state,indent=2,allow_nan=False));tmp.replace(out)
save()
for serial,s in enumerate(selection):
 c=chains[s['chain']-1];z={k:float(c[k][s['index']]) for k in estimator.PARAMETERS};T0=data['T_contest'];theta0=data['theta'];S0=z['sigma']/math.sqrt(z['lambda']);T1=T0*float(selected['time_multiplier']);theta1=theta0*float(selected['prize_ratio']);record=dict(serial=serial,selection=s,point=z,S0=S0,observed_theta=theta0,observed_T=T0,selected_theta=theta1,selected_T=T1)
 try:
  p=Primitives(T=T0,theta=theta0,c_i=z['c_i'],c_j=z['c_j'],sigma=z['sigma'],lamb=z['lambda'],S0=S0,mu0=z['mu_0']);scale=math.sqrt(S0+z['sigma']**2*T0);dy=min(.02*scale,.25*math.sqrt(float(variance(T0,p))));extent=max(7.,(abs(z['mu_0'])+8*math.sqrt(S0))/scale)
  eq=solve_equilibrium_adaptive(p,dy=dy,base_steps=2000,extent=extent);record['observed_nonlinear_B']=float(expected_best_output(eq));record['observed_solver_diagnostics']=eq.diagnostics;del eq
 except Exception as exc:record['observed_error']=str(exc)
 try:
  gh,_=roots_hermitenorm(80);ext=max(12.,float(np.max(np.abs((z['mu_0']+math.sqrt(S0)*gh)/(z['sigma']*math.sqrt(T1)))))+1)
  record['selected_full_result']=full_asymmetric_output(theta1,T1,z['sigma'],z['c_i'],z['c_j'],S0,z['mu_0'],extent=ext,tolerance=1e-8,quadrature=80)
 except Exception as exc:record['selected_error']=str(exc)
 draws={k:np.array([z[k]]) for k in ('c_i','c_j','sigma','lambda','mu_0')};m0=Output(draws,T0,48);m1=Output(draws,T1,48);record['observed_first_order_B']=m0.no_effort+theta0*m0.value(draws['lambda']);record['selected_first_order_B']=m1.no_effort+theta1*m1.value(math.inf)
 record['analytic_observed_baseline']=float(beta(z['mu_0'],S0+z['sigma']**2*T0));record['analytic_selected_baseline']=float(beta(z['mu_0'],S0+z['sigma']**2*T1));record['generic_observed_output_upper']=record['analytic_observed_baseline']+math.sqrt(2*theta0*T0/min(z['c_i'],z['c_j']));record['generic_selected_output_upper']=record['analytic_selected_baseline']+math.sqrt(2*theta1*T1/min(z['c_i'],z['c_j']))
 state['records'].append(record);save()
 if (serial+1)%8==0:print('posterior nonlinear 2551',serial+1,'/128',flush=True)
state['status']='completed';save();print('Posterior nonlinear pressure batch completed.',flush=True)
```
