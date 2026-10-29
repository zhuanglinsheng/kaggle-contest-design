# Task 3：零奖金代表设计的后验非线性核验

8219已由原规则选为零奖金质量余量最大的案例，不依据本次结果选样。从四条接受链各32个时间段预选一个实际联合抽样，共128个；固定随机种子和所有索引。所选零奖金设计的输出为解析无努力期望，原设计逐抽样计算非线性HJB与线性产出方程，做两层空间加密和必要的时间步加密。

失败样本保留，附原设计无努力下界与能量上界；不删除样本或用成功子集估后验均值。数值半径为max(3*两层差,1e-7)，属于诊断，不是严格连续误差界。初始方差保持该抽样原sigma/sqrt(lambda)，不同期限/反馈不重新初始化。

这是在原一阶估计后验中的非线性验证，不能称作非线性重新估计或新的最优零奖金政策。首个大幅质量缺口2551的结果保持原样；本批补足另一种设计家族的后验平均证据。

[实时结果](task3_posterior_nonlinear_8219_20261025.json)。

```python
import csv,json,math,hashlib,os,time
from pathlib import Path
import numpy as np
from pipeline.empirical import estimate as e
from pipeline.synthetic.nonlinear import Primitives,variance
from pipeline.theory.joint_design_exact import solve_equilibrium_adaptive,expected_best_output
from pipeline.empirical.fixed_initial_design import Output,beta
p=Path('docs/review/PaperJK9');out=p/'task3_posterior_nonlinear_8219_20261025.json';assert not out.exists(),'Inspect existing nonlinear batch before resume';ap=p/'reconstruction_20261025/estimation/full/contest_8219/accepted.json';a=json.loads(ap.read_text());assert a['diagnostics']['passed'];digest=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest();assert all(digest(f)==h for f,h in zip(a['chain_paths'],a['chain_sha256']));cc=[e.read_columns(Path(f)) for f in a['chain_paths']];data=json.loads((Path(a['chain_paths'][0]).parent/'data.json').read_text());sel=next(r for r in csv.DictReader((p/'task3_representative_input_inventory_20261025.csv').open()) if r['contest_id']=='8219');assert float(sel['prize_ratio'])==0;seed=202610258219;rng=np.random.default_rng(seed);selection=[]
for chain,c in enumerate(cc):
 bins=np.linspace(0,len(c['sigma']),33,dtype=int)
 for j in range(32):selection.append(dict(chain=chain+1,index=int(rng.integers(bins[j],bins[j+1]))))
state=dict(status='running',driver_pid=os.getpid(),scope='128 preselected actual first-order posterior draws: nonlinear original design versus saved zero-prize design; fixed initial distribution within each draw',selection_seed=seed,selection=selection,selected_policy=sel,chain_sha256=a['chain_sha256'],PDE_source_sha256=digest('pipeline/theory/joint_design_exact.py'),records=[],no_draws_dropped=True,strict_continuous_error_certificate=False,posterior_average_quality_verified=False)
def save():
 tmp=out.with_suffix('.json.tmp');tmp.write_text(json.dumps(state,indent=2,allow_nan=False));tmp.replace(out)
save();clock=time.monotonic()
for serial,s in enumerate(selection):
 c=cc[s['chain']-1];z={k:float(c[k][s['index']]) for k in e.PARAMETERS};S0=z['sigma']/math.sqrt(z['lambda']);T0=data['T_contest'];theta=data['theta'];T1=T0*float(sel['time_multiplier']);pr=Primitives(T=T0,theta=theta,c_i=z['c_i'],c_j=z['c_j'],sigma=z['sigma'],lamb=z['lambda'],S0=S0,mu0=z['mu_0']);q0=S0+z['sigma']**2*T0;scale=math.sqrt(q0);dy=min(.02*scale,.25*math.sqrt(float(variance(T0,pr))));extent=max(7.,(abs(z['mu_0'])+8*math.sqrt(S0))/scale);Bzero=float(beta(z['mu_0'],S0+z['sigma']**2*T1));Bbase=float(beta(z['mu_0'],q0));item=dict(serial=serial,selection=s,point=z,S0=S0,original_theta=theta,original_T=T0,selected_T=T1,selected_nonlinear_B=Bzero,analytic_original_baseline=Bbase,analytic_original_upper=Bbase+math.sqrt(2*theta*T0/min(z['c_i'],z['c_j'])),original_trials=[])
 for level in [1,2]:
  for steps in ([2000,4000,8000] if level==1 else [4000,8000,16000]):
   trial=dict(mesh_level=level,dy=dy/level,base_steps=steps,extent=extent)
   try:
    eq=solve_equilibrium_adaptive(pr,dy=dy/level,base_steps=steps,extent=extent);assert eq.diagnostics['value_min']>=-1e-7 and eq.diagnostics['value_max']<=1+1e-7;trial['B']=float(expected_best_output(eq));trial['diagnostics']=eq.diagnostics;del eq
   except Exception as exc:trial['error']=str(exc)
   item['original_trials'].append(trial)
   if 'B' in trial:break
 good={level:[t for t in item['original_trials'] if t['mesh_level']==level and 'B' in t] for level in [1,2]}
 if all(good.values()):
  coarse=good[1][-1]['B'];fine=good[2][-1]['B'];item.update(observed_nonlinear_B=fine,nonlinear_gap=Bzero-fine,observed_diagnostic_radius=max(3*abs(fine-coarse),1e-7))
 else:item['observed_nonlinear_unresolved']=True
 draws={k:np.array([z[k]]) for k in ['c_i','c_j','sigma','lambda','mu_0']};fo=Output(draws,T0,96);item['observed_first_order_B']=fo.no_effort+theta*fo.value(draws['lambda']);item['first_order_gap']=Bzero-item['observed_first_order_B'];state['records'].append(item);state['elapsed_seconds']=time.monotonic()-clock;save()
 if (serial+1)%8==0:print('NONLINEAR 8219',serial+1,'/128','unresolved',sum(r.get('observed_nonlinear_unresolved',False) for r in state['records']),flush=True)
state['status']='completed';state['all_128_resolved']=all('nonlinear_gap' in r for r in state['records']);save();print('NONLINEAR 8219 BATCH FINISHED',state['all_128_resolved'],flush=True)

```
