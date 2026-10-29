# Task 3：其余全披露代表案例的后验配对核验

23652此前只有一个联合代表点的非线性结果；不能代替其预先选定案例的后验平均检查。四条已接受full链各32个分层随机样本，输出前固定索引，不删失败。每样本保留原初值均值/方差，原有限精度使用既有HJB与独立线性产出方程；所选完全披露使用既有解析策略/BVP或价值坐标输出方程。两层网格与求积逐样本加密，保留失败及解析基线/能量范围。

这仍是在一阶估计后验上的非线性设计评价，不是非线性重新估计。数值半径为诊断，不是严格连续误差界；极小质量差需结合后验MCSE及数值诊断解释，不强行给出节省正负。只读模型源码，审查目录保存索引、过程和报告。

```python
import json,csv,math,time,hashlib,os
from pathlib import Path
import numpy as np
from scipy.special import roots_hermitenorm
from pipeline.empirical import estimate as e
from pipeline.empirical.fixed_initial_design import Output,beta
from pipeline.synthetic.nonlinear import Primitives,variance
from pipeline.theory.joint_design_exact import solve_equilibrium_adaptive,expected_best_output
p=Path('docs/review/PaperJK9');CID=globals().get('CID',23652);status=p/f'task3_posterior_remaining_full_{CID}_20261025.json';assert not status.exists()
def code(f):return f.read_text().partition(chr(10)+chr(96)*3+'python'+chr(10))[2].rpartition(chr(10)+chr(96)*3)[0]
def digest(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
fullenv={};exec(compile(code(p/'task3_full_disclosure_asymmetric_method_20261025.md'),'existing-full-disclosure-output','exec'),fullenv);valueenv={};exec(compile(code(p/'task3_full_disclosure_value_grid_method_20261025.md'),'existing-value-coordinate-output','exec'),valueenv)
ix=json.loads((p/'reconstruction_20261025/accepted_index.json').read_text());entry=next(r for r in ix['entries'] if r['contest_id']==CID);ap=Path(entry['accepted']);ap=ap if ap.is_absolute() else p/'reconstruction_20261025'/ap;assert digest(ap)==entry['accepted_sha256'];a=json.loads(ap.read_text());assert a['diagnostics']['passed'];assert all(digest(f)==h for f,h in zip(a['chain_paths'],a['chain_sha256']));chains=[e.read_columns(Path(f)) for f in a['chain_paths']];data=json.loads((Path(a['chain_paths'][0]).parent/'data.json').read_text());policy=next(r for r in csv.DictReader((p/'task3_representative_input_inventory_20261025.csv').open()) if int(r['contest_id'])==CID);assert policy['choice']=='full_disclosure';rng=np.random.default_rng(20261025+CID*1009);selection=[]
for chain,c in enumerate(chains):
 bins=np.linspace(0,len(c['c_i']),33,dtype=int)
 for j in range(32):selection.append(dict(chain=chain+1,index=int(rng.integers(bins[j],bins[j+1]))))
state=dict(status='running',driver_pid=os.getpid(),contest_id=CID,scope='128 preselected actual first-order-posterior draws; paired observed and saved full-disclosure design, all draws retained',selection_seed=20261025+CID*1009,selection=selection,accepted=str(ap.resolve()),accepted_sha256=digest(ap),chain_sha256=a['chain_sha256'],selected_policy=policy,records=[],strict_continuous_error_certificate=False)
def save():
 tmp=status.with_suffix('.json.tmp');tmp.write_text(json.dumps(state,indent=2,allow_nan=False));tmp.replace(status)
save();clock=time.monotonic()
for serial,s in enumerate(selection):
 c=chains[s['chain']-1];z={k:float(c[k][s['index']]) for k in e.PARAMETERS};T0=data['T_contest'];theta0=data['theta'];T1=T0*float(policy['time_multiplier']);theta1=theta0*float(policy['prize_ratio']);S0=z['sigma']/math.sqrt(z['lambda']);item=dict(serial=serial,selection=s,point=z,S0=S0,observed_theta=theta0,observed_T=T0,selected_theta=theta1,selected_T=T1,observed_trials=[],selected_trials=[]);pr=Primitives(T=T0,theta=theta0,c_i=z['c_i'],c_j=z['c_j'],sigma=z['sigma'],lamb=z['lambda'],S0=S0,mu0=z['mu_0']);scale=math.sqrt(S0+z['sigma']**2*T0);dy=min(.02*scale,.25*math.sqrt(float(variance(T0,pr))));extent=max(7.,(abs(z['mu_0'])+8*math.sqrt(S0))/scale)
 for level in [1,2]:
  for steps in ([2000,4000,8000] if level==1 else [4000,8000,16000]):
   trial=dict(level=level,dy=dy/level,base_steps=steps,extent=extent)
   try:
    eq=solve_equilibrium_adaptive(pr,dy=dy/level,base_steps=steps,extent=extent);assert eq.diagnostics['value_min']>=-1e-7 and eq.diagnostics['value_max']<=1+1e-7;trial.update(B=float(expected_best_output(eq)),diagnostics=eq.diagnostics);del eq
   except Exception as ex:trial['error']=repr(ex)
   item['observed_trials'].append(trial)
   if 'B' in trial:break
  quadrature=80*level;gh,_=roots_hermitenorm(quadrature);xx=(z['mu_0']+math.sqrt(S0)*gh)/(z['sigma']*math.sqrt(T1));full_extent=max(12.,float(np.max(np.abs(xx))))+1;trial=dict(level=level,quadrature=quadrature,extent=full_extent)
  try:
   result=fullenv['full_asymmetric_output'](theta1,T1,z['sigma'],z['c_i'],z['c_j'],S0,z['mu_0'],extent=full_extent,tolerance=1e-8/level,quadrature=quadrature);trial.update(B=float(result['B']),method='analytic_strategy_output_BVP',details=result)
  except Exception as ex:
   trial['BVP_error']=repr(ex)
   try:
    result=valueenv['full_asymmetric_value_grid'](theta1,T1,z['sigma'],z['c_i'],z['c_j'],S0,z['mu_0'],N=8001*level,cutoff=24.+4*(level-1),quadrature=160*level);trial.update(B=float(result['B']),method='analytic_strategy_value_coordinate',details=result)
   except Exception as ex2:trial['error']=repr(ex2)
  item['selected_trials'].append(trial)
 obs={level:[r for r in item['observed_trials'] if r['level']==level and 'B' in r] for level in [1,2]};sel={r['level']:r for r in item['selected_trials'] if 'B' in r}
 if all(obs.values()) and len(sel)==2:
  ob=obs[2][-1]['B'];sb=sel[2]['B'];item.update(observed_nonlinear_B=ob,selected_nonlinear_B=sb,nonlinear_gap=sb-ob,diagnostic_numerical_radius=max(3*abs(ob-obs[1][-1]['B']),1e-7)+max(3*abs(sb-sel[1]['B']),1e-7))
 else:item['nonlinear_unresolved']=True
 draws={k:np.array([z[k]]) for k in ['c_i','c_j','sigma','lambda','mu_0']};fo0=Output(draws,T0,96);fo1=Output(draws,T1,96);item['first_order_gap']=fo1.no_effort+theta1*fo1.value(math.inf)-(fo0.no_effort+theta0*fo0.value(draws['lambda']));item['observed_baseline']=float(beta(z['mu_0'],S0+z['sigma']**2*T0));item['selected_baseline']=float(beta(z['mu_0'],S0+z['sigma']**2*T1));item['observed_energy_upper']=item['observed_baseline']+math.sqrt(2*theta0*T0/min(z['c_i'],z['c_j']));item['selected_energy_upper']=item['selected_baseline']+math.sqrt(2*theta1*T1/min(z['c_i'],z['c_j']));state['records'].append(item);state['elapsed_seconds']=time.monotonic()-clock;save()
 if (serial+1)%8==0:print('REMAINING FULL NONLINEAR',CID,serial+1,'unresolved',sum(r.get('nonlinear_unresolved',False) for r in state['records']),flush=True)
state['status']='completed';state['all128_resolved']=all('nonlinear_gap' in r for r in state['records']);save();print('REMAINING FULL NONLINEAR FINISHED',CID,state['all128_resolved'],flush=True)
```
