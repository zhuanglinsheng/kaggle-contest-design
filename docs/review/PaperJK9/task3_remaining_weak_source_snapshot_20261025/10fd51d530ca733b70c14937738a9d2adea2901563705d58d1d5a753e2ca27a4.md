# Task 3：剩余无反馈代表案例的128抽样后验配对验证

20270/2464/5144按原预选案例保留。每例4链各32个分层随机实际样本，先固定索引，保持原初始均值/方差；原设计两层HJB，所选无反馈设计两层谱方程直接取零精度，并用每样本的小正精度nu=1e-8作数值延拓对照。两层分别N1024/2048、时间步2000/4000；每16个样本预定一次双方独立最佳回应诊断。失败样本不删除，不用全披露替代零反馈，不用单点替代后验平均；数值诊断不是严格误差证书或一般零精度极限证明。

保留每样本解析无努力基线和能量上界，以便在未解决抽样存在时报告整个预选样本的范围，不能只平均成功子集。最终需检查空间/时间/阈值敏感性、子集最佳回应及失败样本；本批次结束不自动代表质量可行性成立。只在审查目录保存过程，不改论文或生产实现。

```python
import json,csv,math,time,hashlib,os
from pathlib import Path
import numpy as np
from scipy.special import roots_hermitenorm
from pipeline.empirical import estimate as e
from pipeline.empirical.fixed_initial_design import Output,beta
from pipeline.synthetic.nonlinear import Primitives,variance
from pipeline.theory.joint_design_exact import solve_equilibrium_adaptive,expected_best_output
p=Path('docs/review/PaperJK9');CID=globals().get('CID',20270);status=p/f'task3_posterior_remaining_weak_{CID}_20261025.json';assert not status.exists()
def code(f):return f.read_text().partition(chr(10)+chr(96)*3+'python'+chr(10))[2].rpartition(chr(10)+chr(96)*3)[0]
def digest(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
fullenv={};exec(compile(code(p/'task3_full_disclosure_asymmetric_method_20261025.md'),'existing-full-disclosure-output','exec'),fullenv);valueenv={};exec(compile(code(p/'task3_full_disclosure_value_grid_method_20261025.md'),'existing-value-coordinate-output','exec'),valueenv)
weak={};spectral_code=code(p/'task3_asymmetric_weak_feedback_method_20261025.md');response_record=json.loads((p/'task3_zero_precision_interface_check_20261025.json').read_text());assert response_record['status']=='completed';response_code=response_record['response_source_snapshot'];exec(compile(spectral_code,'reported-asymmetric-spectral-with-exact-zero','exec'),weak);exec(compile(response_code,'independent-response-exact-zero-Riccati','exec'),weak)
ix=json.loads((p/'reconstruction_20261025/accepted_index.json').read_text());entry=next(r for r in ix['entries'] if r['contest_id']==CID);ap=Path(entry['accepted']);ap=ap if ap.is_absolute() else p/'reconstruction_20261025'/ap;assert digest(ap)==entry['accepted_sha256'];a=json.loads(ap.read_text());assert a['diagnostics']['passed'];assert all(digest(f)==h for f,h in zip(a['chain_paths'],a['chain_sha256']));chains=[e.read_columns(Path(f)) for f in a['chain_paths']];data=json.loads((Path(a['chain_paths'][0]).parent/'data.json').read_text());policy=next(r for r in csv.DictReader((p/'task3_representative_input_inventory_20261025.csv').open()) if int(r['contest_id'])==CID);assert policy['choice']=='no_feedback_limit';rng=np.random.default_rng(20261025+CID*1009);selection=[]
for chain,c in enumerate(chains):
 bins=np.linspace(0,len(c['c_i']),33,dtype=int)
 for j in range(32):selection.append(dict(chain=chain+1,index=int(rng.integers(bins[j],bins[j+1]))))
state=dict(status='running',driver_pid=os.getpid(),contest_id=CID,scope='128 preselected actual first-order-posterior draws; paired observed and saved no-feedback design, all draws retained',selection_seed=20261025+CID*1009,selection=selection,accepted=str(ap.resolve()),accepted_sha256=digest(ap),chain_sha256=a['chain_sha256'],selected_policy=policy,records=[],strict_continuous_error_certificate=False,spectral_source_sha256=hashlib.sha256(spectral_code.encode()).hexdigest(),zero_response_source_sha256=hashlib.sha256(response_code.encode()).hexdigest(),source_sha256={name:digest(p/name) for name in ['task3_asymmetric_weak_feedback_method_20261025.md','task3_zero_precision_interface_check_20261025.json','task3_posterior_remaining_weak_method_20261025.md']})
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
  q0=S0+z['sigma']**2*T1;weak_extent=max(12.,abs(z['mu_0'])/math.sqrt(q0)+5.);N=1024*level;steps=2000*level;trial=dict(level=level,N=N,steps=steps,precision=0.,extent=weak_extent,tolerance=1e-14);args=dict(sigma=z['sigma'],theta=theta1,T=T1,ci=z['c_i'],cj=z['c_j'],S0=S0,mu=z['mu_0'])
  try:
   result,path=weak['asymmetric_spectral'](**args,precision=0.,N=N,steps=steps,tolerance=1e-14,extent=weak_extent,keep_path=(level==2 and serial%16==0));trial['details']=result
   if not result['physical_screen_passed']:raise RuntimeError('Selected zero-precision policy fails physical/independent-payoff screen')
   trial['B']=float(result['B']);trial['method']='spectral_exact_zero_precision'
   if level==2:
    small,_=weak['asymmetric_spectral'](**args,precision=1e-8/(q0*T1),N=N,steps=steps,tolerance=1e-14,extent=weak_extent);trial['small_positive_result']=small
    if not small['physical_screen_passed']:raise RuntimeError('Small-positive continuation fails physical screen')
    trial['zero_small_positive_difference']=abs(result['B']-small['B'])
    if path is not None:
     qi=path['policies'][:,:,0];qj=path['policies'][:,:,1];trial['best_responses']={}
     for player,own,opp,cost in [('i',qi,qj,z['c_i']),('j',qj[::-1,:],qi[::-1,:],z['c_j'])]:trial['best_responses'][player]=weak['evaluate_asymmetric_policy'](z['sigma'],theta1/cost,T1,0.,S0,path['x'],path['times'],own,opp,steps=4000)
    del path
  except Exception as ex:
   trial['error']=repr(ex);trial.pop('B',None)
  item['selected_trials'].append(trial)
 obs={level:[r for r in item['observed_trials'] if r['level']==level and 'B' in r] for level in [1,2]};sel={r['level']:r for r in item['selected_trials'] if 'B' in r}
 if all(obs.values()) and len(sel)==2:
  ob=obs[2][-1]['B'];sb=sel[2]['B'];item.update(observed_nonlinear_B=ob,selected_nonlinear_B=sb,nonlinear_gap=sb-ob,diagnostic_numerical_radius=max(3*abs(ob-obs[1][-1]['B']),1e-7)+max(3*abs(sb-sel[1]['B']),1e-7)+3*sel[2]['zero_small_positive_difference'])
 else:item['nonlinear_unresolved']=True
 draws={k:np.array([z[k]]) for k in ['c_i','c_j','sigma','lambda','mu_0']};fo0=Output(draws,T0,96);fo1=Output(draws,T1,96);item['first_order_gap']=fo1.no_effort+theta1*fo1.value(0.)-(fo0.no_effort+theta0*fo0.value(draws['lambda']));item['observed_baseline']=float(beta(z['mu_0'],S0+z['sigma']**2*T0));item['selected_baseline']=float(beta(z['mu_0'],S0+z['sigma']**2*T1));item['observed_energy_upper']=item['observed_baseline']+math.sqrt(2*theta0*T0/min(z['c_i'],z['c_j']));item['selected_energy_upper']=item['selected_baseline']+math.sqrt(2*theta1*T1/min(z['c_i'],z['c_j']));state['records'].append(item);state['elapsed_seconds']=time.monotonic()-clock;save()
 if (serial+1)%8==0:print('REMAINING WEAK NONLINEAR',CID,serial+1,'unresolved',sum(r.get('nonlinear_unresolved',False) for r in state['records']),flush=True)
state['status']='completed';state['all128_resolved']=all('nonlinear_gap' in r for r in state['records']);save();print('REMAINING WEAK NONLINEAR FINISHED',CID,state['all128_resolved'],flush=True)
```
