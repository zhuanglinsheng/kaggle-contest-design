# 方向不一致的针对性查因

范围：固定原参数，不重新估计、不重新优化。核查奖金尺度、完整状态传播与策略修正；独立复核2551与2464的反馈符号。沿用原价值HJB及独立线性产出方法，每项保留实际输入及网格。该检查不声称估计参数是真值。

产出分解使用同一Feynman--Kac方程，分开累计双方努力，以及 `(m_i-m_j)*(Phi(y/sqrt(Q))-.5)` 的配置项；产出为无努力基准加半数累计总努力加配置项。它是模型内核算，不能将配置项直接叫作福利损失或某一方“放弃”。

```python
from pathlib import Path
import json,math,os,time,hashlib
import numpy as np
from scipy.special import ndtr
from scipy.linalg import solve_banded
from pipeline.synthetic.nonlinear import Primitives,variance
from pipeline.theory.joint_design_exact import solve_equilibrium_adaptive,expected_best_output,_gradient
from pipeline.empirical.fixed_initial_design import Output,beta
p=Path('docs/review/PaperJK9');out=p/'task3_direction_consistency_20261025';group=os.environ['TASK3_CAUSE_GROUP'];cid=2464 if group=='feedback_2464' else 2551;r=next(v for v in json.loads((p/'task3_representative_joint_points_20261025.json').read_text())['points'] if v['contest_id']==cid);z=r['point'];T=r['observed_duration'];S0=r['fixed_initial_variance'];sig=z['sigma'];dest=out/f'cause_{group}.json';state=dict(status='running',scope='Fixed original parameters, no reestimation or redesign',group=group,input=r,records=[],source_sha256={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in ['pipeline/theory/joint_design_exact.py','pipeline/empirical/fixed_initial_design.py']})
if dest.exists():
 state=json.loads(dest.read_text());assert state['group']==group;state['status']='running'
def save():
 tmp=dest.with_suffix('.json.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(dest)
def channels(eq,effort):
 pp=eq.p;y=eq.y.astype(float);tt=eq.t.astype(float);dy=float(y[1]-y[0]);v=np.zeros((len(y),3));band=np.zeros((3,len(y)-2))
 def flow(values,mi,mj,q):
  rewards=np.column_stack((mi,mj,(mi-mj)*(ndtr(y/math.sqrt(q))-.5)))
  return (mi-mj)[:,None]*_gradient(values,dy)+rewards
 for k in range(len(tt)-1,0,-1):
  dt=float(tt[k]-tt[k-1]);t=(tt[k]+tt[k-1])/2;s=float(variance(t,pp));a=pp.lamb*s*s;ratio=dt*a/(4*dy*dy);band[0,1:]=-ratio;band[1]=1+2*ratio;band[2,:-1]=-ratio;pol=(effort[k].astype(float)+effort[k-1].astype(float))/2;mi,mj=pol.T;q=s+sig*sig*(pp.T-t);base=v[1:-1]+ratio*(v[:-2]-2*v[1:-1]+v[2:]);f=flow(v,mi,mj,q);predict=v.copy();predict[1:-1]=solve_banded((1,1),band,base+dt*f[1:-1],check_finite=False);g=flow(predict,mi,mj,q);v[1:-1]=solve_banded((1,1),band,base+dt*(f[1:-1]+g[1:-1])/2,check_finite=False)
 assert np.isfinite(v).all();vals=np.array([np.interp(pp.mu0,y,v[:,j]) for j in range(3)]);baseline=float(beta(np.array([pp.mu0]),np.array([S0+sig*sig*pp.T]))[0]);return dict(B=baseline+.5*(vals[0]+vals[1])+vals[2],baseline=baseline,integrated_effort_i=float(vals[0]),integrated_effort_j=float(vals[1]),allocation_term=float(vals[2]))
def run(theta_ratio,precision_ratio,spacing,steps,decompose=False):
 if any(abs(v['theta_ratio']-theta_ratio)<1e-14 and abs(v['precision_ratio']-precision_ratio)<1e-14 and abs(v['dy_relative']-spacing)<1e-14 and v['base_steps']==steps for v in state['records']):return
 theta=r['observed_prize']*theta_ratio;lam=z['lambda']*precision_ratio;pp=Primitives(T=T,theta=theta,c_i=z['c_i'],c_j=z['c_j'],sigma=sig,lamb=lam,S0=S0,mu0=z['mu_0']);scale=math.sqrt(S0+sig*sig*T);start=time.monotonic();v=dict(theta_ratio=theta_ratio,precision_ratio=precision_ratio,dy_relative=spacing,base_steps=steps,w_i=theta/(z['c_i']*sig*sig),w_j=theta/(z['c_j']*sig*sig))
 try:
  eq=solve_equilibrium_adaptive(pp,dy=spacing*scale,base_steps=steps,extent=8.);v.update(B=float(expected_best_output(eq)),diagnostics=eq.diagnostics);draws={k:np.array([z[k]]) for k in ['c_i','c_j','sigma','lambda','mu_0']};model=Output(draws,T,192);v['first_order_B']=float(model.no_effort+theta*model.coefficients(lam)[0])
  if decompose:
   v['nonlinear_channels']=channels(eq,eq.effort);assert abs(v['B']-v['nonlinear_channels']['B'])<1e-9
   q=variance(eq.t,pp)+sig*sig*(T-eq.t);density=np.exp(-eq.y[None,:]**2/(2*q[:,None]))/np.sqrt(2*np.pi*q[:,None]);fo=theta*density[:,:,None]*np.array([1/z['c_i'],1/z['c_j']])[None,None,:];v['first_policy_exact_state_channels']=channels(eq,fo)
  del eq
 except Exception as ex:v['error']=repr(ex)
 v['seconds']=time.monotonic()-start;state['records'].append(v);save();print('CAUSE',group,theta_ratio,precision_ratio,spacing,v.get('B'),v.get('error'),flush=True)
save()
if group=='prize_scale':
 for scale in [.01,.1,.3,1.]:
  for ratio in [.9,1.1]:run(scale*ratio,1.,.005,4000,decompose=(scale==1.))
elif group=='feedback_2551':
 for theta_ratio in [.001,.01,.1,1.]:
  for precision_ratio in [.5,1.,2.]:run(theta_ratio,precision_ratio,.005,4000)
elif group=='feedback_2464':
 for theta_ratio in [1.,.01]:
  for spacing,steps in [(.005,4000),(.0025,8000)]:
   for precision_ratio in [.5,1.,2.]:run(theta_ratio,precision_ratio,spacing,steps)
else:raise ValueError(group)
state['status']='completed';save()
```
