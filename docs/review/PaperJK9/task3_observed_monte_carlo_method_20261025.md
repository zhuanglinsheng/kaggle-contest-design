# Task 3：2551 原有限精度设计的独立路径核算

日期：2026-10-25。模拟原策略下的公共后验均值 SDE，从固定公开初始均值开始；原设计 S0=sigma/sqrt(lambda)，故创新扩散等于 sigma。终端用 beta(后验均值,S0) 积分潜在初始误差，而不是把后验均值误当作真实差距。累计平均努力加终端 beta 给出输出，参赛者收益另用条件获奖概率减成本核算。

20,000 路径，2,000/4,000 步共用布朗路径；原 HJB 策略场使用空间和时间线性插值。该评价独立于线性产出 PDE。此处仍只评价联合代表点，不认证后验平均反事实。

```python
import json,math,time
from pathlib import Path
import numpy as np
from scipy.special import ndtr
from pipeline.synthetic.nonlinear import Primitives
from pipeline.theory.joint_design_exact import solve_equilibrium_adaptive
report=Path('docs/review/PaperJK9');r=next(x for x in json.loads((report/'task3_observed_nonlinear_points_20261025.json').read_text())['records'] if x['contest_id']==2551)
p=Primitives(**r['primitive']);diag=r['trials'][1]['diagnostics'];eq=solve_equilibrium_adaptive(p,dy=diag['dy'],base_steps=4000,extent=diag['extent'])
assert abs(p.lamb*p.S0*p.S0-p.sigma*p.sigma)<1e-12
scale=p.sigma*math.sqrt(p.T);xgrid=eq.y/scale;dx=float(xgrid[1]-xgrid[0]);policy=eq.effort.astype(float)*p.T/scale
wi=p.theta/(p.c_i*p.sigma*p.sigma);wj=p.theta/(p.c_j*p.sigma*p.sigma)
def controls(t,x):
 k=min(int(np.searchsorted(eq.t,t,side='right'))-1,len(eq.t)-2);k=max(k,0);f=(t-eq.t[k])/(eq.t[k+1]-eq.t[k]);v=(1-f)*policy[k]+f*policy[k+1]
 pos=np.clip((x-xgrid[0])/dx,0,len(xgrid)-1-1e-9);i=pos.astype(np.int64);g=pos-i
 return (1-g)*v[i,0]+g*v[i+1,0],(1-g)*v[i,1]+g*v[i+1,1]
paths=20000;fine_steps=4000;coarse_steps=2000;rng=np.random.default_rng(202610255511);xf=np.full(paths,p.mu0/scale);xc=xf.copy();af=np.zeros(paths);ac=np.zeros(paths);cif=np.zeros(paths);cjf=np.zeros(paths);pending=np.zeros(paths);maxstate=0.;outside=0;start=time.monotonic()
for j in range(fine_steps):
 dt=1/fine_steps;ci,cj=controls(j*p.T/fine_steps,xf);dw=math.sqrt(dt)*rng.standard_normal(paths)
 xf+=(ci-cj)*dt+dw;af+=(ci+cj)*dt/2;cif+=ci*ci*dt/(2*wi);cjf+=cj*cj*dt/(2*wj);pending+=dw
 maxstate=max(maxstate,float(np.max(np.abs(xf))));outside+=int(np.sum((xf<xgrid[0])|(xf>xgrid[-1])))
 if j%2==1:
  t=(j-1)*p.T/fine_steps;ii,jj=controls(t,xc);xc+=(ii-jj)/coarse_steps+pending;ac+=(ii+jj)/(2*coarse_steps);pending[:]=0
s=p.S0/(p.sigma*p.sigma*p.T)
def terminal(x):return math.sqrt(s/(2*math.pi))*np.exp(-x*x/(2*s))+x*(ndtr(x/math.sqrt(s))-.5)
Bf=scale*(af+terminal(xf));Bc=scale*(ac+terminal(xc));stat=lambda a:dict(mean=float(a.mean()),mcse=float(np.std(a,ddof=1)/math.sqrt(paths)))
pi=eq.effort[0,:,0].astype(float)*p.c_i/p.theta;pj=eq.effort[0,:,1].astype(float)*p.c_j/p.theta
cdfi=np.r_[0.,np.cumsum(.5*(pi[:-1]+pi[1:])*np.diff(eq.y))];cdfj=1-np.r_[0.,np.cumsum(.5*(pj[:-1]+pj[1:])*np.diff(eq.y))]
ui=float(np.interp(p.mu0,eq.y,cdfi));uj=float(np.interp(p.mu0,eq.y,cdfj));win=ndtr(xf/math.sqrt(s))
result=dict(date='2026-10-25',scope='Independent posterior-state SDE evaluation of the finite observed design at contest 2551 representative draw',paths=paths,fine_steps=fine_steps,coarse_steps=coarse_steps,initial_public_mean_fixed=True,initial_latent_uncertainty_integrated_at_terminal=True,linear_backward_reference=r['trials'][1]['B'],fine_output=stat(Bf),coarse_output=stat(Bc),paired_fine_minus_coarse=stat(Bf-Bc),player_i_payoff_minus_gradient_value=stat(win-cif-ui),player_j_payoff_minus_gradient_value=stat(1-win-cjf-uj),maximum_absolute_normalized_state=maxstate,policy_domain=[float(xgrid[0]),float(xgrid[-1])],out_of_policy_grid_steps=outside,seconds=time.monotonic()-start,posterior_average_quality_verified=False)
(report/'task3_observed_monte_carlo_2551_20261025.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result,indent=2),flush=True)
```
