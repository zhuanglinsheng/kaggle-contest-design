# Task 3：2551 完全披露设计的独立路径核算

日期：2026-10-25。对所选设计的解析非对称完全披露策略，直接模拟全信息受控差距 SDE，并计算累计双方平均努力加最终绝对差距的一半。另核对获奖指标减累计努力成本与解析参赛者价值。该路径评价独立于输出边值方程。

20,000 条路径，sqrt(剩余期限) 的均匀网格；2,000/4,000 步共用布朗路径，以配对差检查离散误差。控制剖面预先在 65,537 点上计算并线性插值。误差是 Monte Carlo 与网格诊断，不能认证后验平均反事实。

```python
import json,math,hashlib,time
from pathlib import Path
import numpy as np
from scipy.special import ndtr
report=Path('docs/review/PaperJK9')
records=json.loads((report/'task3_full_disclosure_cases_20261025.json').read_text())['records']
r=next(x for x in records if x['contest_id']==2551 and x['label']=='full_at_selected_prize_duration')
sigma=r['sigma'];theta=r['theta'];T=r['T'];S0=r['S0'];mu=r['mu'];wi=theta/(r['ci']*sigma*sigma);wj=theta/(r['cj']*sigma*sigma)
b=math.log(-math.expm1(-wi))-math.log(-math.expm1(-wj))
zgrid=np.linspace(-12,12,65537);K=lambda d:(d+np.sinh(d+b))/2
lo=float(K(-wj));hi=float(K(wi));span=hi-lo;target=lo+span*ndtr(zgrid)
a=np.full_like(zgrid,-wj);c=np.full_like(zgrid,wi);d=-wj+(wi+wj)*ndtr(zgrid)
for _ in range(65):
 err=K(d)-target;a=np.where(err<0,d,a);c=np.where(err>=0,d,c);candidate=d-err/((1+np.cosh(d+b))/2);d=np.where((candidate>a)&(candidate<c),candidate,(a+c)/2)
dp=span*np.exp(-zgrid*zgrid/2)/math.sqrt(2*math.pi)/((1+np.cosh(d+b))/2);fp=np.tanh((d+b)/2)
qi=(1+fp)*dp/2;qj=(1-fp)*dp/2
lc=lambda z:np.logaddexp(z,-z)-math.log(2)
f=wj+2*(lc((d+b)/2)-lc((b-wj)/2));ui=(f+d)/(2*wi);uj=(f-d)/(2*wj)
def interp(z,array):
 pos=np.clip((z+12)*(len(zgrid)-1)/24,0,len(zgrid)-1-1e-9);i=pos.astype(np.int64);fraction=pos-i
 return (1-fraction)*array[i]+fraction*array[i+1]
paths=20000;coarse_steps=2000;fine_steps=4000;rng=np.random.default_rng(202610252551);scale=sigma*math.sqrt(T)
x0=(mu+math.sqrt(S0)*rng.standard_normal(paths))/scale
xf=x0.copy();xc=x0.copy();af=np.zeros(paths);ac=np.zeros(paths);cif=np.zeros(paths);cjf=np.zeros(paths);cic=np.zeros(paths);cjc=np.zeros(paths);pending=np.zeros(paths)
start=time.monotonic()
for j in range(fine_steps):
 left=(1-j/fine_steps)**2;right=(1-(j+1)/fine_steps)**2;dt=left-right
 z=xf/math.sqrt(left);control_i=interp(z,qi)/math.sqrt(left);control_j=interp(z,qj)/math.sqrt(left)
 dw=math.sqrt(dt)*rng.standard_normal(paths)
 xf+=(control_i-control_j)*dt+dw;af+=(control_i+control_j)*dt/2;cif+=control_i**2*dt/(2*wi);cjf+=control_j**2*dt/(2*wj);pending+=dw
 if j%2==1:
  jj=(j-1)//2;leftc=(1-jj/coarse_steps)**2;rightc=(1-(jj+1)/coarse_steps)**2;dtc=leftc-rightc
  zc=xc/math.sqrt(leftc);cii=interp(zc,qi)/math.sqrt(leftc);cjj=interp(zc,qj)/math.sqrt(leftc)
  xc+=(cii-cjj)*dtc+pending;ac+=(cii+cjj)*dtc/2;cic+=cii**2*dtc/(2*wi);cjc+=cjj**2*dtc/(2*wj);pending[:]=0
Bf=scale*(af+np.abs(xf)/2);Bc=scale*(ac+np.abs(xc)/2)
refi=interp(x0,ui);refj=interp(x0,uj);payi=(xf>0).astype(float)-cif;payj=(xf<0).astype(float)-cjf
stat=lambda a:dict(mean=float(np.mean(a)),mcse=float(np.std(a,ddof=1)/math.sqrt(paths)))
result=dict(date='2026-10-25',scope='Independent full-information Euler SDE output and payoff check, selected design at representative joint draw for contest 2551',paths=paths,coarse_steps=coarse_steps,fine_steps=fine_steps,grid_profile_points=len(zgrid),seed=202610252551,full_bvp_output=r['trials'][-1]['B'],fine_output=stat(Bf),coarse_output=stat(Bc),paired_fine_minus_coarse=stat(Bf-Bc),player_i_payoff_minus_analytic_value=stat(payi-refi),player_j_payoff_minus_analytic_value=stat(payj-refj),seconds=time.monotonic()-start,posterior_average_quality_verified=False)
(report/'task3_full_disclosure_monte_carlo_2551_20261025.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result,indent=2),flush=True)
```
