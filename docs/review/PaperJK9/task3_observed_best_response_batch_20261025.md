# Task 3：代表点原设计的独立最佳回应批次

日期：2026-10-25。复现两层原 HJB 策略路径并逐点核对产出相同，然后对每位参赛者分别求给定对手下的最佳回应。通过反转空间评价玩家 j。状态和时间网格加密，记录全状态/时间偏离、净漂移原策略价值与梯度积分差。先运行零努力有利偏离反例，以排除审查器总是返回零偏离的错误。

此数值诊断仍不构成严格连续模型误差界，也不认证后验平均质量。报告与原策略核算分开保存；不改原实现或数据。

[批次记录](task3_observed_best_response_20261025.json)。

```python
import hashlib,json,math,time
from pathlib import Path
import numpy as np
from pipeline.synthetic.nonlinear import Primitives
from pipeline.theory.joint_design_exact import solve_equilibrium_adaptive,expected_best_output
report=Path('docs/review/PaperJK9')
method=(report/'task3_asymmetric_best_response_method_20261025.md').read_text().split('```python\n',1)[1].split('```',1)[0]
exec(compile(method,'<reported independent asymmetric response>','exec'))
old=json.loads((report/'task3_observed_nonlinear_points_20261025.json').read_text())
lookup={r['contest_id']:r for r in old['records']}
out=report/'task3_observed_best_response_20261025.json'
if out.exists():raise RuntimeError('Existing independent audit; inspect before resume')
state=dict(status='running',scope='Independent finite-state best response and net-drift payoff checks at the six representative observed-design joint draws',method_sha256=hashlib.sha256(method.encode()).hexdigest(),records=[],continuous_error_bound=False,nonlinear_quality_preservation_verified=False)
def save():
 temp=out.with_suffix('.json.tmp');temp.write_text(json.dumps(state,indent=2,allow_nan=False));temp.replace(out)
# A zero-effort policy with no opponent effort must admit a profitable deviation.
x=np.linspace(-7,7,201);times=np.array([0.,1.]);policies=np.zeros((201,2))
smoke=evaluate_asymmetric_policy(.5,.1,1.,.1,.02,x,times,policies,policies,steps=1000)
assert smoke['max_subgame_deviation_gain']>1e-3
state['zero_effort_counterexample_smoke']=smoke;save()
for cid in [2551,20270,5144,23652,2464,8219]:
 r=lookup[cid];p=Primitives(**r['primitive']);q0=p.S0+p.sigma*p.sigma*p.T;scale=math.sqrt(q0)
 for level in (0,1):
  saved=r['trials'][level]
  if 'diagnostics' not in saved:
   state['records'].append(dict(contest_id=cid,level=level,error='Source nonlinear solve failed'));save();continue
  diag=saved['diagnostics'];start=time.monotonic()
  try:
   eq=solve_equilibrium_adaptive(p,dy=diag['dy'],base_steps=2000*2**level,extent=diag['extent'])
   B=float(expected_best_output(eq));assert abs(B-saved['B'])<1e-10
   xx=eq.y/scale;tt=1-eq.t[::-1]/p.T
   qi=eq.effort[::-1,:,0].T.astype(float)*p.T/scale;qj=eq.effort[::-1,:,1].T.astype(float)*p.T/scale
   results=[]
   for player,own,opp,cost in [('i',qi,qj,p.c_i),('j',qj[::-1,:],qi[::-1,:],p.c_j)]:
    response=evaluate_asymmetric_policy(p.sigma,p.theta/cost,p.T,p.lamb,p.S0,xx,tt,own,opp,steps=2000*2**level)
    results.append(dict(player=player,response=response))
   state['records'].append(dict(contest_id=cid,level=level,source_B=B,source_nodes=len(xx),responses=results,seconds=time.monotonic()-start))
   del eq,qi,qj
  except Exception as exc:state['records'].append(dict(contest_id=cid,level=level,error=str(exc),seconds=time.monotonic()-start))
  save();print('BR',cid,'level',level,'done',flush=True)
state['status']='completed';save();print('Observed representative best-response batch completed.',flush=True)
```
