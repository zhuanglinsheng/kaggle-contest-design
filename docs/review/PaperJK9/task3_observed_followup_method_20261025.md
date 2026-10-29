# Task 3：原设计的进一步核验

日期：2026-10-25。对 2551 分别加密时间和空间，并增加一层独立最佳回应/净漂移价值检查。对 23652，初次独立审查因 np.arange 坐标的浮点不对称而被旧接口拒绝；本次保留原失败记录，仅将坐标与反向坐标取平均恢复精确对称，记录最大位移，并要求小于 1e-10。策略值保持原计算值，不把这个几何接口问题解释为均衡失败。

[复核记录](task3_observed_followup_20261025.json)。

```python
import json,math,hashlib,time
from pathlib import Path
import numpy as np
from pipeline.synthetic.nonlinear import Primitives
from pipeline.theory.joint_design_exact import solve_equilibrium_adaptive,expected_best_output
report=Path('docs/review/PaperJK9');old=json.loads((report/'task3_observed_nonlinear_points_20261025.json').read_text());lookup={r['contest_id']:r for r in old['records']}
method=(report/'task3_asymmetric_best_response_method_20261025.md').read_text().split('```python\n',1)[1].split('```',1)[0];exec(compile(method,'<reported independent response>','exec'))
out=report/'task3_observed_followup_20261025.json'
if out.exists():raise RuntimeError('Existing follow-up: inspect before resume')
state=dict(status='running',scope='Additional independent checks for the pressure case and floating-point grid geometry retry',records=[],nonlinear_posterior_average_verified=False)
def save():
 temp=out.with_suffix('.json.tmp');temp.write_text(json.dumps(state,indent=2,allow_nan=False));temp.replace(out)
save()
configs=[(2551,2,8000,True,'joint_refinement'),(2551,2,16000,False,'time_only_refinement'),(2551,3,8000,False,'space_only_refinement'),(23652,0,2000,True,'symmetric_coordinate_roundoff_retry'),(23652,1,4000,True,'symmetric_coordinate_roundoff_retry')]
for cid,level,steps,do_br,label in configs:
 r=lookup[cid];p=Primitives(**r['primitive']);diag=r['trials'][0]['diagnostics'];scale=math.sqrt(p.S0+p.sigma*p.sigma*p.T);start=time.monotonic()
 item=dict(contest_id=cid,label=label,spatial_level=level,base_steps=steps)
 try:
  eq=solve_equilibrium_adaptive(p,dy=diag['dy']/2**level,base_steps=steps,extent=diag['extent']);item['B']=float(expected_best_output(eq));item['diagnostics']=eq.diagnostics
  if do_br:
   raw=eq.y/scale;xx=(raw-raw[::-1])/2;shift=float(np.max(np.abs(xx-raw)));assert shift<1e-10 and len(xx)%2
   item['maximum_coordinate_roundoff_correction']=shift
   tt=1-eq.t[::-1]/p.T;qi=eq.effort[::-1,:,0].T.astype(float)*p.T/scale;qj=eq.effort[::-1,:,1].T.astype(float)*p.T/scale
   item['responses']={}
   for player,own,opp,cost in [('i',qi,qj,p.c_i),('j',qj[::-1,:],qi[::-1,:],p.c_j)]:item['responses'][player]=evaluate_asymmetric_policy(p.sigma,p.theta/cost,p.T,p.lamb,p.S0,xx,tt,own,opp,steps=steps)
   del qi,qj
  del eq
 except Exception as exc:item['error']=str(exc)
 item['seconds']=time.monotonic()-start;state['records'].append(item);save();print('follow-up',cid,label,'done',flush=True)
state['status']='completed';save()
```
