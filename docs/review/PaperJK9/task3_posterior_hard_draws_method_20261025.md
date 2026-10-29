# Task 3：两个困难后验抽样的追加加密

日期：2026-10-25。初次有限精度原设计的抽样 50、99 负梯度失败，已在较细网格上恢复；其中 99 还只有一层成功。这里在固定空间下进一步缩小时间步，再单独加密空间，保留原失败项和所有成功层。按原求解器运行，不放宽负策略检查。

[追加记录](task3_posterior_nonlinear_2551_hard_draws_20261025.json)。

```python
import json,math,time
from pathlib import Path
from pipeline.synthetic.nonlinear import Primitives,variance
from pipeline.theory.joint_design_exact import solve_equilibrium_adaptive,expected_best_output
report=Path('docs/review/PaperJK9');raw=json.loads((report/'task3_posterior_nonlinear_2551_20261025.json').read_text());out=report/'task3_posterior_nonlinear_2551_hard_draws_20261025.json'
if out.exists():raise RuntimeError('Existing hard-draw follow-up: inspect before resume')
state=dict(status='running',scope='Further space/time refinement for the two initially failed observed posterior draws',records=[])
def save():
 temp=out.with_suffix('.json.tmp');temp.write_text(json.dumps(state,indent=2,allow_nan=False));temp.replace(out)
save()
for serial,factor,steps in [(99,4,32000),(99,8,32000),(50,8,32000)]:
 r=raw['records'][serial];z=r['point'];p=Primitives(T=r['observed_T'],theta=r['observed_theta'],c_i=z['c_i'],c_j=z['c_j'],sigma=z['sigma'],lamb=z['lambda'],S0=r['S0'],mu0=z['mu_0']);scale=math.sqrt(p.S0+p.sigma*p.sigma*p.T);dy0=min(.02*scale,.25*math.sqrt(float(variance(p.T,p))));extent=max(7.,(abs(p.mu0)+8*math.sqrt(p.S0))/scale);item=dict(serial=serial,dy_factor=factor,base_steps=steps);start=time.monotonic()
 try:
  eq=solve_equilibrium_adaptive(p,dy=dy0/factor,base_steps=steps,extent=extent);item['B']=float(expected_best_output(eq));item['diagnostics']=eq.diagnostics;del eq
 except Exception as exc:item['error']=str(exc)
 item['seconds']=time.monotonic()-start;state['records'].append(item);save();print('hard draw',serial,factor,steps,'done',flush=True)
state['status']='completed';save()
```
