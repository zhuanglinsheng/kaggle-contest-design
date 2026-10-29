# Task 3：128 个后验原设计的逐抽样空间加密

日期：2026-10-25。为避免以少数代表点的误差诊断支持整个后验平均，对 128 个配对抽样的原有限精度设计逐一加密。两个困难抽样复用已完成的更细结果，其余在半空间步长上运行；如果负梯度失败，继续减小时间步，不删样本。

与原粗层比较后再传播数值诊断半径到配对后验质量差。半径仍不是严格 PDE 误差界；单独时间/空间复核及独立随机路径核算另见代表点报告。

[逐抽样细层记录](task3_posterior_observed_fine_2551_20261025.json)。

```python
import json,math,time,hashlib
from pathlib import Path
from pipeline.synthetic.nonlinear import Primitives,variance
from pipeline.theory.joint_design_exact import solve_equilibrium_adaptive,expected_best_output
report=Path('docs/review/PaperJK9');raw=json.loads((report/'task3_posterior_nonlinear_2551_20261025.json').read_text());hard=json.loads((report/'task3_posterior_nonlinear_2551_hard_draws_20261025.json').read_text());hm={r['serial']:r for r in hard['records'] if 'B' in r}
out=report/'task3_posterior_observed_fine_2551_20261025.json'
if out.exists():raise RuntimeError('Existing posterior fine batch: inspect before resume')
state=dict(status='running',scope='Fine spatial mesh for all 128 posterior observed designs; reuse already finer hard-draw solves, no draw dropped',records=[],strict_error_bound=False)
def save():
 tmp=out.with_suffix('.json.tmp');tmp.write_text(json.dumps(state,indent=2,allow_nan=False));tmp.replace(out)
save()
for r in raw['records']:
 serial=r['serial'];item=dict(serial=serial,trials=[])
 if serial in hm:
  item['reused_hard_draw']=hm[serial]
 else:
  z=r['point'];p=Primitives(T=r['observed_T'],theta=r['observed_theta'],c_i=z['c_i'],c_j=z['c_j'],sigma=z['sigma'],lamb=z['lambda'],S0=r['S0'],mu0=z['mu_0']);scale=math.sqrt(p.S0+p.sigma*p.sigma*p.T);dy0=min(.02*scale,.25*math.sqrt(float(variance(p.T,p))));extent=max(7.,(abs(p.mu0)+8*math.sqrt(p.S0))/scale)
  for steps in (4000,8000,16000):
   trial=dict(dy=dy0/2,base_steps=steps)
   try:
    eq=solve_equilibrium_adaptive(p,dy=dy0/2,base_steps=steps,extent=extent);trial['B']=float(expected_best_output(eq));trial['diagnostics']=eq.diagnostics;del eq
   except Exception as exc:trial['error']=str(exc)
   item['trials'].append(trial)
   if 'B' in trial:break
 state['records'].append(item);save()
 if (serial+1)%16==0:print('observed posterior fine',serial+1,'/128',flush=True)
state['status']='completed';save();print('All posterior observed fine checks completed.',flush=True)
```
