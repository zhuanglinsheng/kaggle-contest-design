# Task 3：后验批次的失败恢复与完全披露交叉核验

日期：2026-10-25。原 128 抽样批次保留全部失败项。对全部抽样用价值坐标输出线性方程独立核算三层网格/截断/求积，与成功的原 z 坐标 BVP 比较，并尝试修复三个 BVP 节点上限失败。对两个原有限精度 HJB 的负梯度失败，缩小时间步并加密空间，要求保留结果和失败，不删除抽样。

待全部核对后计算配对后验均值、MCSE 和未成功项的解析包络；不能仅用成功样本的平均数替代原抽样目标。

[复核数据](task3_posterior_nonlinear_2551_followup_20261025.json)。

```python
import json,math,time,hashlib
from pathlib import Path
from pipeline.synthetic.nonlinear import Primitives,variance
from pipeline.theory.joint_design_exact import solve_equilibrium_adaptive,expected_best_output
report=Path('docs/review/PaperJK9');raw=json.loads((report/'task3_posterior_nonlinear_2551_20261025.json').read_text());assert raw['status']=='completed'
method=(report/'task3_full_disclosure_value_grid_method_20261025.md').read_text().split('```python\n',1)[1].split('```',1)[0];exec(compile(method,'<transformed full output>','exec'))
out=report/'task3_posterior_nonlinear_2551_followup_20261025.json'
if out.exists():raise RuntimeError('Existing posterior follow-up: inspect before resume')
state=dict(status='running',scope='Value-coordinate full-output cross-check on all 128 draws and recovery of both observed solver failures; no samples dropped',raw_sha256=hashlib.sha256((report/'task3_posterior_nonlinear_2551_20261025.json').read_bytes()).hexdigest(),method_sha256=hashlib.sha256(method.encode()).hexdigest(),method_snapshot=method,records=[],posterior_quality_verified=False)
def save():
 tmp=out.with_suffix('.json.tmp');tmp.write_text(json.dumps(state,indent=2,allow_nan=False));tmp.replace(out)
save()
for r in raw['records']:
 z=r['point'];item=dict(serial=r['serial'],selected_full_trials=[],observed_recovery=[])
 for N,cut,quad in [(4001,24.,80),(8001,26.,160),(16001,28.,320)]:
  try:item['selected_full_trials'].append(full_asymmetric_value_grid(r['selected_theta'],r['selected_T'],z['sigma'],z['c_i'],z['c_j'],r['S0'],z['mu_0'],N=N,cutoff=cut,quadrature=quad))
  except Exception as exc:item['selected_full_trials'].append(dict(N=N,cutoff=cut,quadrature=quad,error=str(exc)))
 if 'observed_error' in r:
  p=Primitives(T=r['observed_T'],theta=r['observed_theta'],c_i=z['c_i'],c_j=z['c_j'],sigma=z['sigma'],lamb=z['lambda'],S0=r['S0'],mu0=z['mu_0']);scale=math.sqrt(p.S0+p.sigma*p.sigma*p.T);dy0=min(.02*scale,.25*math.sqrt(float(variance(p.T,p))));extent=max(7.,(abs(p.mu0)+8*math.sqrt(p.S0))/scale)
  for factor,steps in [(2,8000),(4,16000)]:
   t=dict(dy=dy0/factor,base_steps=steps)
   try:
    eq=solve_equilibrium_adaptive(p,dy=dy0/factor,base_steps=steps,extent=extent);t['B']=float(expected_best_output(eq));t['diagnostics']=eq.diagnostics;del eq
   except Exception as exc:t['error']=str(exc)
   item['observed_recovery'].append(t)
 state['records'].append(item);save()
 if (r['serial']+1)%16==0:print('posterior transformed follow-up',r['serial']+1,'/128',flush=True)
state['status']='completed';save();print('Posterior transformed follow-up completed.',flush=True)
```
