# Task 3：原有限精度设计的代表点非线性试算

日期：2026-10-25。复用现有非对称 HJB 求解器与独立线性产出方程，在已选联合抽样点上核算原奖金、期限与精度。初始方差固定为该抽样的 sigma/sqrt(lambda_observed)。只保存报告，不修改求解器、论文或原结果。

两层空间/时间共同加密，先检查原求解器的策略非负与值边界诊断。两层差异不是严格误差界；还需独立最佳回应及分开的时间/空间检查。此处不是后验平均质量验证。原始策略路径只在内存中使用，不作为新的生产结果保存。

[数值记录](task3_observed_nonlinear_points_20261025.json)。

```python
import json,hashlib,math,time
from pathlib import Path
import numpy as np
from pipeline.synthetic.nonlinear import Primitives,variance
from pipeline.theory.joint_design_exact import solve_equilibrium_adaptive,expected_best_output
from pipeline.empirical.fixed_initial_design import Output
report=Path('docs/review/PaperJK9');points=json.loads((report/'task3_representative_joint_points_20261025.json').read_text())['points']
out=report/'task3_observed_nonlinear_points_20261025.json'
if out.exists():raise RuntimeError('Existing point results; inspect before resume')
state=dict(date='2026-10-25',status='running',scope='Nonlinear finite observed design at representative joint draws, not posterior averaged output',source_sha256={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in ['pipeline/synthetic/nonlinear.py','pipeline/theory/joint_design_exact.py']},records=[],nonlinear_quality_preservation_verified=False)
def save():
 temp=out.with_suffix('.json.tmp');temp.write_text(json.dumps(state,indent=2,allow_nan=False));temp.replace(out)
save()
for point in points:
 cid=point['contest_id'];z=point['point'];T=point['observed_duration'];theta=point['observed_prize']
 primitive=Primitives(T=T,theta=theta,c_i=z['c_i'],c_j=z['c_j'],sigma=z['sigma'],lamb=z['lambda'],S0=point['fixed_initial_variance'],mu0=z['mu_0'])
 scale=math.sqrt(primitive.S0+primitive.sigma**2*T);terminal_sd=math.sqrt(float(variance(T,primitive)))
 dy0=min(.02*scale,.25*terminal_sd);extent=max(7.,(abs(primitive.mu0)+8*math.sqrt(primitive.S0))/scale)
 draws={k:np.array([z[k]]) for k in ('c_i','c_j','sigma','lambda','mu_0')};linear=Output(draws,T,96)
 record=dict(contest_id=cid,primitive=vars(primitive),first_order_observed_at_same_point=linear.no_effort+theta*linear.value(draws['lambda']),trials=[])
 for level in range(2):
  start=time.monotonic()
  try:
   eq=solve_equilibrium_adaptive(primitive,dy=dy0/2**level,base_steps=2000*2**level,extent=extent)
   B=float(expected_best_output(eq));record['trials'].append(dict(level=level,B=B,diagnostics=eq.diagnostics,seconds=time.monotonic()-start))
   del eq
  except Exception as exc:record['trials'].append(dict(level=level,error=str(exc),seconds=time.monotonic()-start))
 if all('B' in r for r in record['trials']):
  record['coupled_space_time_difference']=abs(record['trials'][1]['B']-record['trials'][0]['B'])
  record['nonlinear_minus_first_order']=record['trials'][1]['B']-record['first_order_observed_at_same_point']
 record['independent_best_response_audited']=False;state['records'].append(record);save();print('observed',cid,'done',flush=True)
state['status']='completed';save();print('Observed point batch completed.',flush=True)
```
