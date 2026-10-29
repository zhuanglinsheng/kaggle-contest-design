# 2551 奖金方向的独立方法核查

在同一原始代表点上，只改变奖金.9/1/1.1倍。采用仓库原始价值HJB与独立产出传播，核查隐式梯度核出现的符号相反是否由求解方法造成。只核查方向，不重新配奖或重估；失败保留。

```python
from pathlib import Path
import json,math
from pipeline.synthetic.nonlinear import Primitives
from pipeline.theory.joint_design_exact import solve_equilibrium_adaptive,expected_best_output
p=Path('docs/review/PaperJK9');out=p/'task3_direction_consistency_20261025';r=next(v for v in json.loads((p/'task3_representative_joint_points_20261025.json').read_text())['points'] if v['contest_id']==2551);z=r['point'];dest=out/'independent_2551_prize_sign.json';assert not dest.exists();state=dict(status='running',scope='Independent value-HJB confirmation at fixed original representative point',records=[])
for ratio in [.9,1.,1.1]:
 v=dict(prize_ratio=ratio)
 try:
  q=Primitives(T=r['observed_duration'],theta=r['observed_prize']*ratio,c_i=z['c_i'],c_j=z['c_j'],sigma=z['sigma'],lamb=z['lambda'],S0=r['fixed_initial_variance'],mu0=z['mu_0']);scale=math.sqrt(q.S0+q.sigma*q.sigma*q.T);eq=solve_equilibrium_adaptive(q,dy=.005*scale,base_steps=4000,extent=8.);v.update(B=float(expected_best_output(eq)),diagnostics=eq.diagnostics);del eq
 except Exception as ex:v['error']=repr(ex)
 state['records'].append(v);dest.write_text(json.dumps(state,indent=2));print('INDEPENDENT SIGN',v,flush=True)
state['status']='completed';dest.write_text(json.dumps(state,indent=2))
```
