# Task 3：全披露下奖金—产出关系的独立核验

重新匹配时不能预设非线性产出随奖金单调。先在2551同一代表参数下，交叉比较自适应z坐标BVP和价值坐标三层计算；不以任何求解失败补零。记录奖金倍率、产出及实际残差，搜索根的失败记录保留。此检查不外推到全部后验或全部设计。

```python
from pathlib import Path
import json,math,time
p=Path('docs/review/PaperJK9');out=p/'task3_direction_consistency_20261025'
def load(n):
 ns={};s=(p/n).read_text().partition('\n```python\n')[2].rpartition('\n```')[0];exec(compile(s,n,'exec'),ns);return ns
f=load('task3_full_disclosure_asymmetric_method_20261025.md')['full_asymmetric_output'];g=load('task3_full_disclosure_value_grid_method_20261025.md')['full_asymmetric_value_grid'];r=next(v for v in json.loads((p/'task3_representative_joint_points_20261025.json').read_text())['points'] if v['contest_id']==2551);z=r['point'];state=dict(status='running',scope='Same-parameter independent-method prize-response check, not monotonicity theorem or population estimate',records=[]);dest=out/'existing_full_prize_curve.json'
def save():
 tmp=dest.with_suffix('.json.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(dest)
save()
for multiple in [.5,1.]:
 for ratio in [.25,.5,1.,1.6,2.5,4.]:
  args=(r['observed_prize']*ratio,r['observed_duration']*multiple,z['sigma'],z['c_i'],z['c_j'],r['fixed_initial_variance'],z['mu_0']);v=dict(ratio=ratio,time_multiplier=multiple)
  try:
   a=f(*args,extent=14.,tolerance=1e-9,quadrature=160);b=g(*args,N=8001,cutoff=26.,quadrature=160);c=g(*args,N=16001,cutoff=28.,quadrature=320);v.update(bvp_B=a['B'],value_grid_B=c['B'],method_difference=a['B']-c['B'],value_grid_refinement_difference=c['B']-b['B'],bvp_hjb_residual=a['maximum_analytic_hjb_residual'],bvp_ode_residual=a['maximum_bvp_rms_residual'],value_grid_residual=c['maximum_scaled_operator_residual'],boundary_budget=c['boundary_output_budget'])
  except Exception as ex:v['error']=repr(ex)
  state['records'].append(v);save();print('FULL PRIZE CURVE',multiple,ratio,v,flush=True)
state['status']='completed';save()
```
