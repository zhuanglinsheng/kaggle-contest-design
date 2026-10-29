# Task 3：所选无反馈极限设计的小正精度核查

日期：2026-10-25。在 20270、2464、5144 三个已选联合代表点，保持原初始分布，按所选奖金与期限核算。用无量纲信息量 `nu=lambda*(S0+sigma²*T)*T` 的 1e-4、1e-6、1e-8 三层小正精度，分别进行两层空间/时间加密。中间精度额外保存内存策略路径，分别独立审查两位参赛者的最佳回应与净漂移价值。

记录负密度或推进失败，不用完全披露结果代替所选无反馈设计。小正精度的数值延拓不是数学极限证明；联合代表点也不代替后验平均质量。强激励失败时保留未确定。

[批次数据](task3_selected_weak_feedback_points_20261025.json)。

```python
import csv,hashlib,json,math,time
from pathlib import Path
import numpy as np
from pipeline.empirical.fixed_initial_design import Output
report=Path('docs/review/PaperJK9')
method=(report/'task3_asymmetric_weak_feedback_method_20261025.md').read_text().split('```python\n',1)[1].split('```',1)[0]
response_method=(report/'task3_asymmetric_best_response_method_20261025.md').read_text().split('```python\n',1)[1].split('```',1)[0]
exec(compile(method,'<reported asymmetric spectrum>','exec'));exec(compile(response_method,'<reported asymmetric response>','exec'))
points={r['contest_id']:r for r in json.loads((report/'task3_representative_joint_points_20261025.json').read_text())['points']}
selection=[r for r in csv.DictReader((report/'task3_representative_input_inventory_20261025.csv').open()) if r['choice']=='no_feedback_limit']
out=report/'task3_selected_weak_feedback_points_20261025.json'
if out.exists():raise RuntimeError('Existing weak-feedback output: inspect before resume')
state=dict(date='2026-10-25',status='running',scope='Positive precision approaching no-feedback at the selected prize-duration and actual representative joint draws; no posterior-average certification',method_sha256=hashlib.sha256(method.encode()).hexdigest(),best_response_method_sha256=hashlib.sha256(response_method.encode()).hexdigest(),records=[],nonlinear_quality_preservation_verified=False)
def save():
 tmp=out.with_suffix('.json.tmp');tmp.write_text(json.dumps(state,indent=2,allow_nan=False));tmp.replace(out)
save()
for selected in selection:
 cid=int(selected['contest_id']);p=points[cid];z=p['point'];theta=p['observed_prize']*float(selected['prize_ratio']);T=p['observed_duration']*float(selected['time_multiplier']);S0=p['fixed_initial_variance'];q0=S0+z['sigma']**2*T
 args=dict(sigma=z['sigma'],theta=theta,T=T,ci=z['c_i'],cj=z['c_j'],S0=S0,mu=z['mu_0'])
 draws={k:np.array([z[k]]) for k in ('c_i','c_j','sigma','lambda','mu_0')};linear=Output(draws,T,96)
 record=dict(contest_id=cid,inputs=args,first_order_zero_limit_at_same_point=linear.no_effort+theta*linear.value(0.),trials=[])
 for nu in (1e-4,1e-6,1e-8):
  precision=nu/(q0*T)
  for level,(N,steps) in enumerate(((1024,4000),(2048,8000))):
   item=dict(normalized_information=nu,physical_precision=precision,level=level,N=N,steps=steps)
   try:
    value,path=asymmetric_spectral(**args,precision=precision,N=N,steps=steps,keep_path=(nu==1e-6))
    item['result']=value
    if nu==1e-6 and value['physical_screen_passed']:
     x=path['x'];tt=path['times'];qi=path['policies'][:,:,0];qj=path['policies'][:,:,1]
     item['best_responses']={}
     for player,own,opp,cost in [('i',qi,qj,z['c_i']),('j',qj[::-1,:],qi[::-1,:],z['c_j'])]:
      item['best_responses'][player]=evaluate_asymmetric_policy(z['sigma'],theta/cost,T,precision,S0,x,tt,own,opp,steps=2000*2**level)
   except Exception as exc:item['error']=str(exc)
   record['trials'].append(item);state['records']=[r for r in state['records'] if r['contest_id']!=cid]+[record];save()
   print('selected weak',cid,'nu',nu,'level',level,'done',flush=True)
state['status']='completed';save();print('Selected weak-feedback representative batch completed.',flush=True)
```
