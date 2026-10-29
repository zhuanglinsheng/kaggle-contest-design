# Task 2：粗网格筛查失败的补充加密

日期：2026-10-25。只读使用报告中的谱方法，针对集中批次未通过两层检查的候选，在此前最细空间网格基础上再加倍，时间步数同时加倍（至少 4,000）。原失败记录保留；只有最后两层均通过物理检查且产出差不超过 2e-5，才标记通过本次数值筛查。

诊断半径为 max(3*最后两层产出差,1e-7)，不是严格误差界。该步骤仍不替代单独的时间、范围、截断阈值敏感性与最佳回应检查，不更新任何原分区。

[补充记录](task2_focused_spectral_refinement_20261025.json)。驱动在原批次运行时接收新失败项，直到原批次完成并核查全部已出现的失败项。

```python
import json,math,time,hashlib,os
from pathlib import Path
report=Path('docs/review/PaperJK9')
text=(report/'task2_spectral_audit_appendix_20261024.md').read_text()
method=text.split('```python\n',1)[1].split('```',1)[0]
exec(compile(method,'<reported spectral method>','exec'))
source_path=report/'task2_focused_spectral_batch_20261025.json'
output=report/'task2_focused_spectral_refinement_20261025.json'
if output.exists():raise RuntimeError('Existing refinement: inspect before resume')
state=dict(status='running',driver_pid=os.getpid(),scope='Additional fine mesh for coarse-screen failures; report-only diagnostics, no classification update',method_sha256=hashlib.sha256(method.encode()).hexdigest(),records=[],global_classification_verified=False)
def save():
 temp=output.with_suffix('.json.tmp');temp.write_text(json.dumps(state,indent=2,allow_nan=False));temp.replace(output)
def physical(r):
 return (r['B']>=0 and r['minimum_density_all_rk_stages']>=-1e-8 and r['maximum_mass_error_all_rk_stages']<=1e-8 and r['maximum_total_value']<=1+1e-7 and r['global_residual']<=2e-6 and all(math.isfinite(r[k]) for k in ('B','global_residual','minimum_density_all_rk_stages')))
save();completed=set()
while True:
 data=json.loads(source_path.read_text())
 pending=[r for r in data['records'] if not r.get('mesh_output_screen_passed',False) and r['index'] not in completed]
 for r in pending:
  old=r['trials'][-1]
  item=dict(index=r['index'],design=r['design'],earlier_finest=old,additional_trial=None)
  if 'result' not in old:
   item['error']='No successful finest trial; needs separate recovery'
  else:
   N=2*old['result']['N'];steps=max(4000,2*old['result']['steps'])
   try:
    new,_=spectral_audit(**r['design'],N=N,steps=steps,tolerance=1e-14,extent=10)
    item['additional_trial']=dict(result=new,physical_screen_passed=physical(new))
    difference=abs(new['B']-old['result']['B']);item['mesh_output_difference']=difference
    item['last_two_mesh_screens_passed']=bool(old['physical_screen_passed'] and physical(new) and difference<=2e-5)
    item['diagnostic_output_radius']=max(3*difference,1e-7)
   except Exception as exc:item['error']=str(exc);item['last_two_mesh_screens_passed']=False
  item['best_response_audited']=False;item['global_classification_verified']=False
  completed.add(r['index']);state['records'].append(item);save()
  print(json.dumps(dict(refined=r['index'],passed=item.get('last_two_mesh_screens_passed',False),total_refined=len(completed))),flush=True)
 if data['status']=='completed':break
 time.sleep(2)
state['status']='completed';state['source_candidate_count']=len(data['records']);state['source_sha256_at_completion']=hashlib.sha256(source_path.read_bytes()).hexdigest();save()
print('Additional refinement completed.',flush=True)
```
