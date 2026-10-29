# Task 3：小正精度设计的 Fourier 初值评价与敏感性复核

日期：2026-10-25。初次非对称试算在非网格初始均值处使用了线性插值，产生约 1e-5 至 1e-4 的二阶评价误差。原谱场及独立最佳回应不受该输出取值方式影响；原结果与源代码快照完整保留。

本批次使用 Fourier 系数直接评价初始均值处的产出，在 20270、2464 两个能通过物理筛查的代表点，分别检查空间、时间、谱阈值/域及正精度进一步缩小；不混合旧插值与新评价构成收敛序列。5144 的负密度失败不因更换输出评价而被接受。

[复核记录](task3_selected_weak_feedback_fourier_checks_20261025.json)。

```python
import hashlib,json,math
from pathlib import Path
report=Path('docs/review/PaperJK9')
method=(report/'task3_asymmetric_weak_feedback_method_20261025.md').read_text().split('```python\n',1)[1].split('```',1)[0]
exec(compile(method,'<Fourier point evaluation>','exec'))
old=json.loads((report/'task3_selected_weak_feedback_points_20261025.json').read_text())
out=report/'task3_selected_weak_feedback_fourier_checks_20261025.json'
if out.exists():raise RuntimeError('Existing follow-up: inspect before resume')
state=dict(status='running',scope='Correct off-grid output interpolation and separate mesh/time/cutoff checks at two viable representative weak-feedback designs',source_sha256=hashlib.sha256(method.encode()).hexdigest(),source_snapshot=method,records=[],zero_precision_limit_proved=False)
def save():
 tmp=out.with_suffix('.json.tmp');tmp.write_text(json.dumps(state,indent=2,allow_nan=False));tmp.replace(out)
save()
for r in old['records']:
 if r['contest_id'] not in (20270,2464):continue
 args=r['inputs'];q0=args['S0']+args['sigma']**2*args['T']
 configs=[(1e-6,1024,4000,1e-14,10.),(1e-6,2048,4000,1e-14,10.),(1e-6,2048,8000,1e-14,10.),(1e-6,4096,8000,1e-14,10.),(1e-6,2048,8000,1e-16,12.),(1e-8,2048,8000,1e-14,10.)]
 for nu,N,steps,tol,extent in configs:
  item=dict(contest_id=r['contest_id'],normalized_information=nu)
  try:item['result'],_=asymmetric_spectral(**args,precision=nu/(q0*args['T']),N=N,steps=steps,tolerance=tol,extent=extent)
  except Exception as exc:item['error']=str(exc)
  state['records'].append(item);save();print('Fourier output',r['contest_id'],nu,N,steps,tol,'done',flush=True)
state['status']='completed';save()
```
