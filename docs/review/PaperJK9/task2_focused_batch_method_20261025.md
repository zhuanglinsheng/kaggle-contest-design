# Task 2：集中候选只读试算批次方法

日期：2026-10-25。输入为上轮 250 个恢复候选；只保存审查报告，不改论文、实现或原结果。

目的：检查候选产出在空间加密下是否稳定，定位能改变类别比较的设计。两层网格、固定范围 10、谱阈值 1e-14；依据显式扩散稳定性选择时间步。筛查密度、质量、价值边界和独立收益残差。`max(3*网格产出差,1e-7)` 仅为诊断半径，不是严格误差界。两层检查后还须进行时间、阈值、范围敏感性和独立最佳回应审查。

零精度记录只表示直接退化方程；不代表已经验证正精度极限。批次输出不能填回原分区，不能认证原完整候选集或连续设计域的类别。

## 执行驱动快照

```python
import csv, hashlib, json, math, time, traceback
from pathlib import Path
import numpy as np
root = Path.cwd()
report_dir = root / 'docs/review/PaperJK9'
method_text = (report_dir/'task2_spectral_audit_appendix_20261024.md').read_text()
method_source = method_text.split('```python\n',1)[1].split('```',1)[0]
exec(compile(method_source,'<reported spectral prototype>','exec'))
queue_path=report_dir/'task2_focused_candidate_queue_20261024.csv'
queue=list(csv.DictReader(queue_path.open()))
out=report_dir/'task2_focused_spectral_batch_20261025.json'
if out.exists():
    raise RuntimeError('Existing batch: inspect and resume explicitly, do not overwrite')
payload=dict(date='2026-10-25',scope='Report-only numerical screening of 250 recovered candidates; no global classification certificate',
    queue_sha256=hashlib.sha256(queue_path.read_bytes()).hexdigest(),
    method_sha256=hashlib.sha256(method_source.encode()).hexdigest(),
    driver_source=Path('docs/review/PaperJK9/task2_focused_batch_method_20261025.md').read_text().split('```python\n',1)[1].split('```',1)[0],
    candidate_count=len(queue),status='running',records=[],global_classification_verified=False)
def persist():
    temp=out.with_suffix('.json.tmp')
    temp.write_text(json.dumps(payload,indent=2,allow_nan=False))
    temp.replace(out)
def check(r):
    return (all(math.isfinite(r[k]) for k in ('B','global_residual','minimum_density_all_rk_stages'))
      and r['B']>=0 and r['minimum_density_all_rk_stages']>=-1e-8
      and r['maximum_mass_error_all_rk_stages']<=1e-8
      and r['maximum_total_value']<=1+1e-7 and r['global_residual']<=2e-6)
persist()
start=time.monotonic()
for index,row in enumerate(queue):
    args={k:float(row[k]) for k in ('sigma','alpha','T','precision')}
    e=float(row['normalized_exposure'])
    base=1024 if e>1.15 and args['precision']<.01 else 512
    record=dict(index=index,design=args,normalized_exposure=e,queue=row,trials=[])
    for N in (base,2*base):
        # RK4 diffusion CFL bound on retained Fourier modes; S is monotone.
        rate=args['sigma']*math.sqrt(args['precision'])*args['T']
        if args['precision']:
            steady=args['sigma']/math.sqrt(args['precision']);z=math.tanh(rate)
            terminal=steady*(1+steady*z)/(steady+z)
        else:
            terminal=1+args['sigma']**2*args['T']
        q0=1+args['sigma']**2*args['T']
        Dmax=args['precision']*max(1.,terminal)**2*args['T']/q0/2
        steps=max(2000 if N<=1024 else 4000,int(math.ceil(Dmax*(math.pi*N/30)**2/1.4)))
        steps=4*int(math.ceil(steps/4))
        try:
            trial,_=spectral_audit(**args,N=N,steps=steps,tolerance=1e-14,extent=10)
            record['trials'].append(dict(result=trial,physical_screen_passed=check(trial)))
        except Exception as exc:
            record['trials'].append(dict(N=N,steps=steps,error=str(exc),physical_screen_passed=False))
    valid=[t for t in record['trials'] if 'result' in t and t['physical_screen_passed']]
    record['both_mesh_physical_screens_passed']=len(valid)==2
    if len(valid)==2:
        difference=abs(valid[1]['result']['B']-valid[0]['result']['B'])
        record['mesh_output_difference']=difference
        record['diagnostic_output_radius']=max(3*difference,1e-7)
        record['mesh_output_screen_passed']=difference<=2e-5
    record['best_response_audited']=False
    record['positive_precision_limit_verified']=False
    record['global_classification_verified']=False
    payload['records'].append(record)
    payload['elapsed_seconds']=time.monotonic()-start
    persist()
    if (index+1)%10==0 or index+1==len(queue):
        print(json.dumps(dict(completed=index+1,total=len(queue),seconds=round(payload['elapsed_seconds'],1),last_precision=args['precision']),ensure_ascii=False),flush=True)
payload['status']='completed';persist()
print('Completed report-only screening batch.',flush=True)
```
