# Task 4：联合积分与旧独立概率不一致时的复核

日期3960联合积分的三个完整预算通过，但1339/1343与旧独立积分不一致。暂不登记联合结果。旧条目标记precision_challenged并保留history，利用联合生产实际最高分子权重点作为当前小时群体提议的附加种子；它们仅作探索种子，不冒充当前小时生产抽样。各小时重新精确计算训练/当前分子密度，三个新独立预算通过，再核对原Stan/独立Python。待差异修复后重新比较旧独立来源与联合估计，任何不一致仍保留。

```python
import json,os,traceback,hashlib,fcntl
from pathlib import Path
p=Path('docs/review/PaperJK9');status=p/'task4_C_day_independent_disagreement_repair_status_20261025.json';assert not status.exists();state=dict(status='running',driver_pid=os.getpid(),finished=[],failed=[],current=None);joint=p/'global_path_importance_20261025/C_complete_day_3960_day_joint_direct2';cf=joint/'independent_tail_compatibility.json';disagreement=json.loads(cf.read_text());assert not disagreement['passed'];tasks=[r for r in disagreement['records'] if not r['consistent']];assert {r['hour'] for r in tasks}=={1339,1343};preserved=joint/'independent_tail_compatibility_before_repair.json';assert not preserved.exists();preserved.write_bytes(cf.read_bytes());index=p/'task4_tail_probability_index_20261025.json'
def digest(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def code(f):return f.read_text().partition(chr(10)+chr(96)*3+'python'+chr(10))[2].rpartition(chr(10)+chr(96)*3)[0]
def save():
 tmp=status.with_suffix('.json.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(status)
with index.with_suffix('.json.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);ix=json.loads(index.read_text())
 for r in tasks:
  key=f"C:3960:complete_day:{r['hour']}";old=ix['entries'][key];assert old['integral_status_sha256']==r['old_integral_status_sha256'];ix.setdefault('history',{}).setdefault(key,[]).append(old);ix['entries'][key]=dict(old,precision_challenged=True,challenge_source_status=str(preserved.resolve()),challenge_source_sha256=digest(preserved))
 tmp=index.with_suffix('.json.tmp');tmp.write_text(json.dumps(ix,indent=2));tmp.replace(index)
save();st=json.loads((joint/'status.json').read_text());extra=joint/f"production_{st['records'][-1]['samples']}.npz"
for r in tasks:
 h=r['hour'];state['current']=h;save();base=Path(r['old_integral_status']).parent
 try:
  env=dict(KIND='C',CID=3960,HOUR=h,SPLIT='complete_day',RUN_LABEL='_day_independent_countercheck4',BASE_STAGE=str(base),LARGE_BUDGET_STAGE=str(base),EXTRA_SEED_POINTS_NPZ=str(extra),RNG_RUN_OFFSET=395000000,PRODUCTION_BUDGETS=[131072,262144,524288,1048576]);exec(compile(code(p/'task4_population_tail_proposal_method_20261025.md'),'independent-current-hour-coverage-countercheck','exec'),env)
  if not env['state']['passed']:raise RuntimeError('Three independent countercheck budgets not established')
  exec(compile(code(p/'task4_tail_coverage_registration_method_20261025.md'),'countercheck-original-target-and-history-registration','exec'),dict(KIND='C',CID=3960,HOUR=h,SPLIT='complete_day',RUN_LABEL='_day_independent_countercheck4'));state['finished'].append(dict(hour=h,stage=str(env['stage'])))
 except Exception as ex:state['failed'].append(dict(hour=h,reason=repr(ex),traceback=traceback.format_exc()));print('INDEPENDENT COUNTERCHECK ERROR',h,repr(ex),flush=True)
 save()
state['current']=None;state['status']='completed' if not state['failed'] else 'completed_with_uncertainty';save();print('INDEPENDENT COUNTERCHECK FINISHED',len(state['finished']),len(state['failed']),flush=True)
```
