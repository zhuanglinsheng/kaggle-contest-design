# Task 4：大预算异常的跟踪与重新核验

持续读取既有精度批次中已终止的失败任务。只有新预算违反原门槛/一致性时，才将旧条目标为precision_challenged；只未达额外分配精度或提议计算中断，另记为precision-only，不据此宣称旧模型结果错误。挑战条目不再被评分当作已验证尾部积分。

使用新大样本中最高分子重要性权重点及高分子密度点重新搜索模式，三个独立新预算通过，再原模型核验后更新历史及当前索引。所有更新用共享文件锁，不能覆盖其他小时。未解决挑战保留。

```python
import json,time,os,hashlib,traceback,fcntl
from pathlib import Path
import numpy as np
p=Path('docs/review/PaperJK9');upstream=p/'task4_C_aggregate_tail_precision_status_20261025.json';status=p/'task4_C_tail_coverage_followup_status_20261025.json';assert not status.exists();state=dict(status='watching_larger_budget_failures',driver_pid=os.getpid(),finished=[],failed=[],precision_only_failures=[],current=None,challenged=[])
def digest(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def save():
 tmp=status.with_suffix('.json.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(status)
def code(f):return Path(f).read_text().partition(chr(10)+chr(96)*3+'python'+chr(10))[2].rpartition(chr(10)+chr(96)*3)[0]
method=code(p/'task4_tail_coverage_repair_method_20261025.md');verify=code(p/'task4_tail_coverage_registration_method_20261025.md');seen=set();save()
while True:
 up=json.loads(upstream.read_text());queue=[r for r in up['failed'] if (r['contest_id'],r['hour']) not in seen]
 if not queue:
  if up['status']!='running':break
  state['current']=None;save();time.sleep(2);continue
 for task in queue:
  cid=task['contest_id'];h=task['hour'];seen.add((cid,h));root=p/'predictive_integration_20261025'/f'C_{cid}_hour_{h}_precision_refine1';st=json.loads((root/'status.json').read_text());records=st['records']
  if not records or all(r['numerical_screen_passed'] and r['consistent_with_previous'] for r in records):state['precision_only_failures'].append(dict(contest_id=cid,hour=h,reason='No violation of original per-hour gates; extra allocation not achieved or proposal evaluation interrupted'));save();continue
  state['current']=dict(contest_id=cid,hour=h);index=p/'task4_tail_probability_index_20261025.json';key=f'C:{cid}:primary_hour:{h}'
  with (p/'task4_tail_probability_index_20261025.json.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX);ix=json.loads(index.read_text());old=ix['entries'][key]
   if not old.get('precision_challenged',False):ix.setdefault('history',{}).setdefault(key,[]).append(old);ix['entries'][key]=dict(old,precision_challenged=True,challenge_source_status=str((root/'status.json').resolve()),challenge_source_sha256=digest(root/'status.json'));tmp=index.with_suffix('.json.tmp');tmp.write_text(json.dumps(ix,indent=2));tmp.replace(index)
  state['challenged'].append(dict(contest_id=cid,hour=h,source=str((root/'status.json').resolve())));save();starts=[]
  try:
   for r in records:
    with np.load(root/f"production_{r['samples']}.npz") as prod:
     lw=prod['log_training']+prod['log_predictive']-prod['log_proposal'];top=np.argpartition(lw,-8)[-8:];starts.extend(prod['z'][top].tolist());top=np.argpartition(prod['log_training']+prod['log_predictive'],-3)[-3:];starts.extend(prod['z'][top].tolist())
   env=dict(KIND='C',CID=cid,HOUR=h,SPLIT='primary_hour',RUN_LABEL='_coverage_repair1',EXTRA_START_POINTS=starts,MAX_NUMERATOR_MODES=8,RNG_RUN_OFFSET=215000000,PRODUCTION_BUDGETS=[131072,262144,524288,1048576]);exec(compile(method,'new-region-proposal-coverage-repair','exec'),env)
   if not env['state']['passed']:raise RuntimeError('Three fresh independent budgets did not establish coverage')
   exec(compile(verify,'coverage-original-target-locked-registration','exec'),dict(KIND='C',CID=cid,HOUR=h,SPLIT='primary_hour',RUN_LABEL='_coverage_repair1'));state['finished'].append(dict(contest_id=cid,hour=h,coverage_repaired_and_registered=True))
  except Exception as ex:state['failed'].append(dict(contest_id=cid,hour=h,reason=repr(ex),traceback=traceback.format_exc()));print('COVERAGE FOLLOWUP ERROR',cid,h,repr(ex),flush=True)
  save()
state['current']=None;state['status']='completed' if not state['failed'] else 'completed_with_coverage_uncertainty';save();print('COVERAGE FOLLOWUP ENDED',len(state['finished']),len(state['failed']),flush=True)

```
