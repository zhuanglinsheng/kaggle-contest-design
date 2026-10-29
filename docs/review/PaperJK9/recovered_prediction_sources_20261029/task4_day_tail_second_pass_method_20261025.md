# Task 4：完整日期切分剩余概率的第二轮积分

保留首轮失败，单独探索当前日期训练目标的分子区域。3960先对1344小时运行退火提议，再复用同一训练后验的全支持提议；其他案例逐小时运行。复用失败则对当前小时独立探索。退火只拟合提议，最终概率仍来自独立精确密度重要性积分及原Stan/独立Python核验；不复用不同截止的概率。不降低原精度门槛。

```python
import json,os,traceback
from pathlib import Path
p=Path('docs/review/PaperJK9');group=globals()['GROUP'];assert group in ['3960','other'];status=p/f'task4_C_day_tail_second_pass_{group}_status_20261025.json';assert not status.exists();state=dict(status='running',driver_pid=os.getpid(),finished=[],failed=[],current=None)
def save():
 tmp=status.with_suffix('.json.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(status)
def code(f):return f.read_text().partition(chr(10)+chr(96)*3+'python'+chr(10))[2].rpartition(chr(10)+chr(96)*3)[0]
tasks=[(3960,h) for h in [1344,1338,1340,1342,1346,1347,1348,1349,1350,1351,1352,1353,1354,1355,1357]] if group=='3960' else [(5357,1063),(5357,1064),(9120,1611),(9120,1612),(9120,1613),(10200,1160),(37077,1774)]
state['tasks']=tasks;save();basis=None
for cid,h in tasks:
 state['current']=dict(contest_id=cid,hour=h);save();base=p/f'predictive_integration_20261025/C_{cid}_complete_day_hour_{h}';label='_day_tempered2'
 try:
  env=dict(KIND='C',CID=cid,HOUR=h,SPLIT='complete_day',RUN_LABEL=label,BASE_STAGE=str(base),PARTICLE_COUNT=2048,PILOT_MOVES=32,PILOT_RNG_OFFSET=315000000,RNG_RUN_OFFSET=315000000,PRODUCTION_BUDGETS=[131072,262144,524288,1048576])
  if basis is not None:
   label='_day_shared2';env.update(RUN_LABEL=label,BASE_PROPOSAL_STAGE=str(basis));method='task4_shared_tempered_tail_method_20261025.md'
  else:method='task4_tempered_tail_proposal_method_20261025.md'
  exec(compile(code(p/method),'day-same-target-proposal-integration','exec'),env)
  if not env['state']['passed'] and basis is not None:
   label='_day_tempered2';env=dict(KIND='C',CID=cid,HOUR=h,SPLIT='complete_day',RUN_LABEL=label,BASE_STAGE=str(base),PARTICLE_COUNT=2048,PILOT_MOVES=32,PILOT_RNG_OFFSET=325000000,RNG_RUN_OFFSET=325000000,PRODUCTION_BUDGETS=[131072,262144,524288,1048576]);exec(compile(code(p/'task4_tempered_tail_proposal_method_20261025.md'),'day-current-hour-proposal-fallback','exec'),env)
  if not env['state']['passed']:raise RuntimeError('Independent production precision not established')
  exec(compile(code(p/'task4_tail_registration_all_folds_method_20261025.md'),'day-current-target-original-parity','exec'),dict(KIND='C',CID=cid,HOUR=h,SPLIT='complete_day',RUN_LABEL=label))
  if group=='3960' and basis is None:basis=env['stage']
  state['finished'].append(dict(contest_id=cid,hour=h,stage=str(env['stage'])))
 except Exception as ex:state['failed'].append(dict(contest_id=cid,hour=h,reason=repr(ex),traceback=traceback.format_exc()));print('DAY SECOND PASS ERROR',cid,h,repr(ex),flush=True)
 save()
state['current']=None;state['status']='completed' if not state['failed'] else 'completed_with_uncertainty';save();print('DAY SECOND PASS FINISHED',group,len(state['finished']),len(state['failed']),flush=True)
```
