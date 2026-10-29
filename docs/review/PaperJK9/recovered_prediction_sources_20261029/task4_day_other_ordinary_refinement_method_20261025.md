# Task 4：日期非3960案例的残余整段精度

等待既有7小时第二轮批次结束，再对5357/9120/10200/37077计算独立全路径积分；与3960的不同目标修复独立进行。尾部失败者跳过，不删除案例；接受仍需原目标及完整路径检查。后续统一评分读取相同索引，避免重复运行。

```python
import json,time,os,traceback
from pathlib import Path
p=Path('docs/review/PaperJK9');status=p/'task4_C_day_other_ordinary_refinement_status_20261025.json';assert not status.exists();state=dict(status='waiting_for_tail_batch',driver_pid=os.getpid(),finished=[],failed=[],current=None)
def save():
 tmp=status.with_suffix('.json.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(status)
def code(f):return f.read_text().partition(chr(10)+chr(96)*3+'python'+chr(10))[2].rpartition(chr(10)+chr(96)*3)[0]
save()
while json.loads((p/'task4_C_day_tail_second_pass_other_status_20261025.json').read_text())['status']=='running':time.sleep(2)
tail=json.loads((p/'task4_C_day_tail_second_pass_other_status_20261025.json').read_text());blocked={r['contest_id'] for r in tail['failed']}
for cid in [5357,9120,10200,37077]:
 if cid in blocked:state['failed'].append(dict(contest_id=cid,reason='Tail probability remains unresolved'));save();continue
 state['status']='integrating';state['current']=cid;save()
 try:
  env=dict(KIND='C',CID=cid,SPLIT='complete_day',RUN_LABEL='_day_other_ordinary2',RNG_RUN_OFFSET=365000000,PRODUCTION_BUDGETS=[262144,524288,1048576,2097152]);exec(compile(code(p/'task4_global_path_importance_method_20261025.md'),'day-ordinary-exact-integration','exec'),env)
  if not env['state']['passed']:raise RuntimeError('Whole-case precision not established')
  exec(compile(code(p/'task4_global_path_registration_method_20261025.md'),'day-ordinary-original-target-parity','exec'),dict(ESTIMATE_STAGE=str(env['stage'])));state['finished'].append(dict(contest_id=cid,stage=str(env['stage'])))
 except Exception as ex:state['failed'].append(dict(contest_id=cid,reason=repr(ex),traceback=traceback.format_exc()));print('DAY ORDINARY ERROR',cid,repr(ex),flush=True)
 save()
state['current']=None;state['status']='completed' if not state['failed'] else 'completed_with_uncertainty';save();print('DAY ORDINARY COMPLETE',len(state['finished']),len(state['failed']),flush=True)
```
