# Task 4：日期3960局部提议失败的群体后续核验

仅跟踪已经终止的当前小时失败；不重复正在运行的目标。以该日期目标的实际生产高权重点探索弯曲分子区域，群体仅拟合冻结全支持提议，最终使用三个独立新预算及原目标核验。保持首轮及退火失败记录。即使平均得分影响较小也不直接接受失败概率。

```python
import json,time,os,traceback
from pathlib import Path
p=Path('docs/review/PaperJK9');status=p/'task4_C_day3960_population_followup_status_20261025.json';assert not status.exists();state=dict(status='watching_day_second_pass',driver_pid=os.getpid(),finished=[],failed=[],current=None,attempted=[])
def save():
 tmp=status.with_suffix('.json.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(status)
def code(f):return f.read_text().partition(chr(10)+chr(96)*3+'python'+chr(10))[2].rpartition(chr(10)+chr(96)*3)[0]
save()
while True:
 upstream=json.loads((p/'task4_C_day_tail_second_pass_3960_status_20261025.json').read_text());todo=[r for r in upstream['failed'] if r['hour'] not in state['attempted']]
 if not todo:
  if upstream['status']!='running':break
  time.sleep(2);continue
 r=todo[0];h=r['hour'];state['current']=h;state['attempted'].append(h);save();base=p/f'predictive_integration_20261025/C_3960_complete_day_hour_{h}_day_tempered2';large=p/f'predictive_integration_20261025/C_3960_complete_day_hour_{h}_day_shared2'
 if not large.exists():large=p/f'predictive_integration_20261025/C_3960_complete_day_hour_{h}'
 try:
  env=dict(KIND='C',CID=3960,HOUR=h,SPLIT='complete_day',RUN_LABEL='_day_population3',BASE_STAGE=str(base),LARGE_BUDGET_STAGE=str(large),RNG_RUN_OFFSET=355000000,PRODUCTION_BUDGETS=[131072,262144,524288,1048576]);exec(compile(code(p/'task4_population_tail_proposal_method_20261025.md'),'day-current-hour-population-proposal','exec'),env)
  if not env['state']['passed']:raise RuntimeError('Three independent population budgets still uncertain')
  exec(compile(code(p/'task4_tail_registration_all_folds_method_20261025.md'),'day-population-original-target-parity','exec'),dict(KIND='C',CID=3960,HOUR=h,SPLIT='complete_day',RUN_LABEL='_day_population3'));state['finished'].append(dict(hour=h,stage=str(env['stage'])))
 except Exception as ex:state['failed'].append(dict(hour=h,reason=repr(ex),traceback=traceback.format_exc()));print('DAY POPULATION ERROR',h,repr(ex),flush=True)
 state['current']=None;save()
state['status']='completed' if not state['failed'] else 'completed_with_uncertainty';save();print('DAY POPULATION FOLLOWUP COMPLETE',len(state['finished']),len(state['failed']),flush=True)
```
