# Task 4：3960四个被挑战小时的群体提议核验

等待原局部修复结束后登记1337的群体提议积分，再检查尚未被其他方法修复的1339/1343/1345。每小时三个独立预算、原Stan/Python核验和同目标历史更新均保留；失败继续标为待核验。不能因为主均值影响较小就直接接受旧概率。

```python
import json,time,os,traceback
from pathlib import Path
p=Path('docs/review/PaperJK9');status=p/'task4_C3960_population_coverage_status_20261025.json';assert not status.exists();state=dict(status='waiting_for_local_coverage_batch',driver_pid=os.getpid(),finished=[],failed=[],current=None)
def save():
 tmp=status.with_suffix('.json.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(status)
def code(f):return Path(f).read_text().partition(chr(10)+chr(96)*3+'python'+chr(10))[2].rpartition(chr(10)+chr(96)*3)[0]
save()
while json.loads((p/'task4_C_tail_coverage_followup_status_20261025.json').read_text())['status']=='watching_larger_budget_failures':time.sleep(2)
verify=code(p/'task4_tail_coverage_registration_method_20261025.md');basis=p/'predictive_integration_20261025/C_3960_hour_1337_population1';assert json.loads((basis/'status.json').read_text())['passed'];index=p/'task4_tail_probability_index_20261025.json'
for h in [1337,1339,1343,1345]:
 state['status']='checking_population_coverage';state['current']=h;save();entry=json.loads(index.read_text())['entries'][f'C:3960:primary_hour:{h}']
 if not entry.get('precision_challenged',False):state['finished'].append(dict(hour=h,already_repaired=True));save();continue
 try:
  if h==1337:label='_population1'
  else:
   label='_shared_population1';env=dict(KIND='C',CID=3960,HOUR=h,SPLIT='primary_hour',RUN_LABEL=label,BASE_PROPOSAL_STAGE=str(basis),RNG_RUN_OFFSET=275000000,PRODUCTION_BUDGETS=[131072,262144,524288,1048576]);exec(compile(code(p/'task4_shared_population_coverage_method_20261025.md'),'shared-population-exact-probability','exec'),env)
   if not env['state']['passed']:raise RuntimeError('Three independent shared-population budgets not accepted')
  exec(compile(verify,'population-important-region-original-target-parity','exec'),dict(KIND='C',CID=3960,HOUR=h,SPLIT='primary_hour',RUN_LABEL=label));state['finished'].append(dict(hour=h,repaired=True))
 except Exception as ex:state['failed'].append(dict(hour=h,reason=repr(ex),traceback=traceback.format_exc()));print('POPULATION COVERAGE ERROR',h,repr(ex),flush=True)
 save()
state['current']=None;state['status']='completed' if not state['failed'] else 'completed_with_uncertainty';save();print('POPULATION COVERAGE BATCH ENDED',len(state['finished']),len(state['failed']),flush=True)

```
