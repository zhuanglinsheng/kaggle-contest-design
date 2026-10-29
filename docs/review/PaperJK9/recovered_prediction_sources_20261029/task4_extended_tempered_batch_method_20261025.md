# Task 4：剩余10个稀有小时的独立加密

33个局部修复未决目标中，退火修复已验证17项，6项提议数值异常另行修复，剩10项为生产积分仍未达门槛。本批以4096粒子、每层36次移动独立探索提议，按131072/262144/524288/1048576新随机数预算补足。使用提议条件数限制，不改目标；全部验收门槛不变，保留各阶段失败。生产通过后等待上一登记批次终止，再核对原Stan/Python并顺序登记。

```python
import json,os,time,traceback
from pathlib import Path
p=Path('docs/review/PaperJK9');up=json.loads((p/'task4_C_primary_tempered_repair_status_20261025.json').read_text());assert up['status']!='running';tasks=[r for r in up['failed'] if r['reason']=='Tempered proposal production still failed unchanged gates'];status=p/'task4_C_extended_tempered_status_20261025.json';assert not status.exists();state=dict(status='running',driver_pid=os.getpid(),tasks=[dict(contest_id=r['contest_id'],hour=r['hour']) for r in tasks],finished=[],failed=[],pending_registration=[],current=None,particle_count=4096,pilot_moves=36,production_budgets=[131072,262144,524288,1048576])
def save():
 tmp=status.with_suffix('.json.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(status)
def code(f):return Path(f).read_text().partition(chr(10)+chr(96)*3+'python'+chr(10))[2].rpartition(chr(10)+chr(96)*3)[0]
method=code(p/'task4_tempered_tail_proposal_method_20261025.md');verify=code(p/'task4_tail_registration_all_folds_method_20261025.md');save()
for task in tasks:
 cid=task['contest_id'];hour=task['hour'];state['current']=dict(contest_id=cid,hour=hour);save()
 try:
  base=p/'predictive_integration_20261025'/f'C_{cid}_hour_{hour}_tempered2';env=dict(KIND='C',CID=cid,HOUR=hour,SPLIT='primary_hour',RUN_LABEL='_extended_tempered1',BASE_STAGE=str(base),RNG_RUN_OFFSET=95000000,PILOT_RNG_OFFSET=90000000,PARTICLE_COUNT=4096,PILOT_MOVES=36,PRODUCTION_BUDGETS=[131072,262144,524288,1048576]);exec(compile(method,'extended-same-target-importance','exec'),env)
  if env['state']['passed']:state['pending_registration'].append(dict(contest_id=cid,hour=hour))
  else:state['failed'].append(dict(contest_id=cid,hour=hour,reason='Extended independent budgets still failed unchanged gates'))
 except Exception as ex:state['failed'].append(dict(contest_id=cid,hour=hour,reason=repr(ex),traceback=traceback.format_exc()));print('EXTENDED TEMPERED ERROR',cid,hour,repr(ex),flush=True)
 save()
while json.loads((p/'task4_C_proposal_numerics_repair_status_20261025.json').read_text())['status']=='running':state['phase']='waiting_for_numeric_repair_registration';save();time.sleep(2)
for task in state['pending_registration']:
 try:
  key=f"C:{task['contest_id']}:primary_hour:{task['hour']}";ix=json.loads((p/'task4_tail_probability_index_20261025.json').read_text())['entries']
  if key not in ix:exec(compile(verify,'extended-original-target-check','exec'),dict(KIND='C',CID=task['contest_id'],HOUR=task['hour'],SPLIT='primary_hour',RUN_LABEL='_extended_tempered1'))
  state['finished'].append(dict(task,original_target_verified=True))
 except Exception as ex:state['failed'].append(dict(task,reason=repr(ex),traceback=traceback.format_exc()))
 save()
state['current']=None;state['status']='completed' if not state['failed'] else 'completed_with_uncertainty';save();print('EXTENDED TEMPERED ENDED',len(state['finished']),len(state['failed']),flush=True)

```
