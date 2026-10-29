# Task 4：局部提议失败后的退火修复批次

仅处理已完成且仍不确定的局部修复。C3960优先复用同训练目标、已通过生产门槛的1344退火提议；若仍失败则当前小时独立退火探索。其他竞赛必须独立建立提议。提议探索不用于最终概率/参数推断；最终仍是冻结提议下独立精确密度积分，门槛不变。

等待上游完成后再顺序核对原目标并登记，避免并发索引覆盖；原始及各修复失败均保留。批次记录明确区分生产通过待核验和已登记，不将前者当成评分已修复。

```python
import json,time,os,traceback
from pathlib import Path
p=Path('docs/review/PaperJK9');upstream=p/'task4_C_primary_tail_repair_status_20261025.json';status=p/'task4_C_primary_tempered_repair_status_20261025.json';assert not status.exists();state=dict(status='running',driver_pid=os.getpid(),finished=[],failed=[],pending_registration=[],current=None,scope='Finish cases still uncertain after local proposal repair; same-target shared tempered proposal or separate tempered proposal pilot; unchanged production gates')
def save():
 tmp=status.with_suffix('.json.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(status)
def code(f):return Path(f).read_text().partition(chr(10)+chr(96)*3+'python'+chr(10))[2].rpartition(chr(10)+chr(96)*3)[0]
shared=code(p/'task4_shared_tempered_tail_method_20261025.md');tempered=code(p/'task4_tempered_tail_proposal_method_20261025.md');verify=code(p/'task4_tail_registration_all_folds_method_20261025.md');seen=set();save()
while True:
 up=json.loads(upstream.read_text());queue=[];ix=json.loads((p/'task4_tail_probability_index_20261025.json').read_text())['entries']
 for r in up['failed']:
  key=f"C:{r['contest_id']}:primary_hour:{r['hour']}"
  if key in seen:continue
  if key in ix:seen.add(key);state['finished'].append(dict(contest_id=r['contest_id'],hour=r['hour'],reused_existing_validation=True));save();continue
  queue.append(r)
 if not queue:
  if up['status']!='running':break
  state['phase']='waiting_for_completed_local_failures';state['current']=None;save();time.sleep(2);continue
 for task in queue:
  cid=task['contest_id'];hour=task['hour'];seen.add(f'C:{cid}:primary_hour:{hour}');state['phase']='tempered_refinement';state['current']=dict(contest_id=cid,hour=hour);save()
  try:
   known=p/'predictive_integration_20261025'/f'C_{cid}_hour_{hour}_tempered1'
   if known.exists() and json.loads((known/'status.json').read_text()).get('passed'):state['pending_registration'].append(dict(contest_id=cid,hour=hour,run_label='_tempered1'));save();continue
   passed=False
   if cid==3960:
    label='_shared_tempered1';env=dict(KIND='C',CID=cid,HOUR=hour,SPLIT='primary_hour',RUN_LABEL=label,BASE_PROPOSAL_STAGE=str(p/'predictive_integration_20261025/C_3960_hour_1344_tempered1'),RNG_RUN_OFFSET=55000000);exec(compile(shared,'shared-tempered-proposal','exec'),env);passed=env['state']['passed']
   if not passed:
    label='_tempered2';base=p/'predictive_integration_20261025'/f'C_{cid}_hour_{hour}_repair1';env=dict(KIND='C',CID=cid,HOUR=hour,SPLIT='primary_hour',RUN_LABEL=label,BASE_STAGE=str(base),RNG_RUN_OFFSET=65000000);exec(compile(tempered,'separate-tempered-proposal','exec'),env);passed=env['state']['passed']
   if passed:state['pending_registration'].append(dict(contest_id=cid,hour=hour,run_label=label))
   else:state['failed'].append(dict(contest_id=cid,hour=hour,reason='Tempered proposal production still failed unchanged gates'))
  except Exception as ex:state['failed'].append(dict(contest_id=cid,hour=hour,reason=repr(ex),traceback=traceback.format_exc()));print('TEMPERED REPAIR ERROR',cid,hour,repr(ex),flush=True)
  save()
state['phase']='registering_after_local_repair_terminal';save()
for task in state['pending_registration']:
 try:
  ix=json.loads((p/'task4_tail_probability_index_20261025.json').read_text())['entries'];key=f"C:{task['contest_id']}:primary_hour:{task['hour']}"
  if key not in ix:exec(compile(verify,'tempered-original-target-check','exec'),dict(KIND='C',CID=task['contest_id'],HOUR=task['hour'],SPLIT='primary_hour',RUN_LABEL=task['run_label']))
  state['finished'].append(dict(task,original_target_verified=True))
 except Exception as ex:state['failed'].append(dict(task,reason=repr(ex),traceback=traceback.format_exc()));print('TEMPERED REGISTRATION ERROR',task,repr(ex),flush=True)
 save()
state['current']=None;state['status']='completed' if not state['failed'] else 'completed_with_uncertainty';save();print('TEMPERED REPAIR BATCH ENDED',len(state['finished']),len(state['failed']),flush=True)

```
