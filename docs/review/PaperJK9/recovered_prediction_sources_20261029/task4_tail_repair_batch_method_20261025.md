# Task 4：遗漏分子模式的独立修复

首次提议未通过的目标使用其最高实际重要性权重点、较高分子密度点增加多起点搜索。C3960小时1342已发现并核验原多起点遗漏的模式，独立新预算通过；同竞赛其他小时可将该点作为额外数值起点，仍重新求当前概率的精确分子密度。目标后验不更新，提议再次冻结，门槛不变。失败状态及生产数据完整保留，修复输出到repair1目录。

为避免并发覆盖统一索引，原批次运行期间只计算修复；原批次终止后再顺序做原Stan/Python重要区域核验并登记。仅登记通过者；其他保留未决。

```python
import json,os,time,traceback
from pathlib import Path
import numpy as np
p=Path('docs/review/PaperJK9');original=p/'task4_C_primary_hour_tail_batch_status_20261025.json';status=p/'task4_C_primary_tail_repair_status_20261025.json';assert not status.exists();state=dict(status='running',driver_pid=os.getpid(),finished=[],failed=[],pending_registration=[],current=None,scope='Repair failed numerator proposal coverage using actual high-importance production points; independent new proposals/budgets, same posterior target; registration serialized after original batch')
def save():
 tmp=status.with_suffix('.json.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(status)
def code(f):return Path(f).read_text().partition(chr(10)+chr(96)*3+'python'+chr(10))[2].rpartition(chr(10)+chr(96)*3)[0]
integrate=code(p/'task4_tail_integration_all_folds_method_20261025.md');verify=code(p/'task4_tail_registration_all_folds_method_20261025.md');seen=set();save()
while True:
 orig=json.loads(original.read_text());queue=[];ix=json.loads((p/'task4_tail_probability_index_20261025.json').read_text())['entries']
 for r in orig['failed']:
  key=f"C:{r['contest_id']}:primary_hour:{r['hour']}"
  if key in seen:continue
  if key in ix:seen.add(key);state['finished'].append(dict(contest_id=r['contest_id'],hour=r['hour'],reused_existing_validated_repair=True));save();continue
  queue.append(r)
 if not queue:
  if orig['status']!='running':break
  state['phase']='waiting_for_next_completed_failed_target';state['current']=None;save();time.sleep(2);continue
 for task in queue:
  cid=task['contest_id'];hour=task['hour'];key=f'C:{cid}:primary_hour:{hour}';seen.add(key);state['phase']='repairing';state['current']=dict(contest_id=cid,hour=hour);save()
  try:
   root=p/'predictive_integration_20261025'/f'C_{cid}_hour_{hour}';s=json.loads((root/'status.json').read_text());starts=[]
   for r in s['records']:
    prod=np.load(root/f"production_{r['samples']}.npz");lw=prod['log_training']+prod['log_predictive']-prod['log_proposal'];starts.extend(prod['z'][np.argsort(lw)[-5:]].tolist());starts.extend(prod['z'][np.argsort(prod['log_training']+prod['log_predictive'])[-2:]].tolist())
   learned=p/'predictive_integration_20261025/C_3960_hour_1342_repair1/proposal.json'
   if cid==3960 and learned.exists():starts.extend([r['point'] for r in sorted(json.loads(learned.read_text())['modes'],key=lambda r:r['negative_log_numerator'])[:3]])
   env=dict(KIND='C',CID=cid,HOUR=hour,SPLIT='primary_hour',RUN_LABEL='_repair1',EXTRA_START_POINTS=starts,MAX_NUMERATOR_MODES=8,RNG_RUN_OFFSET=25000000);exec(compile(integrate,'tail-proposal-repair','exec'),env)
   if env['state']['passed']:state['pending_registration'].append(dict(contest_id=cid,hour=hour,run_label='_repair1'))
   else:state['failed'].append(dict(contest_id=cid,hour=hour,reason='Repaired proposal still did not meet unchanged gates'))
  except Exception as ex:state['failed'].append(dict(contest_id=cid,hour=hour,reason=repr(ex),traceback=traceback.format_exc()));print('TAIL REPAIR ERROR',cid,hour,repr(ex),flush=True)
  save()
state['phase']='registering_after_original_batch_terminal';save()
for task in state['pending_registration']:
 try:
  exec(compile(verify,'repaired-tail-original-target','exec'),dict(KIND='C',CID=task['contest_id'],HOUR=task['hour'],SPLIT='primary_hour',RUN_LABEL=task['run_label']));state['finished'].append(dict(task,original_target_verified=True))
 except Exception as ex:state['failed'].append(dict(task,reason=repr(ex),traceback=traceback.format_exc()));print('TAIL REPAIR REGISTRATION ERROR',task,repr(ex),flush=True)
 save()
state['current']=None;state['status']='completed' if not state['failed'] else 'completed_with_uncertainty';save();print('TAIL REPAIR BATCH ENDED',len(state['finished']),len(state['failed']),flush=True)

```
