# Task 4：完整73例评分之后的尾部积分批次

仅按提前固定的数值门槛识别小时，不按预测优劣删案例。批次保存评分来源哈希、完整任务表、逐项积分/原目标核验与失败原因；先处理平均计算误差较大案例。每小时积分与核验有各自完整证据。重要性失败保留为未决，不通过放宽门槛消除。完成后重算全部73案例分数，才能检查总体是否可报告。

```python
import json,hashlib,time,os,traceback
from pathlib import Path
p=Path('docs/review/PaperJK9');KIND=globals().get('KIND','D');SPLIT=globals().get('SPLIT','primary_hour');suffix='' if SPLIT=='primary_hour' else '_complete_day';src=p/f'task4_{KIND}{suffix}_calibrated_forecast_20261025.json';forecast=json.loads(src.read_text());assert forecast['all_73_scored'];tasks=[dict(contest_id=r['contest_id'],hour=h,case_mcse=r['mean_logscore_batch_influence_mcse']) for r in forecast['records'] for h in r['unrepaired_rare_probability_hours']];tasks.sort(key=lambda r:(-r['case_mcse'],r['contest_id'],r['hour']));status=p/f'task4_{KIND}_{SPLIT}_tail_batch_status_20261025.json';assert not status.exists();state=dict(status='running',driver_pid=os.getpid(),model=KIND,split=SPLIT,forecast_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),tasks=tasks,finished=[],failed=[],current=None)
def save():
 tmp=status.with_suffix('.json.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(status)
def code(f):return Path(f).read_text().partition(chr(10)+chr(96)*3+'python'+chr(10))[2].rpartition(chr(10)+chr(96)*3)[0]
integrate=code(p/'task4_tail_integration_all_folds_method_20261025.md');verify=code(p/'task4_tail_registration_all_folds_method_20261025.md');save()
for task in tasks:
 cid=task['contest_id'];hour=task['hour'];state['current']=task;save();env=dict(KIND=KIND,CID=cid,HOUR=hour,SPLIT=SPLIT)
 try:
  exec(compile(integrate,'frozen-target-tail-integration','exec'),env)
  if not env['state']['passed']:state['failed'].append(dict(task,reason='Importance integration gates not met'));save();continue
  exec(compile(verify,'original-target-tail-verification','exec'),dict(KIND=KIND,CID=cid,HOUR=hour,SPLIT=SPLIT));state['finished'].append(task)
 except Exception as ex:
  state['failed'].append(dict(task,reason=repr(ex),traceback=traceback.format_exc()));print('TAIL BATCH ERROR',cid,hour,repr(ex),flush=True)
 save()
state['current']=None;state['status']='completed' if not state['failed'] else 'completed_with_uncertainty';save();print('TAIL BATCH ENDED',KIND,SPLIT,len(state['finished']),len(state['failed']),flush=True)

```
