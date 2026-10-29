# Task 4：当前两种切分的全样本刷新

主窗等待3960新全路径积分完成并登记；日期窗等待两组既有第二轮积分终止。完整73例重新评分后，仅对没有未解决稀有小时但平均精度仍不足的案例增加独立全路径积分。所有失败保留，不通过删除案例或放宽门槛得到排名。统一汇总继续保留ranking_reportable门槛和来源SHA快照。

```python
import json,time,os,traceback
from pathlib import Path
p=Path('docs/review/PaperJK9');split=globals()['SPLIT'];assert split in ['primary_hour','complete_day'];status=p/f'task4_C_{split}_current_completion_status_20261025.json';assert not status.exists();state=dict(status='waiting_for_existing_integrations',split=split,driver_pid=os.getpid(),finished=[],failed=[],current=None)
def save():
 tmp=status.with_suffix('.json.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(status)
def code(f):return f.read_text().partition(chr(10)+chr(96)*3+'python'+chr(10))[2].rpartition(chr(10)+chr(96)*3)[0]
save()
if split=='primary_hour':
 while True:
  stage=p/'global_path_importance_20261025/C_primary_hour_3960_post_coverage2';s=json.loads((stage/'status.json').read_text())
  if s.get('status','').startswith('completed'):
   if not s.get('passed',False):break
   if 'C:3960:primary_hour' in json.loads((p/'task4_global_path_probability_index_20261025.json').read_text())['entries']:break
  time.sleep(2)
else:
 while any(json.loads((p/f'task4_C_day_tail_second_pass_{g}_status_20261025.json').read_text())['status']=='running' for g in ['3960','other']):time.sleep(2)
suffix='' if split=='primary_hour' else '_complete_day';forecast=p/f'task4_C{suffix}_calibrated_forecast_20261025.json';method='task4_calibrated_posterior_prediction_method_20261025.md' if split=='primary_hour' else 'task4_complete_day_prediction_method_20261025.md';score=code(p/method);state['status']='refreshing_all73';save();exec(compile(score,'C-current-all73-scores','exec'),dict(MODEL_KIND='C'));sc=json.loads(forecast.read_text());assert sc['all_73_scored'];pending=[r for r in sc['records'] if not r['mean_score_numerical_screen_passed']];state['initial_pending']=[dict(contest_id=r['contest_id'],rare_hours=r['unrepaired_rare_probability_hours'],mcse=r['mean_logscore_batch_influence_mcse']) for r in pending];save()
for r in pending:
 cid=r['contest_id']
 if r['unrepaired_rare_probability_hours']:continue
 state['status']='refining_remaining_ordinary_probability';state['current']=cid;save()
 try:
  env=dict(KIND='C',CID=cid,SPLIT=split,RUN_LABEL='_current_complete3',RNG_RUN_OFFSET=345000000,PRODUCTION_BUDGETS=[1048576,2097152,4194304]);exec(compile(code(p/'task4_global_path_importance_method_20261025.md'),'current-fold-independent-refinement','exec'),env)
  if not env['state']['passed']:raise RuntimeError('Whole-case precision still not established')
  # Existing accepted global entries are not overwritten by this registrar.
  key=f'C:{cid}:{split}';ix=json.loads((p/'task4_global_path_probability_index_20261025.json').read_text())
  if key in ix['entries']:raise RuntimeError('Existing accepted integral needs explicit history-preserving replacement; new estimate retained')
  exec(compile(code(p/'task4_global_path_registration_method_20261025.md'),'current-fold-original-target-registration','exec'),dict(ESTIMATE_STAGE=str(env['stage'])));state['finished'].append(dict(contest_id=cid,stage=str(env['stage'])))
 except Exception as ex:state['failed'].append(dict(contest_id=cid,reason=repr(ex),traceback=traceback.format_exc()));print('CURRENT FOLD ERROR',split,cid,repr(ex),flush=True)
 save()
state['status']='refreshing_final_all73';state['current']=None;save();exec(compile(score,'C-final-current-fold-scores','exec'),dict(MODEL_KIND='C'));sc=json.loads(forecast.read_text());state['pending_cases']=[r['contest_id'] for r in sc['records'] if not r['mean_score_numerical_screen_passed']];state['all73_numerical_passed']=not state['pending_cases'];state['status']='completed' if state['all73_numerical_passed'] else 'completed_with_uncertainty';save();print('CURRENT FOLD COMPLETE',split,state['all73_numerical_passed'],state['pending_cases'],flush=True)
```
