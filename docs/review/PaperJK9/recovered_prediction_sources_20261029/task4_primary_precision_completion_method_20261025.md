# Task 4：尾部平均精度补足后的两例普通积分与整体刷新

等待已知存活的尾部精度批次终止，按当时冻结的全部尾部结果对3509/3960做同训练后验整段独立积分，最高2,097,152预算；门槛不变。通过才核验原目标并登记，失败明确保留。最后刷新C全73分数和四模型比较，只有所有案例通过才标记主切分精度完成。该流程不声称完整日期切分或长期目标完成。

```python
import json,time,os,traceback,hashlib
from pathlib import Path
p=Path('docs/review/PaperJK9');status=p/'task4_primary_precision_completion_status_20261025.json';assert not status.exists();state=dict(status='waiting_for_tail_precision',driver_pid=os.getpid(),finished=[],failed=[],current=None)
def save():
 tmp=status.with_suffix('.json.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(status)
def code(f):return Path(f).read_text().partition(chr(10)+chr(96)*3+'python'+chr(10))[2].rpartition(chr(10)+chr(96)*3)[0]
save()
while json.loads((p/'task4_C_aggregate_tail_precision_status_20261025.json').read_text())['status']=='running':time.sleep(2)
tail=json.loads((p/'task4_C_aggregate_tail_precision_status_20261025.json').read_text());state['tail_precision_status']=tail['status'];state['tail_components']=tail.get('actual_tail_components');save();method=code(p/'task4_global_path_importance_method_20261025.md');verify=code(p/'task4_global_path_registration_method_20261025.md')
for cid in [3509,3960]:
 state['status']='integrating_residual_case';state['current']=cid;save()
 try:
  env=dict(KIND='C',CID=cid,SPLIT='primary_hour',RUN_LABEL='_post_tail_precision1',RNG_RUN_OFFSET=175000000,PRODUCTION_BUDGETS=[32768,65536,131072,262144,524288,1048576,2097152]);exec(compile(method,'same-target-whole-path-after-tail-precision','exec'),env)
  if env['state']['passed']:
   exec(compile(verify,'post-tail-original-target-parity','exec'),dict(ESTIMATE_STAGE=str(env['stage'])));state['finished'].append(dict(contest_id=cid,registered=True))
  else:state['failed'].append(dict(contest_id=cid,reason='Whole-case precision gates still not met',stage=str(env['stage'])))
 except Exception as ex:state['failed'].append(dict(contest_id=cid,reason=repr(ex),traceback=traceback.format_exc()));print('PRIMARY PRECISION ERROR',cid,repr(ex),flush=True)
 save()
state['status']='refreshing_all_73_C_scores';state['current']=None;save();exec(compile(code(p/'task4_calibrated_posterior_prediction_method_20261025.md'),'C-all73-after-precision','exec'),{'MODEL_KIND':'C'});sc=json.loads((p/'task4_C_calibrated_forecast_20261025.json').read_text());state['all_C_case_screens_passed']=all(r['mean_score_numerical_screen_passed'] for r in sc['records']);state['C_pending_cases']=[r['contest_id'] for r in sc['records'] if not r['mean_score_numerical_screen_passed']];state['status']='refreshing_gated_comparison';save();exec(compile(code(p/'task4_calibration_all_folds_method_20261025.md'),'all-models-comparison-after-precision','exec'),{});state['status']='completed_primary_precision' if state['all_C_case_screens_passed'] else 'completed_with_remaining_precision_uncertainty';save();print('PRIMARY PRECISION PIPELINE ENDED',state['all_C_case_screens_passed'],state['C_pending_cases'],flush=True)

```
