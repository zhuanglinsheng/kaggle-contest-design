# Task 4：D完整日期切分评分与精度核验

等待已经启动的余19例接续完成，经统一只读入口合并原53例、同日期8219补充与余19例。73例齐备才评分，不能用较长主窗口后验替代。先实际完整评分，再按原有数值门槛处理被标记稀有小时；无未修复稀有小时但平均精度不足的案例用同目标独立整段积分。通过原目标核验才登记，剩余失败明确列出；不把脚本终止当成所有案例通过。

```python
import json,time,os,hashlib,traceback
from pathlib import Path
p=Path('docs/review/PaperJK9');status=p/'task4_D_day_prediction_pipeline_status_20261025.json';assert not status.exists();state=dict(status='waiting_for_all73_day_posteriors',driver_pid=os.getpid(),finished=[],failed=[],current=None)
def save():
 tmp=status.with_suffix('.json.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(status)
def code(f):return Path(f).read_text().partition(chr(10)+chr(96)*3+'python'+chr(10))[2].rpartition(chr(10)+chr(96)*3)[0]
helper={};exec(code(p/'task4_day_acceptance_merge_method_20261025.md'),helper);save()
while True:
 entries=helper['accepted_day_entries'](p,'D');state['accepted_day_posteriors']=len(entries);save()
 if len(entries)==73:break
 resume=json.loads((p/'task4_D_complete_day_resume_status_20261025.json').read_text())
 if resume['status']!='running':state['status']='incomplete_day_training_needs_attention';save();raise SystemExit('D day posterior set incomplete')
 time.sleep(2)
state['status']='scoring_complete_day';save();score=code(p/'task4_complete_day_prediction_method_20261025.md');exec(compile(score,'D-all73-complete-day-predictions','exec'),{'MODEL_KIND':'D'});forecast=p/'task4_D_complete_day_calibrated_forecast_20261025.json';sc=json.loads(forecast.read_text());assert sc['all_73_scored']
if any(r['unrepaired_rare_probability_hours'] for r in sc['records']):
 state['status']='integrating_day_rare_hours';save();exec(compile(code(p/'task4_tail_batch_method_20261025.md'),'D-day-tail-integrals','exec'),{'KIND':'D','SPLIT':'complete_day'});state['status']='refreshing_after_day_tail_integrals';save();exec(compile(score,'D-day-score-after-rare-integrals','exec'),{'MODEL_KIND':'D'});sc=json.loads(forecast.read_text())
state['ordinary_precision_tasks']=[r['contest_id'] for r in sc['records'] if not r['mean_score_numerical_screen_passed'] and not r['unrepaired_rare_probability_hours']];save();method=code(p/'task4_global_path_importance_method_20261025.md');verify=code(p/'task4_global_path_registration_method_20261025.md')
for cid in state['ordinary_precision_tasks']:
 state['status']='integrating_day_ordinary_probabilities';state['current']=cid;save()
 try:
  env=dict(KIND='D',CID=cid,SPLIT='complete_day',RUN_LABEL='_day_pipeline1',RNG_RUN_OFFSET=195000000);exec(compile(method,'D-day-independent-whole-path-integral','exec'),env)
  if env['state']['passed']:exec(compile(verify,'D-day-original-target-parity','exec'),dict(ESTIMATE_STAGE=str(env['stage'])));state['finished'].append(dict(contest_id=cid,independent_joint_integral_registered=True))
  else:state['failed'].append(dict(contest_id=cid,reason='Independent whole-path integration gates not met',stage=str(env['stage'])))
 except Exception as ex:state['failed'].append(dict(contest_id=cid,reason=repr(ex),traceback=traceback.format_exc()));print('D DAY PREDICTION ERROR',cid,repr(ex),flush=True)
 save()
state['current']=None;state['status']='refreshing_final_day_scores';save();exec(compile(score,'D-day-final-current-integrals','exec'),{'MODEL_KIND':'D'});sc=json.loads(forecast.read_text());state['all_case_screens_passed']=all(r['mean_score_numerical_screen_passed'] for r in sc['records']);state['pending_cases']=[r['contest_id'] for r in sc['records'] if not r['mean_score_numerical_screen_passed']];state['forecast_sha256']=hashlib.sha256(forecast.read_bytes()).hexdigest();state['status']='completed_day_prediction' if state['all_case_screens_passed'] else 'completed_with_day_precision_uncertainty';save();print('D DAY PREDICTION PIPELINE ENDED',state['all_case_screens_passed'],state['pending_cases'],flush=True)

```
