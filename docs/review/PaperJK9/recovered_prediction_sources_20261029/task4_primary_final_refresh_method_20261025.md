# Task 4：独立884核验、串行登记与完整评分刷新

单例3509/884独立扩展积分已通过，先核验其原Stan密度及独立Python概率；等待10小时扩展批次登记终止后才加入该条，避免并发写入。随后等待3例普通积分终止，再完整重算C73例并刷新四模型比较与五种PIT随机化。保留数值门槛：刷新完成不等于C全通过，更不等于长期目标完成；剩余案例在结果中明确列出。

```python
import json,hashlib,time,os,traceback
from pathlib import Path
p=Path('docs/review/PaperJK9');status=p/'task4_primary_final_refresh_status_20261025.json';assert not status.exists();state=dict(status='verifying_standalone_884',driver_pid=os.getpid(),finished=[],failed=[])
def digest(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def save():
 tmp=status.with_suffix('.json.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(status)
def code(f):return Path(f).read_text().partition(chr(10)+chr(96)*3+'python'+chr(10))[2].rpartition(chr(10)+chr(96)*3)[0]
save();method=code(p/'task4_tail_registration_all_folds_before_registry_lock_20261025.md');prefix=method.split(";index=p/'task4_tail_probability_index_20261025.json'")[0];assert prefix!=method;env=dict(KIND='C',CID=3509,HOUR=884,SPLIT='primary_hour',RUN_LABEL='_extended_tempered1')
try:
 exec(compile(prefix,'standalone-884-original-target-check','exec'),env);root=env['root'];st=env['st'];last=env['last'];rawpath=env['rawpath'];cut=env['cut'];pf=env['pf'];entry=dict(model='C',contest_id=3509,split='primary_hour',hour_index=884,joint_logscore=last['joint_logscore'],logscore_mcse=last['logscore_mcse'],integral_status=str((root/'status.json').resolve()),integral_status_sha256=digest(root/'status.json'),rare_mode_parity=str(pf.resolve()),rare_mode_parity_sha256=digest(pf),input_sha256=digest(rawpath),model_sha256=st['model_sha256'],training_cutoff=cut,training_prior=st['training_prior']);state['standalone_parity_passed']=True;state['pending_884_registry_entry']=entry;save()
 while json.loads((p/'task4_C_extended_tempered_status_20261025.json').read_text())['status']=='running':state['status']='waiting_for_extended_registry_writer';save();time.sleep(2)
 index=p/'task4_tail_probability_index_20261025.json';ix=json.loads(index.read_text());key='C:3509:primary_hour:884'
 if key in ix['entries']:assert ix['entries'][key]==entry
 else:
  ix['entries'][key]=entry;tmp=index.with_suffix('.json.tmp');tmp.write_text(json.dumps(ix,indent=2));tmp.replace(index)
 state['finished'].append('884_original_target_and_registry');state['status']='waiting_for_residual_global_integrations';save()
 while json.loads((p/'task4_C_primary_residual_global_batch2_status_20261025.json').read_text())['status']=='running':time.sleep(2)
 state['status']='refreshing_complete_C_primary_score';save();exec(compile(code(p/'task4_calibrated_posterior_prediction_method_20261025.md'),'C-current-validated-integral-score','exec'),{'MODEL_KIND':'C'});sc=json.loads((p/'task4_C_calibrated_forecast_20261025.json').read_text());state['complete_score_passed_cases']=sum(r['mean_score_numerical_screen_passed'] for r in sc['records']);state['complete_score_pending_cases']=[r['contest_id'] for r in sc['records'] if not r['mean_score_numerical_screen_passed']];state['complete_score_sha256']=digest(p/'task4_C_calibrated_forecast_20261025.json');state['status']='refreshing_comparison_with_numeric_gates';save();exec(compile(code(p/'task4_calibration_all_folds_method_20261025.md'),'four-model-score-calibration-refresh','exec'),{});state['finished'].append('C_full_score_and_gated_comparison');state['status']='completed_refresh_not_goal_completion';save();print('PRIMARY REFRESH FINISHED',state['complete_score_passed_cases'],state['complete_score_pending_cases'],flush=True)
except Exception as ex:state['status']='completed_with_error';state['failed'].append(dict(reason=repr(ex),traceback=traceback.format_exc()));save();print('PRIMARY REFRESH ERROR',repr(ex),flush=True)

```
