# Task 4：日期3960联合积分及完整样本收尾

等待既有逐小时及群体后续核验终止，以实际已接受的当前小时退火/群体提议覆盖不同区域；四区域试算已保留且未接受。新提议各数值分量直接保存，均为全支持Student密度，保留半数训练后验提议。各分子提议最多取最后16分量以减少重复训练分量，重新归一化混合权重；这是数值提议选择，不改变被积密度。

三个新预算全部通过后，核对原Stan/独立Python及已有独立尾部概率，整段登记协方差估计；再完整评分73例，刷新统一比较及固定500,000次bootstrap。失败必须明示，不能自动宣布任务完成或修改论文。

```python
import json,time,os,traceback,hashlib
from pathlib import Path
p=Path('docs/review/PaperJK9');status=p/'task4_C_day_joint_completion_status_20261025.json';assert not status.exists();state=dict(status='waiting_for_existing_tail_proposals',driver_pid=os.getpid(),current=None,finished=[],failed=[])
def save():
 tmp=status.with_suffix('.json.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(status)
def code(f):return f.read_text().partition(chr(10)+chr(96)*3+'python'+chr(10))[2].rpartition(chr(10)+chr(96)*3)[0]
save()
while True:
 a=json.loads((p/'task4_C_day_tail_second_pass_3960_status_20261025.json').read_text());b=json.loads((p/'task4_C_day3960_population_followup_status_20261025.json').read_text())
 if a['status']!='running' and b['status']!='watching_day_second_pass':break
 time.sleep(2)
try:
 idx=json.loads((p/'task4_tail_probability_index_20261025.json').read_text());basis=[];seen=set()
 for key,t in idx['entries'].items():
  if not key.startswith('C:3960:complete_day:') or t.get('precision_challenged',False):continue
  folder=Path(t['integral_status']).parent
  if not any(label in folder.name for label in ['_day_tempered2','_day_population3']):continue
  st=json.loads((folder/'status.json').read_text());assert st['passed'];bp=json.loads((folder/'proposal.json').read_text());unique=hashlib.sha256(json.dumps({k:bp[k] for k in ['means','scales','weights','df']},sort_keys=True).encode()).hexdigest()
  if unique not in seen:basis.append(str(folder));seen.add(unique)
 assert len(basis)>=7;state['basis']=basis;state['status']='joint_independent_production';save();env=dict(RUN_LABEL='_day_joint_direct2',RNG_RUN_OFFSET=385000000,BASIS_PATHS=basis,NUMERATOR_COMPONENTS_ONLY=True);exec(compile(code(p/'task4_joint_tail_path_importance_method_20261025.md'),'expanded-all-hour-frozen-proposal','exec'),env);state['joint_stage']=str(env['stage']);save()
 if not env['state']['passed']:raise RuntimeError('Expanded joint estimator has not passed all-hour and whole-score gates')
 state['status']='original_target_and_tail_crosschecks';save();exec(compile(code(p/'task4_joint_tail_path_registration_method_20261025.md'),'joint-all-hour-original-target-registration','exec'),dict(ESTIMATE_STAGE=str(env['stage'])));state['finished'].append('joint_all_hour_integral_registered');state['status']='refreshing_all73_day_scores';save();exec(compile(code(p/'task4_complete_day_prediction_method_20261025.md'),'final-C-all73-day-score','exec'),dict(MODEL_KIND='C'));score=json.loads((p/'task4_C_complete_day_calibrated_forecast_20261025.json').read_text());state['pending_cases']=[r['contest_id'] for r in score['records'] if not r['mean_score_numerical_screen_passed']];save();assert len(score['records'])==73 and not state['pending_cases'],'All73 day score gates not met'
 state['finished'].append('all73_day_score_gates_passed');state['status']='refreshing_paired_comparisons';save();exec(compile(code(p/'task4_calibration_all_folds_method_20261025.md'),'final-all-fold-matched-comparison','exec'),{});exec(compile(code(p/'task4_cluster_bootstrap_precision_method_20261025.md'),'final-fixed-cluster-bootstrap','exec'),{});state['finished'].append('all_fold_comparison_and_bootstrap_refreshed');state['status']='completed';save();print('DATE JOINT COMPLETION ALL73 VERIFIED',flush=True)
except Exception as ex:
 state['failed'].append(dict(reason=repr(ex),traceback=traceback.format_exc()));state['status']='completed_with_uncertainty';save();print('DATE JOINT COMPLETION NEEDS REVIEW',repr(ex),flush=True)
```
