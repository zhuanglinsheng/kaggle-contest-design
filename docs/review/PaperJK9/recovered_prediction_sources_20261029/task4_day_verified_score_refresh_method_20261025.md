# Task 4：已登记联合结果的最终全样本刷新

独立争议已修复，43项交叉对照及原Stan/独立Python核验已通过，联合估计已正式登记。本阶段不重复生产积分，只完整评分73例并刷新八项成对比较与固定500,000次bootstrap。保持各项门槛与全部案例，不自动修改论文或标记整个长期目标完成。

```python
import json,hashlib,os,traceback
from pathlib import Path
p=Path('docs/review/PaperJK9');status=p/'task4_C_day_verified_score_refresh_status_20261025.json';assert not status.exists();state=dict(status='checking_registered_joint_source',driver_pid=os.getpid(),finished=[],failed=[],current=None)
def save():
 tmp=status.with_suffix('.json.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(status)
def code(f):return f.read_text().partition(chr(10)+chr(96)*3+'python'+chr(10))[2].rpartition(chr(10)+chr(96)*3)[0]
def digest(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
save()
try:
 g=json.loads((p/'task4_global_path_probability_index_20261025.json').read_text())['entries']['C:3960:complete_day'];assert g['direct_all_hours'];assert digest(g['estimate_status'])==g['estimate_status_sha256'] and digest(g['production'])==g['production_sha256'];assert json.loads(Path(g['original_target_parity']).read_text())['passed'] and json.loads(Path(g['independent_tail_compatibility']).read_text())['passed'];state['status']='scoring_all73';save();exec(compile(code(p/'task4_complete_day_prediction_method_20261025.md'),'C-all73-final-day-score-after-verified-joint','exec'),dict(MODEL_KIND='C'));sc=json.loads((p/'task4_C_complete_day_calibrated_forecast_20261025.json').read_text());state['pending_cases']=[r['contest_id'] for r in sc['records'] if not r['mean_score_numerical_screen_passed']];assert len(sc['records'])==73 and not state['pending_cases'];state['finished'].append('all73_day_scores_passed');state['status']='refreshing_comparisons';save();exec(compile(code(p/'task4_calibration_all_folds_method_20261025.md'),'all-fold-score-and-calibration-final','exec'),{});exec(compile(code(p/'task4_cluster_bootstrap_precision_method_20261025.md'),'all-fold500k-bootstrap-final','exec'),{});cmp=json.loads((p/'task4_calibration_comparison_20261025.json').read_text());assert len(cmp['comparisons'])==8 and all(r['ranking_reportable'] for r in cmp['comparisons']);state['finished'].append('all8_comparisons_reportable');state['status']='completed';save();print('ALL FOUR MODELS BOTH FOLDS ALL73 VERIFIED',flush=True)
except Exception as ex:
 state['failed'].append(dict(reason=repr(ex),traceback=traceback.format_exc()));state['status']='completed_with_uncertainty';save();print('FINAL DAY SCORE NEEDS REVIEW',repr(ex),flush=True)
```
