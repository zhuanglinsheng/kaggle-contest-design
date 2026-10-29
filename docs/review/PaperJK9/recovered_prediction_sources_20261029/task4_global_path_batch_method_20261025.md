# Task 4：C模型普通小时积分不足的固定批次

在完整73例评分后，按数值门槛预先选出没有稀有小时标记、但平均精度不足的案例。C13333试算已通过独立扩展预算和原Stan/整段Python核验；其他案例采用同方法，未通过即记录不确定性。已验证积分按哈希索引复用，不重复重建。此批次不改变后验、先验或预测模型。

```python
import json,os,traceback,hashlib
from pathlib import Path
p=Path('docs/review/PaperJK9');KIND='C';SPLIT='primary_hour';forecast=p/'task4_C_calibrated_forecast_20261025.json';f=json.loads(forecast.read_text());assert f['all_73_scored'];tasks=[r['contest_id'] for r in sorted(f['records'],key=lambda r:-r['mean_logscore_batch_influence_mcse']) if not r['mean_score_numerical_screen_passed'] and not r['unrepaired_rare_probability_hours']];status=p/'task4_C_primary_global_integration_batch_status_20261025.json';assert not status.exists();state=dict(status='running',driver_pid=os.getpid(),tasks=tasks,forecast_sha256=hashlib.sha256(forecast.read_bytes()).hexdigest(),finished=[],failed=[],current=None)
def save():
 tmp=status.with_suffix('.json.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(status)
def code(f):return Path(f).read_text().partition(chr(10)+chr(96)*3+'python'+chr(10))[2].rpartition(chr(10)+chr(96)*3)[0]
integrate=code(p/'task4_global_path_importance_method_20261025.md');verify=code(p/'task4_global_path_registration_method_20261025.md');save()
for cid in tasks:
 index=p/'task4_global_path_probability_index_20261025.json';ix=json.loads(index.read_text()) if index.exists() else dict(entries={});key=f'C:{cid}:primary_hour'
 if key in ix['entries']:state['finished'].append(dict(contest_id=cid,reused_existing_validated_integral=True));save();continue
 state['current']=cid;save()
 try:
  env=dict(KIND=KIND,CID=cid,SPLIT=SPLIT,RUN_LABEL='_batch1',RNG_RUN_OFFSET=30000000);exec(compile(integrate,'global-path-importance','exec'),env)
  if not env['state']['passed']:state['failed'].append(dict(contest_id=cid,reason='Independent integration gates not met'));save();continue
  stage=env['stage'];exec(compile(verify,'global-path-original-target','exec'),dict(ESTIMATE_STAGE=str(stage)));state['finished'].append(dict(contest_id=cid,estimate_stage=str(stage.resolve()),reused_existing_validated_integral=False))
 except Exception as ex:state['failed'].append(dict(contest_id=cid,reason=repr(ex),traceback=traceback.format_exc()));print('GLOBAL PATH BATCH ERROR',cid,repr(ex),flush=True)
 save()
state['current']=None;state['status']='completed' if not state['failed'] else 'completed_with_uncertainty';save();print('GLOBAL PATH BATCH ENDED',len(state['finished']),len(state['failed']),flush=True)

```
