# Task 4：提议计算本身的数值异常修复

六项中断来自审查提议的数值问题或不完整前置缓存：某些极端Hessian尺度使自写Student密度与SciPy核对未过；失败在提议保存前触发，后续错误假定该proposal.json存在。两者不作为论文模型失败证据，也没有放宽验收阈值。

新数值提议的协方差条件数限制为1e6，只改变提议，精确目标密度不变；仍要求相同密度核对精度。缺少前置提议时，只选输入/训练窗口/模型/先验/冻结SHA全部一致的其他小时提议作为探索基础。每目标独立新目录与随机数，保留原异常及全部成功/失败记录。生产积分门槛和原Stan/Python核验保持不变，等待上游登记结束再串行登记。

```python
import json,time,os,traceback
from pathlib import Path
p=Path('docs/review/PaperJK9');upstream=p/'task4_C_primary_tempered_repair_status_20261025.json';up=json.loads(upstream.read_text());tasks=[r for r in up['failed'] if 'traceback' in r and any(t in r['reason'] for t in ['FileNotFoundError','AssertionError'])];status=p/'task4_C_proposal_numerics_repair_status_20261025.json';assert not status.exists();state=dict(status='running',driver_pid=os.getpid(),tasks=[dict(contest_id=r['contest_id'],hour=r['hour'],previous_reason=r['reason']) for r in tasks],finished=[],failed=[],pending_registration=[],current=None)
def save():
 tmp=status.with_suffix('.json.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(status)
def code(f):return Path(f).read_text().partition(chr(10)+chr(96)*3+'python'+chr(10))[2].rpartition(chr(10)+chr(96)*3)[0]
method=code(p/'task4_tempered_tail_proposal_method_20261025.md');verify=code(p/'task4_tail_registration_all_folds_method_20261025.md');save()
for task in tasks:
 cid=task['contest_id'];hour=task['hour'];state['current']=dict(contest_id=cid,hour=hour);save()
 try:
  env=dict(KIND='C',CID=cid,HOUR=hour,SPLIT='primary_hour',RUN_LABEL='_proposal_numeric_fix1',BASE_STAGE=str(p/'predictive_integration_20261025'/f'C_{cid}_hour_{hour}_repair1'),RNG_RUN_OFFSET=85000000);exec(compile(method,'conditioned-proposal-independent-integral','exec'),env)
  if env['state']['passed']:state['pending_registration'].append(dict(contest_id=cid,hour=hour))
  else:state['failed'].append(dict(contest_id=cid,hour=hour,reason='Independent production gates still not met'))
 except Exception as ex:state['failed'].append(dict(contest_id=cid,hour=hour,reason=repr(ex),traceback=traceback.format_exc()));print('PROPOSAL NUMERIC REPAIR ERROR',cid,hour,repr(ex),flush=True)
 save()
while json.loads(upstream.read_text())['status']=='running':state['phase']='waiting_for_upstream_registration';save();time.sleep(2)
for task in state['pending_registration']:
 try:
  key=f"C:{task['contest_id']}:primary_hour:{task['hour']}";ix=json.loads((p/'task4_tail_probability_index_20261025.json').read_text())['entries']
  if key not in ix:exec(compile(verify,'proposal-numeric-original-target-parity','exec'),dict(KIND='C',CID=task['contest_id'],HOUR=task['hour'],SPLIT='primary_hour',RUN_LABEL='_proposal_numeric_fix1'))
  state['finished'].append(dict(task,original_target_verified=True))
 except Exception as ex:state['failed'].append(dict(task,reason=repr(ex),traceback=traceback.format_exc()))
 save()
state['current']=None;state['status']='completed' if not state['failed'] else 'completed_with_uncertainty';save();print('PROPOSAL NUMERICS REPAIR ENDED',len(state['finished']),len(state['failed']),flush=True)

```
