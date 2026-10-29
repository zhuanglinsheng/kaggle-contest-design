# Task 4：3509平均分数的独立大预算接续

上一整段积分至2,097,152才首次通过全案例门槛，因此整阶段尚未满足连续两个独立预算要求。新阶段只有在训练目标、冻结提议SHA及全部尾部条目完全一致时，才能以该已通过单预算为参考，增加独立4,194,304/8,388,608预算；新预算通过且一致才登记。没有把上阶段整体标成通过，也没有重复已失败的小预算。

```python
import json,traceback,os
from pathlib import Path
p=Path('docs/review/PaperJK9');status=p/'task4_C3509_whole_case_extension_status_20261025.json';assert not status.exists();state=dict(status='running',driver_pid=os.getpid(),finished=[],failed=[])
def save():status.write_text(json.dumps(state,indent=2))
def code(f):return Path(f).read_text().partition(chr(10)+chr(96)*3+'python'+chr(10))[2].rpartition(chr(10)+chr(96)*3)[0]
save()
try:
 env=dict(KIND='C',CID=3509,SPLIT='primary_hour',RUN_LABEL='_post_tail_precision2',RNG_RUN_OFFSET=235000000,PRODUCTION_BUDGETS=[4194304,8388608],PREVIOUS_STAGE=str(p/'global_path_importance_20261025/C_primary_hour_3509_post_tail_precision1'));exec(compile(code(p/'task4_global_path_importance_method_20261025.md'),'C3509-extra-independent-budget','exec'),env)
 if env['state']['passed']:exec(compile(code(p/'task4_global_path_registration_method_20261025.md'),'C3509-final-whole-case-parity','exec'),dict(ESTIMATE_STAGE=str(env['stage'])));state['finished'].append('whole_case_integral_registered');state['status']='completed'
 else:state['status']='completed_with_uncertainty';state['failed'].append('Independent full-case precision still not established')
except Exception as ex:state['status']='completed_with_error';state['failed'].append(dict(reason=repr(ex),traceback=traceback.format_exc()));print('C3509 EXTENSION ERROR',repr(ex),flush=True)
save()

```
