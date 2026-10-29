# Task 4：完整源日期切分的 C/D 训练输入

对全部73例固定更早的完整源日期截止，沿用原likelihood字段、count_start_idx和完整构造期限，仅截短保留的公共输入与训练提交。每个初始联合读数的源提交在新的截止之前；扰动截止后公共输入以及全部私人路径不改变训练数据。两模型将使用逐字节相同的训练输入，不复用较长原小时切分的后验。

这是预备与时序核验，不是拟合完成。后续冷启动仅使用本窗口先验中心及分散初值；避免以更长训练窗口后验为初值影响有限样本比较。所有新输入保存在审查目录，原冻结输入保持不变。

[逐例输入清单及哈希](task4_complete_day_inputs_20261025.json)。

```python
import json,hashlib,copy
from pathlib import Path
from pipeline.empirical import estimate as e,unified_prior_reestimate as u
p=Path('docs/review/PaperJK9');folder=p/'prediction_reconstruction_20261025/day_inputs';assert not folder.exists(),'Inspect prepared date-fold inputs before replacement';folder.mkdir(parents=True);audit=json.loads((p/'task4_source_information_audit_20261025.json').read_text());assert audit['status']=='completed';records=[]
def digest(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def construct(raw,cut):
 d=e.stan_data(raw,'full');d['N_Delta']=cut;d['hat_y']=d['hat_y'][:cut]
 for k in ['i','j']:
  d['hat_t_'+k]=[t for t in d['hat_t_'+k] if t<=cut];d['N'+k]=len(d['hat_t_'+k])
 return d
for record in audit['records']:
 cid=record['contest_id'];cut=record['complete_source_day_cutoff_hour'];rawpath=next(Path('results/reestimation/lambda100_sd225/input_snapshots/main').glob(f'contest_{cid}_*.json'));raw=json.loads(rawpath.read_text());assert digest(rawpath)==record['input_sha256'];assert all(t['imputed_hours_from_window_start']<=cut for t in record['initial_source_trace']);d=construct(raw,cut);full=e.stan_data(raw,'full');assert d['T_contest']==full['T_contest'] and d['N_Delta']<raw['N_Delta'];assert d['hat_y']==full['hat_y'][:cut];assert d['count_start_idx']==full['count_start_idx']
 changed=copy.deepcopy(raw);changed['hat_y'][cut:]=[x+100 for x in changed['hat_y'][cut:]];changed['real_y']=[x-100 for x in raw['real_y']];assert construct(changed,cut)==d
 for k in ['i','j']:assert all(0<t<=cut and t>d['count_start_idx']-1 for t in d['hat_t_'+k])
 inputs=dict(d,**u.prior(d));target=folder/f'contest_{cid}.json';target.write_text(json.dumps(inputs,indent=2));records.append(dict(contest_id=cid,cutoff_hour=cut,primary_cutoff_hour=record['training_cutoff_hour'],N_delta=d['N_Delta'],full_deadline_days=d['T_contest'],training_count_i=d['Ni'],training_count_j=d['Nj'],source_input_sha256=digest(rawpath),prepared_input=str(target.resolve()),prepared_input_sha256=digest(target),future_public_and_all_private_perturbation_invariant=True,all_initial_sources_before_day_cutoff=True))
result=dict(status='completed',fold='complete_source_day',contests=len(records),records=records,data_builder_source_sha256=digest(Path(e.__file__)),prior_source_sha256=digest(Path(u.__file__)),source_audit_sha256=digest(p/'task4_source_information_audit_20261025.json'),scope='Same training likelihood conventions as primary, earlier full-source-day boundary, unchanged full constructed horizon; no fits or rankings implied',initialization_rule_for_future_fits='Use prior-centered dispersed starts from this training data only; avoid the longer primary-window posterior even for starting values');(p/'task4_complete_day_inputs_20261025.json').write_text(json.dumps(result,indent=2));print('Complete-day training inputs prepared and checked for',len(records),'contests')

```
