# Task 4：完全相同目标的跨标签积分复用

仅处理3509：两个切分实际截止均为672小时。逐项验证完整Stan训练输入、完整预测原始输入、先验、模型与可执行文件、已接受链和预测函数源码相等；这证明训练后验及全部被评分的概率目标相同。不能用于不同训练窗。保留原日期切分失败记录；仅将已核验主切分积分登记为相同目标的别名，不将其称为新的独立验证。

```python
import ast,json,hashlib,fcntl
from pathlib import Path
from pipeline.empirical import estimate as e,unified_prior_reestimate as u
p=Path('docs/review/PaperJK9');cid=3509;proof=p/'task4_3509_identical_fold_integral_proof_20261025.json';assert not proof.exists()
def digest(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def code(f):return f.read_text().partition(chr(10)+chr(96)*3+'python'+chr(10))[2].rpartition(chr(10)+chr(96)*3)[0]
day=p/f'prediction_reconstruction_20261025/C/complete_day/contest_{cid}/accepted.json';da=json.loads(day.read_text());primary=Path(da['reused_identical_training_target']);pa=json.loads(primary.read_text());assert digest(primary)==da['primary_accepted_sha256'];assert da['diagnostics']['passed'] and pa['diagnostics']['passed']
rawpath=next(Path('results/reestimation/lambda100_sd225/input_snapshots/main').glob(f'contest_{cid}_*.json'));raw=json.loads(rawpath.read_text());daydata=p/f'prediction_reconstruction_20261025/day_inputs/contest_{cid}.json';inputs=json.loads(daydata.read_text());data=e.stan_data(raw,'early');data.update(u.prior(data));assert data==inputs
folder=Path(pa['chain_paths'][0]).parent;dp=folder/'data.json'
if not dp.exists():dp=folder/'parity/data.json'
assert json.loads(dp.read_text())==inputs;assert pa['chain_paths']==da['chain_paths'] and pa['chain_sha256']==da['chain_sha256'];assert all(digest(f)==h for f,h in zip(pa['chain_paths'],pa['chain_sha256']))
for k in ['input_sha256','training_cutoff','model_sha256','executable_sha256','prior']:
 assert pa['configuration'][k]==da['configuration'][k]
assert digest(rawpath)==pa['configuration']['input_sha256'];assert data['N_Delta']==672
exe=p/'prediction_reconstruction_20261025/C/model_snapshot/free_kernel';assert digest(exe)==da['configuration']['executable_sha256'] and digest(exe.with_suffix('.stan'))==da['configuration']['model_sha256']
functions={}
for name in ['task4_calibrated_posterior_prediction_method_20261025.md','task4_complete_day_prediction_method_20261025.md']:
 tree=ast.parse(code(p/name));functions[name]={k:ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==k),include_attributes=False) for k in ['counts_for','predictions']}
assert len(set(json.dumps(x,sort_keys=True) for x in functions.values()))==1
result=dict(status='exact_same_predictive_targets_verified',contest_id=cid,cutoff=672,primary_accepted=str(primary),primary_accepted_sha256=digest(primary),day_accepted=str(day.resolve()),day_accepted_sha256=digest(day),raw_input_sha256=digest(rawpath),day_data_sha256=digest(daydata),primary_training_data_sha256=digest(dp),same_full_training_data=True,same_full_prediction_inputs=True,same_prediction_functions=True,same_priors_model_executable_and_chain_files=True,scope='Memoized identical integrals; not an independent fold replication or a reuse across different training cutoffs.')
proof.write_text(json.dumps(result,indent=2));proofhash=digest(proof)
for name,prefix in [('task4_tail_probability_index_20261025.json',f'C:{cid}:primary_hour:'),('task4_global_path_probability_index_20261025.json',f'C:{cid}:primary_hour')]:
 index=p/name
 with index.with_suffix('.json.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);ix=json.loads(index.read_text());sources=[(k,v) for k,v in ix['entries'].items() if k.startswith(prefix)];assert sources
  for key,v in sources:
   assert not v.get('precision_challenged',False)
   for a,b in [('integral_status','integral_status_sha256'),('rare_mode_parity','rare_mode_parity_sha256'),('estimate_status','estimate_status_sha256'),('production','production_sha256'),('original_target_parity','original_target_parity_sha256')]:
    if a in v:assert digest(v[a])==v[b]
   assert v['training_cutoff']==672 and v['input_sha256']==digest(rawpath) and v['training_prior']==da['configuration']['prior'] and v['model_sha256']==da['configuration']['model_sha256'];newkey=key.replace(':primary_hour',':complete_day');old=ix['entries'].get(newkey)
   if old:ix.setdefault('history',{}).setdefault(newkey,[]).append(old)
   ix['entries'][newkey]=dict(v,split='complete_day',identical_target_alias_of=key,identical_target_proof=str(proof.resolve()),identical_target_proof_sha256=proofhash)
  tmp=index.with_suffix('.json.tmp');tmp.write_text(json.dumps(ix,indent=2));tmp.replace(index);print('IDENTICAL TARGET ALIASES',name,len(sources),flush=True)
```
