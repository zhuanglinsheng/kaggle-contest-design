# Task 4：联合所有小时积分的核验与登记

只有三个新预算全部逐小时和整例通过后，才核验最高分子权重点、训练分母高权重点及训练链参考点。所有点与原Stan训练密度、独立Python完整路径一致；与已有同目标独立尾部积分比较须在合并诊断误差内，不一致保留待查。通过后单独标记direct_all_hours，明确平均误差使用同样本协方差，不能混用旧独立尾部积分。

```python
import json,math,hashlib,fcntl
from pathlib import Path
import numpy as np
p=Path('docs/review/PaperJK9');stage=Path(globals()['ESTIMATE_STAGE'])
def digest(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def code(f):return f.read_text().partition(chr(10)+chr(96)*3+'python'+chr(10))[2].rpartition(chr(10)+chr(96)*3)[0]
st=json.loads((stage/'status.json').read_text());assert st['passed'] and st['direct_all_hours'];assert all(r['screen_passed'] and r['all_hour_precision_passed'] and r['consistent_with_previous'] for r in st['records'][-3:]);prod=stage/f"production_{st['records'][-1]['samples']}.npz";ix=json.loads((p/'task4_tail_probability_index_20261025.json').read_text());comparisons=[]
assert not any(k.startswith(f"{st['model']}:{st['contest_id']}:{st['split']}:") and t.get('precision_challenged',False) for k,t in ix['entries'].items()),'Repair challenged independent integrals before adopting direct estimator'
with np.load(prod) as z:
 lp=z['raw_joint_logscores'];se=z['hour_logscore_mcse'];assert (z['probability_weight_ess']>=1000).all() and (z['maximum_numerator_weights']<.01).all() and (se<=.02).all()
 for key,t in ix['entries'].items():
  if not key.startswith(f"{st['model']}:{st['contest_id']}:{st['split']}:") or t.get('precision_challenged',False):continue
  assert digest(t['integral_status'])==t['integral_status_sha256'];assert t['training_cutoff']==st['training_cutoff'] and t['input_sha256']==st['input_sha256'] and t['model_sha256']==st['model_sha256'] and t['training_prior']==st['training_prior'];j=t['hour_index']-st['training_cutoff']-1;difference=float(lp[j]-t['joint_logscore']);tolerance=5*math.hypot(float(se[j]),t['logscore_mcse'])+.001;comparisons.append(dict(hour=t['hour_index'],difference=difference,tolerance=tolerance,consistent=abs(difference)<=tolerance,old_integral_status=t['integral_status'],old_integral_status_sha256=t['integral_status_sha256']))
compat=dict(passed=all(r['consistent'] for r in comparisons),scope='Same posterior and hourly probability; both Monte Carlo diagnostics, not exact bounds',records=comparisons);cf=stage/'independent_tail_compatibility.json';cf.write_text(json.dumps(compat,indent=2));assert compat['passed'],'Independent tail integrals disagree; investigate before adoption'
source=code(p/'task4_global_path_registration_method_20261025.md');old='points=np.vstack([reference,points]);folder=';new="points=np.vstack([reference,points,prod['numerator_representative_points']]);folder=";assert old in source;source=source.replace(old,new).replace("'largest_training_importance_weight'","'training_or_hour_numerator_importance_point'");exec(compile(source,'all-hour-numerator-and-denominator-original-target-parity','exec'),dict(ESTIMATE_STAGE=str(stage)))
index=p/'task4_global_path_probability_index_20261025.json'
with index.with_suffix('.json.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);g=json.loads(index.read_text());key=f"{st['model']}:{st['contest_id']}:{st['split']}";entry=g['entries'][key];assert entry['estimate_status_sha256']==digest(stage/'status.json');entry.update(direct_all_hours=True,all_hour_precision_passed=True,independent_tail_compatibility=str(cf.resolve()),independent_tail_compatibility_sha256=digest(cf),mean_mcse_method='32 independent importance blocks over all hours, retaining common-normalizer and cross-hour covariance; no independent-tail substitution');tmp=index.with_suffix('.json.tmp');tmp.write_text(json.dumps(g,indent=2));tmp.replace(index)
print('DIRECT ALL-HOUR INTEGRAL REGISTERED',key,len(comparisons),'independent tail comparisons',flush=True)
```
