# Task 4：竞赛bootstrap分位数精度核验

原2,000次bootstrap中D−B日期区间下端接近零，新增C日期来源后共享随机数流位置改变也导致小幅波动。保留原结果，采用每切分独立固定种子、所有模型差共享竞赛抽样索引的500,000次成对bootstrap。10个50,000次批次用于诊断区间端点的模拟误差；不增加原始竞赛样本、不把小时当独立观察、不更改模型或评分。来源全文SHA快照保留；未过全部73例积分门槛的比较仍不可报告。

```python
import json,hashlib
from pathlib import Path
import numpy as np
p=Path('docs/review/PaperJK9');src=p/'task4_calibration_comparison_20261025.json';payload=src.read_bytes();sha=hashlib.sha256(payload).hexdigest();source=json.loads(payload);snapshot=p/'prediction_score_snapshots_20261025'/f'{sha}.json'
if not snapshot.exists():snapshot.write_bytes(payload)
assert hashlib.sha256(snapshot.read_bytes()).hexdigest()==sha
out=dict(status='completed',source=str(snapshot.resolve()),source_sha256=sha,replicates=500000,seed_base=20261025+1904,bootstrap_unit='Whole contest, paired across models; identical indices across contrasts within each fold',comparisons=[])
for fold,split in enumerate(['primary_hour','complete_day']):
 records={(r['model'],r['contest_id']):r for r in source['records'] if r['split']==split};cases=sorted(c for m,c in records if m=='A');assert len(cases)==73 and all((m,c) in records for c in cases for m in ['A','B','C','D']);contrasts=[('D','C'),('D','A'),('D','B'),('B','A')];diff=np.array([[records[a,c]['mean_joint_logscore']-records[b,c]['mean_joint_logscore'] for a,b in contrasts] for c in cases]);rng=np.random.default_rng(out['seed_base']+fold*1000000);samples=[]
 for block in range(50):
  index=rng.integers(0,73,size=(10000,73));samples.append(diff[index].mean(axis=1))
 draws=np.concatenate(samples);ci=np.quantile(draws,[.025,.975],axis=0);batchci=np.array([np.quantile(x,[.025,.975],axis=0) for x in np.array_split(draws,10)]);endpoint_mcse=batchci.std(axis=0,ddof=1)/np.sqrt(10)
 for j,(a,b) in enumerate(contrasts):
  passed=all(records[a,c]['score_numerical_screen_passed'] and records[b,c]['score_numerical_screen_passed'] for c in cases);d=diff[:,j];loomeans=(d.sum()-d)/72;row=dict(split=split,model_difference=a+'-'+b,matched_contests=73,ranking_reportable=passed,mean_joint_logscore_difference=float(d.mean()),contest_bootstrap_95_interval=ci[:,j].tolist(),bootstrap_endpoint_mcse_diagnostic=endpoint_mcse[:,j].tolist(),positive_contest_differences=int(np.sum(d>0)),leave_one_contest_out_mean_range=[float(loomeans.min()),float(loomeans.max())],scope='Bootstrap uncertainty across observed contests, distinct from predictive integration diagnostics; conditional on selected pairs and constructed clocks/deadlines');out['comparisons'].append(row);print(split,a+'-'+b,row['mean_joint_logscore_difference'],row['contest_bootstrap_95_interval'],'quantile_MCSE',row['bootstrap_endpoint_mcse_diagnostic'],'reportable',passed,flush=True)
(p/'task4_cluster_bootstrap_precision_20261025.json').write_text(json.dumps(out,indent=2))
```
