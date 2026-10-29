# Task 3：二元模态占用诊断的正确处理

全局MH四链的所有连续参数和派生量已通过rank/bulk/tail门槛，额外的二元模态指示量却被连续分位尾部ESS误判为0。原因可直接验证：其95%分位点为1，I(value<=1)恒为1，方差为0，不能用该连续量尾部诊断衡量模态占用混合。

对模态占用使用该0/1指示量及其补集自身的MCMC ESS和rank Rhat，门槛仍为ESS>=400、Rhat<=1.01；连续量门槛完全保留。首次失败诊断和状态另存，不重采样、不挑链、不删模态。这是诊断量类型的修正，不把不混合的链算作通过。

```python
import json,hashlib,math
from pathlib import Path
import numpy as np
from pipeline.empirical import estimate as e,nonlinear_reestimate as nd
p=Path('docs/review/PaperJK9');root=p/'prior_reconstruction_20261025/cost_floor_05/contest_8219/global_mh';a=json.loads((root/'diagnostics.json').read_text());assert not a['diagnostics']['passed'];proposal=json.loads((root/'proposal.json').read_text());pilot=np.concatenate([np.log(e.read_columns(Path(f))['sigma']) for f in proposal['source_chain_paths']]);cut=float(np.median(pilot));z=np.array([np.load(root/f'chain_{j}.npz')['z'] for j in range(1,5)]);indicator=(z[:,:,2]<=cut).astype(float);ess=float(nd.effective_sample_size(indicator));essc=float(nd.effective_sample_size(1-indicator));rhat=float(nd.rank_rhat(indicator));q=a['diagnostics']['quantities'].pop('pilot_sigma_side_indicator');mode=dict(rank_rhat=rhat,indicator_ess=ess,complement_ess=essc,occupation_fraction=float(indicator.mean()),pilot_log_sigma_threshold=cut,quantile_tail_ess_not_applicable=True,reason='For binary 0/1 values the .95 quantile is 1 and the <=1 indicator is constant, so continuous quantile-tail ESS is not a valid occupancy diagnostic',initial_continuous_tail_ess=q['tail_ess']);assert np.quantile(indicator,.95)==1 and nd.effective_sample_size((indicator<=1).astype(float))==0
continuous=all(np.isfinite(v['rank_rhat']) and v['rank_rhat']<=1.01 and min(v['bulk_ess'],v['tail_ess'])>=400 for v in a['diagnostics']['quantities'].values());passed=bool(continuous and rhat<=1.01 and min(ess,essc)>=400);assert passed;a['diagnostics']['mode_occupancy_diagnostics']=mode;a['diagnostics']['passed']=passed;a['diagnostics']['diagnostic_correction']='Only additional binary mode-indicator diagnostic corrected; all continuous parameter rank/bulk/tail gates unchanged';a['diagnostic_correction_source']='task3_global_cost_floor_discrete_diagnostic_20261025.md';(root/'diagnostics_discrete_aware.json').write_text(json.dumps(a,indent=2));assert not (root/'accepted.json').exists();(root/'accepted.json').write_text(json.dumps(a,indent=2));old=(root/'status.json').read_text();(root/'status_initial.json').write_text(old);state=json.loads(old);state['status']='completed_corrected_discrete_diagnostic';state['passed']=True;state['mode_occupancy']=mode;state['continuous_minimum_ess']=min(min(v['bulk_ess'],v['tail_ess']) for v in a['diagnostics']['quantities'].values());state['initial_diagnostics_preserved']=True;(root/'status.json').write_text(json.dumps(state,indent=2));print(json.dumps(dict(passed=passed,mode=mode,min_continuous_ess=state['continuous_minimum_ess']),indent=2))

```
