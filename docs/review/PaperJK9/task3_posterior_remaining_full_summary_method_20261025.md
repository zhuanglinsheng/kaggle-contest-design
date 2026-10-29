# Task 3：其余全披露代表案例配对汇总

仅在128个预选样本均取得两层结果、索引逐一相符后汇总。保留原设计/所选设计/配对差；MC95区间是后验均值的Monte Carlo误差诊断，不是后验分布区间。数值半径是两层差异的诊断，不是严格连续误差界。单个样本正差不等于在其数值半径下均可确认质量保持。

```python
import json,csv,hashlib,math
from pathlib import Path
import numpy as np
from pipeline.empirical import nonlinear_reestimate as nd
p=Path('docs/review/PaperJK9');cid=globals().get('CID',23652);source=p/f'task3_posterior_remaining_full_{cid}_20261025.json';s=json.loads(source.read_text());assert s['status']=='completed' and s['all128_resolved'];rr=s['records'];assert len(rr)==len(s['selection'])==128 and [r['serial'] for r in rr]==list(range(128));assert all(r['selection']==s['selection'][r['serial']] for r in rr)
def stats(values):
 v=np.asarray(values);ess=float(nd.effective_sample_size(v.reshape(4,32)));ess=min(128.,max(1.,ess));se=float(v.std(ddof=1)/math.sqrt(ess));mean=float(v.mean());return dict(mean=mean,mcse=se,ess=ess,approximate_mc95=[mean-1.96*se,mean+1.96*se],posterior_draw_q025=float(np.quantile(v,.025)),posterior_draw_q975=float(np.quantile(v,.975)))
g=stats([r['nonlinear_gap'] for r in rr]);radius=float(np.mean([r['diagnostic_numerical_radius'] for r in rr]));out=dict(status='completed128_paired_draws',contest_id=cid,scope=s['scope'],draws=128,all_preselected_draws_retained=True,selected_policy=s['selected_policy'],observed_nonlinear=stats([r['observed_nonlinear_B'] for r in rr]),selected_nonlinear=stats([r['selected_nonlinear_B'] for r in rr]),nonlinear_gap=g,first_order_gap=stats([r['first_order_gap'] for r in rr]),mean_diagnostic_numerical_radius=radius,maximum_diagnostic_numerical_radius=max(r['diagnostic_numerical_radius'] for r in rr),positive_raw_gap_draws=sum(r['nonlinear_gap']>0 for r in rr),draws_positive_after_own_diagnostic_radius=sum(r['nonlinear_gap']>r['diagnostic_numerical_radius'] for r in rr),mc95_lower_minus_mean_diagnostic_radius=g['approximate_mc95'][0]-radius,strict_continuous_error_certificate=False,source=str(source.resolve()),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),posterior_model='First-order estimation posterior; no nonlinear reestimation',overall_task3_complete=False)
target=p/f'task3_posterior_remaining_full_summary_{cid}_20261025.json';assert not target.exists();target.write_text(json.dumps(out,indent=2));rows=[dict(serial=r['serial'],chain=r['selection']['chain'],within_chain_index=r['selection']['index'],observed_nonlinear_B=r['observed_nonlinear_B'],selected_nonlinear_B=r['selected_nonlinear_B'],nonlinear_gap=r['nonlinear_gap'],first_order_gap=r['first_order_gap'],diagnostic_numerical_radius=r['diagnostic_numerical_radius']) for r in rr]
with (p/f'task3_posterior_remaining_full_paired_{cid}_20261025.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
print('REMAINING FULL SUMMARY',cid,g,'mean_radius',radius,'MC95_lower_minus_radius',out['mc95_lower_minus_mean_diagnostic_radius'],flush=True)
```
