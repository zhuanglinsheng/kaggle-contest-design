# Task 3：2551 质量缺口的模型内分解

日期：2026-10-25。为避免把奖金和期限共同变化的全部缺口归因于反馈精度，补算相同 128 抽样在“原奖金、原期限、完全披露”下的输出；同抽样分成原反馈切至完全披露的效应，以及全披露下奖金/期限变化的效应。另列一阶的相同分解。

这是模型内的条件设计比较，不是关于真实反馈因果效应的经验证据。初始分布逐样本固定；两项相加应等于原配对总差。

[分解记录](task3_quality_decomposition_2551_20261025.json)。

```python
import csv,json,math,hashlib
from pathlib import Path
import numpy as np
from pipeline.empirical.fixed_initial_design import Output
from pipeline.empirical.nonlinear_reestimate import effective_sample_size
report=Path('docs/review/PaperJK9');method=(report/'task3_full_disclosure_value_grid_method_20261025.md').read_text().split('```python\n',1)[1].split('```',1)[0];exec(compile(method,'<full output decomposition>','exec'))
raw=json.loads((report/'task3_posterior_nonlinear_2551_20261025.json').read_text());paired=list(csv.DictReader((report/'task3_posterior_paired_2551_20261025.csv').open()));out=report/'task3_quality_decomposition_2551_20261025.json'
if out.exists():raise RuntimeError('Existing decomposition: inspect before resume')
rows=[]
for r,row in zip(raw['records'],paired):
 z=r['point'];trials=[full_asymmetric_value_grid(r['observed_theta'],r['observed_T'],z['sigma'],z['c_i'],z['c_j'],r['S0'],z['mu_0'],N=N,cutoff=cut,quadrature=quad) for N,cut,quad in [(8001,26.,160),(16001,28.,320)]]
 full=trials[-1]['B'];draws={k:np.array([z[k]]) for k in ('c_i','c_j','sigma','lambda','mu_0')};first=Output(draws,r['observed_T'],96);full1=first.no_effort+r['observed_theta']*first.value(math.inf)
 rows.append(dict(serial=r['serial'],full_at_observed_prize_deadline=full,full_original_refinement_difference=abs(trials[1]['B']-trials[0]['B']),nonlinear_precision_switch_effect=full-float(row['observed_nonlinear_B']),nonlinear_prize_deadline_effect=float(row['selected_nonlinear_B'])-full,first_order_precision_switch_effect=full1-r['observed_first_order_B'],first_order_prize_deadline_effect=r['selected_first_order_B']-full1))
def stat(name):
 v=np.array([r[name] for r in rows]);ess=effective_sample_size(v.reshape(4,-1));return dict(mean=float(v.mean()),mcse=float(v.std(ddof=1)/math.sqrt(ess)),negative_draws=int((v<0).sum()))
result=dict(date='2026-10-25',scope='Model-based decomposition of the paired 2551 design gap; not causal empirical evidence',source_sha256=hashlib.sha256(method.encode()).hexdigest(),records=rows,summaries={key:stat(key) for key in ['full_at_observed_prize_deadline','nonlinear_precision_switch_effect','nonlinear_prize_deadline_effect','first_order_precision_switch_effect','first_order_prize_deadline_effect']},strict_error_certificate=False)
(report/'task3_quality_decomposition_2551_20261025.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result['summaries'],indent=2))
```
