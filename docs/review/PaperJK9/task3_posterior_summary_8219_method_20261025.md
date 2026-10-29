# Task 3：8219后验非线性配对结果汇总

128个预设抽样全部保留、原设计两层网格均完成后才汇总。零奖金输出解析计算；原设计每抽样数值半径为max(3*两层差,1e-7)。报告均值Monte Carlo区间与数值诊断半径，二者不等同于严格连续误差证书。

结果只适用于该保存设计和基准一阶估计后验；不推广为全部零奖金设计可行，也不覆盖新成本先验。

```python
import csv,json,math,hashlib,os
from pathlib import Path
import numpy as np
from pipeline.empirical.nonlinear_reestimate import effective_sample_size
p=Path('docs/review/PaperJK9');rawpath=p/'task3_posterior_nonlinear_8219_20261025.json';raw=json.loads(rawpath.read_text());assert raw['status']=='completed' and raw['all_128_resolved'] and len(raw['records'])==128;rows=[]
for r in raw['records']:
 assert 'nonlinear_gap' in r;tr=r['original_trials'];fine=next(x for x in reversed(tr) if x['mesh_level']==2 and 'B' in x);coarse=next(x for x in reversed(tr) if x['mesh_level']==1 and 'B' in x);assert abs(fine['B']-r['observed_nonlinear_B'])<1e-12;assert r['analytic_original_baseline']-r['observed_diagnostic_radius']<=fine['B']<=r['analytic_original_upper']+r['observed_diagnostic_radius'];rows.append(dict(serial=r['serial'],chain=r['selection']['chain'],within_chain_index=r['selection']['index'],observed_nonlinear_B=r['observed_nonlinear_B'],selected_nonlinear_B=r['selected_nonlinear_B'],nonlinear_gap=r['nonlinear_gap'],first_order_gap=r['first_order_gap'],observed_first_order_B=r['observed_first_order_B'],diagnostic_numerical_radius=r['observed_diagnostic_radius'],nonlinear_minus_first_order_gap=r['nonlinear_gap']-r['first_order_gap']))
def stats(key):
 v=np.asarray([r[key] for r in rows]);ess=float(effective_sample_size(v.reshape(4,32)));mcse=float(v.std(ddof=1)/math.sqrt(max(ess,1)));return dict(mean=float(v.mean()),mcse=mcse,ess=ess,approximate_mc95=[float(v.mean()-1.96*mcse),float(v.mean()+1.96*mcse)],draw_q025=float(np.quantile(v,.025)),draw_q975=float(np.quantile(v,.975)))
radius=float(np.mean([r['diagnostic_numerical_radius'] for r in rows]));gap=stats('nonlinear_gap');summary=dict(status='completed',contest_id=8219,draws=128,no_draws_dropped=True,observed_nonlinear=stats('observed_nonlinear_B'),selected_nonlinear=stats('selected_nonlinear_B'),nonlinear_gap=gap,first_order_gap=stats('first_order_gap'),approximation_effect_on_gap=stats('nonlinear_minus_first_order_gap'),mean_diagnostic_numerical_radius=radius,maximum_diagnostic_numerical_radius=max(r['diagnostic_numerical_radius'] for r in rows),positive_nonlinear_gap_draws=sum(r['nonlinear_gap']>0 for r in rows),mc95_lower_minus_diagnostic_radius=gap['approximate_mc95'][0]-radius,quality_surplus_supported_by_this_diagnostic=gap['approximate_mc95'][0]-radius>0,selected_policy=raw['selected_policy'],posterior_model='First-order estimation posterior; not nonlinear reestimated posterior',strict_continuous_error_certificate=False,input_sha256=hashlib.sha256(rawpath.read_bytes()).hexdigest());(p/'task3_posterior_summary_8219_20261025.json').write_text(json.dumps(summary,indent=2))
with (p/'task3_posterior_paired_8219_20261025.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
os.environ.setdefault('MPLCONFIGDIR','/tmp/jk9_audit_mpl');import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
fig,axs=plt.subplots(1,2,figsize=(10,4.3));xx=np.array([r['first_order_gap'] for r in rows]);yy=np.array([r['nonlinear_gap'] for r in rows]);axs[0].scatter(xx,yy,s=18,alpha=.7,color='#245d8e');lo=min(xx.min(),yy.min());hi=max(xx.max(),yy.max());axs[0].plot([lo,hi],[lo,hi],'--',color='.6',lw=.8);axs[0].axhline(0,color='.5',lw=.8);axs[0].axvline(0,color='.5',lw=.8);axs[0].set_xlabel('First-order quality gap');axs[0].set_ylabel('Nonlinear quality gap');axs[0].set_title('128 paired posterior draws');axs[1].errorbar([0,1],[summary['first_order_gap']['mean'],gap['mean']],yerr=[1.96*summary['first_order_gap']['mcse'],1.96*gap['mcse']],fmt='o',capsize=5,color='#245d8e');axs[1].axhline(0,color='.5',lw=.8);axs[1].set_xticks([0,1],['First order','Nonlinear']);axs[1].set_xlim(-.5,1.5);axs[1].set_ylabel('Selected minus observed expected output');axs[1].set_title('Means and 95% Monte Carlo intervals');fig.suptitle('Contest 8219: saved zero-prize design');fig.tight_layout();fig.savefig(p/'task3_posterior_quality_8219_20261025.png',dpi=180,bbox_inches='tight');plt.close(fig);print(json.dumps({k:v for k,v in summary.items() if k not in ['selected_policy']},indent=2))

```
