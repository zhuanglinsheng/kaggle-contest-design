# Task 3：后验配对比较的汇总方法

日期：2026-10-25。仅在四个输入批次全部完成后汇总，禁止删去失败抽样。原设计采用逐样本细层，困难抽样采用追加成功细层；所选全披露设计采用价值坐标最细层。均值 Monte Carlo 误差按四链影响量的 ESS 计算，抽样分位数与均值计算误差区间分开报告。

诊断数值半径：原设计为 max(3*最后两层差,1e-7)，所选设计为 3*最后两层差加边界/先验尾部预算。它不是严格连续模型误差证书。误差区间基于在原一阶估计后验中的 128 个预设联合样本，不是非线性模型重新估计。

[汇总](task3_posterior_summary_2551_20261025.json)、[逐抽样配对表](task3_posterior_paired_2551_20261025.csv)、[图](task3_posterior_quality_2551_20261025.png)。

```python
import csv,json,math,hashlib,os
from pathlib import Path
import numpy as np
from pipeline.empirical.nonlinear_reestimate import effective_sample_size
report=Path('docs/review/PaperJK9')
files=['task3_posterior_nonlinear_2551_20261025.json','task3_posterior_nonlinear_2551_followup_20261025.json','task3_posterior_nonlinear_2551_hard_draws_20261025.json','task3_posterior_observed_fine_2551_20261025.json']
raw,follow,hard,fine=[json.loads((report/f).read_text()) for f in files]
assert all(x['status']=='completed' for x in [raw,follow,hard,fine])
fm={r['serial']:r for r in follow['records']};finem={r['serial']:r for r in fine['records']};hardm={}
for r in hard['records']:
 if 'B' in r:hardm.setdefault(r['serial'],[]).append(r)
rows=[]
for r in raw['records']:
 serial=r['serial'];f=fm[serial];last=f['selected_full_trials'][-1];previous=f['selected_full_trials'][-2];assert 'B' in last and 'B' in previous
 selected_B=last['B'];selected_radius=3*abs(last['B']-previous['B'])+last['boundary_output_budget']+last['prior_tail_output_upper_bound']
 if serial in hardm:
  obs=hardm[serial][-1]['B'];prev=hardm[serial][-2]['B'] if len(hardm[serial])>=2 else next(t['B'] for t in reversed(f['observed_recovery']) if 'B' in t)
 else:
  trials=finem[serial]['trials'];assert 'B' in trials[-1],serial;obs=trials[-1]['B'];prev=r['observed_nonlinear_B']
 obs_radius=max(3*abs(obs-prev),1e-7)
 rows.append(dict(serial=serial,chain=r['selection']['chain'],within_chain_index=r['selection']['index'],observed_nonlinear_B=obs,selected_nonlinear_B=selected_B,nonlinear_gap=selected_B-obs,observed_first_order_B=r['observed_first_order_B'],selected_first_order_B=r['selected_first_order_B'],first_order_gap=r['selected_first_order_B']-r['observed_first_order_B'],observed_diagnostic_radius=obs_radius,selected_diagnostic_radius=selected_radius,diagnostic_gap_radius=obs_radius+selected_radius,nonlinear_minus_first_order_gap=(selected_B-obs)-(r['selected_first_order_B']-r['observed_first_order_B'])))
assert len(rows)==128
with (report/'task3_posterior_paired_2551_20261025.csv').open('w',newline='') as h:
 w=csv.DictWriter(h,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
def stats(key):
 v=np.array([r[key] for r in rows]);ess=float(effective_sample_size(v.reshape(4,-1)));mcse=float(v.std(ddof=1)/math.sqrt(ess));return dict(mean=float(v.mean()),mcse=mcse,ess=ess,approximate_mc95=[float(v.mean()-1.96*mcse),float(v.mean()+1.96*mcse)],draw_q025=float(np.quantile(v,.025)),draw_median=float(np.median(v)),draw_q975=float(np.quantile(v,.975)))
num=float(np.mean([r['diagnostic_gap_radius'] for r in rows]));g=stats('nonlinear_gap')
summary=dict(date='2026-10-25',scope='Paired nonlinear validation of the saved contest 2551 delta=2 full-disclosure design under the reconstructed first-order estimation posterior',input_sha256={f:hashlib.sha256((report/f).read_bytes()).hexdigest() for f in files},draws=128,chains=4,no_posterior_draws_dropped=True,observed_nonlinear=stats('observed_nonlinear_B'),selected_nonlinear=stats('selected_nonlinear_B'),nonlinear_gap=g,first_order_gap=stats('first_order_gap'),approximation_effect_on_gap=stats('nonlinear_minus_first_order_gap'),negative_nonlinear_gap_draws=sum(r['nonlinear_gap']<0 for r in rows),relative_posterior_mean_quality_change_pct=100*(stats('selected_nonlinear_B')['mean']/stats('observed_nonlinear_B')['mean']-1),mean_diagnostic_numerical_gap_radius=num,maximum_individual_numerical_gap_radius=max(r['diagnostic_gap_radius'] for r in rows),mc95_upper_plus_diagnostic_radius=g['approximate_mc95'][1]+num,strict_continuous_error_certificate=False,nonlinear_quality_preserved_in_this_checked_design=False,posterior_model='First-order estimation posterior; not nonlinear reestimated posterior',overall_task3_complete=False)
(report/'task3_posterior_summary_2551_20261025.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
os.environ.setdefault('MPLCONFIGDIR','/tmp/jk9_audit_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
x=np.array([r['first_order_gap'] for r in rows]);y=np.array([r['nonlinear_gap'] for r in rows]);fig,axs=plt.subplots(1,2,figsize=(10,4.3))
axs[0].scatter(x,y,s=18,alpha=.7,color='#245d8e');axs[0].axhline(0,color='.5',lw=.8);axs[0].axvline(0,color='.5',lw=.8);axs[0].plot([-.8,.3],[-.8,.3],ls='--',color='.6',lw=.8);axs[0].scatter([x.mean()],[y.mean()],marker='D',s=60,color='#b94e32',label='Posterior sample mean');axs[0].set_xlim(min(x)-.03,max(x)+.03);axs[0].set_ylim(min(y)-.04,.08);axs[0].set_xlabel('First-order quality gap');axs[0].set_ylabel('Nonlinear quality gap');axs[0].set_title('128 paired posterior draws');axs[0].legend(frameon=False,fontsize=8)
l=summary['first_order_gap'];axs[1].errorbar([0,1],[l['mean'],g['mean']],yerr=[1.96*l['mcse'],1.96*g['mcse']],fmt='o',capsize=5,color='#245d8e');axs[1].axhline(0,color='.5',lw=.8);axs[1].set_xticks([0,1],['First order','Nonlinear']);axs[1].set_ylabel('Selected minus observed expected output');axs[1].set_title('Means and 95% Monte Carlo intervals');axs[1].set_xlim(-.5,1.5)
fig.suptitle('Contest 2551: saved shorter-deadline, higher-prize design',fontsize=12);fig.tight_layout();fig.savefig(report/'task3_posterior_quality_2551_20261025.png',dpi=180,bbox_inches='tight');plt.close(fig)
print(json.dumps(summary,indent=2),flush=True)
```
