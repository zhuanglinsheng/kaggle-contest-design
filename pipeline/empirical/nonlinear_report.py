"""Produce paired-estimation reports only after all ten posterior audits pass."""
from __future__ import annotations
import csv,json,math
from pathlib import Path
import numpy as np
from scipy.special import softmax,ndtr
from pipeline.empirical import nonlinear_reestimate as n
from pipeline.empirical.precision_design import optimal_noise,analyze


def effort_comparison(cid,grid):
    """Compare policies at common states, not estimates of realized effort error."""
    data,_,_,_=n.load_baseline(cid);stage='nonlinear_parameters'
    weights=softmax(grid['fine']-grid['base'])
    _,spread,quantiles=n.weighted_summary(grid['z'],weights)
    center=np.array([q[1] for q in quantiles]);points=grid['z']
    index=int(np.argmin(np.sum(((points-center)/np.maximum(spread,1e-8))**2,axis=1)))
    point=points[index];ci,cj,sig,lam=np.exp(point[:4]);T=data['T_contest']
    wi,wj=data['theta']/(sig*sig*ci),data['theta']/(sig*sig*cj)
    audit=json.loads((n.OUT/f'posterior_grid_{cid}.json').read_text())
    nt=audit['fine_nt'];spacing=audit['fine_dy'];h=1/(sig*T*math.sqrt(lam))
    def profile(spacing,nt):
        policies,dy=n.Likelihood(data,spacing,nt).policy(point)
        values=[]
        for t in (0,.25,.5,.75,.9):
            for y in (-2,-1,0,1,2):
                position=y/dy+(policies.shape[1]-1)/2;j=int(position);f=position-j
                k=int(t*nt);tf=t*nt-k
                values.append((1-tf)*((1-f)*policies[k,j]+f*policies[k,j+1])+tf*((1-f)*policies[k+1,j]+f*policies[k+1,j+1]))
        return np.array(values)
    coarse=profile(spacing,nt)
    for level in range(3):
        spacing/=2;nt*=2;nonlinear=profile(spacing,nt)
        numerical_error=float(np.max(np.abs(nonlinear-coarse))/nonlinear.max())
        if numerical_error<.005:break
        coarse=nonlinear
    else:raise RuntimeError(f'{cid}/{stage}: effort profile grid convergence unresolved')
    rows=[];index=0
    for t in (0,.25,.5,.75,.9):
        for y in (-2,-1,0,1,2):
            leading=np.array([wi,wj])*math.exp(-y*y/(2*(1-t+h)))/math.sqrt(2*math.pi*(1-t+h))
            rows.append(dict(contest_id=cid,parameter_stage=stage,time_fraction=t,normalized_gap=y,
                nonlinear_i=float(nonlinear[index,0]),nonlinear_j=float(nonlinear[index,1]),
                leading_i=float(leading[0]),leading_j=float(leading[1]),
                absolute_error=float(np.max(np.abs(nonlinear[index]-leading)))))
            index+=1
    peak=float(nonlinear.max())
    summary=dict(contest_id=cid,parameter_stage=stage,w_i=float(wi),w_j=float(wj),terminal_normalized_variance=float(h),
        maximum_error_relative_to_checked_nonlinear_peak=max(r['absolute_error'] for r in rows)/peak,
        numerical_grid_error_relative_to_checked_peak=numerical_error,effort_check_dy=spacing,effort_check_nt=nt,
        point_definition='Joint posterior draw nearest log-parameter marginal medians (importance-corrected for nonlinear parameters); common normalized states, not a realized-path/output error estimate.')
    return rows,summary


def design_sensitivity(cid,grid):
    data,_,_,_=n.load_baseline(cid)
    v=n.physical_draws(grid['z'],data['theta']);p=softmax(grid['fine']-grid['base'])
    ci,cj,sigma,lam,mu=v[:,:5].T;T=data['T_contest'];q=sigma*sigma*T;mu2=mu*mu
    cost=2/(1/ci+1/cj);weight=data['theta']*T/(cost*math.sqrt(2*math.pi))
    # optimal_noise averages draws; incorporate posterior correction weights.
    x,inc,_=optimal_noise(q,mu2,sigma,weight*p*len(p))
    xx,inc2,_=optimal_noise(q,mu2,sigma,weight*p*len(p),1025)
    if not np.isclose(inc,inc2,rtol=1e-9,atol=1e-12) or not np.isclose(x,xx,rtol=1e-7,atol=1e-10):
        raise RuntimeError('Precision optimizer failed doubled-grid check')
    baseline_variance=q+sigma/np.sqrt(lam)
    noeff=np.sqrt(q)*np.exp(-mu2/(2*q))/math.sqrt(2*math.pi)+np.abs(mu)*(ndtr(np.abs(mu)/np.sqrt(q))-.5)
    baseline=np.sum(p*(noeff+weight*np.exp(-mu2/(2*baseline_variance))/np.sqrt(baseline_variance)))
    optimized=np.sum(p*noeff)+inc
    old=analyze(cid)
    return dict(contest_id=cid,baseline_optimal_precision=old['optimal_precision'],
        nonlinear_parameters_optimal_precision=math.inf if x==0 else x**-2,
        baseline_gain_pct=old['output_gain_pct'],nonlinear_parameters_gain_pct=100*(optimized/baseline-1),
        baseline_finite_probability=old['drawwise_finite_probability'],
        nonlinear_parameters_finite_probability=float(np.sum(p*(mu2>q))),
        objective='Original Table 7 steady-state leading-order calibration; parameters replaced, nonlinear design objective not evaluated.')


def write_presentation(metrics,wrows,design,effort_summaries):
    """Render checked numerical artifacts without repeating numerical solves."""
    if not len(wrows)==len(design)==len(effort_summaries)==metrics['n_contests']:
        raise RuntimeError('Presentation artifacts have inconsistent contest counts')
    error_min=min(float(a['maximum_error_relative_to_checked_nonlinear_peak']) for a in effort_summaries)
    error_max=max(float(a['maximum_error_relative_to_checked_nonlinear_peak']) for a in effort_summaries)
    old_gain=float(np.median([float(r['baseline_gain_pct']) for r in design]))
    new_gain=float(np.median([float(r['nonlinear_parameters_gain_pct']) for r in design]))
    metrics.update(minimum_reestimated_effort_approximation_error_relative_to_checked_peak=error_min,
                   baseline_selected_median_output_gain_pct=old_gain)
    n.save_json(n.OUT/'comparison_summary.json',metrics)
    lines=['非线性均衡重新估计：原 w 最高的 10 场比赛','',
        '结论：非线性重估显著降低了这10场的激励指标，但参数变化和努力函数差距仍有实质意义。检查结果不能支持“原估计与非线性估计差别很小”的表述。','',
        '保持原数据、10万美元奖金单位、先验、稳态初始化与观测方程一致，仅替换努力函数。m由耦合非线性HJB方程数值求解。选样依据为原后验中max(w_i,w_j)的中位数；指标在每个联合后验抽样上计算。','',
        'contest_id | 原 max(w_i,w_j) 中位数 | 非线性中位数 | 非线性95%区间']
    for r in wrows:lines.append(f"{r['contest_id']} | {float(r['baseline_median']):.3f} | {float(r['nonlinear_median']):.3f} | [{float(r['nonlinear_q025']):.3f}, {float(r['nonlinear_q975']):.3f}]")
    lines+=['',
        f"参数对照：60项原始参数比较中，{metrics['parameters_shifted_over_one_baseline_sd']}项均值变化超过原后验的一个标准差，{metrics['parameters_shifted_over_one_nonlinear_sd']}项超过新后验的一个标准差。{metrics['parameters_with_nonoverlapping_95_intervals']}项的新旧95%区间不重叠。",
        '这衡量模型敏感性；真实参数未知，不能将其解释为真实估计误差或正式显著性检验。完整结果见parameter_comparison.csv。','',
        f"努力函数对照：在每场的新后验代表点上，比较同一参数配置下的两种努力函数。最大绝对差距除以该配置在所检查状态中的非线性努力峰值，得到{100*error_min:.1f}%–{100*error_max:.1f}%。对应数值网格差距最大为{100*metrics['maximum_effort_profile_numerical_grid_error']:.2f}%。",
        '代表点为最接近校正后对数参数边际中位数的联合后验抽样。检查t/T=0,0.25,0.5,0.75,0.9与y/(sigma*sqrt(T))=-2,-1,0,1,2。这是共同状态下的策略差距，并非实现路径、真实努力或预期产出误差。完整结果见effort_approximation_summary.csv及effort_approximation_profiles.csv。','',
        f"Table 7参数敏感性：沿用原稳态一阶设计公式，只替换这10场的参数后验，充分披露建议由{metrics['baseline_full_disclosure']}场变为{metrics['parameter_sensitivity_full_disclosure']}场。6322和8076由有限精度变为充分披露。预期产出改进的中位数从{old_gain:.3f}%变为{new_gain:.3f}%，均指这10场。",
        '6322的改进从3.431%变为1.228%；8076从4.157%变为0.149%。这项分析尚未计算非线性均衡下的设计目标，也没有重估其余63场。原73场主结果和Table 7均保留。完整结果见precision_parameter_sensitivity.csv。','',
        f"计算验证：十场均通过四链诊断、后验网格加密与边界扩展检查。最大rank R-hat={metrics['maximum_rhat']:.5f}；最小采样bulk/tail ESS={metrics['minimum_sampling_ess']:.1f}；最小校正后均值ESS={metrics['minimum_importance_mean_ess']:.1f}。进一步加密造成的参数均值变化最大为{metrics['maximum_grid_shift_sd']:.3f}个后验标准差。十项独立方法检验通过。详细指标见posterior_audit_summary.csv与comparison_summary.json。",'',
        '论文含义：可以据此回应审稿人要求的非线性检查，但不能将结果写成原近似误差可忽略的稳健性验证。若要把Table 7解释为非线性模型的设计预测，需要进一步统一非线性估计与反事实计算。']
    (n.OUT/'comparison_report.txt').write_text('\n'.join(lines)+'\n')
    import os
    cache=n.ROOT/'.cache'/'matplotlib';cache.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(cache))
    os.environ.setdefault('XDG_CACHE_HOME',str(n.ROOT/'.cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5))
    y=np.arange(len(wrows));old=np.array([float(r['baseline_median']) for r in wrows]);new=np.array([float(r['nonlinear_median']) for r in wrows]);low=np.array([float(r['nonlinear_q025']) for r in wrows]);high=np.array([float(r['nonlinear_q975']) for r in wrows])
    ax.scatter(old,y,marker='s',color='#888888',label='Leading-order posterior median')
    ax.errorbar(new,y,xerr=np.array([new-low,high-new]),fmt='o',color='#1265a8',label='Nonlinear posterior median and 95% interval',capsize=3)
    from matplotlib.ticker import ScalarFormatter
    ax.set_xscale('log');ax.set_xticks([.1,.2,.5,1,2,5,10,20,50]);ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_yticks(y,[r['contest_id'] for r in wrows]);ax.invert_yaxis();ax.set_xlabel('max(w_i, w_j), log scale');ax.set_ylabel('Contest ID');ax.legend(loc='upper center',bbox_to_anchor=(.5,1.20),frameon=False);ax.grid(axis='x',alpha=.2);fig.tight_layout();fig.savefig(n.OUT/'w_comparison.png',dpi=180);plt.close(fig)


def main():
    n.OUT=n.ROOT/'results/estimation/nonlinear_top10/fast_v1';n.SOURCE=Path(n.__file__).with_name('nonlinear_fast_kernel.cpp')
    selected=json.loads((n.OUT/'selection.json').read_text())['selected'];ids=[a['contest_id'] for a in selected]
    for cid in ids:
        p=n.OUT/f'posterior_grid_{cid}.json'
        audit=json.loads(p.read_text()) if p.exists() else {}
        if audit.get('status')!='validated' or audit.get('validation_version')!=2:
            raise RuntimeError(f'{cid}: validated posterior required; report not published')
    n.summarize()
    with (n.OUT/'parameter_comparison.csv').open() as f:rows=list(csv.DictReader(f))
    wrows=[r for r in rows if r['parameter']=='max_w'];n.write_csv(n.OUT/'w_comparison.csv',wrows)
    design=[design_sensitivity(cid,np.load(n.OUT/f'posterior_grid_{cid}.npz')) for cid in ids]
    n.write_csv(n.OUT/'precision_parameter_sensitivity.csv',design)
    effort_rows=[];effort_summaries=[]
    for cid in ids:
        details,summary=effort_comparison(cid,np.load(n.OUT/f'posterior_grid_{cid}.npz'))
        effort_rows.extend(details);effort_summaries.append(summary)
    n.write_csv(n.OUT/'effort_approximation_profiles.csv',effort_rows)
    n.write_csv(n.OUT/'effort_approximation_summary.csv',effort_summaries)
    audit=[json.loads((n.OUT/f'posterior_grid_{cid}.json').read_text()) for cid in ids]
    n.write_csv(n.OUT/'posterior_audit_summary.csv',[
        dict(contest_id=a['contest_id'],validation_version=a['validation_version'],
            maximum_rank_rhat=a['sampling_diagnostics']['maximum_rank_rhat'],
            minimum_bulk_ess=a['sampling_diagnostics']['minimum_bulk_ess'],
            minimum_tail_ess=a['sampling_diagnostics']['minimum_tail_ess'],
            importance_ess=a['importance_ess'],importance_mean_ess=a['minimum_importance_mean_ess'],
            maximum_parameter_grid_shift_sd=a['maximum_parameter_shift_in_fine_sd'],
            centered_logdensity_error_q95=a['centered_logdensity_error_q95'],
            centered_logdensity_error_maximum=a['centered_logdensity_error_maximum'],
            maximum_domain_logdensity_change=a['maximum_domain_logdensity_change']) for a in audit])
    baseline_audit=json.loads((n.OUT/'baseline_target_consistency.json').read_text())
    metrics=dict(n_contests=len(ids),maximum_archived_target_consistency_error=max(a['maximum_centered_density_difference'] for a in baseline_audit['contests']),maximum_rhat=max(a['sampling_diagnostics']['maximum_rank_rhat'] for a in audit),
        minimum_sampling_ess=min(min(a['sampling_diagnostics']['minimum_bulk_ess'],a['sampling_diagnostics']['minimum_tail_ess']) for a in audit),
        minimum_importance_ess=min(a['importance_ess'] for a in audit),
        minimum_importance_mean_ess=min(a['minimum_importance_mean_ess'] for a in audit),
        maximum_grid_shift_sd=max(a['maximum_parameter_shift_in_fine_sd'] for a in audit),
        maximum_grid_q95_logdensity_error=max(a['centered_logdensity_error_q95'] for a in audit),
        maximum_domain_logdensity_change=max(a['maximum_domain_logdensity_change'] for a in audit),
        maximum_reestimated_effort_approximation_error_relative_to_checked_peak=max(a['maximum_error_relative_to_checked_nonlinear_peak'] for a in effort_summaries if a['parameter_stage']=='nonlinear_parameters'),
        maximum_effort_profile_numerical_grid_error=max(a['numerical_grid_error_relative_to_checked_peak'] for a in effort_summaries),
        parameters_shifted_over_one_baseline_sd=sum(abs(float(r['shift_in_baseline_sd']))>1 for r in rows if r['parameter'] in n.NAMES),
        parameters_shifted_over_one_nonlinear_sd=sum(abs(float(r['shift_in_nonlinear_sd']))>1 for r in rows if r['parameter'] in n.NAMES),
        parameters_with_nonoverlapping_95_intervals=sum(r['intervals_95_overlap']=='False' for r in rows if r['parameter'] in n.NAMES),
        minimum_reestimated_w_median=min(float(r['nonlinear_median']) for r in wrows),
        maximum_reestimated_w_median=max(float(r['nonlinear_median']) for r in wrows),
        parameter_sensitivity_median_output_gain_pct=float(np.median([r['nonlinear_parameters_gain_pct'] for r in design])),
        baseline_full_disclosure=sum(math.isinf(r['baseline_optimal_precision']) for r in design),
        parameter_sensitivity_full_disclosure=sum(math.isinf(r['nonlinear_parameters_optimal_precision']) for r in design))
    n.save_json(n.OUT/'comparison_summary.json',metrics)
    write_presentation(metrics,wrows,design,effort_summaries)
    print(json.dumps(metrics,indent=2))

if __name__=='__main__':main()
