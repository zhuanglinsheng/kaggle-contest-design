"""Rebuild isolated Chapter 5 outputs only from the complete fresh posterior.

Posterior medians are primary, matching the prior-selection recovery checks.
Means and their stability diagnostics remain sensitivity results.
No original result or manuscript is edited.
"""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pipeline.config import MAIN_ESTIMATION_DIR, MAIN_VALIDATION_DIR, COUNTERFACTUAL_DIR, MAIN_DATA_DIR, resolve_manifest_path
from pipeline.empirical.unified_prior_reestimate import OUT, NAMES, digest, save, assemble


def blue_plot(frame,x,y,fit,path,xlabel,ylabel,show_statistics=False,*,color="blue"):
    from pipeline.validation.chapter5 import plt
    import statsmodels.api as sm
    work=frame[[x,y]].replace([np.inf,-np.inf],np.nan).dropna().sort_values(x)
    grid=np.linspace(float(work[x].min()),float(work[x].max()),200)
    predicted=fit.predict(sm.add_constant(pd.Series(grid,name=x)))
    plt.rcParams.update({'font.size':12})
    fig,ax=plt.subplots(figsize=(5.2,4.2))
    ax.scatter(work[x],work[y],s=20,color=color);ax.plot(grid,predicted,color=color,linestyle='--')
    ax.set_xlabel(xlabel);ax.set_ylabel(ylabel)
    if show_statistics:
        p='$p<0.001$' if fit.pvalues[x]<.001 else f'$p={fit.pvalues[x]:.3f}$'
        ax.text(.04,.96,f'Slope = {fit.params[x]:.3f}\n{p}\n$R^2={fit.rsquared:.3f}$',
                transform=ax.transAxes,va='top',ha='left',fontsize=11,color=color,
                bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=3))
    fig.tight_layout();fig.savefig(path,bbox_inches='tight');plt.close(fig)


def configure():
    from pipeline.validation import chapter5 as ch, chapter5_report as report, chapter5_influence as influence, chapter5_final_audit as audit
    from pipeline.empirical import counterfactual as cf, precision_design as pdsg
    fit=OUT/'estimation'; validation=OUT/'validation'; policy=OUT/'counterfactual'; assets=OUT/'paper_assets'
    for path in (validation, policy, assets):path.mkdir(parents=True,exist_ok=True)
    ch.FIT_PATH=fit/'posterior_summary_final.csv';ch.RESULTS_DIR=validation;ch.PAPER_DIR=assets
    from functools import partial
    ch.save_plot=partial(blue_plot,color="black")
    original_frame=ch.parameter_frame
    def median_frame(mode):
        frame=original_frame(mode)
        medians=pd.read_csv(ch.FIT_PATH).set_index(['contest_id','mode'])
        for name in NAMES:
            frame[name]=[medians.loc[(int(cid),mode),name+'_median'] for cid in frame.contest_id]
        return frame
    ch.original_mean_parameter_frame=original_frame
    ch.parameter_frame=median_frame
    report.MAIN_ESTIMATION_DIR=fit;report.MAIN_VALIDATION_DIR=validation
    influence.MAIN_VALIDATION_DIR=validation
    audit.MAIN_ESTIMATION_DIR=fit;audit.MAIN_VALIDATION_DIR=validation
    audit.parameter_frame=ch.parameter_frame
    cf.FIT_ROOT=fit;cf.FIT_DIR=fit/'full';cf.OUT_DIR=policy
    pdsg.FIT_ROOT=fit;pdsg.OUT_DIR=policy
    return ch,report,influence,audit,pdsg


def precise_appendix_colors():
    from pipeline.config import PAPER_DIR
    path=OUT/'validation/chapter5_paper_numbers.json'
    report=json.loads(path.read_text())
    def value(cell):
        return cell.replace(r'\textcolor{blue}','').strip().removesuffix(r'\\').strip().strip('{}')
    original={int(value(line.split('&')[0])):line.split('&') for line in (PAPER_DIR/'appendix_100k_rows.tex').read_text().splitlines() if '&' in line}
    revised=[]
    for line in report['appendix_rows'].splitlines():
        cells=line.split('&');cid=int(value(cells[0]));old=original[cid]
        output=[old[j].strip().removesuffix(r'\\').strip() if value(cell)==value(old[j]) else r'\textcolor{blue}{'+value(cell)+'}' for j,cell in enumerate(cells)]
        revised.append(' & '.join(output)+r' \\')
    report['appendix_rows']='\n'.join(revised)+'\n'
    save(path,report)


def mean_sensitivity(ch):
    original=ch.FIT_PATH
    frame=pd.read_csv(original)
    meddir=OUT/'validation/mean_sensitivity';meddir.mkdir(parents=True,exist_ok=True)
    medpath=meddir/'posterior_point_summaries.csv';frame.to_csv(medpath,index=False)
    saved_frame=ch.parameter_frame
    ch.FIT_PATH=medpath;ch.RESULTS_DIR=meddir;ch.PAPER_DIR=meddir;ch.parameter_frame=ch.original_mean_parameter_frame
    try:ch.main()
    finally:
        ch.FIT_PATH=original;ch.RESULTS_DIR=OUT/'validation';ch.PAPER_DIR=OUT/'paper_assets';ch.parameter_frame=saved_frame
    (meddir/'INTERPRETATION.txt').write_text('Posterior-mean sensitivity only. Inspect each fit\'s posterior_mean_stability diagnostics: long-tailed lambda means can have large Monte Carlo errors. Primary parameter summaries and regressions use posterior medians. These raw-mean regressions are retained for transparency and are not substituted for the validated median point convention.\n')


def compare():
    new=pd.read_csv(OUT/'estimation/posterior_summary_final.csv')
    old=pd.read_csv(MAIN_ESTIMATION_DIR/'posterior_summary_final.csv')
    merged=new.merge(old,on=['contest_id','mode'],suffixes=('_new','_old'),validate='one_to_one')
    rows=[]
    from pipeline.empirical import unified_prior_reestimate as runner
    for _,row in merged.iterrows():
        oldrun,_=runner.baseline(int(row.contest_id),row['mode'])
        oldchains=[runner.e.read_columns(oldrun/f'chain_{j}.csv') for j in range(1,5)]
        for p in NAMES:
            a,b=row[f'{p}_mean_old'],row[f'{p}_mean_new']
            oldmedian=float(np.median(np.concatenate([np.exp(c['log_r']) if p=='r' else c[p] for c in oldchains])))
            rows.append(dict(contest_id=int(row.contest_id),mode=row['mode'],parameter=p,old_mean=a,new_mean=b,
                old_median=oldmedian,relative_median_change=row[f'{p}_median']/oldmedian-1 if p!='mu_0' else np.nan,
                new_median=row[f'{p}_median'],new_q025=row[f'{p}_q025_new'],new_q975=row[f'{p}_q975_new'],
                old_q025=row[f'{p}_q025_old'],old_q975=row[f'{p}_q975_old'],
                relative_mean_change=b/a-1 if p!='mu_0' else np.nan,
                shift_in_old_sd=(b-a)/row[f'{p}_sd_old'],
                credible_intervals_overlap=max(row[f'{p}_q025_old'],row[f'{p}_q025_new'])<=min(row[f'{p}_q975_old'],row[f'{p}_q975_new'])))
    comparison=pd.DataFrame(rows);comparison.to_csv(OUT/'validation/parameter_comparison.csv',index=False)
    oldreg=pd.read_csv(MAIN_VALIDATION_DIR/'chapter5_regressions.csv').assign(version='original prior; posterior means')
    newreg=pd.read_csv(OUT/'validation/chapter5_regressions.csv').assign(version='new prior; posterior medians (primary)')
    medreg=pd.read_csv(OUT/'validation/mean_sensitivity/chapter5_regressions.csv').assign(version='new prior; raw posterior means (sensitivity; inspect mean MCSE)')
    pd.concat([oldreg,newreg,medreg],ignore_index=True).to_csv(OUT/'validation/regression_comparison.csv',index=False)
    oldpol=pd.read_csv(COUNTERFACTUAL_DIR/'precision_policy_contests.csv')
    newpol=pd.read_csv(OUT/'counterfactual/precision_policy_contests.csv')
    policies=newpol.merge(oldpol,on='contest_id',suffixes=('_new','_old'),validate='one_to_one')
    policies['classification_changed']=np.isinf(policies.optimal_precision_new)!=np.isinf(policies.optimal_precision_old)
    policies.to_csv(OUT/'counterfactual/policy_comparison.csv',index=False)
    return new,comparison,policies


def completion(new,comparison,policies):
    main=new[new['mode']=='full']
    rows=[]
    diagnostics=[]
    manifest=OUT/'estimation/accepted_chain_sources.csv'
    for entry in csv.DictReader(manifest.open()):
        run=resolve_manifest_path(entry['run_dir'],manifest)
        result=json.loads((OUT/'estimation'/entry['mode']/f"contest_{entry['contest_id']}"/'accepted.json').read_text())
        diagnostics.append(result['diagnostics'])
        for j in range(1,5):
            path=run/f'chain_{j}.csv';rows.append(dict(path=str(path.relative_to(OUT)),sha256=digest(path)))
    policy=json.loads((OUT/'counterfactual/precision_policy_summary.json').read_text())
    paper_numbers=json.loads((OUT/'validation/chapter5_paper_numbers.json').read_text())
    report=dict(fits=len(new),contests=len(main),prior={'log_lambda_mean':float(np.log(100)),'log_lambda_sd':2.25,
               'other_priors':'unchanged','selection':'Previous targeted nonlinear-generated simulation research; not an independent 73-contest truth test.'},
        point_convention='Posterior medians are primary, matching the prior-selection recovery experiments. Raw posterior means, their MCSEs, and corresponding regressions are separately labeled sensitivity results.',
        lambda_mean_method='Raw NUTS means with explicit Monte Carlo errors; earlier conditional quadrature experiments preserved, but some long-tailed means remain unstable and are not primary point estimates.',
        sampler='Four-chain NUTS, with same-target fixed-mixture Metropolis-Hastings for selected multimodal posteriors',
        sampler_counts={family:sum(d.get('sampler_family','nuts')==family for d in diagnostics) for family in sorted({d.get('sampler_family','nuts') for d in diagnostics})},
        maximum_rank_rhat=max(v['rank_rhat'] for d in diagnostics for v in d['quantities'].values()),
        minimum_bulk_ess=min(v['bulk_ess'] for d in diagnostics for v in d['quantities'].values()),
        minimum_tail_ess=min(v['tail_ess'] for d in diagnostics for v in d['quantities'].values()),
        maximum_lambda_relative_mean_mcse=max(d['quantities']['lambda']['relative_mean_mcse'] for d in diagnostics),
        lambda_mean_relative_mcse_over_three_percent=sum(d['quantities']['lambda']['relative_mean_mcse']>.03 for d in diagnostics),
        maximum_log_lambda_median_mcse=max(d['median_monte_carlo_diagnostics']['lambda']['approximate_log_median_mcse'] for d in diagnostics),
        divergences=int(new.divergences.sum()),saturated_transitions=sum(d['saturated_transitions'] or 0 for d in diagnostics),
        minimum_energy_bfmi=min(v for d in diagnostics for v in d['energy_bfmi']),
        policy=policy,changed_policy_contests=policies.loc[policies.classification_changed,'contest_id'].astype(int).tolist(),
        approximation_indices={key:paper_numbers[key] for key in ('median_player_posterior_median_w','players_with_posterior_median_w_below_one','contests_with_posterior_median_max_w_below_one')},
        parameters={p:dict(new_mean_cross_contest_median=float(main[f'{p}_mean'].median()),
                           new_median_cross_contest_median=float(main[f'{p}_median'].median()),
                           median_absolute_relative_median_change=float(comparison.loc[(comparison.parameter==p)&(comparison['mode']=='full'),'relative_median_change'].abs().median()) if p!='mu_0' else None,
                           median_absolute_relative_mean_change=float(comparison.loc[(comparison.parameter==p)&(comparison['mode']=='full'),'relative_mean_change'].abs().median()) if p!='mu_0' else None)
                    for p in NAMES},
        interpretation='Prior sensitivity and computation are checked here. Real-data truth is unobserved; parameter change is not approximation error or improved truth recovery. No automatic guarantee for a nonlinear equilibrium counterfactual.',
        chain_checksums=rows)
    save(OUT/'report.json',report)
    groups=policy['groups'];overall=groups[0]
    text=['统一 lambda 先验：主样本重估完成', '',f"完成{len(main)}场完整样本及{len(new)-len(main)}场早期样本。所有后验均由四条链重新采样；方法数量为{report['sampler_counts']}。",
          'log(lambda) ~ N(log100, 2.25^2)；其他先验、数据、奖金单位（10万美元）、似然和一阶努力函数保持既定设定。',
          f"最大rank Rhat={report['maximum_rank_rhat']:.6f}，最小bulk/tail ESS={report['minimum_bulk_ess']:.0f}/{report['minimum_tail_ess']:.0f}；发散{report['divergences']}，树深饱和{report['saturated_transitions']}。",
          f"lambda后验均值相对Monte Carlo标准误超过3%的拟合有{report['lambda_mean_relative_mcse_over_three_percent']}组；均值最大相对MCSE为{100*report['maximum_lambda_relative_mean_mcse']:.2f}%。",'',
          '主参数表及回归统一采用后验中位数，与此前模拟误差检验一致；原始均值及其回归作为单列敏感性结果，尾部不稳定的均值不作精确结论。此前条件积分实验保留，不冒充所有均值已稳定。反事实直接使用联合后验，不能用中位数或均值代替积分。',
          f"新Table7：全披露{policy['full_disclosure_policies']}场，有限精度{policy['finite_policies']}场；跨比赛预期产出改善中位数{overall['gain_median_pct']:.4f}%。",
          f"全披露/有限精度分类变化比赛：{report['changed_policy_contests']}。",'',
          f"玩家-比赛层面的w后验中位数，其跨样本中位数为{paper_numbers['median_player_posterior_median_w']:.3f}；低于1的有{paper_numbers['players_with_posterior_median_w_below_one']}/146对。它是描述性诊断，不是误差上界。",
          '新旧先验下的同口径中位估计变化（不是对真值的误差）：'+', '.join(f"{p}各场绝对相对变化的中位数{100*report['parameters'][p]['median_absolute_relative_median_change']:.1f}%" for p in ('lambda','sigma'))+'。','',
          '主要文件：estimation/posterior_summary_final.csv；validation/parameter_comparison.csv；validation/regression_comparison.csv；counterfactual/policy_comparison.csv。',
          '该结果只表示统一先验后的重估。真实比赛的参数真值未知，不能将新旧参数差异解释为真实误差，或由此断言一阶和非线性反事实误差已达标。',
          '原主估计、模拟实验和论文版本均保留。论文正式替换还应同步其稳健性及时间插补检验，不能混用旧先验结果。']
    auxiliary=OUT/'auxiliary/reports/report.json'
    if auxiliary.exists() and json.loads(auxiliary.read_text()).get('complete'):
        text[-1]='原主估计、模拟实验和论文版本均保留。219组稳健性和32组时间插补估计已同步完成；标蓝论文副本位于manuscript目录。'
    (OUT/'report.txt').write_text('\n'.join(text)+'\n')
    artifactpaths=[p for folder in ('validation','counterfactual','paper_assets') for p in (OUT/folder).rglob('*') if p.is_file() and p.suffix in ('.csv','.pdf','.json','.txt')]
    artifactpaths.extend(OUT/name for name in ('report.json','report.txt','target_parity.json','target_parity_all146.json','target_parity_joint_all146.json','median_protocol.json'))
    save(OUT/'completion.json',dict(complete=True,source_sha256=digest(__file__),report_sha256=digest(OUT/'report.json'),
                                  artifacts=[dict(path=str(p.relative_to(OUT)),sha256=digest(p)) for p in sorted(artifactpaths)]))
    print('\n'.join(text),flush=True)


def refresh_completion():
    completion(pd.read_csv(OUT/'estimation/posterior_summary_final.csv'),
               pd.read_csv(OUT/'validation/parameter_comparison.csv'),
               pd.read_csv(OUT/'counterfactual/policy_comparison.csv'))


def main():
    assemble()
    progress=json.loads((OUT/'progress.json').read_text())
    if not progress['complete']:
        raise RuntimeError('All 146 accepted fits required before derived outputs')
    ch,report,influence,audit,pdsg=configure()
    import subprocess
    from pipeline.config import ROOT
    marker=Path('/Users/linsheng/.codex/plugins/cache/openai-primary-runtime/pdf/26.1007.11041/skills/pdf/container_tools/mark_artifact_operation_started.mjs')
    marker_record=OUT/'pdf_authoring_started.json'
    if marker.exists() and not marker_record.exists():
        subprocess.run(['node',str(marker),'--operation-kind','create','--expected-output-count','1','--output-format','pdf'],check=True)
        save(marker_record,dict(operation='create',expected_output_count=1,output_format='pdf',marker=str(marker)))
    ch.main();report.main();precise_appendix_colors();influence.main();audit.main();pdsg.main()
    mean_sensitivity(ch)
    completion(*compare())


if __name__=='__main__':main()
