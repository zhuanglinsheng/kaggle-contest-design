"""Summarize matched auxiliary fits using posterior medians, like recovery tests."""
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from pipeline.config import manifest_path
from pipeline.empirical import unified_prior_reestimate as u
from pipeline.empirical.unified_prior_auxiliary import BASE_OUT, assemble, input_directory


def report():
    from pipeline.validation.chapter5 import benchmark_row,fit_row
    from pipeline.empirical.unified_prior_outputs import blue_plot as save_plot
    from pipeline.empirical import counterfactual as cf,precision_design as design
    fits=assemble()
    if len(fits)!=251:raise RuntimeError('All auxiliary fits required')
    main=pd.read_csv(BASE_OUT/'estimation/posterior_summary_final.csv')
    if len(main)!=146:raise RuntimeError('All main fits required')
    baseline_full=main[main['mode']=='full'].set_index('contest_id')
    out=BASE_OUT/'auxiliary/reports';out.mkdir(parents=True,exist_ok=True)
    details=[];regressions=[];policyrows=[];scales=[]
    for scenario in ('pair','L1','L2'):
        work=fits[fits.scenario==scenario]
        directory=BASE_OUT/'auxiliary'/scenario/'accepted';directory.mkdir(parents=True,exist_ok=True)
        manifest=directory/'accepted_chain_sources.csv';sources=[]
        for row in work.itertuples():
            raw=json.loads((input_directory(scenario)/f'contest_{row.contest_id}.json').read_text())
            params=pd.Series({p:getattr(row,p+'_median') for p in u.NAMES})
            details.append(dict(scenario=scenario,contest_id=row.contest_id,**params.to_dict(),**benchmark_row(raw,params)))
            sources.append(dict(contest_id=row.contest_id,mode='full',accepted_source='unified_lambda_prior',
                                run_dir=manifest_path(Path(row.run_dir),manifest),input_sha256=row.input_sha256))
        pd.DataFrame(sources).to_csv(manifest,index=False)
        frame=pd.DataFrame([d for d in details if d['scenario']==scenario])
        for p in ('lambda','sigma'):
            frame['x']=np.log1p(frame[p]);frame['y']=np.log1p(frame[p+'_bench'])
            regression,fit=fit_row(frame,'x','y',scenario+'_'+p+'_benchmark');regressions.append(regression)
            if scenario=='pair':save_plot(frame,'x','y',fit,out/f'validate_{p}_pair.pdf',rf'$\log(1+\hat{{\{p}}})$',rf'$\log(1+\{p}^{{bench}})$',show_statistics=True,color="black")
        if scenario!='pair':
            cf.FIT_ROOT=directory;cf.DATA_DIR=input_directory(scenario);design.DATA_DIR=cf.DATA_DIR
            policies=[dict(design.analyze(int(cid)),scenario=scenario) for cid in work.contest_id];policyrows.extend(policies)
            basepol=pd.read_csv(BASE_OUT/'counterfactual/precision_policy_contests.csv').set_index('contest_id')
            from scipy.stats import spearmanr
            a=float(raw.get('score_scale_a',1))
            frame=frame.set_index('contest_id')
            scales.append(dict(scenario=scenario,scale_a=a,full_disclosure_policies=sum(math.isinf(p['optimal_precision']) for p in policies),
                lambda_rank_correlation=float(spearmanr(a*a*frame['lambda'],baseline_full.loc[frame.index,'lambda_median']).statistic),
                sigma_rank_correlation=float(spearmanr(frame['sigma']/a,baseline_full.loc[frame.index,'sigma_median']).statistic),
                median_output_gain_pct=float(np.median([p['output_gain_pct'] for p in policies])),
                classification_agreement=sum(math.isinf(p['optimal_precision'])==math.isinf(basepol.loc[p['contest_id'],'optimal_precision']) for p in policies),
                direction_counts={cat:sum(p['recommendation']==cat for p in policies) for cat in ('Increase precision','Decrease precision','Direction uncertain')}))
    pd.DataFrame(details).to_csv(out/'contest_diagnostics.csv',index=False)
    pd.DataFrame(regressions).to_csv(out/'regressions.csv',index=False)
    pd.DataFrame(policyrows).to_csv(out/'scale_precision_policies.csv',index=False)
    baseline_all=main.set_index(['contest_id','mode']);timing=[]
    for row in fits[fits.scenario.str.startswith('seed_')].itertuples():
        base=baseline_all.loc[(row.contest_id,row.mode)]
        for p in u.NAMES:
            newmean=getattr(row,p+'_median');oldmean=base[p+'_median']
            timing.append(dict(seed=int(row.scenario.split('_')[1]),contest_id=row.contest_id,mode=row.mode,parameter=p,
                baseline_median=oldmean,alternate_median=newmean,absolute_shift_in_baseline_sd=abs(newmean-oldmean)/base[p+'_sd'],
                relative_median_change=newmean/oldmean-1 if p!='mu_0' else np.nan,
                intervals_overlap=max(base[p+'_q025'],getattr(row,p+'_q025'))<=min(base[p+'_q975'],getattr(row,p+'_q975'))))
    timing=pd.DataFrame(timing);timing.to_csv(out/'time_imputation.csv',index=False)
    time_summaries=[]
    for (mode,p),work in timing.groupby(['mode','parameter']):
        time_summaries.append(dict(mode=mode,parameter=p,comparisons=len(work),intervals_overlap=int(work.intervals_overlap.sum()),
            max_absolute_sd_shift=float(work.absolute_shift_in_baseline_sd.max()),
            median_absolute_relative_change=float(work.relative_median_change.abs().median()) if p!='mu_0' else None))
    result=dict(complete=True,accepted_fits=251,robustness_fits=219,imputation_fits=32,
                baseline_sha256=u.digest(BASE_OUT/'estimation/posterior_summary_final.csv'),regressions=regressions,scales=scales,time_imputation=time_summaries,
                interpretation='Same numerical empirical priors in transformed score units; posterior medians for all primitive point summaries; original joint draws for counterfactuals. Original inputs and focal-pair selection preserved.')
    u.save(out/'report.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":report()
