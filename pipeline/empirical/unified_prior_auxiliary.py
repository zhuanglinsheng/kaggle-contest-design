"""Synchronize empirical robustness and imputation fits with the new prior.

Reuse immutable reconstructed inputs; never regenerate focal pairs or dates.
"""
import argparse
import concurrent.futures
import csv
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from pipeline.config import ROOT, MAIN_ESTIMATION_DIR, manifest_path
from pipeline.empirical import unified_prior_reestimate as u
from pipeline.empirical import unified_prior_batch as batch
from pipeline.empirical import unified_prior_moments as rb

BASE_OUT=u.OUT
BASELINE=u.baseline
STARTS=u.initializations
SCENARIOS=('pair','L1','L2','seed_20261008','seed_20261009')
TIME_IDS=(7878,4453,4986,3641,23652,37077,21669,3960)
CURRENT=None


def input_directory(scenario):
    if scenario.startswith('seed_'):
        return ROOT/'results/estimation/chapter5_score_audit'/scenario
    return ROOT/'results/robustness/chapter6/inputs'/scenario


def baseline(cid,mode):
    run,_=BASELINE(cid,mode)
    return run,input_directory(CURRENT)/f'contest_{cid}.json'


def starts(cid,mode,previous=None):
    ans=STARTS(cid,mode,previous)
    if previous is not None:return ans
    raw=json.loads((input_directory(CURRENT)/f'contest_{cid}.json').read_text())
    a=float(raw.get('score_scale_a',1))
    for initial in ans:
        if CURRENT in ('L1','L2'):
            initial['c_i']=max(.0101,initial['c_i']/a**2)
            initial['c_j']=max(.0101,initial['c_j']/a**2)
            initial['sigma']=max(.0101,initial['sigma']*a)
            initial['lambda']=max(.0101,initial['lambda']/a**2)
            initial['log_r']=max(math.log(.01)+.001,initial['log_r']-math.log(a))
            initial['mu_0']*=a
        elif CURRENT=='pair':initial['mu_0']=raw['hat_y'][0]
    return ans


def fit(task):
    global CURRENT
    scenario,cid,mode=task;CURRENT=scenario
    u.OUT=BASE_OUT/'auxiliary'/scenario
    u.baseline=baseline;u.initializations=starts
    result=batch.fit((cid,mode,False))
    return dict(scenario=scenario,contest_id=cid,mode=mode,**result)


def assemble():
    rows=[]
    for scenario in SCENARIOS:
        for p in sorted((BASE_OUT/'auxiliary'/scenario/'estimation').glob('*/contest_*/accepted.json')):
            result=json.loads(p.read_text());row=result['summary'].copy()
            if not result['diagnostics']['passed']:raise RuntimeError('Rejected auxiliary fit')
            for path,h in zip(result['chain_paths'],result['chain_sha256']):
                if u.digest(path)!=h:raise RuntimeError('Changed auxiliary chain')
            rows.append(dict(scenario=scenario,**row,run_dir=str(Path(result['chain_paths'][0]).parent),
                             input_sha256=result['configuration']['original_input_sha256']))
    if rows:
        path=BASE_OUT/'auxiliary/posterior_summary.csv';path.parent.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(rows).to_csv(path,index=False)
    u.save(BASE_OUT/'auxiliary/progress.json',dict(accepted=len(rows),required=251,complete=len(rows)==251))
    print(f'AUXILIARY ACCEPTED {len(rows)}/251',flush=True)
    return pd.DataFrame(rows)


def report():
    from pipeline.validation.chapter5 import benchmark_row,fit_row,save_plot
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
            params=pd.Series({p:getattr(row,p+'_mean') for p in u.NAMES})
            details.append(dict(scenario=scenario,contest_id=row.contest_id,**params.to_dict(),**benchmark_row(raw,params)))
            sources.append(dict(contest_id=row.contest_id,mode='full',accepted_source='unified_lambda_prior',
                                run_dir=manifest_path(Path(row.run_dir),manifest),input_sha256=row.input_sha256))
        pd.DataFrame(sources).to_csv(manifest,index=False)
        frame=pd.DataFrame([d for d in details if d['scenario']==scenario])
        for p in ('lambda','sigma'):
            frame['x']=np.log1p(frame[p]);frame['y']=np.log1p(frame[p+'_bench'])
            regression,fit=fit_row(frame,'x','y',scenario+'_'+p+'_benchmark');regressions.append(regression)
            if scenario=='pair':save_plot(frame,'x','y',fit,out/f'validate_{p}_pair.pdf',rf'$\log(1+\hat{{\{p}}})$',rf'$\log(1+\{p}^{{bench}})$',show_statistics=True)
        if scenario!='pair':
            cf.FIT_ROOT=directory;cf.DATA_DIR=input_directory(scenario);design.DATA_DIR=cf.DATA_DIR
            policies=[dict(design.analyze(int(cid)),scenario=scenario) for cid in work.contest_id];policyrows.extend(policies)
            basepol=pd.read_csv(BASE_OUT/'counterfactual/precision_policy_contests.csv').set_index('contest_id')
            from scipy.stats import spearmanr
            a=float(raw.get('score_scale_a',1))
            frame=frame.set_index('contest_id')
            scales.append(dict(scenario=scenario,scale_a=a,full_disclosure_policies=sum(math.isinf(p['optimal_precision']) for p in policies),
                lambda_rank_correlation=float(spearmanr(a*a*frame['lambda'],baseline_full.loc[frame.index,'lambda_mean']).statistic),
                sigma_rank_correlation=float(spearmanr(frame['sigma']/a,baseline_full.loc[frame.index,'sigma_mean']).statistic),
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
            newmean=getattr(row,p+'_mean');oldmean=base[p+'_mean']
            timing.append(dict(seed=int(row.scenario.split('_')[1]),contest_id=row.contest_id,mode=row.mode,parameter=p,
                baseline_mean=oldmean,alternate_mean=newmean,absolute_shift_in_baseline_sd=abs(newmean-oldmean)/base[p+'_sd'],
                relative_mean_change=newmean/oldmean-1 if p!='mu_0' else np.nan,
                intervals_overlap=max(base[p+'_q025'],getattr(row,p+'_q025'))<=min(base[p+'_q975'],getattr(row,p+'_q975'))))
    timing=pd.DataFrame(timing);timing.to_csv(out/'time_imputation.csv',index=False)
    time_summaries=[]
    for (mode,p),work in timing.groupby(['mode','parameter']):
        time_summaries.append(dict(mode=mode,parameter=p,comparisons=len(work),intervals_overlap=int(work.intervals_overlap.sum()),
            max_absolute_sd_shift=float(work.absolute_shift_in_baseline_sd.max()),
            median_absolute_relative_change=float(work.relative_mean_change.abs().median()) if p!='mu_0' else None))
    result=dict(complete=True,accepted_fits=251,robustness_fits=219,imputation_fits=32,
                baseline_sha256=u.digest(BASE_OUT/'estimation/posterior_summary_final.csv'),regressions=regressions,scales=scales,time_imputation=time_summaries,
                interpretation='Same numerical empirical priors in transformed score units; conditional-mean variance reduction for lambda. Original inputs and focal-pair selection preserved.')
    u.save(out/'report.json',result)
    print(json.dumps(result,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=1);parser.add_argument('--report-only',action='store_true');args=parser.parse_args()
    if args.report_only:report();return
    ids=sorted({int(r['contest_id']) for r in csv.DictReader((MAIN_ESTIMATION_DIR/'accepted_chain_sources.csv').open())})
    tasks=[(scenario,cid,'full') for scenario in ('pair','L1','L2') for cid in ids]
    tasks.extend((scenario,cid,mode) for scenario in ('seed_20261008','seed_20261009') for cid in TIME_IDS for mode in ('full','early'))
    design=dict(tasks=tasks,source_sha256=u.digest(__file__),batch_sha256=u.digest(batch.__file__),moments_sha256=u.digest(rb.__file__),
                input_sha256={f'{scenario}/{cid}':u.digest(input_directory(scenario)/f'contest_{cid}.json') for scenario,cid,_ in tasks})
    signature=BASE_OUT/'auxiliary/design.json'
    if signature.exists() and json.loads(signature.read_text())!=json.loads(json.dumps(design)):raise RuntimeError('Auxiliary protocol changed')
    u.save(signature,design);failures=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(fit,t):t for t in tasks}
        for future in concurrent.futures.as_completed(futures):
            try:future.result()
            except Exception as error:
                failures.append(dict(task=futures[future],error=str(error)));print('FAILED',futures[future],error,flush=True)
            assemble();u.save(BASE_OUT/'auxiliary/failures.json',failures)
    if failures:raise RuntimeError('Auxiliary fits require refinement')
    report()


if __name__=='__main__':main()
