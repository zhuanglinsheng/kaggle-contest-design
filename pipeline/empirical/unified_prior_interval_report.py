"""Carry forward frozen simulation calibration without relabeling credible CIs."""
import json
import math
import numpy as np
import pandas as pd
from pipeline.config import ROOT
from pipeline.empirical.unified_prior_reestimate import OUT,save,digest


def expand(q,k):
    m=math.log(q['median'])
    return math.exp(m+k*(math.log(q['q025'])-m)),math.exp(m+k*(math.log(q['q975'])-m))


def main():
    research=ROOT/'results/validation/sigma_lambda_trials'
    paths={'sigma':research/'sigma_interval_calibration/frozen_multiplier.json',
           'lambda':research/'lambda_refinement/lambda_interval_frozen.json'}
    multipliers={p:json.loads(path.read_text())['multiplier'] for p,path in paths.items()}
    rows=[]
    for replicate in (2,3,4,5):
        for case in sorted((research/'posterior/recovery').glob(f'*_{replicate}')):
            path=case/'lambda_external/summary.json'
            if not path.exists():continue
            a=json.loads(path.read_text())
            if not a['passed']:raise RuntimeError('Rejected recovery case')
            for p,k in multipliers.items():
                s=a['summaries'][p];truth=a['truth'][p];lo,hi=expand(s,k)
                rows.append(dict(case=case.name,replicate=replicate,parameter=p,truth=truth,
                    posterior_median=s['median'],posterior_mean=s['mean'],median_relative_error=abs(s['median']/truth-1),
                    mean_relative_error=abs(s['mean']/truth-1),original_credible_interval_covered=s['q025']<=truth<=s['q975'],
                    simulation_calibrated_interval_covered=lo<=truth<=hi,calibration_multiplier=k,
                    source_summary_sha256=digest(path)))
    frame=pd.DataFrame(rows)
    if len(frame)!=80:raise RuntimeError('Expected four realizations of the ten fixed configurations, two parameters each')
    destination=OUT/'validation';destination.mkdir(parents=True,exist_ok=True)
    frame.to_csv(destination/'simulation_recovery_reference.csv',index=False)
    report={}
    for p,group in frame.groupby('parameter'):
        report[p]=dict(configurations=10,realizations=40,median_point_mean_absolute_relative_error=float(group.median_relative_error.mean()),
            median_point_worst_absolute_relative_error=float(group.median_relative_error.max()),
            mean_point_mean_absolute_relative_error=float(group.mean_relative_error.mean()),
            original_95_credible_interval_covered=int(group.original_credible_interval_covered.sum()),
            simulation_calibrated_interval_covered=int(group.simulation_calibrated_interval_covered.sum()),
            multiplier=multipliers[p],calibration_source_sha256=digest(paths[p]))
    primary=pd.read_csv(OUT/'estimation/posterior_summary_final.csv')
    application=[]
    for row in primary[primary['mode']=='full'].itertuples():
        for p,k in multipliers.items():
            s={key:getattr(row,p+'_'+key) for key in ('median','q025','q975')};lo,hi=expand(s,k)
            application.append(dict(contest_id=row.contest_id,parameter=p,posterior_median=s['median'],
                original_posterior_q025=s['q025'],original_posterior_q975=s['q975'],
                simulation_calibrated_lower=lo,simulation_calibrated_upper=hi,multiplier=k,
                interval_label='Simulation-calibrated interval; not the original 95% credible interval',empirical_truth_coverage='unknown'))
    pd.DataFrame(application).to_csv(destination/'empirical_calibrated_intervals.csv',index=False)
    save(destination/'interval_and_recovery_report.json',dict(parameters=report,
        scope='Previously retained simulations at ten targeted parameter configurations, not a representative validation sample of all73 contests. Average error is not a per-contest error bound.',
        feedback_alignment='Recovery exercises retain their documented simulator-aligned interval-start feedback. Empirical re-estimation retains the existing interval-endpoint convention. The reference recovery rates are not an end-to-end validation of empirical timing and preprocessing.',
        intervals='Original Bayesian equal-tailed credible intervals and separately labeled simulation-calibrated intervals both retained.40/40 observed coverage does not establish a population95% guarantee. Marginal interval expansion does not alter the joint posterior or counterfactuals.',
        source_sha256=digest(__file__)))


if __name__=='__main__':main()
