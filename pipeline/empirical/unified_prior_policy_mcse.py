"""Check Monte Carlo stability of posterior-mean precision-design quantities.

This is a computation check, not a nonlinear approximation-error test.
"""
import json
import math
import numpy as np
import pandas as pd
from scipy.special import ndtr
from pipeline.empirical import unified_prior_reestimate as u
from pipeline.empirical import counterfactual as cf,precision_design as design


def main():
    cf.FIT_ROOT=u.OUT/'estimation'
    frame=pd.read_csv(u.OUT/'counterfactual/precision_policy_contests.csv')
    rows=[]
    for r in frame.itertuples():
        d=cf.read_draws(int(r.contest_id));sigma=d['sigma'];mu2=d['mu_0']**2
        q=sigma*sigma*r.duration;cost=2/(1/d['c_i']+1/d['c_j'])
        weight=r.theta*r.duration/(cost*math.sqrt(2*math.pi))
        noeff=np.sqrt(q)*np.exp(-mu2/(2*q))/math.sqrt(2*math.pi)+np.sqrt(mu2)*(ndtr(np.sqrt(mu2/q))-.5)
        currentq=q+sigma/np.sqrt(d['lambda']);policyq=q+sigma*r.optimal_noise
        current=noeff+weight*np.exp(-mu2/(2*currentq))/np.sqrt(currentq)
        chosen=noeff+weight*np.exp(-mu2/(2*policyq))/np.sqrt(policyq)
        ratio=np.mean(chosen)/np.mean(current)
        influence=((chosen-ratio*current)/np.mean(current)).reshape(4,-1)
        ess=u.n.effective_sample_size(influence)
        mcse=100*float(np.std(influence,ddof=1)/math.sqrt(max(ess,1)))
        recommendations=[];objective_changes=[]
        for j in range(4):
            mask=np.ones(len(q),bool);n=len(q)//4;mask[j*n:(j+1)*n]=False
            x,value,_=design.optimal_noise(q[mask],mu2[mask],sigma[mask],weight[mask])
            recommendations.append('full' if x==0 else 'finite')
            objective_changes.append(float(np.mean(noeff[mask])+value))
        main_class='full' if r.optimal_noise==0 else 'finite'
        rows.append(dict(contest_id=int(r.contest_id),output_gain_pct=float(r.output_gain_pct),gain_monte_carlo_se_pct=mcse,
                         influence_ess=float(ess),policy_class=main_class,
                         leave_one_chain_out_classes=json.dumps(recommendations),
                         all_leave_one_chain_out_classes_agree=all(c==main_class for c in recommendations)))
        print('POLICY MCSE',r.contest_id,mcse,recommendations,flush=True)
    pd.DataFrame(rows).to_csv(u.OUT/'counterfactual/policy_monte_carlo_diagnostics.csv',index=False)
    u.save(u.OUT/'counterfactual/policy_monte_carlo_report.json',dict(contests=len(rows),
        maximum_gain_monte_carlo_se_pct=max(r['gain_monte_carlo_se_pct'] for r in rows),
        median_gain_monte_carlo_se_pct=float(np.median([r['gain_monte_carlo_se_pct'] for r in rows])),
        classifications_stable_under_all_four_leave_one_chain_out_checks=sum(r['all_leave_one_chain_out_classes_agree'] for r in rows),
        interpretation='Monte Carlo error of posterior integration and leave-one-chain-out decision stability. Not posterior credible intervals or nonlinear model-approximation error.',
        source_sha256=u.digest(__file__),posterior_summary_sha256=u.digest(u.OUT/'estimation/posterior_summary_final.csv')))


if __name__=='__main__':main()
