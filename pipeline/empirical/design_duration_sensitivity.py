"""Conditional duration/prize design and output-preserving prize frontiers.

Uses the existing steady-state leading-order calibration, not a new joint
optimum or a fixed-initial-variance nonlinear design calculation.
"""
from __future__ import annotations

import csv
import json
import math

import numpy as np
import pandas as pd
from scipy.special import ndtr

from pipeline.empirical import counterfactual as cf, precision_design as design
from pipeline.empirical.unified_prior_reestimate import OUT, digest, save

DEST=OUT/'design_sensitivity'
TIME_MULTIPLIERS=(.1,.25,.5,.75,1.,1.25,1.5,2.)
PRIZE_MULTIPLIERS=(.5,1.,2.)


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    cf.FIT_ROOT=OUT/'estimation'
    manifest=cf.FIT_ROOT/'accepted_chain_sources.csv'
    ids=sorted(int(r['contest_id']) for r in csv.DictReader(manifest.open()) if r['mode']=='full')
    baseline=pd.read_csv(OUT/'counterfactual/precision_policy_contests.csv').set_index('contest_id')
    rows=[];grid=[]
    save(DEST/'design.json',dict(source_sha256=digest(__file__),manifest_sha256=digest(manifest),
        time_multipliers=TIME_MULTIPLIERS,prize_multipliers=PRIZE_MULTIPLIERS,
        scope='Existing steady-state leading-order calibration; initial perceived gap fixed. Candidate precision affects steady-state incentive variance, not the no-effort term. Not a fixed-S0 joint optimum.',
        organizer_preferences='Not estimated; output-preserving prize frontiers allow explicit delay-cost thresholds without specifying quality utility.'))
    for index,cid in enumerate(ids,1):
        accepted=OUT/'estimation/full'/f'contest_{cid}'/'accepted.json'
        a=json.loads(accepted.read_text())
        for path,sha in zip(a['chain_paths'],a['chain_sha256']):
            if digest(path)!=sha:raise RuntimeError('Changed posterior chain')
        d=cf.read_draws(cid);sigma=d['sigma'];mu2=d['mu_0']**2
        cost=2/(1/d['c_i']+1/d['c_j']);current_x=1/np.sqrt(d['lambda'])
        theta0=float(baseline.loc[cid,'theta']);T0=float(baseline.loc[cid,'duration'])
        target=float(baseline.loc[cid,'current_output'])
        for multiple in TIME_MULTIPLIERS:
            T=multiple*T0;q=sigma*sigma*T;sd=np.sqrt(q);mu=np.sqrt(mu2)
            b0=float(np.mean(sd*np.exp(-mu2/(2*q))/math.sqrt(2*math.pi)+mu*(ndtr(mu/sd)-.5)))
            x,value,_=design.optimal_noise(q,mu2,sigma,1/cost)
            x2,v2,_=design.optimal_noise(q,mu2,sigma,1/cost,1025)
            if not np.isclose(value,v2,rtol=1e-9,atol=1e-12) or not np.isclose(x,x2,rtol=1e-7,atol=1e-10):
                raise RuntimeError(f'Unstable precision solution {cid} {multiple}')
            A=T/math.sqrt(2*math.pi)*value
            qcur=q+sigma*current_x
            Acur=float(np.mean(T*np.exp(-mu2/(2*qcur))/(cost*np.sqrt(2*math.pi*qcur))))
            required=max(0.,(target-b0)/A)
            draw_x=np.maximum(0.,(mu2-q)/sigma)
            decrease=float(np.mean(draw_x>current_x));increase=float(np.mean(draw_x<current_x))
            # A common fixed precision recommendation compared with current
            # uncertainty is distinct from draw-specific directional support.
            fixed_decrease=float(np.mean(x>current_x))
            if multiple==1:
                if not np.isclose(x,float(baseline.loc[cid,'optimal_noise']),rtol=1e-7,atol=1e-9):raise RuntimeError('Baseline policy mismatch')
                if not np.isclose(b0+theta0*Acur,target,rtol=1e-11,atol=1e-12):raise RuntimeError('Baseline output mismatch')
            row=dict(contest_id=cid,time_multiplier=multiple,duration=T,observed_duration=T0,
                observed_prize=theta0,optimal_noise=x,optimal_precision=None if x==0 else x**-2,
                finite_precision=x>0,draw_decrease_probability=decrease,draw_increase_probability=increase,
                fixed_policy_decrease_probability=fixed_decrease,
                decrease_supported=decrease>.75,increase_supported=increase>.75,
                current_precision_median=float(np.median(d['lambda'])),
                finite_precision_to_current_median=None if x==0 else x**-2/float(np.median(d['lambda'])),
                no_effort_output=b0,marginal_prize_output=A,current_regime_marginal_prize_output=Acur,
                target_current_output=target,required_prize=required,required_prize_ratio=required/theta0,
                zero_prize_meets_target=required==0,
                normalized_delay_cost_threshold=None if multiple==1 else (required/theta0-1)/(1-multiple**2))
            rows.append(row)
            for prize_multiple in PRIZE_MULTIPLIERS:
                theta=theta0*prize_multiple
                grid.append(dict(contest_id=cid,time_multiplier=multiple,prize_multiplier=prize_multiple,
                    optimal_noise=x,finite_precision=x>0,
                    precision_policy_output=b0+theta*A,
                    quality_change_from_observed_pct=100*((b0+theta*A)/target-1),
                    precision_only_gain_at_same_prize_time_pct=100*((b0+theta*A)/(b0+theta*Acur)-1)))
        pd.DataFrame(rows).to_csv(DEST/'contest_frontiers.csv',index=False)
        print('COMPLETED',index,'/73',cid,flush=True)
    frame=pd.DataFrame(rows);summary=[]
    for multiple,work in frame.groupby('time_multiplier'):
        finite=work[work.finite_precision]
        summary.append(dict(time_multiplier=float(multiple),contests=len(work),
            finite_precision=int(work.finite_precision.sum()),full_disclosure=int((~work.finite_precision).sum()),
            decrease_supported=int(work.decrease_supported.sum()),increase_supported=int(work.increase_supported.sum()),
            direction_uncertain=int((~work.decrease_supported & ~work.increase_supported).sum()),
            fixed_policy_decrease_supported=int((work.fixed_policy_decrease_probability>.75).sum()),
            median_required_prize_ratio=float(work.required_prize_ratio.median()),
            q25_required_prize_ratio=float(work.required_prize_ratio.quantile(.25)),
            q75_required_prize_ratio=float(work.required_prize_ratio.quantile(.75)),
            zero_prize_meets_target=int(work.zero_prize_meets_target.sum()),
            finite_with_positive_required_prize=int((work.finite_precision & ~work.zero_prize_meets_target).sum()),
            finite_precision_ratio_median=None if finite.empty else float(finite.finite_precision_to_current_median.median())))
    pd.DataFrame(grid).to_csv(DEST/'prize_time_grid.csv',index=False)
    pd.DataFrame(summary).to_csv(DEST/'summary.csv',index=False)
    save(DEST/'summary.json',dict(complete=True,contests=73,summary=summary,
        prize_precision_invariance='For each fixed positive T, B=B0+theta*A implies precision maximizes E[A], independently of every positive prize. At zero prize precision is irrelevant.',
        delay_threshold='Let D_l(T)=delta_l*theta_obs*(T/T_obs)^2. Along the output-preserving frontier, shorter duration improves payoff iff delta_l>(theta_required/theta_obs-1)/(1-(T/T_obs)^2). Utility cancels because posterior-average quality is preserved.',
        no_claim='This is a conditional sensitivity analysis, not identified organizer preferences or a fixed-S0 joint-design solution. No scenarios or draws selected based on desirable outcomes.'))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
