"""Same-prior NUTS versus prior-research MH crosscheck, not a truth test."""
import argparse
import json
import math
import numpy as np
import pandas as pd
from pipeline.empirical import unified_prior_reestimate as u,sigma_lambda_trials as s


def median_se(values):
    ess=u.n.effective_sample_size((values<=np.median(values)).astype(float))
    width=float(np.diff(np.quantile(values,[.45,.55]))[0])
    return 5*width/math.sqrt(max(ess,1))


def main(partial=False):
    rows=[];checked=[]
    for cid in s.IDS:
        accepted=u.OUT/'estimation/full'/f'contest_{cid}/accepted.json'
        if not accepted.exists():
            if partial:continue
            raise RuntimeError(f'Main fit not complete: {cid}')
        a=json.loads(accepted.read_text())
        raw=u.MAIN_DATA_DIR/f'contest_{cid}.json'
        data=u.e.stan_data(json.loads(raw.read_text()),'full')
        research=s.OUT/f'posterior/empirical/{cid}/lambda_external'
        reference=json.loads((research/'summary.json').read_text())
        if not reference['passed']:raise RuntimeError('Rejected same-prior MH reference')
        if json.loads((s.OUT/f'empirical/{cid}/data.json').read_text())!=data:raise RuntimeError('Sampler targets use different data')
        chains=[u.e.read_columns(u.Path(p)) for p in a['chain_paths']]
        old=np.array([np.load(research/f'contest_{cid}/chain_{j}.npz')['z'] for j in range(1,5)])
        for index,name in enumerate(u.NAMES):
            new=np.array([c['log_r'] if name=='r' else np.log(c[name]) if name!='mu_0' else c[name] for c in chains])
            previous=old[:,:,index]
            newmedian=float(np.median(new));oldmedian=float(np.median(previous))
            error=math.sqrt(median_se(new)**2+median_se(previous)**2)
            statistic=abs(newmedian-oldmedian)/error if error else 0.
            newq=np.quantile(new,[.025,.975]);oldq=np.quantile(previous,[.025,.975])
            rows.append(dict(contest_id=cid,parameter=name,coordinate='physical' if name=='mu_0' else 'log',
                nuts_median=newmedian if name=='mu_0' else math.exp(newmedian),
                mh_median=oldmedian if name=='mu_0' else math.exp(oldmedian),
                absolute_median_coordinate_difference=abs(newmedian-oldmedian),combined_approximate_median_mcse=error,
                standardized_median_difference=statistic,flag_over_five_mcse=bool(statistic>5),
                credible_intervals_overlap=bool(max(newq[0],oldq[0])<=min(newq[1],oldq[1]))))
        checked.append(cid)
        print('SAME-PRIOR SAMPLER CHECK',cid,flush=True)
    out=u.OUT/'validation';out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'same_prior_sampler_comparison.csv',index=False)
    u.save(out/'same_prior_sampler_report.json',dict(complete=len(checked)==10,checked_contests=checked,
        comparisons=len(rows),flagged_over_five_approximate_median_mcse=[r for r in rows if r['flag_over_five_mcse']],
        credible_intervals_overlap=sum(r['credible_intervals_overlap'] for r in rows),
        interpretation='Independent computation of the same leading-order posterior and same priors/data. Flags use approximate quantile MCSEs for investigation; this is neither nonlinear-versus-leading comparison nor empirical truth recovery.',
        source_sha256=u.digest(__file__)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');args=parser.parse_args();main(args.partial)
