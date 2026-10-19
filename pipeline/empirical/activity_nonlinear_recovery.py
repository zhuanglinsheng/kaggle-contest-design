"""Common activity-scale prior in nonlinear recovery; no policy substitution."""
import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from pipeline.empirical import nonlinear_reestimate as n
from pipeline.empirical import paired_prior_recovery as r
from pipeline.empirical import paired_prior_improvement as p
from pipeline.empirical import paired_activity_sensitivity as a


class ActivityRecoveryLikelihood(r.RecoveryLikelihood):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.prior_mean[5]=math.log(80)


def configure():
    r.configure('nonlinear','scale_weak')
    n.PRIOR_SD=np.array([1.25,1.25,.5,.75,1.,1.5])
    n.Likelihood=ActivityRecoveryLikelihood
    n.OUT=p.OUT/'recovery_4657/nonlinear/activity_scale/nonlinear'
    n.OUT.mkdir(parents=True,exist_ok=True)
    n._grid_worker=grid_worker


def grid_worker(payload):
    configure()
    source,data,dy,nt,extent,index,z=payload
    return index,ActivityRecoveryLikelihood(data,dy,nt,extent).posterior(z)


def prepare():
    configure();data,_,old,provenance=n.load_baseline(r.CID)
    from cmdstanpy import CmdStanModel
    model=CmdStanModel(stan_file=str(r.a.OUT/'feedback_timing/feedback_model.stan'),exe_file=str(r.a.OUT/'feedback_timing/feedback_model'))
    truth=json.loads((r.FIXTURE/'nonlinear/truth.json').read_text())['truth']
    point=np.array([math.log(truth[k]) for k in ('c_i','c_j','sigma','lambda')]+[truth['mu_0'],truth['log_r']])
    evaluator=ActivityRecoveryLikelihood(data,.0125,256)
    differences=[]
    for z in (point,point+np.array([.1,-.1,.05,.1,.02,.1])):
        stan=float(model.log_prob(r.a.physical(z),data=dict(data,**a.target(data)),jacobian=False,sig_figs=18).iloc[0,0])
        differences.append(evaluator.loglik(z,False)+evaluator.prior(z)-z[:4].sum()-stan)
    error=abs(differences[1]-differences[0])
    if error>1e-7:raise RuntimeError('Activity recovery prior parity failed')
    n.save_json(n.OUT/'configuration.json',dict(profile='activity_scale',prior=a.target(data),
       target_parity_centered_error=error,script_sha256=n.digest(__file__),
       kernel_sha256=n.digest(n.SOURCE),**provenance))
    if not (n.OUT/f'proposal_{r.CID}.json').exists():
        n.save_json(n.OUT/f'explore_{r.CID}.json',dict(best=dict(z=point.tolist()),
          note='Truth initializes optimizer only, not a posterior constraint.'))
        n.calibrate(r.CID,.0125,256)
    print('ACTIVITY NONLINEAR PREPARED',error,flush=True)


def chain(task):
    j,warmup,samples=task;configure()
    n.sample(r.CID,j,warmup,samples,.0125,256,221100+r.CID,pilot=True)


def summarize():
    configure();gate=json.loads((n.OUT/f'posterior_grid_{r.CID}.json').read_text())
    if gate['status']!='validated':
        print('NONLINEAR RECOVERY NOT VALIDATED',gate,flush=True);return
    from scipy.special import softmax
    from pipeline.empirical.paired_prior_report import summary
    data,_,_,_=n.load_baseline(r.CID);grid=np.load(n.OUT/f'posterior_grid_{r.CID}.npz')
    result=summary(grid['z'],softmax(grid['fine']-grid['base']),data)
    truth=json.loads((r.FIXTURE/'nonlinear/truth.json').read_text())['truth']
    report=dict(summaries=result,truth=truth,validation=gate,
        truth_in_95_interval={key:result[key]['q025']<=truth[key]<=result[key]['q975'] for key in n.NAMES},
        median_relative_error={key:abs(result[key]['median']/truth[key]-1) for key in ('c_i','c_j','sigma','lambda','r')},
        scope='Single retained nonlinear-policy realization. Different realized paths prevent subtracting errors of the leading-policy control as an approximation-bias decomposition.')
    n.save_json(n.OUT/'recovery_report.json',report)
    print('NONLINEAR ACTIVITY RECOVERY',json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('prepare','run','grid','report'),default='run')
    parser.add_argument('--workers',type=int,default=4);parser.add_argument('--warmup',type=int,default=1000)
    parser.add_argument('--samples',type=int,default=6000);args=parser.parse_args()
    if args.stage in ('prepare','run'):prepare()
    if args.stage=='run':
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(chain,[(j,args.warmup,args.samples) for j in range(1,5)]))
    if args.stage in ('run','grid'):
        configure();n.posterior_grid(r.CID,draws=2048,workers=args.workers)
    if args.stage in ('run','grid','report'):summarize()
