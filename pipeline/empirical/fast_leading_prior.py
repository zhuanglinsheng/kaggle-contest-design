"""Fast, exactly equivalent leading likelihood; common-prior sensitivity fits.

Four-chain Monte Carlo changes computational method, not the statistical
model. Density parity with Stan is required at dispersed parameter points.
"""
import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from cmdstanpy import CmdStanModel
from pipeline.empirical import nonlinear_reestimate as n
from pipeline.empirical import estimation_detail_audit as a
from pipeline.empirical import paired_prior_improvement as p
from pipeline.empirical import paired_prior_recovery as recovery
from pipeline.empirical import paired_activity_sensitivity as activity

BASE_LIKELIHOOD=n.Likelihood
BASE_LOAD=n.load_baseline


def configuration(data,profile):
    return activity.target(data) if profile=='activity_scale' else a.settings(data,profile)


class LeadingTarget(BASE_LIKELIHOOD):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        cfg=configuration(self.data,os.environ['KAGGLE_FAST_LEADING_PROFILE'])
        self.prior_mean=np.array(cfg['prior_mean']);self.sd=np.array(cfg['prior_sd'])

    def prior(self,z):
        if np.any(z[[0,1,2,3,5]]<math.log(.01)):return -np.inf
        return float(-.5*np.sum(((z-self.prior_mean)/self.sd)**2)-np.log(self.sd).sum()-3*math.log(2*math.pi))

    def loglik(self,z,nonlinear=True):
        return super().loglik(z,False)


class RecoveryLeadingTarget(recovery.RecoveryLikelihood):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        cfg=configuration(self.data,os.environ['KAGGLE_FAST_LEADING_PROFILE'])
        self.prior_mean=np.array(cfg['prior_mean']);self.sd=np.array(cfg['prior_sd'])

    def prior(self,z):return LeadingTarget.prior(self,z)

    def loglik(self,z,nonlinear=True):return super().loglik(z,False)


def configure(profile,kind):
    os.environ['KAGGLE_FAST_LEADING_PROFILE']=profile
    os.environ['KAGGLE_FAST_LEADING_KIND']=kind
    if kind=='empirical':
        n.SOURCE=n.ROOT/'pipeline/empirical/nonlinear_fast_kernel.cpp'
        n.Likelihood=LeadingTarget;n.load_baseline=BASE_LOAD;n.kernel=recovery.BASE_KERNEL
        n.OUT=p.OUT/profile/'leading_fast'
    else:
        recovery.configure('leading_control','scale_weak')
        n.Likelihood=RecoveryLeadingTarget
        n.OUT=p.OUT/'recovery_4657/leading_control'/profile/'leading_fast'
    n.OUT.mkdir(parents=True,exist_ok=True)


def prepare(cid,profile,kind):
    configure(profile,kind)
    data,_,old,provenance=n.load_baseline(cid)
    cfg=configuration(data,profile);evaluator=n.Likelihood(data)
    if kind=='empirical':
        model=CmdStanModel(stan_file=str(a.OUT/'detail_model.stan'),exe_file=str(a.OUT/'detail_model'))
        raw=np.load(a.NONLINEAR/f'posterior_grid_{cid}.npz')['z']
        points=[n.representative_point(old),n.representative_point(raw),old[len(old)//3]]
    else:
        model=CmdStanModel(stan_file=str(a.OUT/'feedback_timing/feedback_model.stan'),exe_file=str(a.OUT/'feedback_timing/feedback_model'))
        truth=json.loads((recovery.FIXTURE/'leading_control/truth.json').read_text())['truth']
        truthz=np.array([math.log(truth[k]) for k in ('c_i','c_j','sigma','lambda')]+[truth['mu_0'],truth['log_r']])
        points=[n.representative_point(old),truthz,old[len(old)//3]]
    differences=[]
    for point in points:
        stan=float(model.log_prob(a.physical(point),data=dict(data,**cfg),jacobian=False,sig_figs=18).iloc[0,0])
        differences.append(evaluator.posterior(point)-point[:4].sum()-stan)
    error=max(abs(v-differences[0]) for v in differences)
    if error>1e-7:raise RuntimeError('Fast leading Stan target mismatch')
    signature=dict(profile=profile,kind=kind,prior=cfg,target_parity_centered_error=error,
         script_sha256=n.digest(__file__),kernel_sha256=n.digest(n.SOURCE),**provenance)
    n.save_json(n.OUT/f'target_{cid}.json',signature)
    proposal=n.OUT/f'proposal_{cid}.json'
    if proposal.exists():return
    # Multiple starts screen modes; optimizer values are never final estimates.
    from scipy.optimize import minimize
    results=[]
    for point in points:
        result=minimize(lambda z:-evaluator.posterior(z),point,method='L-BFGS-B',
            bounds=[(math.log(.01)+1e-6,None)]*4+[(None,None),(math.log(.01)+1e-6,None)],
            options=dict(maxiter=200,ftol=1e-11))
        results.append(dict(z=result.x.tolist(),logposterior=-float(result.fun),converged=bool(result.success)))
    best=max(results,key=lambda r:r['logposterior'])
    n.save_json(n.OUT/f'explore_{cid}.json',dict(best=best,starts=results))
    n.calibrate(cid,.025,64)
    print('FAST TARGET PREPARED',cid,profile,kind,error,flush=True)


def chain(task):
    cid,profile,kind,j,warmup,samples=task;configure(profile,kind)
    n.sample(cid,j,warmup,samples,.025,64,211100+cid,pilot=True)


def summarize(cid,profile,kind):
    configure(profile,kind);data,_,_,_=n.load_baseline(cid)
    folder=n.OUT/f'contest_{cid}';paths=[folder/f'chain_{j}.npz' for j in range(1,5)]
    z=np.array([np.load(path)['z'] for path in paths]);flat=z.reshape(-1,6)
    values=n.physical_draws(flat,data['theta']).reshape(4,z.shape[1],-1)
    rhat=max(n.rank_rhat(values[:,:,k]) for k in range(values.shape[2]))
    ess=min(min(n.rank_ess(values[:,:,k])) for k in range(values.shape[2]))
    gate=dict(max_rank_rhat=rhat,min_ess=ess,passed=rhat<=1.01 and ess>=400,
               algorithm='Four-chain delayed-acceptance independence/RWM, frozen post-warmup transition settings; exact leading likelihood.')
    from pipeline.empirical.paired_prior_report import summary
    results=summary(flat,np.ones(len(flat))/len(flat),data)
    config=json.loads((n.OUT/f'target_{cid}.json').read_text())
    result=dict(diagnostics=gate,summaries=results,target=config,
                chain_sha256=[n.digest(path) for path in paths])
    if kind=='recovery':
        truth=json.loads((recovery.FIXTURE/'leading_control/truth.json').read_text())['truth']
        result['truth']=truth
        result['truth_in_95_interval']={key:results[key]['q025']<=truth[key]<=results[key]['q975'] for key in n.NAMES}
        result['median_relative_error']={key:abs(results[key]['median']/truth[key]-1) for key in ('c_i','c_j','sigma','lambda','r')}
    n.save_json(folder/'summary.json',result)
    print('FAST LEADING RESULT',cid,profile,kind,json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--profile',choices=('r_weak','scale_weak','activity_scale'),default='activity_scale')
    parser.add_argument('--kind',choices=('empirical','recovery'),default='recovery')
    parser.add_argument('--contests',type=int,nargs='+',default=[4657]);parser.add_argument('--warmup',type=int,default=2000)
    parser.add_argument('--samples',type=int,default=30000);parser.add_argument('--stage',choices=('prepare','run','report'),default='run')
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    for cid in args.contests:
        if args.stage in ('prepare','run'):prepare(cid,args.profile,args.kind)
        if args.stage=='run':
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                list(pool.map(chain,[(cid,args.profile,args.kind,j,args.warmup,args.samples) for j in range(1,5)]))
        if args.stage in ('run','report'):summarize(cid,args.profile,args.kind)
