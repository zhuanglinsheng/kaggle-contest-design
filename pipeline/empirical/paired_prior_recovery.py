"""Nonlinear recovery on preserved known-parameter data, with common priors.

The numerical policy is unchanged. A separate kernel accepts interval-start
feedback for the simulator-consistent belief update; endpoint observations stay
at the endpoint. Publication kernels and completed audits are never edited.
"""
from __future__ import annotations
import argparse
import ctypes
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from cmdstanpy import CmdStanModel, from_csv
from pipeline.empirical import nonlinear_reestimate as n
from pipeline.empirical import estimation_detail_audit as a
from pipeline.empirical import paired_prior_improvement as p

BASE_LIKELIHOOD=n.Likelihood
BASE_KERNEL=n.kernel
CID=4657
OUT=p.OUT/'recovery_4657'
FIXTURE=a.OUT/'feedback_timing/recovery_4657'


def prepare_source():
    original=p.OUT.parent.parent.parent/'pipeline/empirical/nonlinear_fast_kernel.cpp'
    original=n.ROOT/'pipeline/empirical/nonlinear_fast_kernel.cpp'
    source=original.read_text()
    assert source.count('const double* held,const int* counts_i')==1
    assert source.count('weight*(held[i]-perceived)')==1
    source=source.replace('const double* held,const int* counts_i','const double* held,const double* feedback,const int* counts_i')
    source=source.replace('weight*(held[i]-perceived)','weight*(feedback[i]-perceived)')
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'recovery_feedback_kernel.cpp'
    if path.exists() and path.read_text()!=source:raise RuntimeError('Recovery source changed')
    if not path.exists():path.write_text(source)
    return path


def recovery_kernel():
    lib=BASE_KERNEL()
    double_array=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    args=lib.empirical_likelihood.argtypes.copy();args.insert(11,double_array)
    lib.empirical_likelihood.argtypes=args
    return lib


class RecoveryLikelihood(BASE_LIKELIHOOD):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.feedback=np.ascontiguousarray(self.data['feedback_gap'],dtype=np.float64)

    def loglik(self,z,nonlinear=True):
        ci,cj,sig,lam=np.exp(z[:4]);d=self.data
        policy,dy=self.policy(z) if nonlinear else (None,self.dy)
        status=ctypes.c_int()
        value=self.lib.empirical_likelihood(d['N_Delta'],d['Delta2f'],d['T_contest'],
          d['theta'],ci,cj,sig,lam,float(z[4]),float(z[5]),self.held,self.feedback,*self.counts,
          d['count_start_idx']-1,None if policy is None else policy.ctypes.data,
          0 if policy is None else policy.shape[1],self.nt,dy,ctypes.byref(status))
        self.calls+=1
        if status.value:raise RuntimeError(f'Recovery numerical failure {status.value}')
        return value


def load_fixture(cid):
    if cid!=CID:raise ValueError(cid)
    policy=os.environ['KAGGLE_RECOVERY_POLICY']
    folder=FIXTURE/policy/'interval_start'
    config=json.loads((folder/'run.json').read_text())
    data=dict(config['inputs']);data.pop('prior_mean');data.pop('prior_sd');data.pop('sigma_floor')
    values=np.load(folder/'posterior.npz')['values']
    z=values.transpose(1,0,2).reshape(-1,6).copy();z[:,:4]=np.log(z[:,:4])
    chains=[values[:,j].copy() for j in range(4)]
    provenance=dict(input_sha256=n.digest(folder/'run.json'),fixture_posterior_sha256=n.digest(folder/'posterior.npz'),
                    truth_sha256=n.digest(FIXTURE/policy/'truth.json'))
    return data,chains,z,provenance


def recovery_grid_worker(payload):
    configure(os.environ['KAGGLE_RECOVERY_POLICY'],os.environ['KAGGLE_PAIRED_PRIOR_PROFILE'])
    source,data,dy,nt,extent,index,z=payload
    return index,RecoveryLikelihood(data,dy,nt,extent).posterior(z)


def configure(policy,profile):
    os.environ['KAGGLE_RECOVERY_POLICY']=policy
    os.environ['KAGGLE_PAIRED_PRIOR_PROFILE']=profile
    n.SOURCE=prepare_source();n.OUT=OUT/policy/profile/'nonlinear';n.OUT.mkdir(parents=True,exist_ok=True)
    n.PRIOR_SD=np.array(a.settings({'hat_y':[0.]},profile)['prior_sd'])
    n.Likelihood=RecoveryLikelihood;n.kernel=recovery_kernel;n.load_baseline=load_fixture
    n._grid_worker=recovery_grid_worker


def prepare(policy,profile):
    configure(policy,profile)
    data,_,_,provenance=load_fixture(CID)
    model=CmdStanModel(stan_file=str(a.OUT/'feedback_timing/feedback_model.stan'),exe_file=str(a.OUT/'feedback_timing/feedback_model'))
    original_inputs=json.loads((FIXTURE/policy/'interval_start/run.json').read_text())['inputs']
    truth=json.loads((FIXTURE/policy/'truth.json').read_text())['truth']
    z=np.array([math.log(truth[k]) for k in ('c_i','c_j','sigma','lambda')]+[truth['mu_0'],truth['log_r']])
    check=RecoveryLikelihood(data,.0125,256)
    params={k:truth[k] for k in ('c_i','c_j','sigma','lambda','mu_0','log_r')}
    stan=float(model.log_prob(params,data=original_inputs,jacobian=False,sig_figs=18).iloc[0,0])
    ours=check.loglik(z,False)+check.prior(z)-z[:4].sum()
    points=[z,z+np.array([.1,-.1,.05,.1,.02,.1])]
    errors=[]
    for point in points:
        params=a.physical(point);inputs=dict(data,**a.settings(data,profile))
        reference=float(model.log_prob(params,data=inputs,jacobian=False,sig_figs=18).iloc[0,0])
        errors.append(check.loglik(point,False)+check.prior(point)-point[:4].sum()-reference)
    centered=max(abs(v-errors[0]) for v in errors)
    if centered>1e-7:raise RuntimeError('Recovery target parity failed')
    n.save_json(n.OUT/'configuration.json',dict(profile=profile,prior_sd=n.PRIOR_SD.tolist(),
        script_sha256=n.digest(__file__),kernel_sha256=n.digest(n.SOURCE),target_parity_centered_error=centered,**provenance))
    proposal=n.OUT/f'proposal_{CID}.json'
    if not proposal.exists():
        n.save_json(n.OUT/f'explore_{CID}.json',dict(best=dict(z=z.tolist()),note='Known truth initializes optimizer only. Chains sample the entire posterior.'))
        n.calibrate(CID,.0125,256)
    print('RECOVERY PREPARED',policy,profile,'parity',centered,flush=True)


def chain(task):
    policy,profile,j,warmup,samples=task;configure(policy,profile)
    n.sample_prior_screen(CID,j,warmup,samples,.0125,256,191100+CID,pilot=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--policy',choices=('leading_control','nonlinear'),default='nonlinear')
    parser.add_argument('--profile',choices=p.PROFILES,default='scale_weak')
    parser.add_argument('--stage',choices=('prepare','sample','grid'),default='sample')
    parser.add_argument('--warmup',type=int,default=1000);parser.add_argument('--samples',type=int,default=4000)
    args=parser.parse_args()
    if args.stage in ('prepare','sample'):prepare(args.policy,args.profile)
    if args.stage=='sample':
        with ProcessPoolExecutor(max_workers=4) as pool:
            list(pool.map(chain,[(args.policy,args.profile,j,args.warmup,args.samples) for j in range(1,5)]))
    if args.stage in ('sample','grid'):
        configure(args.policy,args.profile);n.posterior_grid(CID,draws=2048,workers=4)
