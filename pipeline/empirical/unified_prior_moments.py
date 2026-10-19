"""Numerically integrate lambda conditional on the other five parameters.

Rao--Blackwellization stabilizes posterior means without changing the target,
the NUTS draws, the median, credible bounds, or counterfactual integration.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from scipy.special import roots_legendre
from pipeline.empirical import sigma_lambda_trials as s
from pipeline.empirical import nonlinear_reestimate as n
from pipeline.empirical import unified_prior_reestimate as u


def quadrature_grid(size):
    lo=math.log(.01);hi=math.log(100)+12*2.25
    x,w=roots_legendre(size)
    z=lo+(x+1)*(hi-lo)/2
    logprior=-.5*((z-math.log(100))/2.25)**2
    return z,np.log(w*(hi-lo)/2)+logprior


def conditional_moments(loglik, size=192):
    grid,weights=quadrature_grid(size)
    lp=np.asarray([loglik(v) for v in grid])+weights
    normalizer=logsumexp(lp)
    return np.exp([logsumexp(lp+order*grid)-normalizer for order in (1,2)])


def calculate(paths,data,points=128):
    target=s.Target(data,'lambda_external')
    estimates=[];coarse=[];locations=[]
    for path in paths:
        chain=u.e.read_columns(path)
        indices=np.linspace(0,len(chain['lambda'])-1,min(points,len(chain['lambda'])),dtype=int)
        moments=[];checks=[]
        for index in indices:
            z=np.array([math.log(chain[p][index]) for p in ('c_i','c_j','sigma','lambda')]+[chain['mu_0'][index],chain['log_r'][index]])
            def likelihood(lam):
                z[3]=lam
                return target.components(z)[0]
            fine=conditional_moments(likelihood,192)
            low=conditional_moments(likelihood,96)
            if np.max(np.abs(low/fine-1))>.001:
                higher=conditional_moments(likelihood,384)
                if np.max(np.abs(fine/higher-1))>.001:
                    highest=conditional_moments(likelihood,768)
                    if np.max(np.abs(higher/highest-1))>.001:
                        raise RuntimeError('Conditional lambda quadrature unresolved')
                    fine=higher;higher=highest
                low=fine;fine=higher
            moments.append(fine);checks.append(low)
        estimates.append(moments);coarse.append(checks);locations.append(indices.tolist())
    estimates=np.asarray(estimates);coarse=np.asarray(coarse)
    first=estimates[:,:,0]
    mean=float(np.mean(first));second=float(np.mean(estimates[:,:,1]))
    ess=n.effective_sample_size(first)
    mcse=float(np.std(first,ddof=1)/math.sqrt(max(ess,1)))
    return dict(mean=mean,sd=math.sqrt(max(0,second-mean*mean)),mean_mcse=mcse,
                relative_mean_mcse=mcse/mean,conditional_mean_ess=float(ess),
                doubled_grid_relative_mean_difference=float(abs(np.mean(coarse[:,:,0])/mean-1)),
                conditional_moments=estimates.tolist(),draw_indices=locations,
                points_per_chain=len(locations[0]),source_sha256=u.digest(__file__),
                method='E[lambda|other five parameters,data] by checked 1D quadrature, averaged over fresh NUTS marginal draws; posterior unchanged.')


def augment(cid,mode,paths,data,row,diagnostics):
    folder=Path(paths[0]).parent
    cache=folder/'lambda_conditional_moments.json'
    hashes=[u.digest(p) for p in paths]
    result=None
    if cache.exists():
        saved=json.loads(cache.read_text())
        if saved['chain_sha256']==hashes and saved['source_sha256']==u.digest(__file__):result=saved
    if result is None:
        for points in (128,512,2048):
            result=calculate(paths,data,points)
            if result['relative_mean_mcse']<=.03:break
        result['chain_sha256']=hashes
        u.save(cache,result)
    diagnostics['lambda_raw_mean']=row['lambda_mean']
    diagnostics['lambda_raw_sd']=row['lambda_sd']
    diagnostics['lambda_raw_mean_mcse']=diagnostics['quantities']['lambda']['mean_mcse']
    diagnostics['lambda_mean_calculation']=dict(file=str(cache),source_sha256=result['source_sha256'],relative_mean_mcse=result['relative_mean_mcse'],doubled_grid_relative_mean_difference=result['doubled_grid_relative_mean_difference'])
    row['lambda_mean']=result['mean'];row['lambda_sd']=result['sd'];row['lambda_mean_mcse']=result['mean_mcse']
    diagnostics['quantities']['lambda']['mean_mcse']=result['mean_mcse']
    diagnostics['quantities']['lambda']['relative_mean_mcse']=result['relative_mean_mcse']
    diagnostics['passed']=bool(row['divergences']==0 and diagnostics['saturated_transitions']==0
        and min(diagnostics['energy_bfmi'])>=.3
        and all(v['rank_rhat']<=1.01 and min(v['bulk_ess'],v['tail_ess'])>=400 for v in diagnostics['quantities'].values())
        and all(diagnostics['quantities'][p]['relative_mean_mcse']<=.03 for p in ('sigma','lambda'))
        and result['doubled_grid_relative_mean_difference']<=.001)
    return row,diagnostics
