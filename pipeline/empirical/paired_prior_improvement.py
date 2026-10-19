"""Common-prior comparisons; publication and previous audits are immutable inputs.

Only priors change. Numerical nonlinear estimates require fresh four-chain
sampling AND posterior grid validation before being reported as usable.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.special import softmax
from cmdstanpy import CmdStanModel, from_csv
from pipeline.empirical import estimation_detail_audit as a
from pipeline.empirical import nonlinear_reestimate as n

OUT=n.ROOT/'results/validation/paired_prior_improvement'
PROFILES=('r_weak','scale_weak')
BASE_GRID_WORKER=n._grid_worker


def paired_grid_worker(payload):
    # macOS spawn workers import fresh globals; explicitly propagate the target.
    n.PRIOR_SD=np.array(a.settings({'hat_y':[0.]},os.environ['KAGGLE_PAIRED_PRIOR_PROFILE'])['prior_sd'])
    return BASE_GRID_WORKER(payload)


def configure(profile):
    os.environ['KAGGLE_PAIRED_PRIOR_PROFILE']=profile
    n._grid_worker=paired_grid_worker
    n.SOURCE=Path(__file__).with_name('nonlinear_fast_kernel.cpp')
    n.OUT=OUT/profile/'nonlinear'
    n.OUT.mkdir(parents=True,exist_ok=True)
    n.PRIOR_SD=np.array(a.settings({'hat_y':[0.]},profile)['prior_sd'])


def screen():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[]
    for cid in a.IDS:
        data,_,_,_=n.load_baseline(cid)
        g=np.load(a.NONLINEAR/f'posterior_grid_{cid}.npz');z=g['z']
        base=a.settings(data,'baseline')
        for profile in PROFILES:
            cfg=a.settings(data,profile)
            change=-.5*np.sum(((z-np.array(cfg['prior_mean']))/cfg['prior_sd'])**2-
                             ((z-np.array(base['prior_mean']))/base['prior_sd'])**2,axis=1)
            weights=softmax(g['fine']-g['base']+change)
            values=n.physical_draws(z,data['theta']);mean,sd,q=n.weighted_summary(values,weights)
            ess=[]
            for k in range(values.shape[1]):
                influence=(len(z)*weights*(values[:,k]-mean[k])).reshape(4,-1)
                ess.append(n.effective_sample_size(influence)*sd[k]**2/max(influence.var(ddof=1),1e-30))
            rows.append(dict(contest_id=cid,profile=profile,weight_ess=float(1/(weights@weights)),
                             minimum_mean_ess=float(min(ess)),maximum_weight=float(weights.max()),
                             usable=False,note='Proposal screening only; fresh nonlinear MCMC required.',
                             screened_max_w_quantiles=q[-1].tolist()))
    n.save_json(OUT/'screen.json',rows)
    n.save_json(OUT/'design.json',dict(profiles={p:a.settings({'hat_y':[0.]},p) for p in PROFILES},
        selected_contests=list(a.IDS),selection='Previously fixed high-w representative contests; no reselection.',
        rationale='Prior sensitivity already identified in known-parameter control; widen r alone or r and costs. Means, innovation prior, data, units and effort formulas unchanged.',
        numerical_gate='Four-chain rank Rhat <=1.01, bulk/tail ESS >=400, validated posterior fine-grid correction.',
        evaluation='Known-parameter recovery, observed submission allocation, common-prior parameter and identifiable-combination comparisons. No acceptance threshold on w.',
        caveat='Better agreement alone does not establish correctness. Single recovery realization cannot establish repeated-sample coverage.',
        script_sha256=n.digest(__file__),baseline_source_sha256=n.digest(n.ROOT/'metakaggle/real_data.stan')))
    print(json.dumps(rows,indent=2),flush=True)


def leading(cid,profile,warmup=1500,samples=2500):
    data,_,old,provenance=n.load_baseline(cid)
    folder=OUT/profile/'leading'/f'contest_{cid}';folder.mkdir(parents=True,exist_ok=True)
    model=CmdStanModel(stan_file=str(a.OUT/'detail_model.stan'),exe_file=str(a.OUT/'detail_model'))
    inputs=dict(data,**a.settings(data,profile))
    rng=np.random.default_rng(171100+cid)
    starts=[a.physical(old[j]) for j in rng.choice(len(old),4,replace=False)]
    if cid==4699 and profile=='scale_weak':
        modes=json.loads((a.OUT/'mode_screen.json').read_text())
        starts[0]=next(r['log_coordinate_mode'] for r in modes if r['contest_id']==cid and r['variant']==profile)
    config=dict(inputs=inputs,starts=starts,warmup=warmup,samples=samples,seed=171100+cid,
                adapt_delta=.99,model_sha256=n.digest(a.OUT/'detail_model.stan'),**provenance)
    path=folder/'run.json'
    if path.exists() and json.loads(path.read_text())!=config:raise RuntimeError('Leading configuration changed')
    n.save_json(path,config)
    files=sorted(folder.glob('detail_model-*.csv'))
    if len(files)==4:fit=from_csv([str(p) for p in files])
    elif files:raise RuntimeError('Partial leading run; inspect before restart')
    else:fit=model.sample(data=inputs,inits=starts,chains=4,parallel_chains=4,
         iter_warmup=warmup,iter_sampling=samples,seed=config['seed'],adapt_delta=.99,
         metric='dense_e',max_treedepth=12,sig_figs=18,output_dir=str(folder),show_progress=False,refresh=500)
    if fit.num_draws_sampling!=samples:raise RuntimeError('Incomplete chains')
    draws=fit.draws();values=np.stack([draws[:,:,fit.column_names.index(k)] for k in ('c_i','c_j','sigma','lambda','mu_0','log_r')],axis=2)
    np.savez_compressed(folder/'posterior.npz',values=values)
    method=fit.method_variables();energy=method['energy__']
    diagnostics=dict(divergences=int(method['divergent__'].sum()),
       treedepth_hits=int((method['treedepth__']>=12).sum()),
       minimum_bfmi=float(np.min(np.mean(np.diff(energy,axis=0)**2,axis=0)/np.var(energy,axis=0))))
    n.save_json(folder/'sampler.json',diagnostics)
    print('LEADING DONE',cid,profile,diagnostics,flush=True)


def prepare(cid,profile):
    configure(profile)
    target=n.OUT/f'proposal_{cid}.json'
    config=dict(profile=profile,prior_sd=n.PRIOR_SD.tolist(),script_sha256=n.digest(__file__),
                base_source_sha256=n.digest(Path(n.__file__)),kernel_sha256=n.digest(n.SOURCE))
    signature=n.OUT/'configuration.json'
    if signature.exists() and json.loads(signature.read_text())!=config:raise RuntimeError('Nonlinear source/target changed')
    n.save_json(signature,config)
    if target.exists():return
    data,_,_,_=n.load_baseline(cid)
    g=np.load(a.NONLINEAR/f'posterior_grid_{cid}.npz');z=g['z']
    base=a.settings(data,'baseline');cfg=a.settings(data,profile)
    lw=g['fine']-g['base']-.5*np.sum(((z-np.array(cfg['prior_mean']))/cfg['prior_sd'])**2-
                                  ((z-np.array(base['prior_mean']))/base['prior_sd'])**2,axis=1)
    weights=softmax(lw);center=(weights[:,None]*z).sum(axis=0)
    n.save_json(n.OUT/f'explore_{cid}.json',dict(best=dict(z=center.tolist()),
                note='Old-posterior reweighting initializes optimization only; not an inferential result.'))
    n.calibrate(cid,.0125,256)


def chain_task(task):
    cid,profile,chain,warmup,samples=task
    configure(profile)
    n.sample_prior_screen(cid,chain,warmup,samples,.0125,256,181100+cid,pilot=True)


def nonlinear(cid,profile,warmup=1000,samples=4000):
    prepare(cid,profile)
    with ProcessPoolExecutor(max_workers=4) as pool:
        list(pool.map(chain_task,[(cid,profile,j,warmup,samples) for j in range(1,5)]))
    configure(profile)
    n.posterior_grid(cid,draws=2048,workers=4)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('screen','leading','nonlinear','grid'),required=True)
    p.add_argument('--profile',choices=PROFILES,default='scale_weak')
    p.add_argument('--contests',type=int,nargs='+',choices=a.IDS,default=list(a.IDS))
    p.add_argument('--warmup',type=int);p.add_argument('--samples',type=int)
    args=p.parse_args()
    if args.stage=='screen':screen()
    else:
        for cid in args.contests:
            if args.stage=='leading':leading(cid,args.profile,args.warmup or 1500,args.samples or 2500)
            elif args.stage=='nonlinear':nonlinear(cid,args.profile,args.warmup or 1000,args.samples or 4000)
            else:
                configure(args.profile);n.posterior_grid(cid,draws=2048,workers=4)
