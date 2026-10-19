"""Isolate endpoint/start-of-interval feedback alignment; no publication edits.

The empirical input stores interval-end public gaps. Chapter 4 advances beliefs
using the held gap at interval start. Only the feedback input changes here.
Paired small recovery controls reuse the preserved Chapter 4 hourly simulator.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta

import numpy as np
from scipy.special import softmax
from cmdstanpy import CmdStanModel, from_csv

from pipeline.empirical import estimation_detail_audit as a
from pipeline.empirical import nonlinear_reestimate as n
from synthetic_data.synthetic_data import synthetic_data_simulation

OUT=a.OUT/"feedback_timing"


def model():
    source=(a.OUT/"detail_model.stan").read_text()
    source=source.replace("vector[N_Delta] hat_y;", "vector[N_Delta] hat_y;\n  vector[N_Delta] feedback_gap;")
    assert source.count("(hat_y[i] - tilde_y[i])")==1
    source=source.replace("(hat_y[i] - tilde_y[i])", "(feedback_gap[i] - tilde_y[i])")
    path=OUT/"feedback_model.stan";OUT.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.read_text()!=source: raise RuntimeError("Timing audit model changed")
    if not path.exists(): path.write_text(source)
    return CmdStanModel(stan_file=str(path))


def run_fit(m,folder,data,initial,*,warmup,samples,seed):
    folder.mkdir(parents=True,exist_ok=True)
    signature=dict(inputs=data,starts=initial,warmup=warmup,samples=samples,seed=seed,
        model_sha256=n.digest(OUT/"feedback_model.stan"),metric="dense_e",adapt_delta=.95,max_treedepth=12)
    path=folder/"run.json"
    if path.exists() and json.loads(path.read_text())!=signature: raise RuntimeError("Timing fit changed")
    path.write_text(json.dumps(signature,indent=2))
    files=sorted(folder.glob("feedback_model-*.csv"))
    if len(files)==4:
        fit=from_csv([str(p) for p in files])
        if fit.num_draws_sampling!=samples: raise RuntimeError("Incomplete timing fit")
    elif files: raise RuntimeError("Partial timing fit requires inspection")
    else:
        fit=m.sample(data=data,inits=initial,chains=4,parallel_chains=4,iter_warmup=warmup,
            iter_sampling=samples,seed=seed,metric="dense_e",adapt_delta=.95,max_treedepth=12,
            sig_figs=18,output_dir=str(folder),show_progress=False,refresh=250)
    draws=fit.draws()
    values=np.stack([draws[:,:,fit.column_names.index(k)] for k in ("c_i","c_j","sigma","lambda","mu_0","log_r")],axis=2)
    np.savez_compressed(folder/"posterior.npz",values=values)
    x=values.copy();x[:,:,5]=np.exp(x[:,:,5]);w=data["theta"]/(x[:,:,2,None]**2*x[:,:,:2])
    x=np.concatenate((x,w,w.max(axis=2,keepdims=True)),axis=2)
    summary={}
    for k,key in enumerate(n.NAMES+("w_i","w_j","max_w")):
        v=x[:,:,k];ess=n.rank_ess(v.T)
        summary[key]=dict(mean=float(v.mean()),sd=float(v.std(ddof=1)),median=float(np.median(v)),
            q025=float(np.quantile(v,.025)),q975=float(np.quantile(v,.975)),
            rank_rhat=n.rank_rhat(v.T),bulk_ess=ess[0],tail_ess=ess[1])
    method=fit.method_variables()
    summary["diagnostics"]=dict(max_rank_rhat=max(v["rank_rhat"] for v in summary.values()),
        min_ess=min(min(v["bulk_ess"],v["tail_ess"]) for v in summary.values()),
        divergences=int(method["divergent__"].sum()),treedepth_hits=int((method["treedepth__"]>=12).sum()))
    (folder/"summary.json").write_text(json.dumps(summary,indent=2))
    print(folder.name,summary["max_w"],summary["diagnostics"],flush=True)


def empirical(warmup,samples):
    m=model();base=CmdStanModel(stan_file=str(a.OUT/"detail_model.stan"),exe_file=str(a.OUT/"detail_model"))
    for cid in a.IDS:
        data,_,z,provenance=n.load_baseline(cid)
        inputs=dict(data,**a.settings(data,"baseline"))
        point=a.physical(n.representative_point(z))
        original=float(base.log_prob(point,data=inputs,jacobian=False,sig_figs=18).iloc[0,0])
        unchanged=float(m.log_prob(point,data=dict(inputs,feedback_gap=data["hat_y"]),jacobian=False,sig_figs=18).iloc[0,0])
        if abs(original-unchanged)>1e-7: raise RuntimeError("Feedback model parity failed")
        rng=np.random.default_rng(98100+cid)
        initial=[a.physical(z[k]) for k in rng.choice(len(z),4,replace=False)]
        # Before the observed window, retain the same declared backfilled public input.
        inputs["feedback_gap"]=[data["hat_y"][0]]+data["hat_y"][:-1]
        folder=OUT/f"contest_{cid}"/"interval_start"
        folder.mkdir(parents=True,exist_ok=True)
        (folder/"input_provenance.json").write_text(json.dumps(dict(parity_error=unchanged-original,**provenance),indent=2))
        run_fit(m,folder,inputs,initial,warmup=warmup,samples=samples,seed=98100+cid)


def recovery(warmup,samples,contests=None):
    m=model()
    # Use the already audited fast numerical equilibrium implementation.
    n.SOURCE=n.ROOT/"pipeline/empirical/nonlinear_fast_kernel.cpp"
    for cid in a.IDS if contests is None else contests:
        raw,_,_,_=n.load_baseline(cid)
        grid=np.load(a.NONLINEAR/f"posterior_grid_{cid}.npz");weights=softmax(grid["fine"]-grid["base"])
        _,spread,quantiles=n.weighted_summary(grid["z"],weights)
        center=np.array([q[1] for q in quantiles])
        index=int(np.argmin(np.sum(((grid["z"]-center)/np.maximum(spread,1e-8))**2,axis=1)))
        z=grid["z"][index];truth=a.physical(z);ratio=math.exp(z[5]);T=raw["T_contest"]
        evaluator=n.Likelihood(raw,dy=.003125,nt=4096)
        policies,dy=evaluator.policy(z)
        sigma=truth["sigma"];scale=sigma/math.sqrt(T);nt=policies.shape[0]-1
        def numerical(t,y):
            position=y/(sigma*math.sqrt(T))/dy+(policies.shape[1]-1)/2
            if not 0<=position<policies.shape[1]-1: raise RuntimeError("Recovery leaves HJB state grid")
            j=int(position);f=position-j;time=t/T*nt;k=min(int(time),nt-1);tf=time-k
            return scale*((1-tf)*((1-f)*policies[k,j]+f*policies[k,j+1])+tf*((1-f)*policies[k+1,j]+f*policies[k+1,j+1]))
        # Same Poisson candidates and noise seeds for the two policy versions.
        leading_peak=raw["theta"]/min(truth["c_i"],truth["c_j"])/math.sqrt(2*math.pi*sigma/math.sqrt(truth["lambda"]))
        bound=max(1.,1.01*ratio*max(float(policies.max())*scale,leading_peak)/24)
        for policy_name,policy in (("leading_control",None),("nonlinear",numerical)):
            folder=OUT/f"recovery_{cid}"/policy_name;folder.mkdir(parents=True,exist_ok=True)
            seed=99100+cid
            result=synthetic_data_simulation(theta=raw["theta"],c_i=truth["c_i"],c_j=truth["c_j"],
                sigma=sigma,lamb=truth["lambda"],intensity_effort_ratio=ratio,hour_arrival_ub=bound,
                start_time=datetime(2026,1,1),end_time=datetime(2026,1,1)+timedelta(hours=raw["N_Delta"]),
                mu_0=truth["mu_0"],effort_policy=policy,seed_brownian=seed,seed_poisson=seed+1,
                seed_uniform=seed+2,seed_initial=seed+3)
            times,mi,mj,latent,perceived,public,ei,ej=result
            initial_time=datetime(2026,1,1)
            data=dict(theta=raw["theta"],Delta2f=raw["Delta2f"],N_Delta=len(times),count_start_idx=1,T_contest=T,
                Ni=len(ei),Nj=len(ej),hat_t_i=[(t-initial_time).total_seconds()/3600 for t in ei],
                hat_t_j=[(t-initial_time).total_seconds()/3600 for t in ej],hat_y=public[1:].tolist())
            if len(times)!=raw["N_Delta"]: raise RuntimeError("Recovery time grid mismatch")
            np.savez_compressed(folder/"paths.npz",latent=latent,perceived=perceived,public=public,mi=mi,mj=mj)
            info=dict(truth=dict(truth,r=ratio),theta=raw["theta"],w_i=raw["theta"]/(sigma*sigma*truth["c_i"]),
                w_j=raw["theta"]/(sigma*sigma*truth["c_j"]),seed=seed,policy=policy_name,
                submissions_i=len(ei),submissions_j=len(ej),true_initial_public=truth["mu_0"],
                note="One realization per configuration; diagnostic, not a repeated-sample bias estimate.",
                simulator_sha256=n.digest(n.ROOT/"synthetic_data/synthetic_data.py"))
            (folder/"truth.json").write_text(json.dumps(info,indent=2))
            inputs=dict(data,**a.settings(data,"baseline"))
            # Fit the simulator-consistent timing first; endpoint timing is isolated on real data above.
            inputs["feedback_gap"]=public[:-1].tolist()
            starts=[truth.copy() for _ in range(4)]
            for j in range(4):
                starts[j]["sigma"]*=math.exp((j-1.5)*.1)
                starts[j]["c_i"]*=math.exp((j-1.5)*.1)
            run_fit(m,folder/"interval_start",inputs,starts,warmup=warmup,samples=samples,seed=seed+20)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=("empirical","recovery"),default="empirical")
    p.add_argument("--warmup",type=int,default=1000);p.add_argument("--samples",type=int,default=2000)
    p.add_argument("--contests",nargs="+",type=int,choices=a.IDS)
    args=p.parse_args()
    if args.stage=="empirical": empirical(args.warmup,args.samples)
    else: recovery(args.warmup,args.samples,args.contests)
