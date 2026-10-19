"""Small, isolated Bayesian-detail audit; the leading effort formula is unchanged.

Run with .venv/bin/python -m pipeline.empirical.estimation_detail_audit.
Publication estimates and validated nonlinear results are read-only inputs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import softmax
from cmdstanpy import CmdStanModel, from_csv

from pipeline.config import ROOT
from pipeline.empirical import nonlinear_reestimate as n

OUT = ROOT / "results/validation/estimation_details"
NONLINEAR = ROOT / "results/estimation/nonlinear_top10/fast_v1"
IDS = (4699, 4657, 5390)
VARIANTS = ("baseline", "r_sim", "r_wide", "sigma_wide", "mu_zero", "r_weak", "scale_weak")


def settings(data, variant):
    mean = [0., 0., math.log(2), 0., data["hat_y"][0], math.log(10)]
    sd = [.75, .75, .5, .75, 1., .5]
    floor = .01
    if variant == "r_sim": mean[5], sd[5] = math.log(15), .35
    elif variant == "r_wide": sd[5] = .75
    elif variant == "r_weak": sd[5] = 1.
    elif variant == "scale_weak": sd[0],sd[1],sd[5] = 1.25,1.25,1.
    elif variant == "sigma_wide": sd[2] = 1.
    elif variant == "mu_zero": mean[4] = 0.
    elif variant == "sigma_lower": floor = .001
    elif variant != "baseline": raise ValueError(variant)
    return dict(prior_mean=mean, prior_sd=sd, sigma_floor=floor)


def make_model():
    source = (ROOT / "metakaggle/real_data.stan").read_text()
    source = source.replace("data {\n", "data {\n  vector[6] prior_mean;\n  vector<lower=0>[6] prior_sd;\n  real<lower=0> sigma_floor;\n", 1)
    source = source.replace("real<lower=0.01> sigma;", "real<lower=sigma_floor> sigma;")
    for j, key in enumerate(("c_i", "c_j", "sigma", "lambda"), 1):
        import re
        source, count = re.subn(rf"{key} ~ lognormal\([^;]+;", f"{key} ~ lognormal(prior_mean[{j}], prior_sd[{j}]);", source)
        assert count == 1
    source = source.replace("mu_0 ~ normal(hat_y[1], 1.0);", "mu_0 ~ normal(prior_mean[5], prior_sd[5]);")
    source = source.replace("log_r ~ normal(log(10.0), 0.5);", "log_r ~ normal(prior_mean[6], prior_sd[6]);")
    path = OUT / "detail_model.stan"
    OUT.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != source: raise RuntimeError("Audit source changed; preserve this run")
    if not path.exists(): path.write_text(source)
    return CmdStanModel(stan_file=str(path))


def physical(z):
    return dict(zip(("c_i", "c_j", "sigma", "lambda", "mu_0", "log_r"), np.r_[np.exp(z[:4]), z[4:]].tolist()))


def screen():
    rows = []
    for cid in IDS:
        data, _, z, _ = n.load_baseline(cid)
        base = settings(data, "baseline")
        for variant in VARIANTS:
            cfg = settings(data, variant)
            a, b = np.array(cfg["prior_mean"]), np.array(cfg["prior_sd"])
            a0, b0 = np.array(base["prior_mean"]), np.array(base["prior_sd"])
            lw = -.5*np.sum(((z-a)/b)**2-((z-a0)/b0)**2, axis=1)
            p = softmax(lw)
            w = np.max(data["theta"]/np.exp(2*z[:,2:3]+z[:,:2]), axis=1)
            order = np.argsort(w)
            rows.append(dict(contest_id=cid, variant=variant, weight_ess=float(1/(p@p)),
                max_weight=float(p.max()), screened_max_w_median=float(np.interp(.5,np.cumsum(p[order]),w[order])),
                warning="Screen only; tail extrapolation requires fresh MCMC"))
    write_csv(OUT / "prior_screen.csv", rows)


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def input_audit():
    checks=[]
    for cid in IDS:
        data, _, z, provenance=n.load_baseline(cid)
        y=np.array(data["hat_y"]);changes=np.flatnonzero(np.diff(y)!=0)+1
        events=np.ceil(data["hat_t_i"]+data["hat_t_j"]).astype(int)
        if not np.all(np.isin(changes+1,events)): raise RuntimeError("Endpoint changes outside event intervals")
        point=n.representative_point(z)
        evaluator=n.Likelihood(data)
        before=evaluator.loglik(point,False)
        converted=dict(data,theta=data["theta"]*10)
        after_point=point.copy();after_point[:2]+=math.log(10)
        after=n.Likelihood(converted).loglik(after_point,False)
        if abs(after-before)>1e-7: raise RuntimeError("Monetary-unit likelihood invariance failed")
        checks.append(dict(contest_id=cid,score_changes=len(changes),
            all_changes_at_submission_endpoints=True,monetary_rescaling_loglik_difference=after-before,
            initialization_hours=data["count_start_idx"]-1,
            minimum_submission_hour=min(data["hat_t_i"]+data["hat_t_j"]),
            cost_r_log_correlation=float(np.corrcoef(z[:,0],z[:,5])[0,1]),**provenance))
    (OUT/"input_and_units_audit.json").write_text(json.dumps(checks,indent=2))


def fit_all(warmup, samples, requested=None, *, adapt_delta=.95, run_label=None):
    model = make_model()
    original = CmdStanModel(stan_file=str(ROOT/"metakaggle/real_data.stan"), exe_file=str(ROOT/"metakaggle/real_data"))
    parity = []
    for cid in IDS:
        data, _, z, provenance = n.load_baseline(cid)
        # Baseline target must agree with the existing Stan model at dispersed points.
        differences = []
        for index in (len(z)//7, len(z)//2, 6*len(z)//7):
            params = physical(z[index])
            a = float(original.log_prob(params, data=data, jacobian=False, sig_figs=18).iloc[0,0])
            b = float(model.log_prob(params, data=dict(data, **settings(data,"baseline")), jacobian=False, sig_figs=18).iloc[0,0])
            differences.append(b-a)
        if max(abs(x) for x in differences)>1e-7: raise RuntimeError("Baseline Stan target parity failed")
        parity.append(dict(contest_id=cid, differences=differences))
        variants = VARIANTS + (("sigma_lower",) if cid == 4699 else ())
        for variant in variants:
            if requested is not None and variant not in requested: continue
            number=variants.index(variant)
            tag=variant if run_label is None else variant+"_"+run_label
            folder = OUT/f"contest_{cid}"/tag; folder.mkdir(parents=True, exist_ok=True)
            inputs = dict(data, **settings(data, variant))
            rng = np.random.default_rng(87000+cid)
            starts = [physical(z[i]) for i in rng.choice(len(z),3,replace=False)]
            grid = np.load(NONLINEAR/f"posterior_grid_{cid}.npz")
            weights = softmax(grid["fine"]-grid["base"])
            center = (weights[:,None]*grid["z"]).sum(axis=0)
            # Fourth chain starts in the nonlinear-posterior region to probe initialization sensitivity.
            starts.append(physical(center))
            if variant == "scale_weak":
                modes=json.loads((OUT/"mode_screen.json").read_text())
                candidate=next(row for row in modes if row["contest_id"]==cid and row["variant"]==variant)
                starts[0]=candidate["log_coordinate_mode"]
            config = dict(contest_id=cid,variant=variant,warmup=warmup,samples=samples,
                seed=87000+cid+100*number,metric="dense_e",adapt_delta=adapt_delta,max_treedepth=12,
                model_sha256=n.digest(OUT/"detail_model.stan"), original_model_sha256=n.digest(ROOT/"metakaggle/real_data.stan"),
                inputs=inputs,starts=starts,**provenance)
            if run_label is not None: config["run_label"]=run_label
            signature = folder/"run.json"
            if signature.exists() and json.loads(signature.read_text()) != config: raise RuntimeError("Audit configuration changed")
            signature.write_text(json.dumps(config, indent=2))
            files = sorted(folder.glob("detail_model-*.csv"))
            if len(files) == 4:
                fit = from_csv([str(p) for p in files])
                if fit.num_draws_sampling != samples: raise RuntimeError("Incomplete archived chains")
            elif files: raise RuntimeError("Partial run: preserve files and inspect before retry")
            else:
                fit = model.sample(data=inputs, chains=4, parallel_chains=4, iter_warmup=warmup,
                    iter_sampling=samples, seed=config["seed"], inits=starts, metric="dense_e", adapt_delta=adapt_delta,
                    max_treedepth=12, sig_figs=18, output_dir=str(folder), show_progress=False, refresh=250)
            names=("c_i","c_j","sigma","lambda","mu_0","log_r")
            draws=fit.draws(); values=np.stack([draws[:,:,fit.column_names.index(key)] for key in names],axis=2)
            np.savez_compressed(folder/"posterior.npz", values=values)
            print(f"Completed {cid}/{tag}: {samples} draws x 4 chains",flush=True)
    (OUT/"stan_target_parity.json").write_text(json.dumps(parity,indent=2))


def report():
    rows=[]; diagnostics=[]
    for cid in IDS:
        data, baseline_chains, baseline, _ = n.load_baseline(cid)
        baseline_values=np.array(baseline_chains).transpose(1,0,2)
        nonlinear=np.load(NONLINEAR/f"posterior_grid_{cid}.npz")
        weights=softmax(nonlinear["fine"]-nonlinear["base"])
        targets=nonlinear["z"].copy();targets[:,:4]=np.exp(targets[:,:4]);targets[:,5]=np.exp(targets[:,5])
        variants=("archived",)+tuple(sorted(path.name for path in (OUT/f"contest_{cid}").glob("*") if (path/"posterior.npz").exists()))
        for variant in variants:
            folder=OUT/f"contest_{cid}"/variant
            path=folder/"posterior.npz"
            if variant=="archived": values=baseline_values.copy()
            elif path.exists(): values=np.load(path)["values"]
            else: continue
            values=values.copy();values[:,:,5]=np.exp(values[:,:,5])
            w=data["theta"]/(values[:,:,2,None]**2*values[:,:,:2])
            aug=np.concatenate((values,w,w.max(axis=2,keepdims=True)),axis=2)
            target_w=data["theta"]/(targets[:,2,None]**2*targets[:,:2])
            target_aug=np.column_stack((targets,target_w,target_w.max(axis=1)))
            diag_values=[]
            for k,key in enumerate(n.NAMES+("w_i","w_j","max_w")):
                x=aug[:,:,k];ref=target_aug[:,k];order=np.argsort(ref)
                median=float(np.interp(.5,np.cumsum(weights[order]),ref[order]))
                rh=n.rank_rhat(x.T);ess=n.rank_ess(x.T);diag_values.append((rh,*ess))
                rows.append(dict(contest_id=cid,variant=variant,parameter=key,mean=float(x.mean()),
                    median=float(np.median(x)),sd=float(x.std(ddof=1)),q025=float(np.quantile(x,.025)),q975=float(np.quantile(x,.975)),
                    nonlinear_original_prior_median=median,rank_rhat=rh,bulk_ess=ess[0],tail_ess=ess[1]))
            if variant!="archived":
                fit=from_csv([str(p) for p in sorted(folder.glob("detail_model-*.csv"))])
                method=fit.method_variables()
                diagnostics.append(dict(contest_id=cid,variant=variant,
                    max_rank_rhat=max(t[0] for t in diag_values),min_ess=min(min(t[1:]) for t in diag_values),
                    divergences=int(method["divergent__"].sum()),treedepth_hits=int((method["treedepth__"]>=12).sum()),
                    min_bfmi=float(np.min(np.mean(np.diff(method["energy__"],axis=0)**2,axis=0)/np.var(method["energy__"],axis=0)))))
    write_csv(OUT/"posterior_comparison.csv",rows)
    write_csv(OUT/"sampling_diagnostics.csv",diagnostics)
    print(json.dumps([r for r in rows if r["parameter"]=="max_w"],indent=2),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("all","screen","run","report"), default="all")
    parser.add_argument("--warmup",type=int,default=1000)
    parser.add_argument("--samples",type=int,default=2000)
    parser.add_argument("--variants",nargs="+",choices=VARIANTS+("sigma_lower",))
    parser.add_argument("--adapt-delta",type=float,default=.95)
    parser.add_argument("--run-label")
    parser.add_argument("--contests",type=int,nargs="+",choices=IDS)
    args=parser.parse_args()
    if args.contests is not None: IDS=tuple(args.contests)
    OUT.mkdir(parents=True,exist_ok=True)
    if args.stage in ("all","screen"): screen(); input_audit()
    if args.stage in ("all","run"): fit_all(args.warmup,args.samples,args.variants,adapt_delta=args.adapt_delta,run_label=args.run_label)
    if args.stage in ("all","report"): report()
