"""Paired nonlinear-policy estimation, separate from the publication release.

Only the effort policy changes. Priors, units, hourly recursion, count exposure,
and Gaussian innovations match real_data.stan. Numerical HJB errors are audited
separately; a finite grid is not advertised as an analytic exact equilibrium.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import inspect
import json
import math
import os
import subprocess
import shutil
import time
from pathlib import Path

import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata

from pipeline.config import MAIN_DATA_DIR, MAIN_ESTIMATION_DIR, ROOT, resolve_manifest_path
from pipeline.empirical.estimate import stan_data

OUT = ROOT / "results" / "estimation" / "nonlinear_top10" / "implicit_v1"
NAMES = ("c_i", "c_j", "sigma", "lambda", "mu_0", "r")
SOURCE = Path(__file__).with_name("nonlinear_implicit_kernel.cpp")
PRIOR_SD = np.array([.75, .75, .5, .75, 1., .5])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def read_chain(path):
    with Path(path).open() as f:
        reader = csv.DictReader(line for line in f if not line.startswith("#"))
        return np.array([[float(row[k]) for k in
                         ("c_i", "c_j", "sigma", "lambda", "mu_0", "log_r")]
                        for row in reader])


def load_baseline(cid):
    manifest = MAIN_ESTIMATION_DIR / "accepted_chain_sources.csv"
    with manifest.open() as f:
        row = next(r for r in csv.DictReader(f)
                   if int(r["contest_id"]) == cid and r["mode"] == "full")
    raw_path = MAIN_DATA_DIR / f"contest_{cid}.json"
    if digest(raw_path) != row["input_sha256"]:
        raise RuntimeError("Input provenance mismatch")
    run_dir = resolve_manifest_path(row["run_dir"], manifest)
    raw = json.loads(raw_path.read_text())
    data = stan_data(raw, "full")
    if json.loads((run_dir / "data.json").read_text()) != data:
        raise RuntimeError(f"Accepted Stan input differs for {cid}")
    paths = [run_dir / f"chain_{j}.csv" for j in range(1, 5)]
    chains = [read_chain(p) for p in paths]
    x = np.concatenate(chains)
    z = x.copy()
    z[:, :4] = np.log(z[:, :4])
    return data, chains, z, dict(input_sha256=digest(raw_path),
                                 chain_sha256=[digest(p) for p in paths],
                                 accepted_run_dir=str(run_dir))


def representative_point(z):
    center=np.median(z,axis=0)
    spread=np.maximum(np.quantile(z,.75,axis=0)-np.quantile(z,.25,axis=0),1e-8)
    return z[np.argmin(np.sum(((z-center)/spread)**2,axis=1))].copy()


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    with (MAIN_ESTIMATION_DIR / "accepted_chain_sources.csv").open() as f:
        ids = sorted({int(r["contest_id"]) for r in csv.DictReader(f) if r["mode"] == "full"})
    rows = []
    for cid in ids:
        data, _, z, provenance = load_baseline(cid)
        x = np.exp(z[:, :4])
        w = data["theta"] / (x[:, 2:3]**2 * x[:, :2])
        wm = np.max(w, axis=1)
        rows.append(dict(contest_id=cid, median_max_w=float(np.median(wm)),
                         q025_max_w=float(np.quantile(wm,.025)),
                         q975_max_w=float(np.quantile(wm,.975)), **provenance))
    rows.sort(key=lambda r: (-r["median_max_w"], r["contest_id"]))
    save_json(OUT / "selection.json", dict(
        rule="Top 10 by posterior median of max(theta/(sigma^2*c_i),theta/(sigma^2*c_j)); full accepted posterior, ties by ID.",
        manifest_sha256=digest(MAIN_ESTIMATION_DIR / "accepted_chain_sources.csv"),
        baseline_model_sha256=digest(ROOT / "metakaggle" / "real_data.stan"),
        all_ranked=rows, selected=rows[:10]))
    print(json.dumps([{k:r[k] for k in ("contest_id","median_max_w")} for r in rows[:10]],indent=2))


def kernel():
    cache = ROOT / ".cache" / "nonlinear_reestimate"
    cache.mkdir(parents=True, exist_ok=True)
    library = cache / (digest(SOURCE)[:16] + ".so")
    if not library.exists():
        temporary = library.with_name(library.name + f".{os.getpid()}.tmp")
        subprocess.run(["c++", "-O3", "-std=c++17", "-shared", "-fPIC",
                        str(SOURCE), "-o", str(temporary)], check=True)
        temporary.replace(library)
    lib = ctypes.CDLL(str(library))
    double_array = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    int_array = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    int_pointer = ctypes.POINTER(ctypes.c_int)
    lib.solve_hjb.argtypes = [ctypes.c_double]*4 + [ctypes.c_int, ctypes.c_double, double_array, int_pointer]
    lib.solve_hjb.restype = ctypes.c_int
    lib.empirical_likelihood.argtypes = [ctypes.c_int]+[ctypes.c_double]*9+[
        double_array,int_array,int_array,ctypes.c_int,ctypes.c_void_p,
        ctypes.c_int,ctypes.c_int,ctypes.c_double,int_pointer]
    lib.empirical_likelihood.restype = ctypes.c_double
    return lib


class Likelihood:
    def __init__(self,data,dy=.025,nt=512,extent=7):
        self.data=data; self.dy=dy; self.nt=nt; self.extent=extent
        self.lib=kernel(); self.calls=0; self.hjb_steps=0
        self.held=np.ascontiguousarray(data["hat_y"],dtype=np.float64)
        self.counts=[]
        for key in ("hat_t_i","hat_t_j"):
            count=np.zeros(data["N_Delta"],dtype=np.int32)
            for t in data[key]: count[max(1,math.ceil(t))-1]+=1
            self.counts.append(count)
        self.prior_mean=np.array([0,0,math.log(2),0,self.held[0],math.log(10)])

    def prior(self,z):
        if np.any(z[[0,1,2,3,5]]<math.log(.01)): return -np.inf
        return float(-.5*np.sum(((z-self.prior_mean)/PRIOR_SD)**2)
                     -np.log(PRIOR_SD).sum()-3*math.log(2*math.pi))

    def policy(self,z):
        ci,cj,sig,lam=np.exp(z[:4]); T=self.data["T_contest"]
        h=1/(sig*T*math.sqrt(lam)); theta=self.data["theta"]
        wi,wj=theta/(sig*sig*ci),theta/(sig*sig*cj)
        # Resolve the terminal Gaussian; no candidate-dependent rounding of parameters.
        dy=min(self.dy,.3*math.sqrt(h))
        extent=max(self.extent,(abs(z[4])/(sig*math.sqrt(T))+3)/math.sqrt(1+h))
        for attempt in range(3):
            ny=2*math.ceil(extent*math.sqrt(1+h)/dy)+1
            if ny*(self.nt+1)>50000000:
                raise RuntimeError("HJB allocation too large; numerical audit required")
            policy=np.empty((self.nt+1,ny,2),dtype=np.float64)
            steps=ctypes.c_int()
            status=self.lib.solve_hjb(wi,wj,h,dy,self.nt,extent,policy,ctypes.byref(steps))
            self.hjb_steps+=steps.value
            if status==ny: return policy,dy
            if status==-2: raise RuntimeError("HJB step limit; do not truncate posterior")
            dy/=2
        raise RuntimeError("Non-monotone HJB solution after refinement; do not truncate posterior")

    def loglik(self,z,nonlinear=True):
        ci,cj,sig,lam=np.exp(z[:4]); d=self.data
        policy,dy=self.policy(z) if nonlinear else (None,self.dy)
        status=ctypes.c_int()
        value=self.lib.empirical_likelihood(d["N_Delta"],d["Delta2f"],d["T_contest"],
            d["theta"],ci,cj,sig,lam,float(z[4]),float(z[5]),self.held,*self.counts,
            d["count_start_idx"]-1,None if policy is None else policy.ctypes.data,
            0 if policy is None else policy.shape[1],self.nt,dy,ctypes.byref(status))
        self.calls+=1
        if status.value: raise RuntimeError(f"Likelihood numerical failure {status.value}")
        return value

    def posterior(self,z):
        prior=self.prior(z)
        return prior+self.loglik(z) if np.isfinite(prior) else -np.inf


def audit(cid):
    import cmdstanpy
    from pipeline.synthetic.nonlinear import Primitives, solve_equilibrium
    data,_,z,provenance=load_baseline(cid); center=representative_point(z)
    center_source='representative accepted baseline posterior draw'
    exploratory=OUT/f'explore_{cid}.json'
    if exploratory.exists():
        candidate=json.loads(exploratory.read_text())
        if candidate['status']=='exploratory_only' and candidate['kernel_sha256']==digest(SOURCE):
            center=np.array(candidate['best']['z'])
            center_source='exploratory nonlinear mode; posterior-region audit still required'
    model=cmdstanpy.CmdStanModel(stan_file=str(ROOT/"metakaggle"/"real_data.stan"),
                               exe_file=str(ROOT/"metakaggle"/"real_data"))
    L=Likelihood(data)
    paired=[]
    for shift in (np.zeros(6),np.array([.05,-.05,.03,.08,.01,-.04])):
        zz=center+shift
        params={k:float(v) for k,v in zip(("c_i","c_j","sigma","lambda"),np.exp(zz[:4]))}
        params.update(mu_0=float(zz[4]),log_r=float(zz[5]))
        reference=float(model.log_prob(params,data=data,jacobian=False,sig_figs=18).iloc[0,0])
        ours=L.loglik(zz,False)+L.prior(zz)-float(zz[:4].sum())
        paired.append((reference,ours))
    delta_error=(paired[1][0]-paired[0][0])-(paired[1][1]-paired[0][1])
    if abs(delta_error)>1e-7: raise RuntimeError(f"Stan parity failed: {delta_error}")
    # Independent production solver comparison in a smooth, weak-incentive case.
    p=Primitives(T=1,theta=.001,c_i=10,c_j=12,sigma=1,lamb=4,S0=.5)
    reference=solve_equilibrium(p,dy=.05,dt=1/2000,extent=7)
    lib=L.lib; ny=2*math.ceil(7*math.sqrt(1.5)/.05)+1
    values=np.empty((513,ny,2)); work=ctypes.c_int()
    if lib.solve_hjb(.0001,.001/12,.5,.05,512,7,values,ctypes.byref(work))!=ny:
        raise RuntimeError("Kernel test failed")
    max_policy_error=0
    for t in (0,.25,.5,.75):
        for y in (-1,0,1):
            pos=y/.05+(ny-1)/2; j=int(pos); f=pos-j
            actual=(1-f)*values[int(t*512),j]+f*values[int(t*512),j+1]
            max_policy_error=max(max_policy_error,float(np.max(np.abs(actual-reference.at(t,y)))))
    if max_policy_error>1e-7: raise RuntimeError(f"Independent weak-incentive HJB check failed: {max_policy_error}")
    refinements=[]
    audit_path=OUT/f"audit_{cid}.json"
    for dy,nt,extent in ((.05,512,7),(.025,1024,7),(.0125,2048,7),(.0125,2048,9)):
        evaluator=Likelihood(data,dy,nt,extent); start=time.perf_counter()
        ll=evaluator.loglik(center)
        refinements.append(dict(dy=dy,nt=nt,extent=extent,loglik=ll,
                                seconds=time.perf_counter()-start,hjb_steps=evaluator.hjb_steps))
        OUT.mkdir(parents=True,exist_ok=True)
        save_json(audit_path,dict(contest_id=cid,kernel_sha256=digest(SOURCE),**provenance,
              status="in_progress",stan_logdensity_difference_error=delta_error,
              independent_solver_max_policy_error=max_policy_error,center_source=center_source,
              center_z=center.tolist(),refinements=refinements))
        print(f"{cid}: grid ({dy}, {nt}, {extent}), loglik={ll:.6f}, seconds={refinements[-1]['seconds']:.2f}",flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    spatial_change=abs(refinements[-2]['loglik']-refinements[-3]['loglik'])
    boundary_change=abs(refinements[-1]['loglik']-refinements[-2]['loglik'])
    save_json(audit_path,dict(contest_id=cid,kernel_sha256=digest(SOURCE),**provenance,
              status="complete",grid_loglik_change=spatial_change,boundary_loglik_change=boundary_change,
              initial_grid_check_passed=spatial_change<.1 and boundary_change<.1,
              stan_logdensity_difference_error=delta_error,
              independent_solver_max_policy_error=max_policy_error,center_source=center_source,
              center_z=center.tolist(),refinements=refinements))
    print(json.dumps(refinements,indent=2))


def sample(cid,chain,warmup,samples,dy,nt,seed,pilot=False):
    audit_path=OUT/f"audit_{cid}.json"
    if not audit_path.exists() and not pilot:raise RuntimeError("Run numerical audit before sampling")
    numerical=json.loads(audit_path.read_text()) if audit_path.exists() else {}
    if not pilot and (numerical.get('kernel_sha256')!=digest(SOURCE) or not numerical.get('initial_grid_check_passed',False)):
        raise RuntimeError("Numerical grid check unresolved; posterior comparison is not yet valid")
    if not pilot and (dy>.0125 or nt<2048):
        raise RuntimeError("Sampling grid must be at least as fine as the audited reference")
    data,_,baseline,provenance=load_baseline(cid)
    folder=OUT/f"contest_{cid}";folder.mkdir(parents=True,exist_ok=True)
    dest=folder/f"chain_{chain}.npz"; checkpoint=folder/f"checkpoint_{chain}.npz"
    proposal_path=OUT/f'proposal_{cid}.json'
    if not proposal_path.exists():raise RuntimeError('Calibrate a nonlinear proposal before sampling')
    sampler_hash=hashlib.sha256((inspect.getsource(sample)+inspect.getsource(Likelihood)+inspect.getsource(kernel)).encode()).hexdigest()
    config=dict(contest_id=cid,chain=chain,warmup=warmup,samples=samples,dy=dy,nt=nt,
                seed=seed,pilot_requires_posterior_grid_validation=pilot,
                proposal_sha256=digest(proposal_path),sampler_sha256=sampler_hash,
                kernel_sha256=digest(SOURCE),**provenance)
    settings=folder/f"settings_{chain}.json"
    if settings.exists() and json.loads(settings.read_text())!=config:
        raise RuntimeError("Existing settings differ; use a separate output directory")
    save_json(settings,config)
    if dest.exists(): print(f"{cid}/{chain}: complete",flush=True);return
    rng=np.random.default_rng(seed+chain)
    evaluator=Likelihood(data,dy,nt)
    base_cov=np.cov(baseline,rowvar=False)+np.eye(6)*1e-8
    proposal_info=json.loads(proposal_path.read_text())
    if proposal_info['kernel_sha256']!=digest(SOURCE):raise RuntimeError('Proposal kernel hash differs')
    mode=np.array(proposal_info['mode_z']);proposal_cov=np.array(proposal_info['covariance'])*1.5
    proposal_chol=np.linalg.cholesky(proposal_cov);proposal_precision=np.linalg.inv(proposal_cov)
    def logq(x):return -5.5*math.log1p(float((x-mode)@proposal_precision@(x-mode))/5)
    def screen(x):return -.75*float((x-mode)@proposal_precision@(x-mode))
    if checkpoint.exists():
        saved=np.load(checkpoint,allow_pickle=False)
        z=saved['z'];logp=float(saved['logp']);accepted=int(saved['accepted']);iteration=int(saved['iteration'])
        history=list(saved['history']);records=list(saved['records']);scale=float(saved['scale']);cov=saved['cov']
        rng.bit_generator.state=json.loads(str(saved['rng']))
    else:
        # Dispersed actual baseline posterior draws, not a shared posterior mean.
        z=mode.copy() if chain==1 else mode+.5*proposal_chol@rng.normal(size=6)
        logp=evaluator.posterior(z)
        accepted=0;iteration=0;history=[];records=[];scale=1.;cov=proposal_cov
    start=time.perf_counter();block_accepted=0;screen_rejections=0
    for i in range(iteration,warmup+samples):
        independent=rng.uniform()<.7
        proposal=(mode+proposal_chol@rng.normal(size=6)*math.sqrt(5/rng.chisquare(5))) if independent else (
            z+rng.multivariate_normal(np.zeros(6),cov)*(2.38/math.sqrt(6))*scale)
        correction=logq(z)-logq(proposal) if independent else 0.
        approximate_difference=screen(proposal)-screen(z) if independent else 0.
        # Delayed acceptance: the first stage includes the proposal ratio;
        # the exact second stage removes the screening density ratio. Their
        # product obeys detailed balance for the unchanged nonlinear target.
        screened=independent and math.log(rng.uniform())>=approximate_difference+correction
        if screened:
            screen_rejections+=1;threshold=-np.inf
        else:
            proposed=evaluator.posterior(proposal)
            threshold=proposed-logp-approximate_difference
        if math.log(rng.uniform())<threshold:
            z=proposal;logp=proposed;accepted+=1;block_accepted+=1
        if i<warmup:
            history.append(z.copy())
            if (i+1)%100==0:
                rate=block_accepted/100
                scale*=math.exp(max(-.3,min(.3,rate-.234)))
                if i>=199:
                    empirical=np.cov(np.array(history)[len(history)//2:],rowvar=False)
                    cov=.9*empirical+.1*proposal_cov+np.eye(6)*1e-8
        else: records.append(np.r_[z,logp])
        if (i+1)%100==0 or i+1==warmup+samples:
            np.savez_compressed(checkpoint,z=z,logp=logp,accepted=accepted,iteration=i+1,
                history=np.asarray(history),records=np.asarray(records),scale=scale,cov=cov,
                rng=json.dumps(rng.bit_generator.state))
            print(f"{cid}/{chain}: {i+1}/{warmup+samples}, acceptance={accepted/(i+1):.3f}, "
                  f"logp={logp:.3f}, seconds={time.perf_counter()-start:.1f}",flush=True)
            block_accepted=0
    np.savez_compressed(dest,z=np.asarray(records)[:,:6],logp=np.asarray(records)[:,6])
    save_json(folder/f"diagnostics_{chain}.json",dict(iterations=warmup+samples,
        acceptance=accepted/(warmup+samples),likelihood_calls=evaluator.calls,
        cheap_screen_rejections_since_resume=screen_rejections,
        elapsed_seconds=time.perf_counter()-start))


def sample_prior_screen(cid,chain,warmup,samples,dy,nt,seed,pilot=False):
    audit_path=OUT/f"audit_{cid}.json"
    if not audit_path.exists() and not pilot:raise RuntimeError("Run numerical audit before sampling")
    numerical=json.loads(audit_path.read_text()) if audit_path.exists() else {}
    if not pilot and (numerical.get('kernel_sha256')!=digest(SOURCE) or not numerical.get('initial_grid_check_passed',False)):
        raise RuntimeError("Numerical grid check unresolved; posterior comparison is not yet valid")
    if not pilot and (dy>.0125 or nt<2048):
        raise RuntimeError("Sampling grid must be at least as fine as the audited reference")
    data,_,baseline,provenance=load_baseline(cid)
    folder=OUT/f"contest_{cid}";folder.mkdir(parents=True,exist_ok=True)
    dest=folder/f"chain_{chain}.npz"; checkpoint=folder/f"checkpoint_{chain}.npz"
    proposal_path=OUT/f'proposal_{cid}.json'
    if not proposal_path.exists():raise RuntimeError('Calibrate a nonlinear proposal before sampling')
    sampler_hash=hashlib.sha256((inspect.getsource(sample_prior_screen)+inspect.getsource(Likelihood)+inspect.getsource(kernel)).encode()).hexdigest()
    config=dict(contest_id=cid,chain=chain,warmup=warmup,samples=samples,dy=dy,nt=nt,
                seed=seed,pilot_requires_posterior_grid_validation=pilot,sampler_mode="prior-screen",
                proposal_sha256=digest(proposal_path),sampler_sha256=sampler_hash,
                kernel_sha256=digest(SOURCE),**provenance)
    settings=folder/f"settings_{chain}.json"
    if settings.exists() and json.loads(settings.read_text())!=config:
        raise RuntimeError("Existing settings differ; use a separate output directory")
    save_json(settings,config)
    if dest.exists(): print(f"{cid}/{chain}: complete",flush=True);return
    rng=np.random.default_rng(seed+chain)
    evaluator=Likelihood(data,dy,nt)
    base_cov=np.cov(baseline,rowvar=False)+np.eye(6)*1e-8
    proposal_info=json.loads(proposal_path.read_text())
    if proposal_info['kernel_sha256']!=digest(SOURCE):raise RuntimeError('Proposal kernel hash differs')
    mode=np.array(proposal_info['mode_z']);proposal_cov=np.array(proposal_info['covariance'])*1.5
    proposal_chol=np.linalg.cholesky(proposal_cov);proposal_precision=np.linalg.inv(proposal_cov)
    def logq(x):return -5.5*math.log1p(float((x-mode)@proposal_precision@(x-mode))/5)
    def screen(x):return evaluator.prior(x)
    if checkpoint.exists():
        saved=np.load(checkpoint,allow_pickle=False)
        z=saved['z'];logp=float(saved['logp']);accepted=int(saved['accepted']);iteration=int(saved['iteration'])
        history=list(saved['history']);records=list(saved['records']);scale=float(saved['scale']);cov=saved['cov']
        rng.bit_generator.state=json.loads(str(saved['rng']))
    else:
        # Independently seeded starts around the fixed nonlinear proposal center.
        z=mode.copy() if chain==1 else mode+.5*proposal_chol@rng.normal(size=6)
        logp=evaluator.posterior(z)
        accepted=0;iteration=0;history=[];records=[];scale=1.;cov=proposal_cov
    start=time.perf_counter();block_accepted=0;screen_rejections=0
    for i in range(iteration,warmup+samples):
        independent=rng.uniform()<.7
        proposal=(mode+proposal_chol@rng.normal(size=6)*math.sqrt(5/rng.chisquare(5))) if independent else (
            z+rng.multivariate_normal(np.zeros(6),cov)*(2.38/math.sqrt(6))*scale)
        correction=logq(z)-logq(proposal) if independent else 0.
        approximate_difference=screen(proposal)-screen(z) if independent else 0.
        # Delayed acceptance: the first stage includes the proposal ratio;
        # the exact second stage removes the screening density ratio. Their
        # product obeys detailed balance for the unchanged nonlinear target.
        screened=independent and math.log(rng.uniform())>=approximate_difference+correction
        if screened:
            screen_rejections+=1;threshold=-np.inf
        else:
            proposed=evaluator.posterior(proposal)
            threshold=proposed-logp-approximate_difference
        if math.log(rng.uniform())<threshold:
            z=proposal;logp=proposed;accepted+=1;block_accepted+=1
        if i<warmup:
            history.append(z.copy())
            if (i+1)%100==0:
                rate=block_accepted/100
                scale*=math.exp(max(-.3,min(.3,rate-.234)))
                if i>=199:
                    empirical=np.cov(np.array(history)[len(history)//2:],rowvar=False)
                    cov=.9*empirical+.1*proposal_cov+np.eye(6)*1e-8
        else: records.append(np.r_[z,logp])
        if (i+1)%100==0 or i+1==warmup+samples:
            np.savez_compressed(checkpoint,z=z,logp=logp,accepted=accepted,iteration=i+1,
                history=np.asarray(history),records=np.asarray(records),scale=scale,cov=cov,
                rng=json.dumps(rng.bit_generator.state))
            print(f"{cid}/{chain}: {i+1}/{warmup+samples}, acceptance={accepted/(i+1):.3f}, "
                  f"logp={logp:.3f}, seconds={time.perf_counter()-start:.1f}",flush=True)
            block_accepted=0
    np.savez_compressed(dest,z=np.asarray(records)[:,:6],logp=np.asarray(records)[:,6])
    save_json(folder/f"diagnostics_{chain}.json",dict(iterations=warmup+samples,
        acceptance=accepted/(warmup+samples),likelihood_calls=evaluator.calls,
        cheap_screen_rejections_since_resume=screen_rejections,
        elapsed_seconds=time.perf_counter()-start))


class StudentMixture:
    """Fixed multivariate t mixture; normalized component densities matter."""
    def __init__(self,components):
        self.weights=np.array([c['weight'] for c in components]);self.weights/=self.weights.sum()
        self.means=np.array([c['mean'] for c in components])
        self.covs=np.array([c['covariance'] for c in components])*1.5
        self.chols=np.linalg.cholesky(self.covs);self.precisions=np.linalg.inv(self.covs)
        self.log_norm=np.log(self.weights)-.5*np.linalg.slogdet(self.covs)[1]
        self.dimension=self.means.shape[1]
    def logpdf(self,x):
        from scipy.special import logsumexp
        delta=x-self.means
        mahal=np.einsum('ki,kij,kj->k',delta,self.precisions,delta)
        return float(logsumexp(self.log_norm-(5+self.dimension)/2*np.log1p(mahal/5)))
    def draw(self,rng):
        k=rng.choice(len(self.weights),p=self.weights)
        return self.means[k]+self.chols[k]@rng.normal(size=self.dimension)*math.sqrt(5/rng.chisquare(5))


def sample_mixture_screen(cid,chain,warmup,samples,dy,nt,seed,pilot=False):
    audit_path=OUT/f"audit_{cid}.json"
    if not audit_path.exists() and not pilot:raise RuntimeError("Run numerical audit before sampling")
    numerical=json.loads(audit_path.read_text()) if audit_path.exists() else {}
    if not pilot and (numerical.get('kernel_sha256')!=digest(SOURCE) or not numerical.get('initial_grid_check_passed',False)):
        raise RuntimeError("Numerical grid check unresolved; posterior comparison is not yet valid")
    if not pilot and (dy>.0125 or nt<2048):
        raise RuntimeError("Sampling grid must be at least as fine as the audited reference")
    data,_,baseline,provenance=load_baseline(cid)
    folder=OUT/f"contest_{cid}";folder.mkdir(parents=True,exist_ok=True)
    dest=folder/f"chain_{chain}.npz"; checkpoint=folder/f"checkpoint_{chain}.npz"
    proposal_path=OUT/f'proposal_{cid}.json'
    if not proposal_path.exists():raise RuntimeError('Calibrate a nonlinear proposal before sampling')
    sampler_hash=hashlib.sha256((inspect.getsource(sample_mixture_screen)+inspect.getsource(StudentMixture)+inspect.getsource(Likelihood)+inspect.getsource(kernel)).encode()).hexdigest()
    config=dict(contest_id=cid,chain=chain,warmup=warmup,samples=samples,dy=dy,nt=nt,
                seed=seed,pilot_requires_posterior_grid_validation=pilot,sampler_mode="mixture-prior-screen",
                proposal_sha256=digest(proposal_path),sampler_sha256=sampler_hash,
                kernel_sha256=digest(SOURCE),**provenance)
    settings=folder/f"settings_{chain}.json"
    if settings.exists() and json.loads(settings.read_text())!=config:
        raise RuntimeError("Existing settings differ; use a separate output directory")
    save_json(settings,config)
    if dest.exists(): print(f"{cid}/{chain}: complete",flush=True);return
    rng=np.random.default_rng(seed+chain)
    evaluator=Likelihood(data,dy,nt)
    base_cov=np.cov(baseline,rowvar=False)+np.eye(6)*1e-8
    proposal_info=json.loads(proposal_path.read_text())
    if proposal_info['kernel_sha256']!=digest(SOURCE):raise RuntimeError('Proposal kernel hash differs')
    mode=np.array(proposal_info['mode_z']);proposal_cov=np.array(proposal_info['covariance'])*1.5
    proposal_chol=np.linalg.cholesky(proposal_cov);proposal_precision=np.linalg.inv(proposal_cov)
    mixture=StudentMixture(proposal_info["mixture_components"])
    logq=mixture.logpdf
    def screen(x):return evaluator.prior(x)
    if checkpoint.exists():
        saved=np.load(checkpoint,allow_pickle=False)
        z=saved['z'];logp=float(saved['logp']);accepted=int(saved['accepted']);iteration=int(saved['iteration'])
        history=list(saved['history']);records=list(saved['records']);scale=float(saved['scale']);cov=saved['cov']
        rng.bit_generator.state=json.loads(str(saved['rng']))
    else:
        # Independently seeded starts around the fixed nonlinear proposal center.
        z=mixture.means[(chain-1)%len(mixture.means)]+.3*mixture.chols[(chain-1)%len(mixture.means)]@rng.normal(size=6)
        logp=evaluator.posterior(z)
        accepted=0;iteration=0;history=[];records=[];scale=1.;cov=proposal_cov
    start=time.perf_counter();block_accepted=0;screen_rejections=0
    for i in range(iteration,warmup+samples):
        independent=rng.uniform()<.7
        proposal=mixture.draw(rng) if independent else (
            z+rng.multivariate_normal(np.zeros(6),cov)*(2.38/math.sqrt(6))*scale)
        correction=logq(z)-logq(proposal) if independent else 0.
        approximate_difference=screen(proposal)-screen(z) if independent else 0.
        # Delayed acceptance: the first stage includes the proposal ratio;
        # the exact second stage removes the screening density ratio. Their
        # product obeys detailed balance for the unchanged nonlinear target.
        screened=independent and math.log(rng.uniform())>=approximate_difference+correction
        if screened:
            screen_rejections+=1;threshold=-np.inf
        else:
            proposed=evaluator.posterior(proposal)
            threshold=proposed-logp-approximate_difference
        if math.log(rng.uniform())<threshold:
            z=proposal;logp=proposed;accepted+=1;block_accepted+=1
        if i<warmup:
            history.append(z.copy())
            if (i+1)%100==0:
                rate=block_accepted/100
                scale*=math.exp(max(-.3,min(.3,rate-.234)))
                if i>=199:
                    empirical=np.cov(np.array(history)[len(history)//2:],rowvar=False)
                    cov=.9*empirical+.1*proposal_cov+np.eye(6)*1e-8
        else: records.append(np.r_[z,logp])
        if (i+1)%100==0 or i+1==warmup+samples:
            np.savez_compressed(checkpoint,z=z,logp=logp,accepted=accepted,iteration=i+1,
                history=np.asarray(history),records=np.asarray(records),scale=scale,cov=cov,
                rng=json.dumps(rng.bit_generator.state))
            print(f"{cid}/{chain}: {i+1}/{warmup+samples}, acceptance={accepted/(i+1):.3f}, "
                  f"logp={logp:.3f}, seconds={time.perf_counter()-start:.1f}",flush=True)
            block_accepted=0
    np.savez_compressed(dest,z=np.asarray(records)[:,:6],logp=np.asarray(records)[:,6])
    save_json(folder/f"diagnostics_{chain}.json",dict(iterations=warmup+samples,
        acceptance=accepted/(warmup+samples),likelihood_calls=evaluator.calls,
        cheap_screen_rejections_since_resume=screen_rejections,
        elapsed_seconds=time.perf_counter()-start))


def gaussian_mixture_logpdf(mixture,x):
    """Frozen two-region posterior surrogate, distinct from the prior."""
    from scipy.special import logsumexp
    delta=x-mixture.means
    mahal=np.einsum('ki,kij,kj->k',delta,mixture.precisions,delta)
    return float(logsumexp(mixture.log_norm-.5*mahal))


def sample_mixture_gaussian(cid,chain,warmup,samples,dy,nt,seed,pilot=False):
    audit_path=OUT/f"audit_{cid}.json"
    if not audit_path.exists() and not pilot:raise RuntimeError("Run numerical audit before sampling")
    numerical=json.loads(audit_path.read_text()) if audit_path.exists() else {}
    if not pilot and (numerical.get('kernel_sha256')!=digest(SOURCE) or not numerical.get('initial_grid_check_passed',False)):
        raise RuntimeError("Numerical grid check unresolved; posterior comparison is not yet valid")
    if not pilot and (dy>.0125 or nt<2048):
        raise RuntimeError("Sampling grid must be at least as fine as the audited reference")
    data,_,baseline,provenance=load_baseline(cid)
    folder=OUT/f"contest_{cid}";folder.mkdir(parents=True,exist_ok=True)
    dest=folder/f"chain_{chain}.npz"; checkpoint=folder/f"checkpoint_{chain}.npz"
    proposal_path=OUT/f'proposal_{cid}.json'
    if not proposal_path.exists():raise RuntimeError('Calibrate a nonlinear proposal before sampling')
    sampler_hash=hashlib.sha256((inspect.getsource(sample_mixture_gaussian)+inspect.getsource(StudentMixture)+inspect.getsource(gaussian_mixture_logpdf)+inspect.getsource(Likelihood)+inspect.getsource(kernel)).encode()).hexdigest()
    config=dict(contest_id=cid,chain=chain,warmup=warmup,samples=samples,dy=dy,nt=nt,
                seed=seed,pilot_requires_posterior_grid_validation=pilot,sampler_mode="mixture-gaussian-screen",
                proposal_sha256=digest(proposal_path),sampler_sha256=sampler_hash,
                kernel_sha256=digest(SOURCE),**provenance)
    settings=folder/f"settings_{chain}.json"
    if settings.exists() and json.loads(settings.read_text())!=config:
        raise RuntimeError("Existing settings differ; use a separate output directory")
    save_json(settings,config)
    if dest.exists(): print(f"{cid}/{chain}: complete",flush=True);return
    rng=np.random.default_rng(seed+chain)
    evaluator=Likelihood(data,dy,nt)
    base_cov=np.cov(baseline,rowvar=False)+np.eye(6)*1e-8
    proposal_info=json.loads(proposal_path.read_text())
    if proposal_info['kernel_sha256']!=digest(SOURCE):raise RuntimeError('Proposal kernel hash differs')
    mode=np.array(proposal_info['mode_z']);proposal_cov=np.array(proposal_info['covariance'])*1.5
    proposal_chol=np.linalg.cholesky(proposal_cov);proposal_precision=np.linalg.inv(proposal_cov)
    mixture=StudentMixture(proposal_info["mixture_components"])
    logq=mixture.logpdf
    # A frozen Gaussian mixture covers both pilot regions; the exact
    # second stage retains the original posterior rather than this screen.
    def screen(x):return gaussian_mixture_logpdf(mixture,x)
    if checkpoint.exists():
        saved=np.load(checkpoint,allow_pickle=False)
        z=saved['z'];logp=float(saved['logp']);accepted=int(saved['accepted']);iteration=int(saved['iteration'])
        history=list(saved['history']);records=list(saved['records']);scale=float(saved['scale']);cov=saved['cov']
        rng.bit_generator.state=json.loads(str(saved['rng']))
    else:
        # Independently seeded starts around the fixed nonlinear proposal center.
        z=mixture.means[(chain-1)%len(mixture.means)]+.3*mixture.chols[(chain-1)%len(mixture.means)]@rng.normal(size=6)
        logp=evaluator.posterior(z)
        accepted=0;iteration=0;history=[];records=[];scale=1.;cov=proposal_cov
    start=time.perf_counter();block_accepted=0;screen_rejections=0
    for i in range(iteration,warmup+samples):
        independent=rng.uniform()<.7
        proposal=mixture.draw(rng) if independent else (
            z+rng.multivariate_normal(np.zeros(6),cov)*(2.38/math.sqrt(6))*scale)
        correction=logq(z)-logq(proposal) if independent else 0.
        approximate_difference=screen(proposal)-screen(z) if independent else 0.
        # Delayed acceptance: the first stage includes the proposal ratio;
        # the exact second stage removes the screening density ratio. Their
        # product obeys detailed balance for the unchanged nonlinear target.
        screened=independent and math.log(rng.uniform())>=approximate_difference+correction
        if screened:
            screen_rejections+=1;threshold=-np.inf
        else:
            proposed=evaluator.posterior(proposal)
            threshold=proposed-logp-approximate_difference
        if math.log(rng.uniform())<threshold:
            z=proposal;logp=proposed;accepted+=1;block_accepted+=1
        if i<warmup:
            history.append(z.copy())
            if (i+1)%100==0:
                rate=block_accepted/100
                scale*=math.exp(max(-.3,min(.3,rate-.234)))
                if i>=199:
                    empirical=np.cov(np.array(history)[len(history)//2:],rowvar=False)
                    cov=.9*empirical+.1*proposal_cov+np.eye(6)*1e-8
        else: records.append(np.r_[z,logp])
        if (i+1)%100==0 or i+1==warmup+samples:
            np.savez_compressed(checkpoint,z=z,logp=logp,accepted=accepted,iteration=i+1,
                history=np.asarray(history),records=np.asarray(records),scale=scale,cov=cov,
                rng=json.dumps(rng.bit_generator.state))
            print(f"{cid}/{chain}: {i+1}/{warmup+samples}, acceptance={accepted/(i+1):.3f}, "
                  f"logp={logp:.3f}, seconds={time.perf_counter()-start:.1f}",flush=True)
            block_accepted=0
    np.savez_compressed(dest,z=np.asarray(records)[:,:6],logp=np.asarray(records)[:,6])
    save_json(folder/f"diagnostics_{chain}.json",dict(iterations=warmup+samples,
        acceptance=accepted/(warmup+samples),likelihood_calls=evaluator.calls,
        cheap_screen_rejections_since_resume=screen_rejections,
        elapsed_seconds=time.perf_counter()-start))


def refine(cid,dy,nt):
    """Continue an audit at exactly the saved joint parameter point."""
    path=OUT/f"audit_{cid}.json"
    audit_record=json.loads(path.read_text())
    data,_,_,provenance=load_baseline(cid)
    if audit_record['kernel_sha256']!=digest(SOURCE):
        raise RuntimeError("Kernel changed: rerun the audit before refinement")
    for key,value in provenance.items():
        if audit_record[key]!=value:raise RuntimeError("Baseline provenance changed")
    start=time.perf_counter();evaluator=Likelihood(data,dy,nt,7)
    row=dict(dy=dy,nt=nt,extent=7)
    try:
        row['loglik']=evaluator.loglik(np.array(audit_record['center_z']))
    except RuntimeError as error:
        row['error']=str(error)
    row.update(seconds=time.perf_counter()-start,hjb_steps=evaluator.hjb_steps)
    audit_record.setdefault('extra_refinements',[]).append(row)
    save_json(path,audit_record)
    print(json.dumps(row,indent=2))


def explore(cid,dy,nt):
    """Locate a candidate nonlinear mode; this is not posterior estimation."""
    from scipy.optimize import minimize
    data,_,baseline,provenance=load_baseline(cid)
    evaluator=Likelihood(data,dy,nt)
    center=representative_point(baseline);history=[];start=time.perf_counter()
    def objective(z):
        value=evaluator.posterior(z)
        history.append(dict(z=z.tolist(),logposterior=float(value) if np.isfinite(value) else None))
        if len(history)%50==0:
            finite=[r for r in history if r['logposterior'] is not None]
            best=max(finite,key=lambda r:r['logposterior'])
            save_json(OUT/f'explore_{cid}.json',dict(status='in_progress',contest_id=cid,
                dy=dy,nt=nt,kernel_sha256=digest(SOURCE),**provenance,best=best,history=history))
            print(f'{cid}: evaluations={len(history)}, best={best["logposterior"]:.4f}',flush=True)
        return -value
    simplex=np.tile(center,(7,1))
    for j in range(6):simplex[j+1,j]+=.12 if j!=4 else .02
    result=minimize(objective,center,method='Nelder-Mead',options=dict(
        initial_simplex=simplex,maxfev=700,xatol=.002,fatol=.01))
    finite=[r for r in history if r['logposterior'] is not None]
    best=max(finite,key=lambda r:r['logposterior'])
    save_json(OUT/f'explore_{cid}.json',dict(status='exploratory_only',contest_id=cid,
        optimizer_converged=bool(result.success),optimizer_message=str(result.message),
        dy=dy,nt=nt,kernel_sha256=digest(SOURCE),**provenance,best=best,
        seconds=time.perf_counter()-start,history=history))
    print(cid,'candidate',best,flush=True)


def calibrate(cid,dy,nt):
    """Refine the proposal mode and curvature on the actual sampling grid."""
    from scipy.optimize import minimize
    data,_,_,provenance=load_baseline(cid)
    seed_info=json.loads((OUT/f'explore_{cid}.json').read_text())
    center=np.array(seed_info['best']['z']);L=Likelihood(data,dy,nt)
    def objective(z):return -L.posterior(z)
    increments=np.array([.02,.02,.015,.025,.001,.015])
    def curvature(z):
        f=objective(z);H=np.empty((6,6))
        for j in range(6):
            e=np.zeros(6);e[j]=increments[j]
            H[j,j]=(objective(z+e)-2*f+objective(z-e))/increments[j]**2
            for k in range(j):
                v=np.zeros(6);v[k]=increments[k]
                H[j,k]=H[k,j]=(objective(z+e+v)-objective(z+e-v)-objective(z-e+v)+objective(z-e-v))/(4*increments[j]*increments[k])
        val,vec=np.linalg.eigh(H)
        return (vec*(1/np.maximum(val,1)))@vec.T,val
    cov,_=curvature(center);transform=np.linalg.cholesky(cov)
    result=minimize(lambda x:objective(center+transform@x),np.zeros(6),method='BFGS',
                    options=dict(maxiter=80,gtol=.05,eps=.001))
    mode=center+transform@result.x
    covariance,eigenvalues=curvature(mode)
    save_json(OUT/f'proposal_{cid}.json',dict(contest_id=cid,dy=dy,nt=nt,
        kernel_sha256=digest(SOURCE),**provenance,mode_z=mode.tolist(),
        covariance=covariance.tolist(),raw_curvature_eigenvalues=eigenvalues.tolist(),
        logposterior=L.posterior(mode),optimizer_converged=bool(result.success),
        optimizer_message=str(result.message),likelihood_calls=L.calls))
    print(cid,'proposal calibrated',result.success,L.calls,flush=True)


def weighted_summary(values,weights):
    mean=np.sum(values*weights[:,None],axis=0)
    sd=np.sqrt(np.sum((values-mean)**2*weights[:,None],axis=0))
    quantiles=[]
    for j in range(values.shape[1]):
        order=np.argsort(values[:,j]);w=weights[order]
        quantiles.append(np.interp([.025,.5,.975],np.cumsum(w)-.5*w,values[order,j]))
    return mean,sd,np.array(quantiles)


def extend_sampling(cid,chain,extra):
    """Append draws with fixed post-warmup transition settings and preserved RNG."""
    folder=OUT/f'contest_{cid}';settings_path=folder/f'settings_{chain}.json'
    old=json.loads(settings_path.read_text())
    if not (folder/f'chain_{chain}.npz').exists():raise RuntimeError('Finish the chain before extending')
    if old['kernel_sha256']!=digest(SOURCE):raise RuntimeError('Use the original chain backend')
    backup=folder/'sampling_extensions'/f'after_{old["samples"]}'
    backup.mkdir(parents=True,exist_ok=True)
    for name in (f'settings_{chain}.json',f'chain_{chain}.npz',f'diagnostics_{chain}.json'):
        p=folder/name
        if p.exists():shutil.copy2(p,backup/name)
    updated=dict(old,samples=old['samples']+extra)
    save_json(settings_path,updated)
    (folder/f'chain_{chain}.npz').unlink()
    sampler={'prior-screen':sample_prior_screen,'mixture-prior-screen':sample_mixture_screen,'mixture-gaussian-screen':sample_mixture_gaussian}.get(updated.get('sampler_mode'),sample)
    sampler(cid,chain,updated['warmup'],updated['samples'],updated['dy'],updated['nt'],updated['seed'],
            updated['pilot_requires_posterior_grid_validation'])


def learn_proposal(cid,generation=1):
    """Use a completed pilot to improve exploration; retain the pilot separately."""
    folder=OUT/f'contest_{cid}'
    paths=[folder/f'chain_{j}.npz' for j in range(1,5)]
    if not all(p.exists() for p in paths):raise RuntimeError('Four completed pilot chains required')
    z=np.concatenate([np.load(p)['z'] for p in paths])
    proposal_path=OUT/f'proposal_{cid}.json';old=json.loads(proposal_path.read_text())
    covariance=np.cov(z,rowvar=False)+np.eye(6)*1e-10
    phase=OUT/'learned_proposal_pilots'
    if generation>1:phase=phase/f'generation_{generation}'
    phase.mkdir(parents=True,exist_ok=True)
    dest=phase/f'contest_{cid}'
    if dest.exists():raise RuntimeError('Pilot archive already exists; do not overwrite')
    original_settings=json.loads((folder/'settings_1.json').read_text())
    folder.rename(dest)
    shutil.copy2(proposal_path,dest/'original_proposal.json')
    save_json(proposal_path,dict(contest_id=cid,kernel_sha256=old['kernel_sha256'],
        mode_z=z.mean(axis=0).tolist(),covariance=covariance.tolist(),
        center_definition='Pilot transformed-parameter mean used as a fixed proposal center, not as a parameter estimate.',
        pilot_chain_sha256=[digest(dest/p.name) for p in paths],pilot_archive=str(dest),
        original_proposal=old,original_sampling_settings=original_settings))
    print(cid,'proposal learned; pilot retained',flush=True)


def learn_mixture_proposal(cid,generation):
    """Fixed two-component proposal for an observed nonlinear posterior ridge.

    The split is a computational proposal choice, not a posterior restriction.
    Both t components have full support and exact mixture density correction.
    """
    folder=OUT/f'contest_{cid}'
    z=np.concatenate([np.load(folder/f'chain_{j}.npz')['z'] for j in range(1,5)])
    groups=[z[z[:,2]<=.5],z[z[:,2]>.5]]
    if min(len(a) for a in groups)<100:raise RuntimeError('Insufficient pilot points in each proposal component')
    learn_proposal(cid,generation)
    proposal_path=OUT/f'proposal_{cid}.json';proposal=json.loads(proposal_path.read_text())
    proposal['mixture_components']=[dict(weight=w,mean=a.mean(0).tolist(),
        covariance=(np.cov(a,rowvar=False)+np.eye(6)*1e-6).tolist()) for w,a in zip((.8,.2),groups)]
    proposal['mixture_split']=dict(coordinate='log(sigma)',threshold=.5,pilot_component_counts=[len(a) for a in groups],
        interpretation='Fixed proposal components only; target prior and likelihood unchanged.')
    save_json(proposal_path,proposal)


def physical_draws(z,theta):
    values=z.copy();values[:,:4]=np.exp(values[:,:4]);values[:,5]=np.exp(values[:,5])
    w=theta/(values[:,2:3]**2*values[:,:2])
    return np.column_stack([values,w,np.max(w,axis=1)])


def _grid_worker(payload):
    global SOURCE
    source,data,dy,nt,extent,index,point=payload
    SOURCE=Path(source)
    return index,Likelihood(data,dy,nt,extent).posterior(point)


def posterior_grid(cid,draws=1024,level=0,workers=1):
    """Correct the named pilot grid and validate over the posterior, not one mode."""
    from scipy.special import softmax
    folder=OUT/f'contest_{cid}'
    paths=[folder/f'chain_{j}.npz' for j in range(1,5)]
    if not all(p.exists() for p in paths):raise RuntimeError('All four chains must finish first')
    chains=[np.load(p) for p in paths]
    data,_,_,provenance=load_baseline(cid)
    all_z=np.array([a['z'] for a in chains]);flat=all_z.reshape(-1,6)
    full_values=physical_draws(flat,data['theta']).reshape(4,all_z.shape[1],9)
    rhats=[rank_rhat(full_values[:,:,k]) for k in range(9)]
    ess=[rank_ess(full_values[:,:,k]) for k in range(9)]
    diagnostics=dict(maximum_rank_rhat=max(rhats),minimum_bulk_ess=min(x[0] for x in ess),
                     minimum_tail_ess=min(x[1] for x in ess))
    diagnostics['passed']=max(rhats)<1.01 and min(v for pair in ess for v in pair)>=400
    check_path=OUT/f'posterior_grid_{cid}.json'
    if not diagnostics['passed']:
        save_json(check_path,dict(contest_id=cid,status='sampling_diagnostics_unresolved',**diagnostics))
        print(cid,'sampling diagnostics unresolved',diagnostics,flush=True);return
    n=draws//4
    indices=np.linspace(0,all_z.shape[1]-1,n,dtype=int)
    z=np.concatenate([a['z'][indices] for a in chains])
    base=np.concatenate([a['logp'][indices] for a in chains])
    cache_path=OUT/f'posterior_grid_{cid}.npz';config_path=OUT/f'posterior_grid_settings_{cid}.json'
    old_config=json.loads(config_path.read_text()) if config_path.exists() else None
    old_level=old_config.get('refinement_level',0) if old_config else 0
    level=max(level,old_level)
    fine_dy=.00625/2**level;fine_nt=1024*2**level
    config=dict(contest_id=cid,draws=len(z),kernel_sha256=digest(SOURCE),
                nonlinear_chain_sha256=[digest(p) for p in paths],fine_dy=fine_dy,fine_nt=fine_nt,
                check_dy=fine_dy/2,check_nt=fine_nt*2,**provenance)
    if level:config['refinement_level']=level
    fine=np.full(len(z),np.nan);reference=np.full(len(z),np.nan)
    if old_config and old_config!=config:
        grid_keys={'fine_dy','fine_nt','check_dy','check_nt','refinement_level'}
        if {k:v for k,v in old_config.items() if k not in grid_keys}!={k:v for k,v in config.items() if k not in grid_keys} or level!=old_level+1:
            raise RuntimeError('Posterior grid cache provenance differs')
        saved=np.load(cache_path)
        if not np.array_equal(saved['z'],z) or not np.array_equal(saved['base'],base):
            raise RuntimeError('Cannot transfer different posterior draws between grids')
        fine=saved['reference'].copy()
        archive=OUT/'grid_refinement_levels'/f'contest_{cid}'/f'level_{old_level}'
        archive.mkdir(parents=True,exist_ok=True)
        for previous in (cache_path,config_path,check_path,OUT/f'posterior_domain_{cid}.npz'):
            if previous.exists():
                backup=archive/previous.name
                if backup.exists() and digest(backup)!=digest(previous):
                    raise RuntimeError('Refinement archive already differs; do not overwrite')
                shutil.copy2(previous,backup)
        domain_path=OUT/f'posterior_domain_{cid}.npz'
        if domain_path.exists():domain_path.unlink()
        print(cid,'REFINE POSTERIOR GRID',old_level,'to',level,flush=True)
    elif cache_path.exists():
        saved=np.load(cache_path);fine=saved['fine'];reference=saved['reference']
    save_json(config_path,config)
    started=time.perf_counter()
    def checkpoint():
        tmp=cache_path.with_name(cache_path.name+f'.{os.getpid()}.tmp')
        with tmp.open('wb') as stream:np.savez_compressed(stream,z=z,base=base,fine=fine,reference=reference)
        tmp.replace(cache_path)
    checkpoint()
    def evaluate(indices,target,dy,nt,label,chunk):
        pending=[int(j) for j in indices if not np.isfinite(target[j])]
        completed=len(indices)-len(pending)
        def record(j,value):
            nonlocal completed
            target[j]=value;completed+=1
            if completed%chunk==0:
                checkpoint();print(f'{cid}: {label} {completed}/{len(indices)}, seconds={time.perf_counter()-started:.1f}',flush=True)
        if workers>1 and pending:
            from concurrent.futures import ProcessPoolExecutor,as_completed
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures=[pool.submit(_grid_worker,(str(SOURCE),data,dy,nt,7,j,z[j])) for j in pending]
                for future in as_completed(futures):record(*future.result())
        else:
            evaluator=Likelihood(data,dy,nt)
            for j in pending:record(j,evaluator.posterior(z[j]))
        checkpoint()
    evaluate(range(len(z)),fine,fine_dy,fine_nt,'posterior grid',32)
    # Uniformly spaced representative checks retain posterior frequencies.
    representative=np.unique(np.linspace(0,len(z)-1,256,dtype=int))
    # Extremes are additional stress tests, excluded from frequency summaries.
    values=physical_draws(z,data['theta'])
    extremes=np.unique(np.r_[np.argmin(values,axis=0),np.argmax(values,axis=0)])
    checks=np.unique(np.r_[representative,extremes])
    evaluate(checks,reference,fine_dy/2,fine_nt*2,'finer posterior check',16)
    weights=softmax(fine-base);importance_ess=float(1/np.sum(weights**2))
    mean,sd,_=weighted_summary(values,weights)
    # Self-normalized importance means have influence u*(h-E_target h),
    # where u is the density ratio normalized to have sample mean one.
    # Account for residual within-chain dependence after thinning.
    mean_ess=[]
    for k in range(values.shape[1]):
        influence=(len(z)*weights*(values[:,k]-mean[k])).reshape(4,n)
        variance=float(np.var(influence,ddof=1))
        effective=effective_sample_size(influence)
        mean_ess.append(float(min(len(z),effective*sd[k]**2/max(variance,1e-30))))
    a=softmax((fine-base)[representative]);b=softmax((reference-base)[representative])
    ma,sa,_=weighted_summary(values[representative],a);mb,_,_=weighted_summary(values[representative],b)
    shifts=np.abs((mb-ma)/np.maximum(sa,1e-12))
    delta=reference[representative]-fine[representative]
    center=float(np.sum(a*delta));centered=np.abs(delta-center)
    order=np.argsort(centered);q95=float(np.interp(.95,np.cumsum(a[order]),centered[order]))
    maximum=float(np.max(np.abs((reference-fine)[checks]-center)))
    # Boundary sensitivity is distinct from space/time refinement. Include
    # representative points and every parameter extreme in the posterior set.
    domain_indices=np.unique(np.r_[np.linspace(0,len(z)-1,16,dtype=int),extremes])
    domain_path=OUT/f'posterior_domain_{cid}.npz'
    domain=np.full(len(domain_indices),np.nan)
    if domain_path.exists():
        saved=np.load(domain_path)
        if not np.array_equal(saved['z'],z[domain_indices]) or str(saved['kernel_sha256'])!=digest(SOURCE) or (level and (float(saved['fine_dy'])!=fine_dy or int(saved['fine_nt'])!=fine_nt)):
            raise RuntimeError('Posterior domain cache provenance differs')
        domain=saved['logp']
    wider=Likelihood(data,fine_dy,fine_nt,extent=9)
    for j,index in enumerate(domain_indices):
        if not np.isfinite(domain[j]):domain[j]=wider.posterior(z[index])
        np.savez_compressed(domain_path,z=z[domain_indices],logp=domain,kernel_sha256=digest(SOURCE),fine_dy=fine_dy,fine_nt=fine_nt)
    domain_error=float(np.max(np.abs(domain-fine[domain_indices])))
    passed=importance_ess>=400 and min(mean_ess)>=400 and float(weights.max())<.02 and float(shifts.max())<.05 and q95<.1 and maximum<.25 and domain_error<.01
    result=dict(**config,validation_version=2,status='validated' if passed else 'finer_grid_required',sampling_diagnostics=diagnostics,
        importance_ess=importance_ess,maximum_importance_weight=float(weights.max()),
        minimum_importance_mean_ess=min(mean_ess),importance_mean_ess=mean_ess,
        reference_checks=len(checks),reference_representative_checks=len(representative),
        maximum_parameter_shift_in_fine_sd=float(shifts.max()),parameter_grid_shifts=shifts.tolist(),
        centered_logdensity_error_q95=q95,centered_logdensity_error_maximum=maximum,
        domain_checks=len(domain_indices),maximum_domain_logdensity_change=domain_error,
        note='Likelihood offsets constant over parameters cancel in posterior normalization. Reported grid checks concern posterior-relative errors; no claim of analytic exactness.')
    save_json(check_path,result)
    print(cid,'POSTERIOR GRID',result['status'],'ESS',importance_ess,'max shift',shifts.max(),flush=True)


def write_csv(path,rows):
    if not rows:return
    with Path(path).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def rank_rhat(values):
    split=np.concatenate([values[:,:values.shape[1]//2],values[:,values.shape[1]//2:]],axis=0)
    ranked=ndtri((rankdata(split.ravel())-.375)/(split.size+.25)).reshape(split.shape)
    folded=ndtri((rankdata(np.abs(split-np.median(split)).ravel())-.375)/(split.size+.25)).reshape(split.shape)
    def calc(a):
        n=a.shape[1];within=np.var(a,axis=1,ddof=1).mean()
        return math.sqrt(((n-1)/n*within+np.var(a.mean(axis=1),ddof=1))/within) if within>0 else math.inf
    return max(calc(ranked),calc(folded))


def effective_sample_size(values):
    m,n=values.shape
    centered=values-values.mean(axis=1,keepdims=True)
    transformed=np.fft.rfft(centered,n=2*n,axis=1)
    covariance=np.fft.irfft(transformed*np.conjugate(transformed),n=2*n,axis=1)[:,:n]/n
    within=np.var(values,axis=1,ddof=1).mean()
    variance=(n-1)/n*within+np.var(values.mean(axis=1),ddof=1)
    if variance<=0:return 0.
    rho=1-(within-covariance.mean(axis=0))/variance;rho[0]=1.
    total=0.;previous=math.inf
    for k in range(0,n-1,2):
        pair=float(rho[k]+rho[k+1])
        if pair<=0:break
        pair=min(pair,previous);total+=pair;previous=pair
    return float(min(m*n,m*n/max(1.,-1+2*total)))


def rank_ess(values):
    normalized=ndtri((rankdata(values.ravel())-.375)/(values.size+.25)).reshape(values.shape)
    bulk=effective_sample_size(normalized)
    tail=min(effective_sample_size((values<=np.quantile(values,q)).astype(float)) for q in (.05,.95))
    return bulk,tail


def summarize():
    from scipy.special import softmax
    selection=json.loads((OUT/"selection.json").read_text())["selected"]
    rows=[];pilot_rows=[];status=[];baseline_rows=[];audit_rows=[]
    for selected in selection:
        cid=selected["contest_id"];paths=[OUT/f"contest_{cid}"/f"chain_{j}.npz" for j in range(1,5)]
        data,old,_,_=load_baseline(cid)
        old=np.array(old);old[:,:,5]=np.exp(old[:,:,5])
        w=data['theta']/(old[:,:,2:3]**2*old[:,:,:2])
        old_all=np.concatenate([old,w,np.max(w,axis=2)[:,:,None]],axis=2)
        for k,name in enumerate(NAMES+("w_i","w_j","max_w")):
            a=old_all[:,:,k].ravel()
            baseline_rows.append(dict(contest_id=cid,parameter=name,mean=float(a.mean()),
                sd=float(a.std(ddof=1)),median=float(np.median(a)),
                q025=float(np.quantile(a,.025)),q975=float(np.quantile(a,.975))))
        audit_path=OUT/f"audit_{cid}.json"
        numerical=json.loads(audit_path.read_text()) if audit_path.exists() else {}
        refinements=numerical.get('refinements',[])
        spatial=abs(refinements[2]['loglik']-refinements[1]['loglik']) if len(refinements)>=3 else None
        boundary=abs(refinements[3]['loglik']-refinements[2]['loglik']) if len(refinements)>=4 else None
        gate=(numerical.get('kernel_sha256')==digest(SOURCE) and spatial is not None
              and boundary is not None and spatial<.1 and boundary<.1)
        usable=[r for r in refinements+numerical.get('extra_refinements',[])
                if r.get('extent')==7 and 'loglik' in r]
        usable.sort(key=lambda r:(-r['dy'],r['nt']))
        finest=usable[-1] if usable else {}
        latest_change=abs(usable[-1]['loglik']-usable[-2]['loglik']) if len(usable)>=2 else None
        audit_rows.append(dict(contest_id=cid,median_max_w=selected['median_max_w'],
            grid_loglik_change=spatial,boundary_loglik_change=boundary,
            initial_grid_check_passed=gate,stan_logdensity_difference_error=
            numerical.get('stan_logdensity_difference_error'),finest_requested_dy=finest.get('dy'),
            finest_nt=finest.get('nt'),latest_grid_loglik_change=latest_change))
        if not all(p.exists() for p in paths):
            started=any((OUT/f"contest_{cid}"/f"settings_{j}.json").exists() for j in range(1,5))
            status.append(dict(contest_id=cid,status="sampling_in_progress" if started else "not_sampled",
                initial_grid_check_passed=gate,
                next_required="finish four chains and posterior grid validation" if started else (
                "posterior sampling and posterior grid audit" if gate
                else "resolve nonlinear grid sensitivity before posterior sampling")));continue
        z=np.array([np.load(p)['z'] for p in paths])
        posterior_check_path=OUT/f'posterior_grid_{cid}.json'
        posterior_check=json.loads(posterior_check_path.read_text()) if posterior_check_path.exists() else {}
        validated=posterior_check.get('status')=='validated' and posterior_check.get('validation_version')==2
        final_summary=None
        if validated:
            if posterior_check['kernel_sha256']!=digest(SOURCE) or posterior_check['nonlinear_chain_sha256']!=[digest(p) for p in paths]:
                raise RuntimeError('Validated posterior provenance differs')
            grid=np.load(OUT/f'posterior_grid_{cid}.npz')
            values=physical_draws(grid['z'],data['theta'])
            weights=softmax(grid['fine']-grid['base'])
            final_summary=weighted_summary(values,weights)
        new=z.copy();new[:,:,:4]=np.exp(new[:,:,:4]);new[:,:,5]=np.exp(new[:,:,5])
        for values in (old,new):
            w=data['theta']/(values[:,:,2:3]**2*values[:,:,:2])
            # w_i,w_j and draw-specific max(w_i,w_j), preserving joint draws.
            if values is old: old_all=np.concatenate([values,w,np.max(w,axis=2)[:,:,None]],axis=2)
            else: new_all=np.concatenate([values,w,np.max(w,axis=2)[:,:,None]],axis=2)
        rhats=[];bulk_ess=[];tail_ess=[]
        for k,name in enumerate(NAMES+("w_i","w_j","max_w")):
            a=old_all[:,:,k].ravel();b=new_all[:,:,k].ravel();rh=rank_rhat(new_all[:,:,k]);rhats.append(rh)
            bulk,tail=rank_ess(new_all[:,:,k]);bulk_ess.append(bulk);tail_ess.append(tail)
            mean=float(final_summary[0][k]) if validated else float(b.mean())
            sd=float(final_summary[1][k]) if validated else float(b.std(ddof=1))
            quantiles=final_summary[2][k] if validated else np.quantile(b,[.025,.5,.975])
            destination=rows if validated else pilot_rows
            destination.append(dict(contest_id=cid,parameter=name,status='validated' if validated else 'pilot_grid_unvalidated',baseline_mean=float(a.mean()),
                nonlinear_mean=mean,baseline_sd=float(a.std(ddof=1)),nonlinear_sd=sd,
                shift_in_baseline_sd=float((mean-a.mean())/a.std(ddof=1)),
                shift_in_nonlinear_sd=float((mean-a.mean())/max(sd,1e-12)),
                intervals_95_overlap=bool(quantiles[0]<=np.quantile(a,.975) and quantiles[2]>=np.quantile(a,.025)),
                baseline_median=float(np.median(a)),nonlinear_median=float(quantiles[1]),
                baseline_q025=float(np.quantile(a,.025)),baseline_q975=float(np.quantile(a,.975)),
                nonlinear_q025=float(quantiles[0]),nonlinear_q975=float(quantiles[2]),
                rank_rhat=rh,bulk_ess=bulk,tail_ess=tail))
        status.append(dict(contest_id=cid,status="validated" if validated else "sampled_pilot_requires_validation",maximum_rank_rhat=max(rhats),
                           minimum_bulk_ess=min(bulk_ess),minimum_tail_ess=min(tail_ess),
                           sampling_converged=max(rhats)<1.01 and min(bulk_ess+tail_ess)>=400))
    if rows:
        write_csv(OUT/"parameter_comparison.csv",rows)
    if pilot_rows:write_csv(OUT/'pilot_parameter_comparison.csv',pilot_rows)
    write_csv(OUT/"baseline_selected.csv",baseline_rows)
    write_csv(OUT/"audit_summary.csv",audit_rows)
    save_json(OUT/"status.json",dict(contests=status,
        interpretation="Numerical nonlinear-policy posterior; convergence and grid sensitivity must pass before substantive interpretation."))
    print(json.dumps(status,indent=2))


def baseline_target_consistency():
    """Check archived native-Stan targets, allowing rounded CSV coordinates."""
    records=[]
    selected=json.loads((OUT/'selection.json').read_text())['selected']
    for item in selected:
        cid=item['contest_id'];data,_,_,provenance=load_baseline(cid)
        evaluator=Likelihood(data);differences=[]
        for chain in range(1,5):
            path=Path(provenance['accepted_run_dir'])/f'chain_{chain}.csv'
            with path.open() as stream:
                rows=list(csv.DictReader(line for line in stream if not line.startswith('#')))
            for index in np.linspace(0,len(rows)-1,3,dtype=int):
                row=rows[index]
                values=np.array([float(row[k]) for k in ('c_i','c_j','sigma','lambda','mu_0','log_r')])
                point=values.copy();point[:4]=np.log(values[:4])
                if np.any(values[:4]<=.01) or point[5]<=math.log(.01):
                    raise RuntimeError('CSV precision insufficient to reconstruct a lower-bound Jacobian')
                adjustment=np.sum(np.log(values[:4]-.01)-np.log(values[:4]))+math.log(point[5]-math.log(.01))
                predicted=evaluator.prior(point)+evaluator.loglik(point,False)+adjustment
                differences.append(float(row['lp__'])-predicted)
        centered=np.array(differences)-np.median(differences)
        record=dict(contest_id=cid,draws=len(differences),
            maximum_centered_density_difference=float(np.abs(centered).max()),
            sd_density_difference=float(np.std(differences)))
        records.append(record);print(record,flush=True)
    save_json(OUT/'baseline_target_consistency.json',dict(
        method='Compare archived Stan lp__ with current leading-order observation law and priors, adjusting lower-bound transformation Jacobians; common density constants removed; archived CSV values are rounded.',
        contests=records))


def main():
    global OUT,SOURCE
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','baseline-target','audit','refine','explore','calibrate','learn-proposal','learn-mixture','sample','sample-prior','sample-mixture','sample-mixture-gaussian','extend','posterior-grid','summarize'])
    parser.add_argument('--contest',type=int);parser.add_argument('--chain',type=int,default=1)
    parser.add_argument('--warmup',type=int,default=2000);parser.add_argument('--samples',type=int,default=4000)
    parser.add_argument('--dy',type=float,default=.0125);parser.add_argument('--nt',type=int,default=2048)
    parser.add_argument('--seed',type=int,default=20261011)
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--grid-draws',type=int,default=1024)
    parser.add_argument('--grid-level',type=int,default=0)
    parser.add_argument('--grid-workers',type=int,default=1)
    parser.add_argument('--proposal-generation',type=int,default=1)
    parser.add_argument('--backend',choices=['legacy','fast'],default='legacy')
    parser.add_argument('--extra-samples',type=int,default=2000)
    args=parser.parse_args()
    if args.backend=='fast':
        OUT=ROOT/'results'/'estimation'/'nonlinear_top10'/'fast_v1'
        SOURCE=Path(__file__).with_name('nonlinear_fast_kernel.cpp')
    if args.mode=='prepare':prepare()
    elif args.mode=='baseline-target':baseline_target_consistency()
    elif args.mode=='audit':audit(args.contest)
    elif args.mode=='refine':refine(args.contest,args.dy,args.nt)
    elif args.mode=='explore':explore(args.contest,args.dy,args.nt)
    elif args.mode=='calibrate':calibrate(args.contest,args.dy,args.nt)
    elif args.mode=='sample':sample(args.contest,args.chain,args.warmup,args.samples,args.dy,args.nt,args.seed,args.pilot)
    elif args.mode=='sample-prior':sample_prior_screen(args.contest,args.chain,args.warmup,args.samples,args.dy,args.nt,args.seed,args.pilot)
    elif args.mode=='sample-mixture':sample_mixture_screen(args.contest,args.chain,args.warmup,args.samples,args.dy,args.nt,args.seed,args.pilot)
    elif args.mode=='sample-mixture-gaussian':sample_mixture_gaussian(args.contest,args.chain,args.warmup,args.samples,args.dy,args.nt,args.seed,args.pilot)
    elif args.mode=='posterior-grid':
        # Independent resumable drivers may reach the same completed contest.
        # Serialize its cache writes; the second caller reuses saved solves.
        import fcntl
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/f'posterior_grid_{args.contest}.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            posterior_grid(args.contest,args.grid_draws,args.grid_level,args.grid_workers)
    elif args.mode=='extend':extend_sampling(args.contest,args.chain,args.extra_samples)
    elif args.mode=='learn-mixture':learn_mixture_proposal(args.contest,args.proposal_generation)
    elif args.mode=='learn-proposal':learn_proposal(args.contest,args.proposal_generation)
    else:summarize()


if __name__=='__main__':main()
