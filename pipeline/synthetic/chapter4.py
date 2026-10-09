"""Reproducible full-nonlinear/non-steady-state Chapter 4 experiment suite.

Run: python -m pipeline.synthetic.chapter4 --prepare
     python -m pipeline.synthetic.chapter4 --fit all
Outputs are separate from legacy experiments, with raw draws and fixed seeds.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from cmdstanpy import CmdStanModel, from_csv
from pipeline.config import ROOT
from pipeline.synthetic.nonlinear import Primitives, Equilibrium, solve_equilibrium, simulate, likelihood_data, variance

OUT = ROOT / 'results/synthetic/nonsteady_hjb'
STAN = ROOT / 'synthetic_data/nonsteady_recovery.stan'
BASE = Primitives()
NAMES = ('baseline_fixed_r','baseline_estimated_r','pool10','pool20','days180','days360')
PARAMS = ('c_i','c_j','sigma','lambda','mu_0','S0')


def write_json(path, data):
    def encode(x):
        if isinstance(x,np.ndarray): return x.tolist()
        if isinstance(x,np.generic): return x.item()
        raise TypeError(type(x))
    path.write_text(json.dumps(data,indent=2,default=encode)+'\n')


def equilibrium(p, *, dy=0.2, dt=1/48):
    path = OUT / f'equilibrium_T{p.T:g}_dy{dy:g}_dt{dt:.7f}.npz'
    if path.exists():
        with np.load(path) as z:
            if not np.array_equal(z['primitives'],np.array(list(asdict(p).values()))):
                raise ValueError(f'Cached HJB primitives differ from the requested experiment: {path}')
            return Equilibrium(p,z['t'],z['y'],z['effort'],json.loads(path.with_suffix('.json').read_text()))
    eq = solve_equilibrium(p,dy=dy,dt=dt)
    eq.save(path)
    write_json(path.with_suffix('.json'),eq.diagnostics)
    return eq


def grid_check(eq):
    # Double both spatial and temporal resolution; never reroll simulation seeds.
    fine = solve_equilibrium(eq.p,dy=0.1,dt=1/96)
    diffs, levels = [], []
    for t in np.unique(np.r_[0,.1,.5,1,eq.p.T/2,eq.p.T-1,eq.p.T]):
        q = float(variance(t,eq.p)) + eq.p.sigma**2*(eq.p.T-t)
        for y in np.linspace(-3,3,31)*np.sqrt(q):
            actual, reference = eq.at(t,y), fine.at(t,y)
            diffs.extend(abs(actual-reference))
            levels.extend(reference)
    result = dict(max_absolute_error=max(diffs),
                  max_error_over_peak=max(diffs)/max(levels),
                  rms_error_over_rms=np.linalg.norm(diffs)/np.linalg.norm(levels))
    if result['max_error_over_peak'] > 0.01:
        raise RuntimeError(f'HJB grid refinement failed: {result}')
    return result


def save_contest(name,c):
    np.savez_compressed(OUT/f'{name}_paths.npz',path=c['path'],events=c['events'])
    return {k:v for k,v in c.items() if k not in ('path',)}


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest = dict(primitives=asdict(BASE), simulation_dt=1/96, likelihood_dt=1/24,
                    priors='Section 3: lognormal supports >=0.01, mu_0 N(first public gap,1); pooled center is mean of first gaps; prior applied once',
                    grid_checks={},experiments={}, calibration={})
    eq = equilibrium(BASE)
    manifest['grid_checks']['90'] = grid_check(eq)
    print('HJB baseline:',eq.diagnostics,manifest['grid_checks']['90'],flush=True)
    baseline = simulate(eq,seed=1000)
    baseline_meta = save_contest('baseline',baseline)
    for name,free in [('baseline_fixed_r',False),('baseline_estimated_r',True)]:
        data = likelihood_data([baseline],estimate_r=free)
        write_json(OUT/f'{name}.json',data)
        manifest['experiments'][name] = dict(contests=[baseline_meta],mu_prior_center=data['mu_prior_center'])
    pool=[]
    for i in range(20):
        c=simulate(eq,seed=3000+10*i)
        save_contest(f'pool_contest{i+1:02}',c)
        pool.append(c)
        print(f'Pool contest {i+1}/20: {len(c["events"])} events',flush=True)
    for size in (10,20):
        data=likelihood_data(pool[:size])
        write_json(OUT/f'pool{size}.json',data)
        manifest['experiments'][f'pool{size}']=dict(
            contests=[{k:v for k,v in c.items() if k!='path'} for c in pool[:size]],
            mu_prior_center=data['mu_prior_center'])
    # Match PATHWISE integrated intensity per day to the baseline, as specified
    # in the paper. Effort paths do not depend on r in the continuous-signal DGP.
    target=float(np.sum(baseline['expected_counts'])/BASE.T)
    for days in (180,360):
        eq_long=equilibrium(replace(BASE,T=float(days)))
        manifest['grid_checks'][str(days)]=grid_check(eq_long)
        seed=5000+days
        ratio=15.0
        trace=[]
        for attempt in range(10):
            c=simulate(eq_long,seed=seed,ratio=ratio)
            rate=float(np.sum(c['expected_counts'])/days)
            trace.append(dict(ratio=ratio,rate=rate))
            if abs(rate-target)/target < 0.005:
                break
            ratio *= target/rate
        else:
            raise RuntimeError(f'Rate calibration did not converge for T={days}: {trace}')
        name=f'days{days}'
        data=likelihood_data([c])
        write_json(OUT/f'{name}.json',data)
        manifest['experiments'][name]=dict(contests=[save_contest(name,c)],mu_prior_center=data['mu_prior_center'])
        manifest['calibration'][str(days)]=dict(target_per_day=target,trace=trace)
        print('Long horizon:',days,'r=',ratio,'events=',len(c['events']),flush=True)
    manifest['stan_sha256']=hashlib.sha256(STAN.read_bytes()).hexdigest()
    write_json(OUT/'manifest.json',manifest)


def summarize(fit,name):
    summary=fit.summary()
    truths=dict(c_i=1.2,c_j=1.5,sigma=2.0,**{'lambda':1.0},mu_0=0.0,S0=4.0,r_free=15.0)
    names=list(PARAMS)+(['r_free'] if name=='baseline_estimated_r' else [])
    results={}
    for param in names:
        v=fit.stan_variable(param).reshape(-1)
        row=summary.loc['r_free[1]' if param=='r_free' else param]
        lo,hi=np.quantile(v,[.025,.975])
        results['r' if param=='r_free' else param]=dict(
            true=truths[param],mean=v.mean(),sd=v.std(ddof=1),rmsd=np.sqrt(np.mean((v-truths[param])**2)),
            q2_5=lo,q97_5=hi,rhat=row['R_hat'],ess_bulk=row['ESS_bulk'],ess_tail=row['ESS_tail'])
    m=fit.method_variables()
    diag=dict(divergences=int(m['divergent__'].sum()),
              treedepth_hits=int((m['treedepth__']>=12).sum()),
              max_rhat=max(x['rhat'] for x in results.values()),
              min_ess_bulk=min(x['ess_bulk'] for x in results.values()),
              min_ess_tail=min(x['ess_tail'] for x in results.values()),
              bfmi=[float(np.mean(np.diff(m['energy__'][:,j])**2)/np.var(m['energy__'][:,j]))
                    for j in range(m['energy__'].shape[1])])
    diag['passed']=bool(diag['divergences']==0 and diag['treedepth_hits']==0 and diag['max_rhat']<=1.01
                        and diag['min_ess_bulk']>=400 and diag['min_ess_tail']>=400 and min(diag['bfmi'])>=.3)
    results['diagnostics']=diag
    write_json(OUT/f'{name}_summary.json',results)
    summary.to_csv(OUT/f'{name}_diagnostics.csv')
    print(name,diag,flush=True)
    return results


def fit_all(names,warmup,samples):
    model=CmdStanModel(stan_file=str(STAN))
    for name in names:
        folder=OUT/name
        folder.mkdir(exist_ok=True)
        data=json.loads((OUT/f'{name}.json').read_text())
        print(f'Starting {name}: {data["J"]} contests, {data["N"]} integration steps',flush=True)
        fit=model.sample(data=data,chains=4,parallel_chains=4,iter_warmup=warmup,iter_sampling=samples,
                         seed=7100+NAMES.index(name)*100,adapt_delta=.95,max_treedepth=12,
                         output_dir=str(folder),show_progress=False,show_console=False,refresh=100,
                         inits=dict(c_i=1.0,c_j=1.0,sigma=2.0,**{'lambda':1.0},S0=2.0,
                                    mu_0=data['mu_prior_center'],**({'r_free':[10.0]} if data['estimate_r'] else {})))
        summarize(fit,name)
        write_json(OUT/f'{name}_run.json',dict(csv_files=fit.runset.csv_files,warmup=warmup,samples=samples,
                    data_sha256=hashlib.sha256((OUT/f'{name}.json').read_bytes()).hexdigest(),
                    stan_sha256=hashlib.sha256(STAN.read_bytes()).hexdigest()))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--fit',choices=('all',*NAMES))
    parser.add_argument('--warmup',type=int,default=1000)
    parser.add_argument('--samples',type=int,default=1000)
    args=parser.parse_args()
    if args.prepare: prepare()
    if args.fit: fit_all(NAMES if args.fit=='all' else [args.fit],args.warmup,args.samples)


if __name__=='__main__': main()
