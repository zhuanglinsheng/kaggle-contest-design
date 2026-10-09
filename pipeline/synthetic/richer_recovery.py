"""Section 4.1: steady estimation of the saved nonlinear, S0=4 samples.

No simulation, recalibration, private-path fitting, or time trimming occurs.
Independent contests have independent latent states and shared parameters.
"""
import argparse
import hashlib
import json
import time

import numpy as np
from cmdstanpy import CmdStanModel, from_csv

from pipeline.config import ROOT
from pipeline.synthetic.table1 import DATA, TRUTH, load_data, likelihood_check, write_json
from pipeline.synthetic.model_comparison import PARAMS, log_likelihood, log_prior

STAN = ROOT/'synthetic_data/hourly_steady_pooled.stan'
OUT = ROOT/'results/synthetic/richer_steady'
CASES = {
    'pool10': ([f'pool_contest{i:02}' for i in range(1,11)],15.,10100),
    'pool20': ([f'pool_contest{i:02}' for i in range(1,21)],15.,10200),
    'days180': (['days180'],18.6484375,10300),
    'days360': (['days360'],47.4375,10400),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pack_contests(contests, ratio):
    first=contests[0]
    for d in contests:
        for key in ('N','dt','theta','mu_prior_sd'):
            if d[key]!=first[key]: raise ValueError(f'Mismatched {key}')
    return dict(J=len(contests),N=first['N'],dt=first['dt'],theta=first['theta'],
        ratio=ratio,mu_prior_sd=first['mu_prior_sd'],
        public_gap=np.array([d['public_gap'] for d in contests]).T,
        count_i=np.array([d['count_i'] for d in contests],dtype=int).T,
        count_j=np.array([d['count_j'] for d in contests],dtype=int).T)


def independent_joint(p,contests,ratio):
    p=dict(p,S0=p['sigma']/np.sqrt(p['lambda']))
    x=np.array([[p[n] for n in PARAMS[:5]]+[np.log(p['r']),p['S0']]])
    likelihood=sum(float(log_likelihood(x,d)[0]) for d in contests)
    prior=float(log_prior(x,dict(contests[0],free_S0=0,ratio=ratio))[0])
    return likelihood+prior


def checks(model,data,contests,ratio):
    points=[dict(TRUTH,r=ratio),dict(TRUTH,c_i=.8,c_j=1.8,sigma=1.5,
        **{'lambda':1.4},mu_0=.8,r=.85*ratio)]
    result=[]
    for p in points:
        params={n:p[n] for n in PARAMS[:5]}; params['log_r']=np.log(p['r'])
        actual=float(model.log_prob(params=params,data=data,jacobian=False,sig_figs=18)['lp__'].iloc[0])
        expected=independent_joint(p,contests,ratio)
        dense_errors=[]
        for d in contests:
            out=likelihood_check(d,dict(p,S0=p['sigma']/np.sqrt(p['lambda'])))
            dense_errors.append(out['difference'])
        error=actual-expected
        if abs(error)>1e-7 or max(abs(v) for v in dense_errors)>1e-7:
            raise RuntimeError((error,dense_errors))
        result.append(dict(stan_minus_independent_joint=error,
                           max_innovations_minus_dense=max(abs(v) for v in dense_errors)))
    return result


def summarize(fit,truth):
    sm=fit.summary(); ans={}
    for p in PARAMS:
        v=fit.stan_variable(p); row=sm.loc[p]; lo,hi=np.quantile(v,[.025,.975])
        ans[p]=dict(true=truth[p],mean=v.mean(),sd=v.std(ddof=1),
            rmsd=np.sqrt(np.mean((v-truth[p])**2)),q2_5=lo,q97_5=hi,
            covers_truth=bool(lo<=truth[p]<=hi),rhat=row['R_hat'],
            ess_bulk=row['ESS_bulk'],ess_tail=row['ESS_tail'])
    m=fit.method_variables()
    diag=dict(divergences=int(m['divergent__'].sum()),
        treedepth_hits=int((m['treedepth__']>=12).sum()),
        max_rhat=max(v['rhat'] for v in ans.values()),
        min_ess_bulk=min(v['ess_bulk'] for v in ans.values()),
        min_ess_tail=min(v['ess_tail'] for v in ans.values()),
        bfmi=[float(np.mean(np.diff(m['energy__'][:,j])**2)/np.var(m['energy__'][:,j])) for j in range(4)])
    diag['passed']=bool(diag['divergences']==0 and diag['treedepth_hits']==0 and
        diag['max_rhat']<=1.01 and diag['min_ess_bulk']>=400 and
        diag['min_ess_tail']>=400 and min(diag['bfmi'])>=.3)
    ans['diagnostics']=diag
    return ans


def run(model,name,args):
    sources,ratio,seed=CASES[name]
    folder=OUT/name; folder.mkdir(parents=True,exist_ok=True)
    paths={s:DATA/f'{s}_paths.npz' for s in sources}
    hashes={s:sha(p) for s,p in paths.items()}
    contests=[load_data(s,True) for s in sources]
    for d in contests: d['ratio']=ratio
    data=pack_contests(contests,ratio)
    write_json(folder/'data.json',data)
    observed=dict(contests=len(contests),days_each=data['N']*data['dt'],
        submissions_i=int(data['count_i'].sum()),submissions_j=int(data['count_j'].sum()),
        readouts=int(((data['count_i']+data['count_j'])>0).sum()))
    print('Checking',name,observed,flush=True)
    check=checks(model,data,contests,ratio)
    signature=dict(stan_sha256=sha(STAN),data_sha256=sha(folder/'data.json'),source_sha256=hashes)
    if args.reuse and (folder/'run.json').exists():
        run_info=json.loads((folder/'run.json').read_text())
        if any(run_info[k]!=v for k,v in signature.items()):
            raise RuntimeError(f'{name}: changed inputs cannot reuse chains')
        fit=from_csv(run_info['csv_files'])
    else:
        initial=[]
        for ci,cj,sigma,lam,mu,rscale in [(1.,1.4,1.7,.8,-.5,.8),
                (1.2,1.5,2.,1.,0.,1.),(1.4,1.8,2.3,1.3,.5,1.2),(.9,1.25,2.1,1.1,1.,1.)]:
            initial.append(dict(c_i=ci,c_j=cj,sigma=sigma,**{'lambda':lam},mu_0=mu,log_r=np.log(ratio*rscale)))
        started=time.monotonic()
        print('Sampling',name,flush=True)
        fit=model.sample(data=data,chains=4,parallel_chains=4,iter_warmup=args.warmup,
            iter_sampling=args.samples,seed=seed,adapt_delta=.95,max_treedepth=12,
            inits=initial,sig_figs=18,output_dir=str(folder),show_progress=False,refresh=200)
        run_info=dict(**signature,seed=seed,warmup=args.warmup,samples=args.samples,
            csv_files=fit.runset.csv_files,elapsed_seconds=time.monotonic()-started,
            input_sources=sources,observed=observed,likelihood_checks=check,
            estimation='steady S0=sigma/sqrt(lambda); common six parameters, r free',
            dgp='saved optimal_hourly/nonsteady: nonlinear HJB m, S0=4',
            prior_r=dict(logmean=float(np.log(ratio)),logsd=.35),
            pooling='independent initial latent draws; per-contest filters; common prior once')
        write_json(folder/'run.json',run_info)
    result=summarize(fit,dict(TRUTH,r=ratio))
    result['observed']=observed
    write_json(folder/'summary.json',result)
    if hashes!={s:sha(p) for s,p in paths.items()}: raise RuntimeError('Source data changed')
    print('RESULT',name,json.dumps(result),flush=True)
    if not result['diagnostics']['passed']: raise RuntimeError(f'{name}: diagnostics failed')
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cases',nargs='+',choices=CASES,default=list(CASES))
    parser.add_argument('--warmup',type=int,default=1000)
    parser.add_argument('--samples',type=int,default=2000)
    parser.add_argument('--reuse',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    model=CmdStanModel(stan_file=str(STAN))
    for name in args.cases: run(model,name,args)


if __name__=='__main__': main()
