"""Table 1 estimation with original synthetic priors and hourly timing.

The DGP is the accepted optimal_hourly/nonsteady data. No simulations or
seeds are changed here. Repeated fits, when requested, use separate contests.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import multivariate_normal
from cmdstanpy import CmdStanModel
from pipeline.config import ROOT

DATA = ROOT/'results/synthetic/optimal_hourly/nonsteady'
OUT = ROOT/'results/synthetic/table1_optimal_hourly'
STAN = ROOT/'synthetic_data/hourly_nonsteady.stan'
TRUTH = dict(c_i=1.2,c_j=1.5,sigma=2.,lambda_=1.,mu_0=0.,S0=4.,r=15.)
TRUTH['lambda'] = TRUTH.pop('lambda_')


def write_json(path,value):
    def encode(x):
        if isinstance(x,np.ndarray): return x.tolist()
        if isinstance(x,np.generic): return x.item()
        raise TypeError(type(x))
    path.write_text(json.dumps(value,indent=2,default=encode)+'\n')


def load_data(name,estimate_r=False,mu_sd=1.,S0_logmean=np.log(2.)):
    with np.load(DATA/f'{name}_paths.npz') as z:
        N=len(z['effort_i'])
        counts=[]
        for k in ('i','j'):
            steps=np.ceil(z[f'events_{k}_days']*24).astype(int)
            counts.append(np.bincount(steps,minlength=N+1)[1:])
        return dict(N=N,dt=1/24,theta=1.,ratio=15.,estimate_r=int(estimate_r),
                    public_gap=z['public'].tolist(),count_i=counts[0].tolist(),count_j=counts[1].tolist(),
                    mu_prior_sd=mu_sd,S0_prior_logmean=S0_logmean,S0_prior_logsd=.75)


def likelihood_check(data,p):
    """Compare innovations to original dense Gaussian likelihood, holding params fixed."""
    mu=p['mu_0']; var=p['S0']; perceived=mu; total=0.; elapsed=0.; drift_sum=0.; point_process=0.
    times=[]; means=[]; obs=[]; noises=[]
    steady=p['sigma']/np.sqrt(p['lambda'])
    for n in range(data['N']):
        t=n*data['dt']; z=np.tanh(p['sigma']*np.sqrt(p['lambda'])*t)
        S=steady*(p['S0']+steady*z)/(steady+p['S0']*z)
        Q=S+p['sigma']**2*(data['N']*data['dt']-t)
        kernel=data['theta']*np.exp(-perceived**2/(2*Q))/np.sqrt(2*np.pi*Q)
        r=p['r'] if data['estimate_r'] else data['ratio']
        point_process+=data['count_i'][n]*np.log(r*kernel/p['c_i']/24)
        point_process+=data['count_j'][n]*np.log(r*kernel/p['c_j']/24)
        point_process-=r*kernel*(1/p['c_i']+1/p['c_j'])*data['dt']
        drift=kernel*(1/p['c_i']-1/p['c_j'])*data['dt']
        perceived+=drift-np.expm1(-p['lambda']*S*data['dt'])*(data['public_gap'][n]-perceived)
        mu+=drift; var+=p['sigma']**2*data['dt']; elapsed+=data['dt']; drift_sum+=drift
        if data['count_i'][n]+data['count_j'][n]:
            noise=1/(p['lambda']*elapsed); v=var+noise; y=data['public_gap'][n+1]
            total+=-.5*(np.log(2*np.pi*v)+(y-mu)**2/v)
            mu+=var/v*(y-mu); var*=noise/v; elapsed=0.
            times.append((n+1)*data['dt']); means.append(p['mu_0']+drift_sum); obs.append(y); noises.append(noise)
    cov=p['S0']+p['sigma']**2*np.minimum.outer(times,times)+np.diag(noises)
    dense=multivariate_normal.logpdf(obs,mean=means,cov=cov)
    if not np.isclose(total,dense,atol=1e-8,rtol=0): raise RuntimeError((total,dense))
    return dict(innovations_logpdf=total,dense_logpdf=dense,difference=total-dense,
                point_process_loglik=point_process)


def run(model,name,source,estimate_r,*,warmup,samples,seed,mu_sd=1.,S0_logmean=np.log(2.)):
    folder=OUT/name
    folder.mkdir(parents=True,exist_ok=True)
    data=load_data(source,estimate_r,mu_sd,S0_logmean)
    write_json(folder/'data.json',data)
    checks=[likelihood_check(data,p) for p in
            (TRUTH,dict(TRUTH,c_i=.8,c_j=1.8,sigma=1.5,mu_0=.8,S0=1.))]
    print('Starting',name,'counts',sum(data['count_i']),sum(data['count_j']),flush=True)
    initial=dict(c_i=1.,c_j=1.2,sigma=2.,**{'lambda':1.},mu_0=0.,S0=2.)
    if estimate_r: initial['log_r_free']=[np.log(15.)]
    fit=model.sample(data=data,chains=4,parallel_chains=4,iter_warmup=warmup,
        iter_sampling=samples,seed=seed,adapt_delta=.95,max_treedepth=12,
        inits=initial,output_dir=str(folder),show_progress=False,show_console=False,refresh=200)
    summary=fit.summary(); result={}
    for p in ('c_i','c_j','sigma','lambda','mu_0','S0')+(('r',) if estimate_r else ()):
        draws=fit.stan_variable(p).reshape(-1); row=summary.loc[p]
        low,high=np.quantile(draws,[.025,.975]); truth=TRUTH[p]
        result[p]=dict(true=truth,mean=draws.mean(),sd=draws.std(ddof=1),
            error=draws.mean()-truth,rmsd=np.sqrt(np.mean((draws-truth)**2)),
            q2_5=low,q97_5=high,covers_truth=bool(low<=truth<=high),
            rhat=row['R_hat'],ess_bulk=row['ESS_bulk'],ess_tail=row['ESS_tail'])
    method=fit.method_variables()
    diagnostics=dict(divergences=int(method['divergent__'].sum()),
        treedepth_hits=int((method['treedepth__']>=12).sum()),
        max_rhat=max(v['rhat'] for v in result.values()),
        min_ess_bulk=min(v['ess_bulk'] for v in result.values()),
        min_ess_tail=min(v['ess_tail'] for v in result.values()),
        bfmi=[float(np.mean(np.diff(method['energy__'][:,j])**2)/np.var(method['energy__'][:,j])) for j in range(4)])
    diagnostics['passed']=bool(diagnostics['divergences']==0 and diagnostics['treedepth_hits']==0 and
        diagnostics['max_rhat']<=1.01 and diagnostics['min_ess_bulk']>=400 and
        diagnostics['min_ess_tail']>=400 and min(diagnostics['bfmi'])>=.3)
    result['diagnostics']=diagnostics
    write_json(folder/'summary.json',result)
    write_json(folder/'run.json',dict(source=source,seed=seed,warmup=warmup,samples=samples,
        csv_files=fit.runset.csv_files,likelihood_checks=checks,
        stan_sha256=hashlib.sha256(STAN.read_bytes()).hexdigest(),
        data_sha256=hashlib.sha256((folder/'data.json').read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256((DATA/f'{source}_paths.npz').read_bytes()).hexdigest()))
    print(name,json.dumps(result,default=float),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['baseline','sensitivity','replications'],default='baseline')
    parser.add_argument('--warmup',type=int,default=1000)
    parser.add_argument('--samples',type=int,default=1000)
    args=parser.parse_args()
    model=CmdStanModel(stan_file=str(STAN))
    if args.mode=='baseline':
        for name,free,seed in [('fixed_r',False,7100),('estimated_r',True,7200)]:
            run(model,name,'baseline',free,warmup=args.warmup,samples=args.samples,seed=seed)
    elif args.mode=='sensitivity':
        for name,kwargs in [('mu_sd2',dict(mu_sd=2.)),('S0_median4',dict(S0_logmean=np.log(4.)))]:
            run(model,name,'baseline',False,warmup=args.warmup,samples=args.samples,seed=7300,**kwargs)
    else:
        for i in range(20):
            run(model,f'replicate{i+1:02}',f'pool_contest{i+1:02}',False,
                warmup=args.warmup,samples=args.samples,seed=8000+100*i)


if __name__=='__main__': main()
