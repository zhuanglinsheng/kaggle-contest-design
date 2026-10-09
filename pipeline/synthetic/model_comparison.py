"""Same-data NS/SS comparison; never simulates or modifies the accepted DGP.

Normalized bridge sampling is primary. MLE/BIC are auxiliary diagnostics.
Run: python -m pipeline.synthetic.model_comparison
Recompute only evidence/optimization: append --reuse.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from cmdstanpy import CmdStanModel, from_csv
from scipy.special import expit, logit, logsumexp
from scipy.stats import lognorm, norm, multivariate_normal

from pipeline.config import ROOT
from pipeline.synthetic.table1 import DATA, TRUTH, load_data, likelihood_check, write_json

STAN = ROOT / 'synthetic_data/hourly_comparison.stan'
OUT = ROOT / 'results/synthetic/initialization_comparison'
BOUNDS = np.array([[.1,5.],[.1,5.],[.5,10.],[1e-6,100.],[-20.,20.]])
PARAMS = ('c_i','c_j','sigma','lambda','mu_0','r')
# Declared before fitting; all configurations estimate r and use the same data.
CONFIGS = [('steady',0,2.,9100), ('nonsteady',1,2.,9200),
           ('nonsteady_prior1',1,1.,9300), ('nonsteady_prior4',1,4.,9400)]


def log_likelihood(x, data):
    """Vectorized independent implementation, columns: common five, log r, S0."""
    x = np.atleast_2d(x)
    ci,cj,sigma,lam,mu0,logr,S0 = x.T
    perceived=mu0.copy(); mean=mu0.copy(); var=S0.copy()
    steady=sigma/np.sqrt(lam); rate=sigma*np.sqrt(lam)
    r=np.exp(logr); sig2=sigma*sigma; total=np.zeros(len(x)); elapsed=0.
    dt=data['dt']; N=data['N']
    for n in range(N):
        t=n*dt; z=np.tanh(rate*t)
        S=steady*(S0+steady*z)/(steady+S0*z)
        Q=S+sig2*(N*dt-t)
        lk=np.log(data['theta'])-.5*np.log(2*np.pi*Q)-perceived**2/(2*Q)
        mi=np.exp(lk)/ci; mj=np.exp(lk)/cj; drift=(mi-mj)*dt
        if data['count_i'][n]:
            total+=data['count_i'][n]*(logr+lk-np.log(ci)-np.log(24))
        if data['count_j'][n]:
            total+=data['count_j'][n]*(logr+lk-np.log(cj)-np.log(24))
        total-=r*(mi+mj)*dt
        perceived+=drift-np.expm1(-lam*S*dt)*(data['public_gap'][n]-perceived)
        mean+=drift; var+=sig2*dt; elapsed+=dt
        if data['count_i'][n]+data['count_j'][n]:
            noise=1/(lam*elapsed); v=var+noise; innovation=data['public_gap'][n+1]-mean
            total-=.5*(np.log(2*np.pi*v)+innovation**2/v)
            mean+=var/v*innovation; var*=noise/v; elapsed=0.
    return total


def log_prior(x, data):
    """Density w.r.t. d(ci,cj,sigma,lambda,mu0,logr[,S0]), including truncation."""
    x=np.atleast_2d(x); ans=np.zeros(len(x))
    for j,(loc,sd) in enumerate([(0.,.75),(0.,.75),(np.log(2.),.5),(0.,.75)]):
        dist=lognorm(s=sd,scale=np.exp(loc)); lo,hi=BOUNDS[j]
        ans+=dist.logpdf(x[:,j])-np.log(dist.cdf(hi)-dist.cdf(lo))
    dist=norm(0,data['mu_prior_sd'])
    ans+=dist.logpdf(x[:,4])-np.log(dist.cdf(20)-dist.cdf(-20))
    ans+=norm.logpdf(x[:,5],np.log(data['ratio']),.35)
    if data['free_S0']:
        dist=lognorm(s=data['S0_prior_logsd'],scale=np.exp(data['S0_prior_logmean']))
        ans+=dist.logpdf(x[:,6])-dist.logsf(.01)
        ans=np.where(x[:,6]>.01,ans,-np.inf)
    valid=np.all((x[:,:5]>BOUNDS[:,0]) & (x[:,:5]<BOUNDS[:,1]),axis=1)
    return np.where(valid,ans,-np.inf)


def to_unconstrained(x, free):
    x=np.atleast_2d(x); u=x[:,:6+free].copy()
    u[:,:5]=logit((x[:,:5]-BOUNDS[:,0])/(BOUNDS[:,1]-BOUNDS[:,0]))
    if free: u[:,6]=np.log(x[:,6]-.01)
    return u


def from_unconstrained(u, free):
    u=np.atleast_2d(u); x=np.empty((len(u),7))
    width=BOUNDS[:,1]-BOUNDS[:,0]
    x[:,:5]=BOUNDS[:,0]+width*expit(u[:,:5]); x[:,5]=u[:,5]
    jac=np.sum(np.log(width)-np.logaddexp(0,-u[:,:5])-np.logaddexp(0,u[:,:5]),axis=1)
    if free:
        x[:,6]=.01+np.exp(u[:,6]); jac+=u[:,6]
    else: x[:,6]=x[:,2]/np.sqrt(x[:,3])
    return x,jac


def log_target(u,data):
    x,jac=from_unconstrained(u,data['free_S0'])
    return log_likelihood(x,data)+log_prior(x,data)+jac


def bridge_iteration(log_ratio_posterior, log_ratio_proposal, tol=1e-10):
    """Meng--Wong bridge fixed point, with sample-size mixture weights."""
    a=np.asarray(log_ratio_posterior); b=np.asarray(log_ratio_proposal)
    s1=len(a)/(len(a)+len(b)); s2=1-s1
    logz=float(np.median(a))
    for iteration in range(1,10001):
        den_b=np.logaddexp(np.log(s1)+b,np.log(s2)+logz)
        den_a=np.logaddexp(np.log(s1)+a,np.log(s2)+logz)
        new=float(logsumexp(b-den_b)-np.log(len(b))
                  -logsumexp(-den_a)+np.log(len(a)))
        if abs(new-logz)<tol:
            return dict(log_evidence=new,iterations=iteration)
        logz=new
    raise RuntimeError('Bridge iteration did not converge')


def evidence(fit,data,seed):
    # First half of every chain fits the proposal; disjoint second halves
    # estimate evidence. Keep chain identities for Monte Carlo diagnostics.
    arr=fit.draws(concat_chains=False)
    ix={name:fit.column_names.index(name) for name in
        ('c_i','c_j','sigma','lambda','mu_0','log_r','S0')}
    physical=arr[:,:,[ix[n] for n in ix]]
    half=len(physical)//2
    train=to_unconstrained(physical[:half].reshape(-1,7),data['free_S0'])
    post=to_unconstrained(physical[half:].reshape(-1,7),data['free_S0'])
    proposal=multivariate_normal(train.mean(axis=0),np.cov(train,rowvar=False)*1.1)
    target_post=log_target(post,data); ratio_post=target_post-proposal.logpdf(post)
    stan_lp=arr[half:,:,fit.column_names.index('lp__')].reshape(-1)
    target_error=float(np.max(np.abs(target_post-stan_lp)))
    if target_error>1e-7:
        raise RuntimeError(f'Bridge target differs from Stan posterior: {target_error}')
    pooled=[]; chainwise=[]; importance=[]
    for rep in range(4):
        candidate=proposal.rvs(size=len(post),random_state=np.random.default_rng(seed+rep))
        ratio_prop=log_target(candidate,data)-proposal.logpdf(candidate)
        pooled.append(bridge_iteration(ratio_post,ratio_prop))
        importance.append(float(logsumexp(ratio_prop)-np.log(len(ratio_prop))))
        # Each diagnostic uses a different chain and independent proposal draws.
        chainwise.append(bridge_iteration(ratio_post[rep::4],ratio_prop)['log_evidence'])
    vals=np.array([v['log_evidence'] for v in pooled])
    return dict(log_evidence=vals.mean(),repetitions=pooled,
                proposal_repeat_sd=vals.std(ddof=1),chain_log_evidence=chainwise,
                chain_mcse_diagnostic=np.std(chainwise,ddof=1)/2,
                max_abs_target_minus_stan_lp=target_error,
                proposal_importance_log_evidence=importance,
                posterior_evaluation_draws=len(post),proposal_draws_per_repeat=len(post),
                method='normal bridge; disjoint proposal-fit/evaluation halves; covariance inflation 1.1')


def stan_params(p,free):
    result={n:float(p[n]) for n in PARAMS[:5]}
    result['log_r']=float(np.log(p['r']))
    result['S0_free']=[float(p['S0'])] if free else []
    return result


def fit_model(model, name,free,median,seed,args):
    folder=OUT/name; folder.mkdir(parents=True,exist_ok=True)
    data=load_data('baseline',True,S0_logmean=np.log(median))
    data.pop('estimate_r'); data.update(free_S0=free,use_prior=1)
    write_json(folder/'data.json',data)
    if args.reuse and (folder/'run.json').exists():
        run=json.loads((folder/'run.json').read_text())
        for key,path in [('source_sha256',DATA/'baseline_paths.npz'),('stan_sha256',STAN),
                         ('data_sha256',folder/'data.json')]:
            if run[key]!=hashlib.sha256(path.read_bytes()).hexdigest():
                raise RuntimeError(f'Cannot reuse changed {key}')
        fit=from_csv(run['csv_files'])
    else:
        print('Sampling',name,flush=True)
        fit=model.sample(data=data,chains=4,parallel_chains=4,iter_warmup=args.warmup,
                         iter_sampling=args.samples,seed=seed,adapt_delta=.95,max_treedepth=12,
                         inits=stan_params(dict(TRUTH,S0=2.),free),sig_figs=18,
                         output_dir=str(folder),show_progress=False,refresh=200)
        write_json(folder/'run.json',dict(seed=seed,warmup=args.warmup,samples=args.samples,
            csv_files=fit.runset.csv_files,
            source_sha256=hashlib.sha256((DATA/'baseline_paths.npz').read_bytes()).hexdigest(),
            stan_sha256=hashlib.sha256(STAN.read_bytes()).hexdigest(),
            data_sha256=hashlib.sha256((folder/'data.json').read_bytes()).hexdigest()))
    summary=fit.summary(); result={}
    for namep in PARAMS+(('S0',) if free else ()):
        values=fit.stan_variable(namep); row=summary.loc[namep]
        lo,hi=np.quantile(values,[.025,.975])
        result[namep]=dict(true=TRUTH[namep],mean=values.mean(),sd=values.std(ddof=1),
            rmsd=np.sqrt(np.mean((values-TRUTH[namep])**2)),q2_5=lo,q97_5=hi,
            rhat=row['R_hat'],ess_bulk=row['ESS_bulk'],ess_tail=row['ESS_tail'])
    method=fit.method_variables()
    diag=dict(divergences=int(method['divergent__'].sum()),
        treedepth_hits=int((method['treedepth__']>=12).sum()),
        max_rhat=max(v['rhat'] for v in result.values()),
        min_ess_bulk=min(v['ess_bulk'] for v in result.values()),
        min_ess_tail=min(v['ess_tail'] for v in result.values()),
        bfmi=[np.mean(np.diff(method['energy__'][:,j])**2)/np.var(method['energy__'][:,j]) for j in range(4)])
    diag['passed']=bool(diag['divergences']==0 and diag['treedepth_hits']==0 and
        diag['max_rhat']<=1.01 and diag['min_ess_bulk']>=400 and
        diag['min_ess_tail']>=400 and min(diag['bfmi'])>=.3)
    result['diagnostics']=diag
    write_json(folder/'summary.json',result)
    if not diag['passed']: raise RuntimeError(f'{name}: posterior diagnostics failed')
    checks=[]
    for p in (TRUTH,dict(TRUTH,c_i=.8,c_j=1.8,sigma=1.5,mu_0=.8,S0=1.)):
        if not free: p=dict(p,S0=p['sigma']/np.sqrt(p['lambda']))
        x=np.array([[p[n] for n in PARAMS[:5]]+[np.log(p['r']),p['S0']]])
        dense=likelihood_check(dict(data,estimate_r=1),p)
        lik=float(log_likelihood(x,data)[0])
        joint=float((log_likelihood(x,data)+log_prior(x,data))[0])
        stan=float(model.log_prob(params=stan_params(p,free),data=data,
                                  jacobian=False,sig_figs=18)['lp__'].iloc[0])
        check=dict(vector_minus_dense=lik-dense['innovations_logpdf']-dense['point_process_loglik'],
                   normalized_joint_minus_stan=joint-stan)
        if max(abs(v) for v in check.values())>1e-8: raise RuntimeError(check)
        checks.append(check)
    result['likelihood_checks']=checks
    print('Bridge sampling',name,flush=True)
    result['evidence']=evidence(fit,data,seed+50)
    write_json(folder/'summary.json',result)
    print(name,json.dumps(dict(diagnostics=diag,evidence=result['evidence'])),flush=True)
    return data,fit,result


def maximize(model,data,fit,name,extra=None):
    """Multi-start ML, no prior, no unconstraining Jacobian in objective."""
    folder=OUT/name/'mle'; folder.mkdir(exist_ok=True)
    starts=[dict(TRUTH), {n:float(np.median(fit.stan_variable(n))) for n in PARAMS+('S0',)}]
    for idx in np.linspace(0,len(fit.stan_variable('r'))-1,4,dtype=int):
        starts.append({n:float(fit.stan_variable(n)[idx]) for n in PARAMS+('S0',)})
    if extra: starts.append(extra)
    runs=[]
    for idx,p in enumerate(starts):
        try:
            start_folder=folder/f'start{idx+1}'
            start_folder.mkdir(exist_ok=True)
            fitml=model.optimize(data=dict(data,use_prior=0),inits=stan_params(p,data['free_S0']),
                algorithm='lbfgs',iter=10000,tol_rel_grad=1e4,tol_rel_obj=1e-11,
                jacobian=False,seed=15000+idx,output_dir=str(start_folder),sig_figs=18,
                show_console=False,refresh=1000)
            values=fitml.optimized_params_dict
            runs.append(dict(success=True,log_likelihood=values['log_likelihood'],
                             parameters={n:values[n] for n in PARAMS+('S0',)}))
        except RuntimeError as error:
            runs.append(dict(success=False,error=str(error)))
    good=[v for v in runs if v['success']]
    if not good: raise RuntimeError(f'All MLE starts failed: {name}')
    best=max(good,key=lambda v:v['log_likelihood'])
    result=dict(best=best,starts=runs,k=6+data['free_S0'])
    boundary=[]
    for n,(lo,hi) in zip(PARAMS[:5],BOUNDS):
        if best['parameters'][n]-lo<1e-6*(hi-lo): boundary.append(f'{n}: lower {lo}')
        if hi-best['parameters'][n]<1e-6*(hi-lo): boundary.append(f'{n}: upper {hi}')
    if data['free_S0'] and best['parameters']['S0']-.01<1e-6:
        boundary.append('S0: lower 0.01')
    result['boundary_parameters']=boundary
    result['successful_start_loglik_range']=float(np.ptp([v['log_likelihood'] for v in good]))
    p=best['parameters']
    x=np.array([[p[n] for n in PARAMS[:5]]+[np.log(p['r']),p['S0']]])
    result['independent_loglik_error']=float(log_likelihood(x,data)[0]-best['log_likelihood'])
    if abs(result['independent_loglik_error'])>1e-7: raise RuntimeError(result)
    result['bic_by_n']={str(n):-2*best['log_likelihood']+result['k']*np.log(n)
                        for n in (45,46,91)}
    write_json(OUT/name/'mle.json',result)
    print('MLE',name,json.dumps(result),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--warmup',type=int,default=1000)
    parser.add_argument('--samples',type=int,default=2000)
    parser.add_argument('--reuse',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    write_json(OUT/'design.json',dict(configurations=CONFIGS,common_parameters=PARAMS,
        data='optimal_hourly/nonsteady/baseline_paths.npz',drop_initial_data=False,
        main_comparison='nonsteady / steady',primary='normalized bridge marginal likelihood',
        sensitivity='S0 prior log-median log(1), log(2), log(4); common log-SD 0.75',
        bic='Auxiliary convention n=45 hourly readouts, with n=46 and 91 sensitivity; not grid size.',
        warmup=args.warmup,samples=args.samples))
    model=CmdStanModel(stan_file=str(STAN))
    fits={name:fit_model(model,name,free,median,seed,args)
          for name,free,median,seed in CONFIGS}
    ss=maximize(model,*fits['steady'][:2],'steady')
    ns=maximize(model,*fits['nonsteady'][:2],'nonsteady',extra=ss['best']['parameters'])
    if ns['best']['log_likelihood']+1e-6<ss['best']['log_likelihood']:
        raise RuntimeError('Nested-model likelihood inequality failed')
    base=fits['steady'][2]['evidence']; comparisons={}
    for name in ('nonsteady','nonsteady_prior1','nonsteady_prior4'):
        ev=fits[name][2]['evidence']; logbf=ev['log_evidence']-base['log_evidence']
        comparisons[name]=dict(log_bayes_factor_ns_ss=logbf,bayes_factor_ns_ss=np.exp(logbf),
            log_bf_mcse_diagnostic=np.hypot(ev['chain_mcse_diagnostic'],base['chain_mcse_diagnostic']))
    write_json(OUT/'comparison.json',dict(comparisons=comparisons,mle_steady=ss,mle_nonsteady=ns,
        delta_bic_ns_minus_ss={n:ns['bic_by_n'][n]-ss['bic_by_n'][n] for n in ss['bic_by_n']}))
    print('COMPARISON',json.dumps(comparisons),flush=True)


if __name__=='__main__': main()
