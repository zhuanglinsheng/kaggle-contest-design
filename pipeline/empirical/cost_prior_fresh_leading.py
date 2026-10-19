"""Fresh leading fit for the prespecified relative-cost sensitivity."""
import json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from pipeline.empirical import nonlinear_reestimate as n
from pipeline.empirical import fast_leading_prior as f
from pipeline.empirical import paired_prior_improvement as p
from pipeline.empirical import cost_prior_trial as trial
CID=5390
BASE=f.LeadingTarget
CONFIG=trial.CANDIDATES['relative_cost_preserved']


class Target(BASE):
    def prior(self,z):
        old=super().prior(z)
        if not np.isfinite(old):return old
        return old+float(trial.log_ratio(np.asarray(z)[None,:],CONFIG)[0])


def configure():
    f.configure('scale_weak','empirical');n.Likelihood=Target
    n.OUT=trial.OUT/'fresh_relative_cost/leading_fast';n.OUT.mkdir(parents=True,exist_ok=True)


def prepare():
    configure();path=n.OUT/f'proposal_{CID}.json'
    if path.exists():return
    folder=p.OUT/f'scale_weak/leading_fast/contest_{CID}'
    z=np.concatenate([np.load(folder/f'chain_{j}.npz')['z'] for j in range(1,5)])
    w=np.exp(trial.log_ratio(z,CONFIG)-max(trial.log_ratio(z,CONFIG)));w/=w.sum()
    threshold=float(np.median(z[:,2]));groups=[z[:,2]<=threshold,z[:,2]>threshold]
    components=[]
    for mask in groups:
        wg=w[mask];wg/=wg.sum();zg=z[mask];mean=wg@zg
        cov=(zg-mean).T@(wg[:,None]*(zg-mean))+np.eye(6)*1e-6
        components.append(dict(weight=float(w[mask].sum()),mean=mean.tolist(),covariance=cov.tolist()))
    mean=w@z;cov=(z-mean).T@(w[:,None]*(z-mean))+np.eye(6)*1e-6
    n.save_json(path,dict(contest_id=CID,kernel_sha256=n.digest(n.SOURCE),mode_z=mean.tolist(),covariance=cov.tolist(),
        mixture_components=components,proposal_only=True,source_chain_sha256=[n.digest(folder/f'chain_{j}.npz') for j in range(1,5)]))
    n.save_json(n.OUT/'target.json',dict(candidate=CONFIG,source_prior='scale_weak',script_sha256=n.digest(__file__),trial_script_sha256=n.digest(trial.__file__),
        support='Same physical floors and remaining priors as source.'))


def chain(j):
    configure();n.sample_mixture_gaussian(CID,j,2000,60000,.025,64,261100+CID,pilot=True)


def summarize():
    configure();data,_,_,_=n.load_baseline(CID);paths=[n.OUT/f'contest_{CID}/chain_{j}.npz' for j in range(1,5)]
    z=np.array([np.load(path)['z'] for path in paths]);values=trial.quantities(z.reshape(-1,6),data).reshape(4,z.shape[1],-1)
    diag=dict(maximum_rank_rhat=max(n.rank_rhat(values[:,:,k]) for k in range(values.shape[2])),
        minimum_ess=min(min(n.rank_ess(values[:,:,k])) for k in range(values.shape[2])))
    diag['passed']=diag['maximum_rank_rhat']<=1.01 and diag['minimum_ess']>=400
    means,sds,q=n.weighted_summary(values.reshape(-1,values.shape[2]),np.ones(z.shape[0]*z.shape[1])/(z.shape[0]*z.shape[1]))
    names=trial.report.NAMES+('cost_ratio',)
    result=dict(diagnostics=diag,chain_sha256=[n.digest(path) for path in paths])
    if diag['passed']:result['summaries']={name:dict(mean=float(means[k]),sd=float(sds[k]),q025=float(q[k,0]),median=float(q[k,1]),q975=float(q[k,2])) for k,name in enumerate(names)}
    n.save_json(n.OUT/'summary.json',result);print(json.dumps(diag),flush=True)


if __name__=='__main__':
    prepare()
    with ProcessPoolExecutor(max_workers=4) as pool:list(pool.map(chain,range(1,5)))
    summarize()
