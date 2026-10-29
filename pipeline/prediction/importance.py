"""Portable version of the recovered Task4 global-path importance protocol.

Proposal and 32-block diagnostics retain the original definitions and gates.
Historical tail replacements require separate certificates and are not reused.
"""
import json
import math
from pathlib import Path
import numpy as np
from scipy.special import gammaln,logsumexp
from pipeline.publication import ROOT,digest,write_json
from pipeline.config import resolve_manifest_path
from pipeline.jk9_fit import training_data
from pipeline.empirical import estimate as e,nonlinear_reestimate as nd
from pipeline.empirical.unified_prior_reestimate import prior
from pipeline.prediction.native import FullPath


class Proposal:
    def __init__(self,points,seed):
        pilot=points[np.linspace(0,len(points)-1,min(4096,len(points)),dtype=int)]
        self.dim=dim=points.shape[1];cov=np.cov(pilot,rowvar=False)+np.eye(dim)*1e-7
        L=np.linalg.cholesky(cov);center=pilot.mean(0);white=np.linalg.solve(L,(pilot-center).T).T
        rng=np.random.default_rng(seed);centers=white[rng.choice(len(white),16,replace=False)].copy()
        for _ in range(30):
            labels=np.argmin(np.sum((white[:,None,:]-centers[None,:,:])**2,axis=2),axis=1)
            for k in range(16):
                if np.any(labels==k):centers[k]=white[labels==k].mean(0)
        base=[];sizes=[]
        for k in range(16):
            g=pilot[labels==k]
            if len(g)>=20:base.append((g.mean(0),np.cov(g,rowvar=False)+np.eye(dim)*1e-6));sizes.append(len(g))
        if not sizes:raise ValueError('Too few training draws for original 16-cluster proposal')
        base.append((center,4*cov));self.weights=np.r_[.95*np.asarray(sizes)/sum(sizes),.05]
        self.means=np.array([m for m,c in base]);self.df=7.
        self.scales=np.array([c for m,c in base])*(self.df-2)/self.df
        self.chols=np.linalg.cholesky(self.scales);self.inverses=np.linalg.inv(self.scales)
        self.constant=(gammaln((self.df+dim)/2)-gammaln(self.df/2)-dim/2*math.log(self.df*math.pi)
                       -.5*np.linalg.slogdet(self.scales)[1])

    def logq(self,z):
        d=z[:,None,:]-self.means;v=np.einsum('nki,kij,nkj->nk',d,self.inverses,d)
        return logsumexp(np.log(self.weights)[None,:]+self.constant[None,:]
            -(self.df+self.dim)/2*np.log1p(v/self.df),axis=1)

    def sample(self,n,seed):
        rng=np.random.default_rng(seed);idx=rng.choice(len(self.weights),size=n,p=self.weights)
        return self.means[idx]+np.einsum('nij,nj->ni',self.chols[idx],rng.normal(size=(n,self.dim)))/np.sqrt(rng.chisquare(self.df,size=n)/self.df)[:,None]

    def record(self):
        return dict(df=self.df,weights=self.weights.tolist(),means=self.means.tolist(),scales=self.scales.tolist(),
                    frozen_before_production=True,uses_no_test_counts_to_fit_proposal=True)


class Target:
    def __init__(self,raw,cutoff,free):
        self.raw=raw;self.cutoff=cutoff;self.free=free;self.engine=FullPath()
        self.data=training_data(raw,'early',cutoff);self.prior=prior(self.data)
        self.names=(*e.PARAMETERS,'deadline_power','gap_log_scale') if free else e.PARAMETERS
        self.dim=len(self.names)

    def evaluate(self,z):
        out=np.full(len(z),-np.inf);path=np.full((len(z),self.raw['N_Delta']-self.cutoff),-np.inf)
        good=np.isfinite(z).all(axis=1)&(z[:,:4]>=math.log(.01)).all(axis=1)&(z[:,5]>=math.log(.01))
        good&=(z[:,:4]<700).all(axis=1)&(z[:,5]<700)
        if self.free:good&=z[:,7]<700
        indices=np.flatnonzero(good)
        if not len(indices):return out,path
        x=z[good];columns={k:np.exp(x[:,j]) if j<4 else x[:,j] for j,k in enumerate(self.names)}
        likelihood,ll=self.engine.evaluate(self.raw,columns,self.cutoff,self.free)
        means=np.r_[self.prior['prior_mean'],[.5,0.]] if self.free else np.array(self.prior['prior_mean'])
        scales=np.r_[self.prior['prior_sd'],[.5,.5]] if self.free else np.array(self.prior['prior_sd'])
        # Lognormal density and log-coordinate Jacobian cancel for first four.
        logprior=(-.5*((x-means)/scales)**2-np.log(scales)-.5*math.log(2*math.pi)).sum(axis=1)
        out[indices]=likelihood+logprior;path[indices]=ll
        return out,path


def integrate(row,accepted_path,model,stage,budgets=(32768,65536,131072,262144,524288,1048576),seed_offset=0):
    from pipeline.jk9_prediction import raw_input
    stage=Path(stage)
    if stage.exists():raise FileExistsError('Use a new integration run directory; preserve previous diagnostics')
    if len(budgets)<2 or any(n<32 or n%32 for n in budgets):raise ValueError('At least two budgets, each divisible by 32')
    accepted_path=Path(accepted_path).resolve();a=json.loads(accepted_path.read_text());config=a['configuration']
    raw=raw_input(row);target=Target(raw,row['cutoff_hour'],model=='C')
    source=ROOT/'pipeline/models'/('free_kernel.stan' if model=='C' else 'leading_order.stan')
    if (not a['diagnostics']['passed'] or config['model']!=model or config['training_cutoff']!=row['cutoff_hour']
            or config['raw_input_sha256']!=row['input_sha256'] or config['prior']!=target.prior
            or config['source_model_sha256']!=digest(source)):
        raise ValueError('Accepted posterior target differs from frozen input/model/prior/window')
    if len(a['chain_paths'])!=4 or len(a['chain_sha256'])!=4:raise ValueError('Four chains required')
    chains=[]
    for p,h in zip(a['chain_paths'],a['chain_sha256'],strict=True):
        p=resolve_manifest_path(p,accepted_path)
        if digest(p)!=h:raise ValueError(f'Changed chain: {p}')
        cols=e.read_columns(p);z=np.array([cols[k] for k in target.names]).T;z[:,:4]=np.log(z[:,:4]);chains.append(z)
    reference=np.array(chains);points=np.concatenate(chains);reference_mean=points.mean(0)
    reference_mcse=np.array([np.std(reference[:,:,j],ddof=1)/math.sqrt(max(nd.effective_sample_size(reference[:,:,j]),1)) for j in range(target.dim)])
    cid=row['contest_id'];extra=100000 if model=='C' else 0
    proposal=Proposal(points,20261025+cid+7000000+extra)
    stage.mkdir(parents=True);write_json(stage/'proposal.json',proposal.record())
    state=dict(model=model,**row,accepted_sha256=digest(accepted_path),proposal_sha256=digest(stage/'proposal.json'),
               prior=target.prior,source_sha256={str(p.relative_to(ROOT)):digest(p) for p in
                   [source,Path(__file__).resolve(),ROOT/'pipeline/prediction/native.py',ROOT/'pipeline/prediction/full_path.cpp']},
               seed_offset=seed_offset,records=[],passed=False,scope='Fresh independent full-path importance integration; no historical tail overrides')
    previous=None;H=raw['N_Delta']-row['cutoff_hour']
    for repeat,n in enumerate(budgets):
        x=proposal.sample(n,20261025+8000000+seed_offset+cid*1009+repeat*1000000+extra)
        g0=[];g1=[];g2=[];lweights=[];failures=[]
        for ids in np.array_split(np.arange(n),32):
            lp,ll=target.evaluate(x[ids]);w=lp-proposal.logq(x[ids]);lw=w[:,None]+ll
            if np.isnan(w).any() or np.isnan(lw).any():raise RuntimeError('Nonfinite integration arithmetic; no acceptance')
            g0.append(float(logsumexp(w)-math.log(len(ids))))
            g1.append(logsumexp(lw,axis=0)-math.log(len(ids)));g2.append(logsumexp(2*lw,axis=0)-math.log(len(ids)))
            lweights.append(w);failures.append(~np.isfinite(ll).all(axis=1))
        w=np.concatenate(lweights);bad=np.concatenate(failures);den=logsumexp(w)-math.log(n)
        normw=np.exp(w-logsumexp(w));g0=np.array(g0);g1=np.array(g1);g2=np.array(g2)
        lp=logsumexp(g1,axis=0)-math.log(32)-den
        if not np.isfinite(lp).all() or (lp>1e-8).any():raise RuntimeError('Invalid predictive probabilities; no acceptance')
        essp=n*np.exp(2*(lp+den)-(logsumexp(g2,axis=0)-math.log(32)))
        mu=normw@x;se=np.sqrt(n/(n-1)*np.sum(normw[:,None]**2*(x-mu)**2,axis=0))
        tolerance=5*np.hypot(se,reference_mcse)+.005;moment=bool(np.all(abs(mu-reference_mean)<=tolerance))
        influence=np.sum(np.exp(g1-(lp+den)[None,:])-np.exp(g0-den)[:,None],axis=1)/H
        mcse=float(np.std(influence,ddof=1)/math.sqrt(32));half=[]
        for ids in [np.arange(16),np.arange(16,32)]:
            half.append(float((logsumexp(g1[ids],axis=0)-math.log(16)-(logsumexp(g0[ids])-math.log(16))).mean()))
        rare=[row['cutoff_hour']+j+1 for j in range(H) if essp[j]<400 or lp[j]<-20]
        ess=float(1/(normw@normw));failure=float(normw[bad].sum());score=float(lp.mean());halfdiff=abs(half[0]-half[1])
        consistent=previous is None or abs(score-previous['mean_joint_logscore'])<=5*math.hypot(mcse,previous['mean_logscore_mcse'])+1e-4
        screen=bool(ess>=1000 and normw.max()<.01 and moment and failure<1e-8 and not rare and mcse<=.001 and halfdiff<=max(.002,4*mcse))
        rec=dict(samples=n,mean_joint_logscore=score,mean_logscore_mcse=mcse,half_score_difference=halfdiff,
            denominator_ess=ess,maximum_denominator_weight=float(normw.max()),denominator_moment_check=moment,
            denominator_moment_difference=(mu-reference_mean).tolist(),denominator_moment_tolerance=tolerance.tolist(),
            future_nonfinite_path_posterior_mass=failure,unrepaired_rare_probability_hours=rare,
            minimum_probability_ess=float(essp.min()),screen_passed=screen,consistent_with_previous=bool(consistent))
        state['records'].append(rec)
        np.savez_compressed(stage/f'production_{n}.npz',z=x,log_training_importance=w,block_log_denominator=g0,
            block_log_numerators=g1,block_log_numerator_squares=g2,joint_logscores=lp,probability_weight_ess=essp)
        state['passed']=bool(previous and previous['screen_passed'] and screen and consistent)
        write_json(stage/'status.json',state);print(model,cid,n,screen,state['passed'],flush=True)
        if state['passed']:break
        previous=rec
    return state
