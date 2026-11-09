"""Portable version of the recovered Task4 global-path importance protocol.

Proposal and 32-block diagnostics retain the original definitions and gates.
Tail corrections are linked to the posterior/input identities of their run.
A fresh run generates matching corrections through the specialized tail methods.
"""
import json
import math
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
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
        self.modes = {}
        self.prepare_density()

    def prepare_density(self):
        dim = self.dim
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
                    frozen_before_production=True, numerator_modes=self.modes,
                    uses_no_test_counts_to_fit_proposal=not self.modes)


    def include_numerator_modes(self, points, target, hours):
        pilot = points[np.linspace(0, len(points)-1, min(4096, len(points)), dtype=int)]
        cov = np.cov(pilot, rowvar=False) + np.eye(self.dim)*1e-7
        chol = np.linalg.cholesky(cov)
        center = np.median(pilot, axis=0)
        indices = [hour-target.cutoff-1 for hour in hours]
        training, numerators = [], []
        for chunk in np.array_split(pilot, 8):
            log_training, path = target.evaluate(chunk)
            training.append(log_training)
            numerators.append(path[:, indices])
        lp = np.concatenate(training)
        ll = np.concatenate(numerators, axis=0)
        constrained = [0, 1, 2, 3, 5]
        lower = math.log(.01) + 1e-7
        bounds = [(lower, None) if j in constrained else (None, None) for j in range(self.dim)]
        means, scales, weights = [], [], []
        for column, hour in enumerate(hours):
            index = hour - target.cutoff - 1

            def objective(z):
                training, path = target.evaluate(z[None, :])
                value = training[0] + path[0, index]
                return -float(value) if np.isfinite(value) else 1e100

            starts = [*pilot[np.argsort(lp+ll[:, column])[-4:]], center, target.means]
            for factor in (2., 4.):
                start = center.copy()
                start[2:4] += [math.log(factor), -math.log(factor)]
                starts.append(start)
            results = []
            for start in starts:
                start = start.copy()
                start[constrained] = np.maximum(start[constrained], lower+1e-4)
                results.append(minimize(objective, start, method='L-BFGS-B', bounds=bounds,
                    options=dict(maxiter=600, ftol=1e-11, gtol=1e-6, maxls=40)))
            self.modes[str(hour)] = [dict(point=r.x.tolist(), negative_log_numerator=float(r.fun),
                                          optimizer_converged=bool(r.success)) for r in results]
            best = min(r.fun for r in results)
            chosen = []
            for r in sorted(results, key=lambda item: item.fun):
                if r.fun >= 1e100 or r.fun > best+30:
                    break
                if any(np.linalg.norm(np.linalg.solve(chol, r.x-other)) < .05 for other in chosen):
                    continue
                chosen.append(r.x)
                if len(chosen) == 4:
                    break
            if not chosen:
                raise RuntimeError(f'No finite predictive numerator mode for hour {hour}')
            for point in chosen:
                steps = .001*np.maximum(1., np.abs(point))
                steps[constrained] = np.minimum(steps[constrained],
                    np.maximum(1e-8, .2*(point[constrained]-math.log(.01))))
                precision = numerical_hessian(objective, point, steps)
                if np.isfinite(precision).all():
                    values, vectors = np.linalg.eigh(precision)
                    local_cov = (vectors/np.clip(values, 1e-3, 1e8)) @ vectors.T
                    local_cov += np.eye(self.dim)*1e-7
                else:
                    local_cov = cov
                for scale in (.5, 1., 2.):
                    means.append(point)
                    scales.append(scale*local_cov*(self.df-2)/self.df)
                    weights.append(.5/(len(hours)*len(chosen)*3))
        self.means = np.concatenate((self.means, means))
        self.scales = np.concatenate((self.scales, scales))
        self.weights = np.r_[.5*self.weights, weights]
        self.prepare_density()


def numerical_hessian(function, point, steps):
    directions = np.diag(steps)
    matrix = np.empty((len(point), len(point)))
    origin = function(point)
    for i, di in enumerate(directions):
        matrix[i, i] = (function(point+di)-2*origin+function(point-di))/steps[i]**2
        for j, dj in enumerate(directions[:i]):
            matrix[i, j] = matrix[j, i] = (
                function(point+di+dj)-function(point+di-dj)
                -function(point-di+dj)+function(point-di-dj))/(4*steps[i]*steps[j])
    return matrix


class Target:
    def __init__(self,raw,cutoff,free):
        self.raw=raw;self.cutoff=cutoff;self.free=free;self.engine=FullPath()
        self.data=training_data(raw,'early',cutoff);self.prior=prior(self.data)
        self.names=(*e.PARAMETERS,'deadline_power','gap_log_scale') if free else e.PARAMETERS
        self.dim=len(self.names)
        self.means = np.r_[self.prior['prior_mean'], [.5, 0.]] if free else np.array(self.prior['prior_mean'])

    def evaluate(self,z):
        out=np.full(len(z),-np.inf);path=np.full((len(z),self.raw['N_Delta']-self.cutoff),-np.inf)
        good=np.isfinite(z).all(axis=1)&(z[:,:4]>=math.log(.01)).all(axis=1)&(z[:,5]>=math.log(.01))
        good&=(z[:,:4]<700).all(axis=1)&(z[:,5]<700)
        if self.free:good&=z[:,7]<700
        indices=np.flatnonzero(good)
        if not len(indices):return out,path
        x=z[good];columns={k:np.exp(x[:,j]) if j<4 else x[:,j] for j,k in enumerate(self.names)}
        likelihood,ll=self.engine.evaluate(self.raw,columns,self.cutoff,self.free)
        scales=np.r_[self.prior['prior_sd'],[.5,.5]] if self.free else np.array(self.prior['prior_sd'])
        # Lognormal density and log-coordinate Jacobian cancel for first four.
        logprior=(-.5*((x-self.means)/scales)**2-np.log(scales)-.5*math.log(2*math.pi)).sum(axis=1)
        out[indices]=likelihood+logprior;path[indices]=ll
        return out,path


def integrate(row,accepted_path,model,stage,budgets=(32768,65536,131072,262144,524288,1048576),seed_offset=0, *, repair_status=None):
    from pipeline.jk9_prediction import raw_input
    stage=Path(stage)
    if stage.exists():raise FileExistsError('Use a new integration run directory; preserve previous diagnostics')
    if len(budgets)<2 or len(set(budgets))!=len(budgets) or any(n<32 or n%32 for n in budgets):
        raise ValueError('At least two distinct budgets, each divisible by 32')
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
    H = raw['N_Delta']-row['cutoff_hour']
    hours = []
    if repair_status is not None:
        repair_status = Path(repair_status)
        original = json.loads(repair_status.read_text())
        if any(original.get(k) != v for k, v in dict(row, model=model,
                accepted_sha256=digest(accepted_path)).items()):
            raise ValueError('Repair status must match the frozen split, model and accepted posterior')
        hours = sorted(set(original['records'][-1]['unrepaired_rare_probability_hours'])
                       | set(original.get('tail_hours', [])))
        if original['passed'] or not hours:
            raise ValueError('Repair requires an unresolved integration with diagnosed rare hours')
        if any(not row['cutoff_hour'] < hour <= raw['N_Delta'] for hour in hours):
            raise ValueError('Tail hours must lie in the frozen test window')
        old_offsets = {original['seed_offset']+j*1000000 for j in range(len(original['records']))}
        new_offsets = {seed_offset+j*1000000 for j in range(len(budgets))}
        if old_offsets & new_offsets:
            raise ValueError('Use a different seed offset for independent repair production')
        proposal.include_numerator_modes(points, target, hours)
    tail_indices = [hour-row['cutoff_hour']-1 for hour in hours]
    stage.mkdir(parents=True);write_json(stage/'proposal.json',proposal.record())
    state=dict(model=model,**row,accepted_sha256=digest(accepted_path),proposal_sha256=digest(stage/'proposal.json'),
               prior=target.prior,source_sha256={str(p.relative_to(ROOT)):digest(p) for p in
                   [source,Path(__file__).resolve(),ROOT/'pipeline/prediction/native.py',ROOT/'pipeline/prediction/full_path.cpp']},
               seed_offset=seed_offset,records=[],passed=False,tail_hours=hours,
               repair_source_sha256=digest(repair_status) if repair_status is not None else None,
               scope='Independent full-path integration of the fixed training posterior and hourly joint probabilities')
    previous=None
    for repeat,n in enumerate(budgets):
        x=proposal.sample(n,20261025+8000000+seed_offset+cid*1009+repeat*1000000+extra)
        g0=[];g1=[];g2=[];lweights=[];failures=[];tail_weights=[]
        for block in np.array_split(x, 32):
            block_den, block_num, block_sq = [], [], []
            for start in range(0, len(block), 512):
                chunk = block[start:start+512]
                lp,ll=target.evaluate(chunk);w=lp-proposal.logq(chunk);lw=w[:,None]+ll
                if np.isnan(w).any() or np.isnan(lw).any():
                    raise RuntimeError('Nonfinite integration arithmetic; no acceptance')
                block_den.append(logsumexp(w))
                block_num.append(logsumexp(lw,axis=0))
                block_sq.append(logsumexp(2*lw,axis=0))
                lweights.append(w);failures.append(~np.isfinite(ll).all(axis=1))
                if hours:tail_weights.append(lw[:,tail_indices])
            g0.append(float(logsumexp(block_den)-math.log(len(block))))
            g1.append(logsumexp(block_num,axis=0)-math.log(len(block)))
            g2.append(logsumexp(block_sq,axis=0)-math.log(len(block)))
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
        rare=[row['cutoff_hour']+j+1 for j in range(H) if j not in tail_indices and (essp[j]<400 or lp[j]<-20)]
        ess=float(1/(normw@normw));failure=float(normw[bad].sum());score=float(lp.mean());halfdiff=abs(half[0]-half[1])
        consistent=previous is None or abs(score-previous['mean_joint_logscore'])<=5*math.hypot(mcse,previous['mean_logscore_mcse'])+1e-4
        tail = {}
        if hours:
            num = np.concatenate(tail_weights)
            nw = np.exp(num-logsumexp(num,axis=0))
            tail_se = np.sqrt(n/(n-1)*np.sum((nw-normw[:,None])**2,axis=0))
            for k, (hour, index) in enumerate(zip(hours, tail_indices, strict=True)):
                se_hour = float(tail_se[k])
                ess_hour = float(1/np.sum(nw[:,k]**2))
                max_weight = float(nw[:,k].max())
                logscore = float(lp[index])
                prev = previous['tail_diagnostics'][str(hour)] if previous else None
                stable = prev is None or abs(logscore-prev['joint_logscore']) <= (
                    5*math.hypot(se_hour,prev['logscore_mcse'])+.001)
                tail[str(hour)] = dict(joint_logscore=logscore, logscore_mcse=se_hour,
                    numerator_ess=ess_hour, maximum_numerator_weight=max_weight,
                    screen_passed=bool(ess_hour>=1000 and max_weight<.01 and se_hour<=.02))
                consistent = consistent and stable
        screen=bool(ess>=1000 and normw.max()<.01 and moment and failure<1e-8 and not rare
                    and all(t['screen_passed'] for t in tail.values()) and mcse<=.001 and halfdiff<=max(.002,4*mcse))
        rec=dict(samples=n,mean_joint_logscore=score,mean_logscore_mcse=mcse,half_score_difference=halfdiff,
            denominator_ess=ess,maximum_denominator_weight=float(normw.max()),denominator_moment_check=moment,
            denominator_moment_difference=(mu-reference_mean).tolist(),denominator_moment_tolerance=tolerance.tolist(),
            future_nonfinite_path_posterior_mass=failure,unrepaired_rare_probability_hours=rare,
            minimum_probability_ess=float(essp.min()),tail_diagnostics=tail,screen_passed=screen,consistent_with_previous=bool(consistent))
        state['records'].append(rec)
        np.savez_compressed(stage/f'production_{n}.npz',z=x,log_training_importance=w,block_log_denominator=g0,
            block_log_numerators=g1,block_log_numerator_squares=g2,joint_logscores=lp,probability_weight_ess=essp)
        state['passed']=bool(previous and previous['screen_passed'] and screen and consistent)
        write_json(stage/'status.json',state);print(model,cid,n,screen,state['passed'],flush=True)
        if state['passed']:break
        previous=rec
    return state
