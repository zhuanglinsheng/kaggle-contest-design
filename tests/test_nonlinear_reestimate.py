"""Independent likelihood and equilibrium checks for the paired estimator."""
import math
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.stats import multivariate_normal

from pipeline.empirical.nonlinear_reestimate import Likelihood, rank_rhat, rank_ess
from pipeline.synthetic.nonlinear import Primitives, solve_equilibrium
from pipeline.empirical import nonlinear_reestimate as estimator


class GaussianLikelihood:
    def __init__(self,*args):self.calls=0
    def prior(self,z):return 0.
    def posterior(self,z):
        self.calls+=1
        sd=np.array([1.,.7,.4,.3,.25,.5])
        return float(-.5*np.sum((z/sd)**2))


class GaussianWithPrior(GaussianLikelihood):
    def prior(self,z):
        sd=np.array([1.,.7,.4,.3,.25,.5])
        return float(-.5*np.sum(((z-.5*sd)/(1.5*sd))**2))


class BimodalWithPrior(GaussianLikelihood):
    def prior(self,z):return float(-.5*np.sum((z/3)**2))
    def posterior(self,z):
        from scipy.special import logsumexp
        self.calls+=1
        means=np.array([[-3,0,0,0,0,0],[3,0,0,0,0,0]])
        return float(logsumexp(np.log([.75,.25])-.5*np.sum((z-means)**2,axis=1)))


class NonlinearReestimationTests(unittest.TestCase):
    def setUp(self):
        self.source_patch=patch.object(estimator,'SOURCE',Path(estimator.__file__).with_name('nonlinear_fast_kernel.cpp'))
        self.source_patch.start();self.addCleanup(self.source_patch.stop)

    def data(self):
        return dict(theta=.3,Delta2f=.1,N_Delta=8,T_contest=.8,count_start_idx=3,
                    hat_t_i=[2.2,2.9,6.1],hat_t_j=[4.1,6.2],
                    hat_y=[.1,.1,.2,.2,-.15,-.15,.05,.05])

    def test_likelihood_matches_dense_observation_law(self):
        d=self.data();ci,cj,sigma,lam,mu,r=1.2,1.8,1.1,2.,.12,12.
        z=np.array([math.log(ci),math.log(cj),math.log(sigma),math.log(lam),mu,math.log(r)])
        L=Likelihood(d);perceived=mu;drifts=[];efforts=[]
        steady=sigma/math.sqrt(lam);weight=-math.expm1(-math.sqrt(lam)*sigma*d['Delta2f'])
        drift=0
        for i,y in enumerate(d['hat_y']):
            q=steady+sigma*sigma*(d['T_contest']-i*d['Delta2f'])
            kernel=d['theta']*math.exp(-perceived*perceived/(2*q))/math.sqrt(2*math.pi*q)
            mi,mj=kernel/ci,kernel/cj;efforts.append([mi,mj])
            drift+=(mi-mj)*d['Delta2f'];drifts.append(drift)
            perceived+=(mi-mj)*d['Delta2f']+weight*(y-perceived)
        intensity=r*np.array(efforts)*d['Delta2f']
        counts=np.column_stack(L.counts)
        expected=float(np.sum(counts*np.log(np.maximum(intensity,1e-12)))
                       -np.sum(intensity[d['count_start_idx']-1:]))
        indices=np.flatnonzero(counts.sum(axis=1))
        times=(indices+1)*d['Delta2f'];elapsed=np.diff(np.r_[0,times])
        covariance=steady+sigma*sigma*np.minimum.outer(times,times)
        covariance+=np.diag(1/(lam*elapsed)+1e-9)
        residual=np.array(d['hat_y'])[indices]-mu-np.array(drifts)[indices]
        expected+=multivariate_normal.logpdf(residual,mean=np.zeros(len(times)),cov=covariance)
        self.assertAlmostEqual(L.loglik(z,False),expected,places=10)

    def test_gradient_solver_matches_independent_value_solver(self):
        d=self.data();d.update(theta=1.,T_contest=1.)
        z=np.array([math.log(10),math.log(12),0.,math.log(4),0.,math.log(10)])
        L=Likelihood(d,.02,1024);policies,dy=L.policy(z)
        p=Primitives(T=1,theta=1,c_i=10,c_j=12,sigma=1,lamb=4,S0=.5)
        reference=solve_equilibrium(p,dy=.02,dt=1/4000,extent=7)
        for t in (0,.25,.5,.75):
            for y in (-1,0,1):
                position=y/dy+(policies.shape[1]-1)/2;j=int(position);f=position-j
                actual=(1-f)*policies[int(t*1024),j]+f*policies[int(t*1024),j+1]
                np.testing.assert_allclose(actual,reference.at(t,y),rtol=.006,atol=2e-5)

    def test_diagnostics_detect_between_chain_disagreement(self):
        rng=np.random.default_rng(18)
        values=rng.normal(size=(4,4000))
        self.assertLess(rank_rhat(values),1.01)
        bulk,tail=rank_ess(values)
        self.assertGreater(bulk,400);self.assertGreater(tail,400)
        values[0]+=3
        self.assertGreater(rank_rhat(values),1.1)
        self.assertLess(rank_ess(values)[0],400)

    def test_non_small_incentives_match_independent_value_solver(self):
        d=self.data();d.update(theta=2.,T_contest=1.)
        z=np.array([0.,math.log(1.5),0.,math.log(4),0.,math.log(10)])
        L=Likelihood(d,.025,1024);policies,dy=L.policy(z)
        p=Primitives(T=1,theta=2,c_i=1,c_j=1.5,sigma=1,lamb=4,S0=.5)
        reference=solve_equilibrium(p,dy=.025,dt=1/8000,extent=7)
        errors=[]
        for t in (0,.25,.5,.75):
            for y in (-2,-1,0,1,2):
                pos=y/dy+(policies.shape[1]-1)/2;j=int(pos);f=pos-j
                actual=(1-f)*policies[int(t*1024),j]+f*policies[int(t*1024),j+1]
                errors.append(float(np.max(np.abs(actual-reference.at(t,y)))))
        self.assertLess(max(errors)/float(policies.max()),.001)

    def test_actual_mixture_sampler_recovers_known_gaussian(self):
        sd=np.array([1.,.7,.4,.3,.25,.5]);cov=np.diag(sd**2)
        baseline=np.random.default_rng(5).normal(size=(100,6))*sd
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            proposal=dict(kernel_sha256=estimator.digest(estimator.SOURCE),
                          mode_z=[0.]*6,covariance=cov.tolist())
            (out/'proposal_1.json').write_text(json.dumps(proposal))
            with patch.object(estimator,'OUT',out),patch.object(estimator,'Likelihood',GaussianLikelihood),patch.object(
                estimator,'load_baseline',return_value=({},[],baseline,{})),contextlib.redirect_stdout(io.StringIO()):
                estimator.sample(1,1,500,6000,.0125,512,123,pilot=True)
            draws=np.load(out/'contest_1/chain_1.npz')['z']/sd
            self.assertLess(float(np.max(np.abs(draws.mean(axis=0)))),.07)
            self.assertLess(float(np.max(np.abs(draws.var(axis=0)-1))),.10)

    def test_importance_correction_recovers_known_shifted_distribution(self):
        from scipy.special import softmax
        rng=np.random.default_rng(45);values=rng.normal(size=(50000,2))
        target=np.array([.4,-.2]);scale=.85
        log_ratio=-.5*np.sum(((values-target)/scale)**2,axis=1)+.5*np.sum(values**2,axis=1)
        mean,sd,q=estimator.weighted_summary(values,softmax(log_ratio))
        np.testing.assert_allclose(mean,target,atol=.02)
        np.testing.assert_allclose(sd,scale,atol=.02)
        np.testing.assert_allclose(q[:,1],target,atol=.025)

    def test_prior_screen_sampler_recovers_known_gaussian(self):
        sd=np.array([1.,.7,.4,.3,.25,.5]);cov=np.diag(sd**2)
        baseline=np.random.default_rng(5).normal(size=(100,6))*sd
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            (out/'proposal_1.json').write_text(json.dumps(dict(kernel_sha256=estimator.digest(estimator.SOURCE),mode_z=[0.]*6,covariance=cov.tolist())))
            with patch.object(estimator,'OUT',out),patch.object(estimator,'Likelihood',GaussianWithPrior),patch.object(estimator,'load_baseline',return_value=({},[],baseline,{})),contextlib.redirect_stdout(io.StringIO()):
                estimator.sample_prior_screen(1,1,500,12000,.0125,512,125,pilot=True)
            draws=np.load(out/'contest_1/chain_1.npz')['z']/sd
            self.assertLess(float(np.max(np.abs(draws.mean(axis=0)))),.07)
            self.assertLess(float(np.max(np.abs(draws.var(axis=0)-1))),.10)

    def test_fixed_student_mixture_sampler_recovers_both_modes(self):
        components=[dict(weight=w,mean=[m,0,0,0,0,0],covariance=np.eye(6).tolist()) for w,m in [( .6,-3),(.4,3)]]
        mixture=estimator.StudentMixture(components)
        from scipy.stats import multivariate_t
        point=np.array([.7,.1,-.2,.2,.3,-.1])
        expected=sum(c['weight']*multivariate_t.pdf(point,loc=c['mean'],shape=1.5*np.eye(6),df=5) for c in components)
        # A common t normalizing constant cancels in the Metropolis ratio.
        point2=np.zeros(6)
        expected2=sum(c['weight']*multivariate_t.pdf(point2,loc=c['mean'],shape=1.5*np.eye(6),df=5) for c in components)
        self.assertAlmostEqual(mixture.logpdf(point)-mixture.logpdf(point2),math.log(expected/expected2),places=12)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            proposal=dict(kernel_sha256=estimator.digest(estimator.SOURCE),mode_z=[-1.5,0,0,0,0,0],covariance=np.eye(6).tolist(),mixture_components=components)
            (out/'proposal_1.json').write_text(json.dumps(proposal))
            with patch.object(estimator,'OUT',out),patch.object(estimator,'Likelihood',BimodalWithPrior),patch.object(estimator,'load_baseline',return_value=({},[],np.zeros((100,6)),{})),contextlib.redirect_stdout(io.StringIO()):
                estimator.sample_mixture_screen(1,1,600,10000,.0125,256,126,pilot=True)
            draws=np.load(out/'contest_1/chain_1.npz')['z']
            self.assertLess(abs(float(np.mean(draws[:,0]>0))-.25),.025)
            self.assertLess(abs(float(draws[:,0].mean())+1.5),.15)
            self.assertLess(float(np.max(np.abs(draws[:,1:].var(0)-1))),.10)

    def test_gaussian_mixture_screen_sampler_recovers_both_modes(self):
        components=[dict(weight=w,mean=[m,0,0,0,0,0],covariance=np.eye(6).tolist()) for w,m in [( .6,-3),(.4,3)]]
        mixture=estimator.StudentMixture(components)
        from scipy.stats import multivariate_t
        point=np.array([.7,.1,-.2,.2,.3,-.1])
        expected=sum(c['weight']*multivariate_t.pdf(point,loc=c['mean'],shape=1.5*np.eye(6),df=5) for c in components)
        # A common t normalizing constant cancels in the Metropolis ratio.
        point2=np.zeros(6)
        expected2=sum(c['weight']*multivariate_t.pdf(point2,loc=c['mean'],shape=1.5*np.eye(6),df=5) for c in components)
        self.assertAlmostEqual(mixture.logpdf(point)-mixture.logpdf(point2),math.log(expected/expected2),places=12)
        normal=sum(c['weight']*multivariate_normal.pdf(point,mean=c['mean'],cov=1.5*np.eye(6)) for c in components)
        normal2=sum(c['weight']*multivariate_normal.pdf(point2,mean=c['mean'],cov=1.5*np.eye(6)) for c in components)
        self.assertAlmostEqual(estimator.gaussian_mixture_logpdf(mixture,point)-estimator.gaussian_mixture_logpdf(mixture,point2),math.log(normal/normal2),places=12)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            proposal=dict(kernel_sha256=estimator.digest(estimator.SOURCE),mode_z=[-1.5,0,0,0,0,0],covariance=np.eye(6).tolist(),mixture_components=components)
            (out/'proposal_1.json').write_text(json.dumps(proposal))
            with patch.object(estimator,'OUT',out),patch.object(estimator,'Likelihood',BimodalWithPrior),patch.object(estimator,'load_baseline',return_value=({},[],np.zeros((100,6)),{})),contextlib.redirect_stdout(io.StringIO()):
                estimator.sample_mixture_gaussian(1,1,600,10000,.0125,256,126,pilot=True)
            draws=np.load(out/'contest_1/chain_1.npz')['z']
            self.assertLess(abs(float(np.mean(draws[:,0]>0))-.25),.025)
            self.assertLess(abs(float(draws[:,0].mean())+1.5),.15)
            self.assertLess(float(np.max(np.abs(draws[:,1:].var(0)-1))),.10)

    def test_parallel_grid_workers_preserve_target(self):
        from concurrent.futures import ProcessPoolExecutor
        d=self.data();z=np.array([math.log(1.2),math.log(1.8),math.log(1.1),math.log(2),.12,math.log(12)])
        points=[z,z+np.array([.05,-.03,.02,.04,.01,-.02])]
        serial=[Likelihood(d,.05,128).posterior(p) for p in points]
        payloads=[(str(estimator.SOURCE),d,.05,128,7,j,p) for j,p in enumerate(points)]
        with ProcessPoolExecutor(max_workers=2) as pool:parallel=[value for _,value in pool.map(estimator._grid_worker,payloads)]
        np.testing.assert_allclose(parallel,serial,rtol=0,atol=1e-12)


if __name__=='__main__':unittest.main()
