"""Regression tests for the nonlinear, non-steady-state synthetic experiments."""
import unittest
from dataclasses import replace
import numpy as np
from scipy.special import ndtr
from scipy.stats import multivariate_normal, norm
from pipeline.synthetic.nonlinear import Primitives, variance, solve_equilibrium, simulate, likelihood_data


class NonlinearSyntheticTests(unittest.TestCase):
    def test_riccati_initial_value_and_sensitivity(self):
        p=Primitives(T=2)
        t=np.linspace(0,2,2001)
        s=variance(t,p)
        self.assertEqual(s[0],4)
        np.testing.assert_allclose(np.gradient(s,t)[1:-1],p.sigma**2-p.lamb*s[1:-1]**2,atol=2e-4)
        eps=1e-5
        actual=(variance(t,replace(p,S0=p.S0+eps))-s)/eps
        a=p.sigma*np.sqrt(p.lamb)
        expected=(np.cosh(a*t)+(p.S0/(p.sigma/np.sqrt(p.lamb)))*np.sinh(a*t))**-2
        np.testing.assert_allclose(actual,expected,atol=1e-6)

    def test_small_prize_limit_and_symmetry(self):
        p=Primitives(T=2,theta=.001,c_i=1,c_j=1)
        eq=solve_equilibrium(p,dy=.1,dt=1/96)
        for t in (0,1):
            Q=variance(t,p)+p.sigma**2*(p.T-t)
            leading=p.theta/np.sqrt(2*np.pi*Q)
            np.testing.assert_allclose(eq.at(t,0),leading,rtol=.002)
        np.testing.assert_allclose(eq.at(.5,1),eq.at(.5,-1)[::-1],atol=1e-8)

    def test_initial_draw_thinning_and_event_intervals(self):
        p=Primitives(T=2)
        eq=solve_equilibrium(p)
        c=simulate(eq,seed=12,ratio=50)
        seed=np.random.SeedSequence(12).spawn(5)[0]
        self.assertEqual(c['initial_y'],np.random.default_rng(seed).normal(0,2))
        events=c['events']
        self.assertGreater(len(events),0)
        np.testing.assert_allclose(events[:,3],np.diff(np.r_[0,events[:,0]]))
        self.assertGreaterEqual(c['thinning_bound'],50*eq.effort.max())
        data=likelihood_data([c])
        self.assertAlmostEqual(sum(data['dt']),2)
        self.assertEqual(sum(x>0 for x in data['player']),len(events))
        for i in range(len(data['dt'])-1):
            if data['player'][i]>0:
                self.assertEqual(data['held'][i+1],data['obs'][i])

    def test_innovations_equals_gaussian_joint_density(self):
        t=np.array([.2,.7,1.3,2.0])
        h=np.diff(np.r_[0,t])
        sigma,lamb,s0,mu=2.,1.3,4.,.2
        y=np.array([.1,1.,-.5,2.])
        cov=s0+sigma**2*np.minimum.outer(t,t)+np.diag(1/(lamb*h))
        expected=multivariate_normal.logpdf(y,mean=np.full(4,mu),cov=cov)
        mean,var,total=mu,s0,0.
        for step,obs in zip(h,y):
            var+=sigma**2*step
            noise=1/(lamb*step)
            total+=norm.logpdf(obs,mean,np.sqrt(var+noise))
            mean+=var/(var+noise)*(obs-mean)
            var=var*noise/(var+noise)
        self.assertAlmostEqual(total,expected,places=12)


if __name__=='__main__': unittest.main()
