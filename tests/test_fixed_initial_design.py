"""Mathematical checks for the fixed-prior output mapping, not old table values."""
import math
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr

from pipeline.empirical.fixed_initial_design import Output, beta, variance


class FixedInitialOutputTests(unittest.TestCase):
    def draws(self):
        return dict(mu_0=np.array([.7, -1.2]), sigma=np.array([1., .8]),
                    c_i=np.array([.8, 1.4]), c_j=np.array([1.4, .9]),
                    **{'lambda':np.array([1., 2.])})

    def test_steady_initialization_and_endpoints(self):
        self.assertAlmostEqual(float(variance(3., 2., 1., 4.)), 1.)
        self.assertAlmostEqual(float(variance(3., 2., 1., 0.)), 13.)
        self.assertEqual(float(variance(3., 2., 1., math.inf)), 0.)

    def test_symmetric_costs_remove_first_order_precision_effect(self):
        d=self.draws();d['c_j']=d['c_i'].copy()
        model=Output(d, 2., 64)
        expected=np.exp(-d['mu_0']**2/(2*model.q))/np.sqrt(2*np.pi*model.q)*2/d['c_i']
        for precision in [0., .01, 1., 1000., math.inf]:
            np.testing.assert_allclose(model.coefficients(precision),expected,rtol=1e-13)

    def test_relabeling_players_preserves_output(self):
        d=self.draws();other=dict(d,mu_0=-d['mu_0'],c_i=d['c_j'],c_j=d['c_i'])
        for precision in [0., .3, 3., math.inf]:
            a,b=Output(d,2.,64),Output(other,2.,64)
            np.testing.assert_allclose(a.coefficients(precision),b.coefficients(precision),rtol=1e-13)
            self.assertAlmostEqual(a.no_effort,b.no_effort)

    def test_gaussian_identity_against_independent_state_integration(self):
        d={k:v[:1] for k,v in self.draws().items()}
        model=Output(d,1.,64);mu=float(d['mu_0'][0]);q=float(model.q[0])
        def G(x,v):return math.exp(-x*x/(2*v))/math.sqrt(2*math.pi*v)
        for qt in [.2*q,.5*q,.9*q]:
            numeric=quad(lambda x:G(x,qt)*(ndtr(x/math.sqrt(qt))-.5)*G(x-mu,q-qt),
                         -math.inf,math.inf,epsabs=1e-12)[0]
            expected=G(mu,q)*(ndtr(mu*math.sqrt(qt)/math.sqrt(q*(2*q-qt)))-.5)
            self.assertAlmostEqual(numeric,expected,places=12)

    def test_zero_gap_and_zero_prize(self):
        d=self.draws();d['mu_0']=np.zeros(2)
        model=Output(d,2.,64)
        np.testing.assert_allclose(model.coefficients(0.),model.coefficients(math.inf),rtol=1e-13)
        expected=float(np.mean(beta(d['mu_0'],model.q)))
        self.assertAlmostEqual(model.no_effort,expected)

    def test_time_integral_against_independent_adaptive_quadrature(self):
        d={k:v[:1] for k,v in self.draws().items()}
        model=Output(d,1.,48)
        mu=float(d['mu_0'][0]);q=float(model.q[0])
        for precision in [0., .1, 3., math.inf]:
            def integrand(t):
                st=float(variance(t,1.,1.,precision))
                qt=st+1-t
                return ndtr(mu*math.sqrt(qt/(q*(2*q-qt))))-.5
            H=quad(integrand,0.,1.,epsabs=1e-12)[0]
            expected=float(model.base[0]+model.weight[0]*H)
            self.assertAlmostEqual(float(model.coefficients(precision)[0]),expected,places=11)

    def test_single_draw_preferred_endpoint_tracks_gap_cost_alignment(self):
        d={k:v[:1] for k,v in self.draws().items()}
        model=Output(d,1.,48)
        values=[model.value(p) for p in [0., .01, .1, 1., 10., 1e3, math.inf]]
        self.assertTrue(all(a>=b-1e-12 for a,b in zip(values,values[1:])))
        d['mu_0']=-d['mu_0']
        model=Output(d,1.,48)
        values=[model.value(p) for p in [0., .01, .1, 1., 10., 1e3, math.inf]]
        self.assertTrue(all(a<=b+1e-12 for a,b in zip(values,values[1:])))


if __name__=='__main__':unittest.main()
