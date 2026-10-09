"""Consistency checks for the revised PaperJK7 synthetic model."""

from __future__ import annotations

import math
import unittest
from datetime import datetime, timedelta

import numpy as np

from synthetic_data.synthetic_data import (
    filtering_variance,
    leading_order_efforts,
    steady_state_variance,
    synthetic_data_simulation,
)


class RevisedSyntheticModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.start = datetime(2026, 1, 1)
        self.end = self.start + timedelta(days=90)

    def test_effort_matches_theorem_one(self) -> None:
        y = 1.3
        t = self.start + timedelta(days=12)
        theta, c_i, c_j, sigma, lamb = 1.0, 1.2, 1.5, 2.0, 0.8
        actual_i, actual_j = leading_order_efforts(
            y, t, self.end, prize=theta, c_i=c_i, c_j=c_j, sigma=sigma, lamb=lamb
        )
        q_t = sigma**2 * 78 + sigma / math.sqrt(lamb)
        kernel = math.exp(-y**2 / (2 * q_t)) / math.sqrt(2 * math.pi * q_t)
        self.assertAlmostEqual(actual_i, theta * kernel / c_i)
        self.assertAlmostEqual(actual_j, theta * kernel / c_j)

    def test_precision_has_state_dependent_effort_effect(self) -> None:
        common = dict(
            t=self.end - timedelta(days=1),
            T=self.end,
            prize=1.0,
            c_i=1.0,
            c_j=1.0,
            sigma=1.0,
        )
        tie_low, _ = leading_order_efforts(0.0, lamb=0.1, **common)
        tie_high, _ = leading_order_efforts(0.0, lamb=100.0, **common)
        gap_low, _ = leading_order_efforts(3.0, lamb=0.1, **common)
        gap_high, _ = leading_order_efforts(3.0, lamb=100.0, **common)
        self.assertGreater(tie_high, tie_low)
        self.assertGreater(gap_low, gap_high)

    def test_simulator_draws_initial_gap_from_common_prior(self) -> None:
        mu_0, sigma, lamb, seed = 0.7, 2.0, 0.5, 91
        result = synthetic_data_simulation(
            theta=1.0,
            c_i=1.2,
            c_j=1.5,
            sigma=sigma,
            lamb=lamb,
            intensity_effort_ratio=15.0,
            hour_arrival_ub=1.0,
            start_time=self.start,
            end_time=self.start + timedelta(hours=2),
            mu_0=mu_0,
            seed_initial=seed,
        )
        real_gap, perceived_gap, observed_gap = result[3], result[4], result[5]
        expected = np.random.default_rng(seed).normal(
            mu_0, math.sqrt(steady_state_variance(sigma, lamb))
        )
        self.assertAlmostEqual(real_gap[0], expected)
        self.assertEqual(perceived_gap[0], mu_0)
        self.assertEqual(observed_gap[0], mu_0)

    def test_public_gap_is_carried_forward_without_submission(self) -> None:
        result = synthetic_data_simulation(
            theta=1.0,
            c_i=1.2,
            c_j=1.5,
            sigma=2.0,
            lamb=1.0,
            intensity_effort_ratio=100.0,
            hour_arrival_ub=1.0,
            start_time=self.start,
            end_time=self.start + timedelta(days=2),
            seed_brownian=100,
            seed_poisson=101,
            seed_uniform=102,
            seed_initial=103,
        )
        observed_gap = result[5]
        submissions = result[6] + result[7]
        update_steps = {
            math.ceil((event - self.start).total_seconds() / 3600)
            for event in submissions
        }
        self.assertTrue(update_steps)
        for step in range(1, len(observed_gap)):
            if step not in update_steps:
                self.assertEqual(observed_gap[step], observed_gap[step - 1])

    def test_variance_solves_riccati(self) -> None:
        for sigma, lamb, S0 in [(2.,1.,4.), (1.3,.7,.2), (2.,1.,0.)]:
            self.assertAlmostEqual(filtering_variance(0., sigma, lamb, S0), S0)
            for t in (.1,.5,1.):
                h = 1e-6
                derivative = (filtering_variance(t+h,sigma,lamb,S0)
                              - filtering_variance(t-h,sigma,lamb,S0))/(2*h)
                S = filtering_variance(t,sigma,lamb,S0)
                self.assertAlmostEqual(derivative, sigma**2-lamb*S*S, places=7)

    def test_policy_callback_preserves_every_other_simulation_step(self) -> None:
        end = self.start+timedelta(days=3)
        kwargs = dict(theta=1.,c_i=1.2,c_j=1.5,sigma=2.,lamb=1.,S0=4.,
                      intensity_effort_ratio=100.,hour_arrival_ub=1.,
                      start_time=self.start,end_time=end)
        def same_policy(t,y):
            return leading_order_efforts(y,self.start+timedelta(days=t),end,
                prize=1.,c_i=1.2,c_j=1.5,sigma=2.,lamb=1.,
                S_t=filtering_variance(t,2.,1.,4.))
        old=synthetic_data_simulation(**kwargs)
        new=synthetic_data_simulation(**kwargs,effort_policy=same_policy)
        for a,b in zip(old,new):
            np.testing.assert_array_equal(a,b)

    def test_hourly_generator_uses_nonlinear_policy_at_pre_update_state(self) -> None:
        from pipeline.synthetic.nonlinear import Primitives, solve_equilibrium
        eq=solve_equilibrium(Primitives(T=2.),dy=.1,dt=1/96)
        result=synthetic_data_simulation(theta=1.,c_i=1.2,c_j=1.5,sigma=2.,lamb=1.,
            S0=4.,intensity_effort_ratio=15.,hour_arrival_ub=1.,start_time=self.start,
            end_time=self.start+timedelta(days=2),effort_policy=eq.at)
        for n in range(len(result[0])):
            np.testing.assert_array_equal([result[1][n],result[2][n]],eq.at(n*(1/24),result[4][n]))
        approx=leading_order_efforts(0.,self.start,self.start+timedelta(days=2),
            prize=1.,c_i=1.2,c_j=1.5,sigma=2.,lamb=1.,S_t=4.)
        self.assertGreater(float(np.max(abs(eq.at(0.,0.)-approx))),1e-5)

    def test_explicit_steady_initial_variance_reproduces_old_path(self) -> None:
        kwargs = dict(theta=1.,c_i=1.2,c_j=1.5,sigma=2.,lamb=.7,
                      intensity_effort_ratio=15.,hour_arrival_ub=1.,
                      start_time=self.start,end_time=self.start+timedelta(days=4))
        old = synthetic_data_simulation(**kwargs)
        new = synthetic_data_simulation(**kwargs, S0=steady_state_variance(2.,.7))
        for a,b in zip(old,new):
            np.testing.assert_array_equal(a,b)

    def test_nonsteady_changes_only_variance_inputs_to_old_recursions(self) -> None:
        dt = 1/24
        seed = 1000
        result = synthetic_data_simulation(
            theta=1.,c_i=1.2,c_j=1.5,sigma=2.,lamb=1.,
            intensity_effort_ratio=100.,hour_arrival_ub=1.,
            start_time=self.start,end_time=self.start+timedelta(days=3),
            S0=4.,seed_brownian=seed,seed_poisson=seed+1,
            seed_uniform=seed+2,seed_initial=seed+3)
        grid,mi,mj,latent,perceived,public,sub_i,sub_j = result
        rng = np.random.default_rng(seed)
        noise = rng.normal(size=len(grid))
        shock = rng.normal(size=len(grid))
        self.assertEqual(latent[0], np.random.default_rng(seed+3).normal(0.,2.))
        steps = {math.ceil((t-self.start).total_seconds()/3600) for t in sub_i+sub_j}
        last = 0
        for n,t in enumerate(grid):
            S = filtering_variance(n*dt,2.,1.,4.)
            expected_mi,expected_mj = leading_order_efforts(
                perceived[n],t,self.start+timedelta(days=3),prize=1.,
                c_i=1.2,c_j=1.5,sigma=2.,lamb=1.,S_t=S)
            self.assertEqual(mi[n],expected_mi)
            self.assertEqual(mj[n],expected_mj)
            drift=(mi[n]-mj[n])*dt
            self.assertAlmostEqual(latent[n+1],latent[n]+drift+2*dt**.5*shock[n])
            self.assertAlmostEqual(perceived[n+1],perceived[n]+drift
                                   -np.expm1(-S*dt)*(public[n]-perceived[n]))
            if n+1 in steps:
                self.assertAlmostEqual(public[n+1],latent[n+1]+noise[n]/((n+1-last)*dt)**.5)
                last=n+1
            else:
                self.assertEqual(public[n+1],public[n])


if __name__ == "__main__":
    unittest.main()
