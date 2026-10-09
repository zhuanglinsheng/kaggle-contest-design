"""Checks for the numerical plane's normalization, endpoints, and diagnostics."""
import math
import unittest

from pipeline.theory.joint_design_plane import (
    AMPLITUDE, B0, GAMMA, POWER, RATE, calculate, delay, tail, zero_design, upper_output,
    profile,
)
from pipeline.theory.joint_design_exact import full_disclosure_output


class PlaneChecks(unittest.TestCase):
    def test_explicit_shape_floors(self):
        for b in [B0, 1., 3.]:
            self.assertGreaterEqual(RATE*b, GAMMA)
        self.assertEqual(POWER, GAMMA)
        self.assertEqual(tail(B0), AMPLITUDE)
        self.assertEqual(delay(0), 0.)

    def test_calibration_and_small_loss_arithmetic(self):
        t, loss, ce = zero_design(1.)
        self.assertAlmostEqual(t, 1., places=11)
        self.assertGreater(loss, 0.)
        self.assertGreater(ce, 0.)
        self.assertLess(loss, 1e-8)
        for sigma in [.04, 2.6]:
            self.assertTrue(all(math.isfinite(v) and v>0 for v in zero_design(sigma)))

    def test_zero_endpoint_against_small_positive_precision(self):
        zero = calculate((1., .2, 1., 0., 1.))
        nearby = calculate((1., .2, 1., 1e-6, 1.))
        self.assertEqual(zero["status"], "analytic_zero_series")
        self.assertTrue(math.isfinite(nearby["B"]))
        self.assertAlmostEqual(zero["B"], nearby["B"], delta=3e-4)
        self.assertLess(zero["error"], 1e-10)

    def test_full_endpoint_not_finite_precision_proxy(self):
        full = calculate((1., .05, 1., math.inf, 1.))
        self.assertEqual(full["status"], "full_disclosure_bvp")
        nearby = calculate((1., .05, 1., 500., 1.))
        self.assertTrue(math.isfinite(nearby["B"]))
        self.assertAlmostEqual(full["B"], nearby["B"], delta=1e-3)

    def test_effective_prize_normalization(self):
        first = full_disclosure_output(theta=.1, cost=1, horizon=1, initial_variance=1,
                                      dy=.05, dt=1/200)
        second = full_disclosure_output(theta=.4, cost=4, horizon=1, initial_variance=1,
                                       dy=.05, dt=1/200)
        self.assertAlmostEqual(first, second, places=13)

    def test_failure_is_not_an_inferior_output(self):
        row = calculate((.04, 6., 1., math.inf, 0.))
        self.assertTrue(row["status"].startswith("failed"))
        self.assertTrue(math.isnan(row["B"]))

    def test_extended_precision_cap_remains_unresolved(self):
        key = (1., .1, 1., 10000., 0.)
        cache = {key:dict(B=.8,error=0.,status="test")}
        row, _ = profile(1.,1e-11,[(.1,1.,10000.)],cache,6.,precision_cap=10000.)
        self.assertEqual(row["regime"],"finite_tail_unresolved")

    def test_adaptive_search_retains_both_endpoint_families(self):
        from pipeline.theory.joint_design_refined_plane import winning_designs
        designs = {(a,1.,l) for a in [.1,1.] for l in [0.,math.inf]}
        cache = {(1.,a,t,l,0.):dict(B=B0+.1*a,error=0.,status="test")
                 for a,t,l in designs}
        winners = winning_designs(1.,[1e-8,100.],designs,cache)
        self.assertTrue(any(math.isinf(v[2]) for v in winners))
        self.assertTrue(any(v[2]==0 for v in winners))
        self.assertEqual({v[0] for v in winners},{.1,1.})

    def test_vectorized_boundary_profiles_match_scalar(self):
        from pipeline.theory.joint_design_boundary_polish import profiles
        designs = [(a,1.,l) for a in [.001,.1] for l in [0.,math.inf]]
        cache = {(1.,a,t,l,level):dict(B=.6+a*.1+(0 if l==0 else .001),error=0.,status="test")
                 for a,t,l in designs for level in [0.,1.,2.]}
        costs = [1e-12,1e-9,1.]
        batch = profiles(1.,costs,designs,cache,relative_floor=.005,required_level=2,branch_screen=False)
        for cost,row in zip(costs,batch):
            scalar,_ = profile(1.,cost,designs,cache,6.,10000.,relative_floor=.005,required_level=2)
            self.assertEqual(row["regime"],scalar["regime"])
            self.assertAlmostEqual(row["loss"],scalar["loss"])
            self.assertAlmostEqual(row["tolerance"],scalar["tolerance"])

    def test_finer_output_is_independent_resolution(self):
        fine = calculate((1., .05, 1., math.inf, 1.))
        finer = calculate((1., .05, 1., math.inf, 2.))
        self.assertTrue(math.isfinite(finer["B"]))
        self.assertAlmostEqual(fine["B"],finer["B"],delta=2e-6)

    def test_fixed_payoff_floor_can_hide_zero_prize(self):
        from pipeline.theory.joint_design_boundary_polish import profiles
        a = 1e-5
        designs = [(a,1.,l) for l in [0.,math.inf]]
        b = math.sqrt(1/math.pi)+a/math.sqrt(4*math.pi)
        cache = {(1.,a,1.,l,k):dict(B=b,error=0.,status="test") for _,_,l in designs for k in [0.,1.,2.]}
        fixed = profiles(1.,[5e-10],designs,cache,relative_floor=.005)[0]
        adaptive = profiles(1.,[5e-10],designs,cache)[0]
        self.assertEqual(fixed["regime"],"near_tie")
        self.assertEqual(adaptive["regime"],"zero_on_grid")

    def test_failed_winning_family_does_not_obscure_precision_ranking(self):
        from pipeline.theory.joint_design_boundary_polish import profiles
        designs = [(.1,1.,0.),(.1,1.,math.inf),(.01,.5,20.)]
        cache = {(1.,a,t,l,k):dict(B=.8 if l==0 else .565,error=0.,status="test")
                 for a,t,l in designs[:2] for k in [0.,1.,2.]}
        for k in [0.,1.,2.]:
            cache[(1.,.01,.5,20.,k)] = dict(B=math.nan,error=math.nan,status="failed")
        old = profiles(1.,[1e-11],designs,cache,branch_screen=False)[0]
        new = profiles(1.,[1e-11],designs,cache)[0]
        self.assertEqual(old["regime"],"unresolved")
        self.assertEqual(new["regime"],"finite_on_grid")
        self.assertEqual(new["design_unresolved"],1)

    def test_failed_opposite_family_still_blocks_precision_ranking(self):
        from pipeline.theory.joint_design_boundary_polish import profiles
        designs = [(.1,1.,0.),(.1,1.,math.inf),(.01,.5,math.inf)]
        cache = {(1.,a,t,l,k):dict(B=.8 if l==0 else .565,error=0.,status="test")
                 for a,t,l in designs[:2] for k in [0.,1.,2.]}
        for k in [0.,1.,2.]:
            cache[(1.,.01,.5,math.inf,k)] = dict(B=math.nan,error=math.nan,status="failed")
        new = profiles(1.,[1e-11],designs,cache)[0]
        self.assertEqual(new["regime"],"unresolved")

    def test_finite_cap_can_certify_grid_family_not_optimizer_location(self):
        from pipeline.theory.joint_design_boundary_polish import profiles
        designs = [(.1,1.,10000.),(.1,1.,math.inf)]
        cache = {(1.,a,t,l,k):dict(B=.8 if l==10000 else .565,error=0.,status="test")
                 for a,t,l in designs for k in [0.,1.,2.]}
        row = profiles(1.,[1e-11],designs,cache)[0]
        self.assertEqual(row["regime"],"finite_on_grid")
        self.assertEqual(row["design_unresolved"],1)

    def test_failed_full_family_does_not_obscure_full_grid_preference(self):
        from pipeline.theory.joint_design_boundary_polish import profiles
        designs = [(.1,1.,0.),(.1,1.,math.inf),(.01,.5,math.inf)]
        cache = {(1.,a,t,l,k):dict(B=.565 if l==0 else .8,error=0.,status="test")
                 for a,t,l in designs[:2] for k in [0.,1.,2.]}
        for k in [0.,1.,2.]:
            cache[(1.,.01,.5,math.inf,k)] = dict(B=math.nan,error=math.nan,status="failed")
        row = profiles(1.,[1e-11],designs,cache)[0]
        self.assertEqual(row["regime"],"full_on_grid")

    def test_full_profile_obeys_strong_prize_envelope(self):
        job = (.04, .01, .4358329405715108, math.inf, 1.)
        row = calculate(job)
        self.assertTrue(math.isfinite(row["B"]))
        self.assertLessEqual(row["B"], upper_output(*job[:4])+1e-8)
        self.assertLess(row["B"], .401)

    def test_full_profile_obeys_weak_prize_envelope(self):
        alpha = .1
        row = calculate((1., alpha, 1., math.inf, 1.))
        base = math.sqrt(2/(2*math.pi))
        leading = alpha/math.sqrt(4*math.pi)
        self.assertGreater(row["B"], base+leading)
        self.assertLess(row["B"], base+leading+alpha**2/(4*math.sqrt(4*math.pi)))


if __name__ == "__main__":
    unittest.main()
