"""Checks for the moderate-curvature figure's objective and zero-prize baseline."""
import math
import unittest

from pipeline.theory.joint_design_moderate_plane import loss_output, zero_design
from pipeline.theory.joint_design_plane import GAMMA, POWER


class ModeratePlaneChecks(unittest.TestCase):
    def test_original_experiment_is_not_recalibrated(self):
        self.assertEqual(GAMMA, 64.)
        self.assertEqual(POWER, 64.)

    def test_moderate_utility_is_increasing_and_concave(self):
        for b in (0., .4, 1., 3.):
            marginal = 6*math.exp(-2*b)
            second = -12*math.exp(-2*b)
            self.assertGreater(marginal, 0.)
            self.assertLess(second, 0.)
            self.assertAlmostEqual(float(loss_output(b)), 3*math.exp(-2*b))

    def test_zero_prize_deadline_satisfies_first_order_condition(self):
        for sigma in (.06,.4,1.,3.):
            t,loss,b=zero_design(sigma)
            self.assertGreater(t,0.)
            self.assertLess(t,math.sqrt(6.))
            marginal=6*math.exp(-2*b)*sigma**2/(2*math.sqrt(2*math.pi*(1+sigma**2*t)))
            self.assertAlmostEqual(marginal,t,places=10)
            self.assertAlmostEqual(loss,3*math.exp(-2*b)+t*t/2)

    def test_stable_audit_preserves_exact_zero_effort_baseline(self):
        from pipeline.theory.joint_design_stable_audit import stable_output
        for precision in (0.,1.,100.):
            b,diag=stable_output(.4,0.,.7,precision,3)
            self.assertAlmostEqual(b,math.sqrt((1+.4**2*.7)/(2*math.pi)),places=10)
            self.assertEqual(diag["max_effort"],0.)

    def test_streaming_audit_preserves_original_two_pass_outputs(self):
        from pipeline.theory.joint_design_stable_audit import stable_output
        reference=[((1.,.2,1.,0.,3),.6205746700749875,500),
                   ((.3,1.5,.6,.01,3),.763919139810471,999),
                   ((.1,2.16,.765,.0001,3),1.117509905080739,1339)]
        for job,b,steps in reference:
            actual,diagnostics=stable_output(*job)
            self.assertAlmostEqual(actual,b,places=12)
            self.assertEqual(diagnostics["steps"],steps)

    def test_physical_policy_check_recovers_no_signal_identity(self):
        from pipeline.theory.joint_design_stable_audit import stable_output
        _,diag=stable_output(1.,.2,1.,0.,3,verify_policy=True)
        self.assertAlmostEqual(diag["physical_policy_verification"],
                               diag["no_signal_verification"],places=9)

    def test_small_positive_signal_needs_independent_policy_verification(self):
        from pipeline.theory.joint_design_stable_audit import stable_output
        with self.assertRaisesRegex(RuntimeError,"physical-policy value verification"):
            stable_output(.1,2.16,.765,.0001,3,verify_policy=True)

    def test_stable_audit_agrees_with_weak_no_signal_series(self):
        from pipeline.theory.joint_design_stable_audit import stable_output
        from pipeline.theory.joint_design_plane import calculate
        exact=calculate((1.,.2,1.,0.,2.))["B"]
        coarse,dc=stable_output(1.,.2,1.,0.,3)
        fine,df=stable_output(1.,.2,1.,0.,4)
        self.assertLess(abs(fine-exact),abs(coarse-exact))
        self.assertAlmostEqual(fine,exact,delta=2e-4)
        self.assertLess(df["no_signal_verification"],dc["no_signal_verification"])

    def test_no_signal_screen_uses_individual_energy_bound(self):
        from pipeline.theory.joint_design_moderate_plane import candidate_output_upper
        from pipeline.theory.joint_design_plane import calculate,upper_output
        upper=candidate_output_upper(1.,.2,1.,0.)
        self.assertLessEqual(calculate((1.,.2,1.,0.,2.))["B"],upper)
        self.assertLess(upper,upper_output(1.,.2,1.,0.))
        self.assertEqual(candidate_output_upper(1.,.2,1.,1.),upper_output(1.,.2,1.,1.))

    def test_finest_failure_is_not_replaced_by_coarse_success(self):
        from pipeline.theory.joint_design_moderate_plane import latest
        key=(1.,.2,1.,0.)
        coarse=dict(B=.6,level=3.)
        failed=dict(B=math.nan,level=5.)
        cache={key+(3.,):coarse,key+(5.,):failed}
        self.assertIs(latest(cache,key),failed)

    def test_stable_alternative_is_not_hidden_by_noisy_point_winner(self):
        import numpy as np
        from pipeline.theory.joint_design_moderate_plane import interval_indices
        data=np.array([[.1,1.,0.,.8,2.],[.2,1.,1.,.7,2.],[.1,1.,math.inf,.6,2.]])
        scores=np.array([[.5,.6,.8]])
        errors=np.array([.4,.01,.01])
        ((fl,fu),(ul,uu)),lower,upper=interval_indices(data,scores,errors)
        self.assertEqual(int(fl[0]),0)
        self.assertEqual(int(fu[0]),1)
        self.assertLess(upper[0,fu[0]],lower[0,ul[0]])
        self.assertGreater(scores[0,0]+errors[0],lower[0,ul[0]])

    def test_family_interval_contains_all_admissible_perturbations(self):
        import numpy as np
        from pipeline.theory.joint_design_moderate_plane import interval_indices
        data=np.array([[.1,1.,0.,.8,2.],[.2,1.,1.,.7,2.],[.1,1.,math.inf,.6,2.]])
        scores=np.array([[.5,.6,.8]])
        errors=np.array([.04,.01,.03])
        pairs,lower,upper=interval_indices(data,scores,errors)
        rng=np.random.default_rng(125)
        for _ in range(30):
            realized=scores[0]+rng.uniform(-1,1,3)*errors
            for sub,(lo,hi) in zip(([0,1],[2]),pairs):
                actual=float(realized[sub].min())
                self.assertLessEqual(lower[0,lo[0]],actual)
                self.assertGreaterEqual(upper[0,hi[0]],actual)

    def test_cached_no_signal_policy_must_pass_nash_verification(self):
        from pipeline.theory.joint_design_moderate_plane import latest
        key=(.06,2.,1.,0.)
        original=dict(B=1.2,error=.1,level=5.,status="stable_BE_LF_audit")
        cache={key+(5.,):original}
        self.assertTrue(math.isnan(latest(cache,key)["B"]))
        self.assertEqual(original["B"],1.2)

    def test_dense_cost_grid_does_not_change_original_grid(self):
        import numpy as np
        from pipeline.theory.joint_design_moderate_plane import classify,COSTS
        original=COSTS.copy()
        designs={(.02,.5,0.),(.02,.5,math.inf)}
        cache={(1.,a,t,l,1.):dict(B=.51,error=0.,level=1.,status="test")
               for a,t,l in designs}
        grid=np.array([.03,.0325,.035])
        rows=classify(1.,designs,cache,grid)
        np.testing.assert_array_equal([r["c"] for r in rows],grid)
        np.testing.assert_array_equal(COSTS,original)

    def test_single_stable_mesh_is_not_treated_as_zero_error(self):
        from pipeline.theory.joint_design_moderate_plane import families
        designs={(.2,.5,.01),(.2,.5,math.inf)}
        cache={(1.,.2,.5,.01,3.):dict(B=.8,error=0.,level=3.,status="stable_BE_LF_audit"),
               (1.,.2,.5,math.inf,2.):dict(B=.6,error=0.,level=2.,status="full_disclosure_bvp")}
        data,_,_,errors,_=families(1.,designs,cache,[.4])
        i=next(i for i,row in enumerate(data) if row[2]==.01)
        self.assertGreater(errors[i],1.)


if __name__=="__main__":
    unittest.main()
