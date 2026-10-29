"""Independent baselines and rejection checks for the new density solver."""
import math
import unittest
from unittest.mock import patch

from pipeline.theory.joint_design_conservative_audit import solve
from pipeline.theory.joint_design_stable_audit import stable_output


class ConservativeAuditChecks(unittest.TestCase):
    def test_zero_effort_reproduces_gaussian_output_and_half_prize(self):
        result=solve(.7,0.,.6,1.,nodes=401,stretch=4.)
        exact=math.sqrt((1+.7**2*.6)/(2*math.pi))
        self.assertLess(abs(result['B']-exact),1e-5)
        self.assertAlmostEqual(result['player_value'],.5,places=10)
        self.assertAlmostEqual(result['physical_value'],.5,places=10)
        self.assertAlmostEqual(result['mass'],1.,places=10)
        self.assertTrue(result['mesh_checks_passed'])

    def test_weak_incentive_output_agrees_with_independent_value_solver(self):
        args=(.7,.3,.6,1.)
        result=solve(*args,nodes=401,stretch=4.)
        reference,_=stable_output(*args,4,verify_policy=True)
        self.assertLess(abs(result['B']-reference),1e-4)
        self.assertLess(result['global_residual'],1e-4)
        self.assertTrue(result['mesh_checks_passed'])

    def test_unresolved_weak_feedback_is_not_accepted_on_integrator_success(self):
        result=solve(.06,2.164248520857008,.7656981839261894,.01,
                     nodes=201,stretch=4.)
        self.assertTrue(result['success'])
        self.assertGreater(result['global_residual'],.002)
        self.assertFalse(result['mesh_checks_passed'])

    def test_even_grid_and_infeasible_precision_are_rejected(self):
        with self.assertRaises(ValueError):
            solve(.7,.3,.6,1.,nodes=400)
        with self.assertRaises(ValueError):
            solve(.7,.3,.6,0.,nodes=401)

    def test_failed_finest_mesh_cannot_be_replaced_by_two_coarser_successes(self):
        from pipeline.theory.joint_design_conservative_repair import audit_one
        case=dict(sigma=.7,alpha=.3,T=.6,**{'lambda':1.})
        outcomes=[dict(nodes=801,B=.6,mesh_checks_passed=True),
                  dict(nodes=1601,B=.6001,mesh_checks_passed=True),
                  dict(nodes=3201,B=.61,mesh_checks_passed=False)]
        with patch('pipeline.theory.joint_design_conservative_repair.solve',side_effect=outcomes):
            result=audit_one((case,[801,1601,3201],18.,12000))
        self.assertFalse(result['two_mesh_checks_passed'])
        self.assertNotIn('estimated_output_error',result)


if __name__=='__main__':
    unittest.main()
