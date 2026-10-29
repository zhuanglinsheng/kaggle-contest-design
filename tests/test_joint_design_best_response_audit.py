"""Independent strategic-deviation checks for candidate verification."""
import unittest
import numpy as np
from pipeline.theory.joint_design_best_response_audit import audit,evaluate_symmetric_policy


class BestResponseChecks(unittest.TestCase):
    def test_zero_effort_policy_has_profitable_deviation_with_positive_prize(self):
        x=np.linspace(-7,7,201)
        result=evaluate_symmetric_policy(.7,.3,.6,1.,x,[0.,1.],
                                         np.zeros((len(x),2)),steps=200)
        self.assertGreater(result['deviation_gain'],.005)
        self.assertAlmostEqual(result['prescribed'],.5,places=9)
        self.assertLess(result['maximum_hamiltonian_gap'],1e-7)

    def test_weak_incentive_equilibrium_has_small_deviation_gain(self):
        result=audit(.7,.3,.6,1.,nodes=201,stretch=4.,steps=300)
        self.assertLess(result['max_deviation_gain'],1e-4)
        self.assertLess(result['max_subgame_deviation_gain'],1e-4)
        self.assertGreater(result['deviation_gain'],-1e-6)
        self.assertLess(result['maximum_hamiltonian_gap'],1e-7)

    def test_invalid_policy_is_rejected(self):
        x=np.linspace(-7,7,101)
        with self.assertRaises(ValueError):
            evaluate_symmetric_policy(.7,.3,.6,1.,x,[0.,1.],
                                      -np.ones((101,2)),steps=20)


if __name__=='__main__':
    unittest.main()
