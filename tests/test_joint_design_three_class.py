import unittest
import numpy as np

from pipeline.theory.joint_design_three_class import decision
from pipeline.theory.joint_design_moderate_plane import smooth_class_display,fitted_display_boundaries


class ThreeClassDecisionTests(unittest.TestCase):
    def row(self, **updates):
        row=dict(sigma=.1,c=.4,zero_loss=1.,full_loss=1.1,finite_loss=1.2,
                 comparison_upper=1.,full_upper=1.11,finite_upper=1.21,
                 failed_finite_lower=.9,failed_full_lower=2.)
        row.update(updates)
        return row

    def test_zero_is_a_decision_not_a_certificate(self):
        result=decision(self.row())
        self.assertEqual(result["classification"],"zero")
        self.assertTrue(result["competitive_failed_design"])
        self.assertFalse(result["global_classification_verified"])
        self.assertAlmostEqual(result["payoff_margin"],.1)

    def test_positive_prize_families_compared_separately(self):
        self.assertEqual(decision(self.row(full_loss=.8))["classification"],"full")
        self.assertEqual(decision(self.row(finite_loss=.8))["classification"],"finite")

    def test_conservative_disagreement_preserved(self):
        result=decision(self.row(finite_loss=.8,finite_upper=1.3))
        self.assertEqual(result["classification"],"finite")
        self.assertEqual(result["interval_winner"],"zero")

    def test_display_smoothing_preserves_source_and_partition(self):
        z=np.array([[0,0,1],[0,1,2],[2,2,2]])
        original=z.copy()
        sx,cy,weights,display=smooth_class_display([.1,.2,.3],[.1,.2,.3],z)
        np.testing.assert_array_equal(z,original)
        np.testing.assert_allclose(weights.sum(axis=0),1.,atol=1e-12)
        self.assertEqual(display.shape,(len(cy),len(sx)))
        self.assertTrue(set(np.unique(display))<=set(range(3)))

    def test_spline_display_preserves_source_and_region_order(self):
        z=np.array([[0,0,1,1],[0,1,1,2],[2,2,2,2]])
        original=z.copy()
        x,finite,exit=fitted_display_boundaries([.1,.2,.3,.4],[.1,.2,.3],z)
        np.testing.assert_array_equal(z,original)
        self.assertTrue(np.all(finite<=exit))
        self.assertTrue(np.all(finite>=.1))
        self.assertTrue(np.all(exit<=.3))

    def test_initial_shared_boundary_is_exactly_shared(self):
        z=np.array([[0,0,0,1],[0,0,1,2],[2,2,2,2]])
        x,finite,exit=fitted_display_boundaries([.1,.2,.3,.4],[.1,.2,.3],z)
        np.testing.assert_array_equal(finite[x<=.2],exit[x<=.2])


if __name__=="__main__":unittest.main()
