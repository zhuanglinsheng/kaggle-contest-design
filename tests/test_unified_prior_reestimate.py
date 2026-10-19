"""Freeze the intended change: lambda prior only, same empirical likelihood."""
import math
import unittest
from pipeline.empirical.unified_prior_reestimate import prior
from pipeline.empirical.estimate import stan_data


class UnifiedPriorTests(unittest.TestCase):
    def test_only_precision_prior_changes(self):
        cfg=prior({'hat_y':[.37]})
        self.assertEqual(cfg['prior_mean'],[0.,0.,math.log(2),math.log(100),.37,math.log(10)])
        self.assertEqual(cfg['prior_sd'],[.75,.75,.5,2.25,1.,.5])
        self.assertEqual(cfg['sigma_floor'],.01)

    def test_early_data_preserve_full_deadline_and_units(self):
        raw=dict(theta=100,Delta2f=1/24,N_Delta=100,initialization_hours=24,
                 count_start_hour=23,hat_t_i=[24,50,90],hat_t_j=[25,70,99],hat_y=[.1]*100)
        full=stan_data(raw,'full');early=stan_data(raw,'early')
        self.assertEqual(full['T_contest'],early['T_contest'])
        self.assertEqual(early['theta'],1.)
        self.assertEqual(early['N_Delta'],81)
        self.assertEqual(early['hat_t_i'],[24.,50.])
        self.assertEqual(early['hat_t_j'],[25.,70.])
        self.assertEqual(early['count_start_idx'],24)


if __name__=='__main__':unittest.main()
