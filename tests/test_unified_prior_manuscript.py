import re
import unittest
import pandas as pd
from pipeline.config import PAPER_DIR,MAIN_VALIDATION_DIR
from pipeline.empirical.unified_prior_manuscript import update_figure,environment


class ManuscriptPatchTests(unittest.TestCase):
    def test_figure_headers_are_not_coefficient_rows(self):
        source=r'''\begin{figure}
\label{fig-validate-lambda}
{
 & coef. & stderr & 95\% CI \\
 & 0 & 0 & [0, 0] \\
 & 0 & 0 & [0, 0] \\
(3) Condition Number = 1.}
\end{figure}'''
        row=dict(constant=1.,constant_se=.1,constant_ci_low=.8,constant_ci_high=1.2,
                 constant_p=.001,coefficient=.817,std_error=.1,ci_low=.6,ci_high=1.,
                 p_value=.001,condition_number=2.)
        transformed=environment(source,'figure','fig-validate-lambda',lambda block:update_figure(block,row))
        self.assertIn('& coef. & stderr & 95\\% CI',transformed)
        self.assertIn(r'\textcolor{blue}{0.817',transformed)
        self.assertEqual(source.count('{')-source.count('}'),transformed.count('{')-transformed.count('}'))


if __name__=='__main__':unittest.main()
