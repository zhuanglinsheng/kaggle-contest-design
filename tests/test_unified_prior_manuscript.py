import re
import unittest
import pandas as pd
from pipeline.config import PAPER_DIR,MAIN_VALIDATION_DIR
from pipeline.empirical.unified_prior_manuscript import update_figure,environment


class ManuscriptPatchTests(unittest.TestCase):
    def test_figure_headers_are_not_coefficient_rows(self):
        source=(PAPER_DIR/'PaperJK7.tex').read_text()
        row=pd.read_csv(MAIN_VALIDATION_DIR/'chapter5_regressions.csv').set_index('specification').loc['lambda_benchmark'].to_dict()
        row['coefficient']+=.1
        transformed=environment(source,'figure','fig-validate-lambda',lambda block:update_figure(block,row))
        self.assertIn('& coef. & stderr & 95\\% CI',transformed)
        self.assertIn(r'\textcolor{blue}{0.817',transformed)
        self.assertEqual(source.count('{')-source.count('}'),transformed.count('{')-transformed.count('}'))


if __name__=='__main__':unittest.main()
