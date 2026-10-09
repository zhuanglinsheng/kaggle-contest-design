"""Rational inequalities in ledger 114; no intermediate-precision certificate."""
from fractions import Fraction as F


def verify():
    assert F(125,1024)/F(2,5) > F(1,4)
    assert F(1,64) < F(1,3)
    assert F(128,125)**2/F(6) < F(1,5)
    assert F(1,5)/3 == F(1,15)
    assert F(1,4)/15 < F(1,3)
    print('Verified inverse-center window, Gaussian amplification and integrated output constants.')
    print('The analytic inverse identity and zero-equilibrium identification are in ledger 114.')
    print('Endpoint comparison does not rank all intermediate feedback precisions.')


if __name__ == '__main__':
    verify()
