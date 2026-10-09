"""Rational constants for ledger 112, not a global disclosure theorem."""
from fractions import Fraction as F


def verify():
    assert F(1, 12)/(1-F(1, 80)) == F(20, 237) < F(1, 11)
    assert F(2, 11)+F(1, 484) < F(1, 5)
    assert F(1, 6)+F(1, 22)+F(1, 120) == F(97, 440) < F(1, 4)
    # Sharp-coefficient obstruction already at q=7.
    assert F(42, 13)/F(22, 7) > 1
    print('Verified exact-envelope constants and q=7 obstruction margin.')
    print('The profile comparison and Gaussian-resolvent identities are proved in the ledger.')
    print('Finite precision and joint-design regime boundaries are not certified here.')


if __name__ == '__main__':
    verify()
