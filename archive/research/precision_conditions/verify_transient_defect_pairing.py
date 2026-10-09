"""Exact constants for ledger section 94; no output ranking is certified."""

from fractions import Fraction as F
from itertools import product


def verify():
    assert F(47, 384)-F(2, 21) == F(73, 2688) > 0
    assert (1-F(1, 4)/12)/4 == F(47, 192)
    assert 1/(1-F(1, 4)/2) == F(8, 7)
    assert F(1, 12)-F(1, 8) == -F(1, 24)
    # Formal jet of the exact PDE, after F = Y - w*g*E/2.
    # Degree at most two in each variable; the full three-point tensor
    # grid certifies the identity by repeated polynomial interpolation.
    for w, e, ez, g, gz, b, fy in product((F(-1), F(0), F(1)), repeat=7):
        yz = fy+w*(gz*e+g*ez)/2
        direct = w*ez*yz/2-w*g*b*ez/2-3*w*w*g*ez*ez/8
        source = w*w*ez*gz*e/4-w*w*g*ez*ez/8-w*g*b*ez/2
        assert direct == w*ez*fy/2+source
    print("Verified exact transient pairing source and initial negative margin.")
    print("The late-time sign uses the established positive slow-mode remainder.")


if __name__ == "__main__":
    verify()
