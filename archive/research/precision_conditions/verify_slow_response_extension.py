"""Exact rational bounds for the strengthened slow organizer response.

This checks constants, not the differential equations or global precision
ranking; their arguments are in PROOF_LEDGER section 86.
"""

from fractions import Fraction as F


def verify():
    w = F(3, 10)
    flux_ratio = (F(77, 40) + F(1029, 2000) * w**2) / (
        (1 - w**2 / 2) * (1 - F(1029, 2000) * w**4)
    )
    output_slope = (1 - F(11, 10) * w) * (1 - w**2 / 2) / 2
    assert flux_ratio < F(25, 12)
    assert output_slope > F(19, 60)
    assert F(19, 60) - F(25, 24) * w == F(1, 240)
    assert 1 - F(1029, 2000) * w**4 > 0
    residual_constant = F(31, 100)
    assert residual_constant * (1-F(21, 160)) > F(21, 80)
    derivative_error = F(167, 200)*F(8, 7) + F(441, 400)*F(231, 400)*F(8, 21)
    assert derivative_error < F(6, 5)
    assert F(21, 20)*(F(1, 2)+F(6, 5)*F(1, 4)) == F(21, 25)
    assert F(21, 25)/(1-F(21, 25)/8) == F(168, 179) < 1
    w = F(9, 20)
    output_slope = (1-F(11, 10)*w)*(1-w*w/2)/2
    assert output_slope-w/2 > F(1, 600)
    print("Verified rational constants: slow response is negative for 0 < w <= 9/20.")
    print("This does not establish all-precision ranking or a joint-design region.")


if __name__ == "__main__":
    verify()
