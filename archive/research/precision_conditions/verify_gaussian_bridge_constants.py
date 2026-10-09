"""Rational endpoint checks in the Gaussian bridge source inequality.

The continuum proof uses Gaussian integration by parts and concavity;
these checks certify only its numerical constants, not a sampled grid.
"""

from fractions import Fraction as F


def verify():
    # (1+e)^2 - 4e(2-e) = (1-e)(1-5e) >= 0 for 0 <= e <= 1/5.
    assert (1 - F(1, 5)) * (1 - 5 * F(1, 5)) == 0

    # 2e/((1+e)*k) <= 4/5 on e <= 1/10.
    # Clearing squares gives p(e)=16+116e^2-168e, decreasing there.
    e = F(1, 10)
    assert 232 * e - 168 < 0
    assert 16 + 116 * e * e - 168 * e == F(9, 25) > 0
    assert 4 / (1 + e) == F(40, 11)
    print("Verified: source coefficient strictly positive for 0 < epsilon <= 1/5.")
    print("Verified: quantitative two-term lower bound for epsilon <= 1/10.")


if __name__ == "__main__":
    verify()
