"""Algebra in ledger 109; not a global joint-design certificate.

The continuous Riccati, differentiability and equilibrium arguments live
in the ledger. No sampled parameter grid is used to prove a sign.
"""
from fractions import Fraction as F


def multiply(a, b):
    result = [F(0)]*(len(a)+len(b)-1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            result[i+j] += x*y
    return result


def verify():
    # Worst spatial endpoint in 2N-D when its z^2 coefficient is negative.
    assert multiply([1, -1], [1, 0, 3, 8]) == [1, -1, 3, 5, -8]
    # Integral of (1-t)*(s^2*t+s*t^2+t^3/3).
    moments = [F(1, n+1)-F(1, n+2) for n in [1, 2, 3]]
    assert moments == [F(1, 6), F(1, 12), F(1, 20)]
    assert [60*moments[0], 60*moments[1], 20*moments[2]] == [10, 5, 1]
    # Integral over 0<t<r<1 of t*r.
    assert F(1, 2)*F(1, 4) == F(1, 8)
    print("Verified radial factorization, zero-precision coefficient and triangle moment.")
    print("Uniform C1 dependence and exact weak-prize sorting require the continuous proof.")
    print("No operational global disclosure region is certified by this script.")


if __name__ == '__main__':
    verify()
