"""Exact cancellation used in the positive-defect amplitude argument.

Checks algebra, not weighted regularity or all-precision output signs.
"""
from fractions import Fraction as F


def verify():
    for n in range(1, 101):
        w = F(n, 200)
        a = 3*w/2
        assert a*3*w/4-a*a/2 == 0
    assert F(9, 20) < F(1, 2)
    assert F(7, 10)-F(11, 20) == F(3, 20) > 0
    assert F(441, 400)*F(1, 2)**2/6 < F(1, 20)
    print("Verified defect-transform cancellation and separation of slow and fast rates.")


if __name__ == "__main__":
    verify()
