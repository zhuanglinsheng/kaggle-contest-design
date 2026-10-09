"""Exact time-averaged analytic cap; Section 35's deadline exclusion.

The time rectangles bound an integral of proved analytic caps, not an HJB
solution grid. All final inequalities use rational arithmetic and tails.
"""
from fractions import Fraction as F
from math import factorial
from functools import lru_cache

from verify_peak_majorant_and_buffer import cap, sqrt_upper
from verify_gaussian_jet_cap import verify as verify_gaussian_cap


@lru_cache(maxsize=None)
def time_average_cap(exposure):
    total = F(0)
    for j in range(1, 17):
        upper = exposure * F(j, 16)
        total += cap(upper) if upper <= F(61, 100) else F(17, 25)
    return total / 16


def verify():
    verify_gaussian_cap()
    h = F(63, 250)
    ratio = F(27, 10)
    factor = F(53, 25)
    z_upper = F(334, 125)
    xmax = F(73, 125)
    assert factor * F(63, 50) < z_upper
    assert sum((2*z_upper)**j/factorial(j) for j in range(24)) > (
        ratio+z_upper)/(ratio-z_upper)
    assert (factor*F(63, 50))**2/21 < xmax*xmax
    assert ratio * (h-F(1, 8)**2/4) < F(67, 100)
    root_ratio = F(411, 250)
    assert root_ratio*root_ratio > ratio
    pending = [(F(1, 8), xmax)]
    worst = F(0)
    count = 0
    while pending:
        lo, hi = pending.pop()
        w = h-lo*lo/4
        average = time_average_cap(ratio*w)
        upper = sqrt_upper(F(1, 1000)+hi) * (
            1+average*F(251, 100)*root_ratio*w
        )
        if upper < F(99, 100):
            worst = max(worst, upper)
            count += 1
        else:
            assert hi-lo > F(1, 2**20), (float(lo), float(hi), float(upper))
            middle = (lo+hi)/2
            pending.extend(((lo, middle), (middle, hi)))
    assert h/(factor-F(1, 3)) < F(142, 1000)
    print("PASS: competitive remaining designs have T > (53/25)*delta.")
    print("Rational output intervals:", count, "largest upper:", float(worst))
    print("PASS: early exposure alpha*delta/Q_delta < .142.")
    print("Scope: deadline localization, not full joint-optimum existence.")


if __name__ == "__main__":
    verify()
