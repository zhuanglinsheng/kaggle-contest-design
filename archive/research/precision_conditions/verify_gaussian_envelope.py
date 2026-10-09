"""Exact rational checks for the uniform full-disclosure output envelope.

No quadrature or floating-point values enter the certificates.  The classical
bracket 3.14159 < pi < 3.14160 is the only externally fixed numerical bracket.
"""

from fractions import Fraction as F
from math import factorial


def sqrt_interval(x, digits=18):
    lo, hi = 0, 10 ** (digits + 1)
    scale = 10**digits
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if F(mid * mid, scale * scale) <= x:
            lo = mid
        else:
            hi = mid
    return F(lo, scale), F(hi, scale)


def exp_negative_interval(x, n=60):
    upper = sum((-x) ** j / factorial(j) for j in range(n + 1))
    lower = upper - x ** (n + 1) / factorial(n + 1)
    return lower, upper


def gaussian_integral_interval(z, n=60):
    upper = sum(
        (-1) ** j * z ** (2 * j + 1)
        / (2**j * factorial(j) * (2 * j + 1))
        for j in range(n + 1)
    )
    lower = upper - z ** (2 * n + 3) / (
        2 ** (n + 1) * factorial(n + 1) * (2 * n + 3)
    )
    return lower, upper


def log_interval(x, n=40):
    assert 1 <= x <= 2
    t = (x - 1) / (x + 1)
    lower = 2 * sum(t ** (2 * j + 1) / (2 * j + 1) for j in range(n))
    upper = lower + 2 * t ** (2 * n + 1) / ((2 * n + 1) * (1 - t * t))
    return lower, upper


pi_lo, pi_hi = F(314159, 100000), F(3927, 1250)
coef_lo = sqrt_interval(1 / (2 * pi_hi))[0]
coef_hi = sqrt_interval(1 / (2 * pi_lo))[1]
log2_lo, log2_hi = log_interval(F(2))
c_hi = log_interval(pi_hi / 2)[1] / 2


def gaussian_intervals(z):
    el, eu = exp_negative_interval(z * z / 2)
    il, iu = gaussian_integral_interval(z)
    return (F(1, 2) - coef_hi * iu, F(1, 2) - coef_lo * il), (
        coef_lo * el, coef_hi * eu
    )


bounds = []
first = (1 + c_hi + log2_hi) / 4 + F(1, 288)
second = (1 + c_hi - log2_lo) / 4 + F(7, 288)
for name, square, bound in [
    ("0 to 1/2", first, F(6952, 10000)),
    ("1/2 to 1", second, F(3969, 10000)),
]:
    assert 0 < square < bound * bound, name
    bounds.append(bound)
    print(name, "<", bound)

for a, b, bound in [
    (F(1), F(3, 2), F(2581, 10000)),
    (F(3, 2), F(2), F(1498, 10000)),
    (F(2), F(3), F(1290, 10000)),
]:
    qa, pa = gaussian_intervals(a)
    qb, pb = gaussian_intervals(b)
    assert 0 < qa[0] <= qa[1] < F(1, 2)
    factor = (2 - 2 * qa[1]) / (1 - 2 * qa[1])
    integral_hi = b * qb[1] - pb[0] - a * qa[0] + pa[1]
    square = (b - a) * factor * integral_hi
    assert 0 < square < bound * bound, (a, b)
    bounds.append(bound)
    print(a, "to", b, "<", bound)

q3, p3 = gaussian_intervals(F(3))
tail_square = F(4, 27) * (2 - 2 * q3[1]) / (1 - 2 * q3[1]) * p3[1]
tail_bound = F(363, 10000)
assert 0 < tail_square < tail_bound * tail_bound
bounds.append(tail_bound)
assert sum(bounds) < F(5, 3)
print("tail <", tail_bound)
print("half-line integral <", sum(bounds), "< 5/3")
print("C_G < 5; full-disclosure risk multiplier = 11")
