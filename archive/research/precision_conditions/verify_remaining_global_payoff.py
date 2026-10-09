"""Exact norm/time-integral signs for the remaining precision payoff screen.

The kernel domination, Brownian coupling and positive-series comparisons
are mathematical proof inputs, recorded in Sections 40--41 of the note.
This is not a mesh of optimizers or a finite-difference HJB certificate.
"""
from fractions import Fraction as F
from math import ceil
from functools import lru_cache
from verify_mature_spatial_strip import exp_upper


ETA = F(7, 4)
WMAX = F(63, 250)
WEAK = F(3, 25)


def weak_gaussian(rho):
    return F(3991, 10000)*exp_upper(rho*rho/2) + (
        F(50, 157)*rho/(1-rho*rho/3)-F(3, 40)*rho
    )


def verify_weak():
    K, R, d = F(1, 6), F(4, 5), F(1, 10000)
    assert R*R < F(2, 3)
    h0 = F(3991, 5000)*exp_upper((R*R+d*d)/2)
    C = exp_upper(R*R/2)*(
        F(221, 100)+8*d*exp_upper(2*d*d)+2*(R+d)*exp_upper(d*d)
    )
    a1 = K*F(97, 500)*F(10001, 10000)*(
        F(6633, 8000)*exp_upper(3*R*R/4)
        +F(2633, 8000)*exp_upper(7*R*R/4)
    )
    delta, A = F(3, 40), F(13, 100)
    H = h0+delta
    assert K*C*A*H/2 < delta
    assert a1+K*C*(3*A*A+delta*(2*h0+delta))/4 < A
    assert max(K*C*(A+H*F(5, 4))/2,
               K*C*(H/F(5, 4)+3*A)/2) < 1
    radius = F(113, 200)
    assert radius*radius < R*R/2

    scale = F(10, 21)
    r0, r1, r2 = F(1, 25)*scale**2, F(17, 100)*scale**2, scale**2
    a0, a_1, a2 = F(11, 100)*scale, F(6, 25)*scale, scale
    m0 = F(3991, 10000)+(r0+ETA*a0)/2
    rho, M, time = F(7, 125), F(19, 40), F(42, 625)
    ratio = rho/radius
    initial = weak_gaussian(rho)+(r0+ETA*a0)/2+rho*(r1+ETA*a_1)/2
    initial += rho*rho*(r2+ETA*a2)/4
    initial += (delta+ETA*A)/2*ratio**3/(1-ratio)
    assert ETA*time*M < rho
    assert initial < M
    eh = r0+4/ETA**2*(M-m0)
    a = a0+2/ETA*(1+3/ETA**2)*(M-m0)
    assert eh < F(11, 250) and a < F(14, 125)
    # Actual early output: residual and state-displacement terms.
    w = WEAK
    early_coefficient = F(1489, 500)*(F(7, 15)+(
        F(55, 36)+F(243, 2000)*F(14, 25)*F(14, 15)
    )*w)
    assert early_coefficient < 2
    print('PASS: weak early remainder/difference caps:', float(eh), float(a))
    print('PASS: weak early output coefficient:', float(early_coefficient), '<2')


def verify_mature():
    # Gaussian Lipschitz, expectation and variance-ratio constants.
    assert F(237, 200)**2 > F(94, 67)
    assert F(398, 1000)**2*2*F(22, 7) < 1
    assert F(283, 200)**2 > 2
    assert F(1773, 1000)**2 > F(22, 7)
    assert 2*F(1773, 1000)*F(243, 1000) < F(431, 500)
    assert F(1778, 500)*F(398, 1000) > F(283, 200)
    assert F(1489, 500)*F(398, 1000) > F(237, 200)
    assert F(75, 134) < F(14, 25)
    # A positive geometric sum bounds (cosh(K)-1)/(2K^2).
    assert F(1, 4)/(1-F(7, 20)**2/12) < F(1, 2)*F(18, 25)**2
    w = WMAX
    residual = F(11, 50)*w+F(8, 25)*w*w
    factor = (1+w*w/2)*(
        1+F(889, 250)*residual
        +F(1489, 500)*(F(243, 1000)+F(431, 500)*residual)*F(56, 125)*w
    )
    assert (factor-1)/w < F(9, 5)
    print('PASS: mature output coefficient:', float((factor-1)/w), '<1.8')


def psi_strong(rho, q):
    radius = F(159, 500)
    ratio = rho/radius
    return (
        F(2, 5)*exp_upper(rho*rho)
        +(F(1, 25)*q*q+ETA*F(11, 100)*q)/2
        +rho*(F(17, 100)*q*q+ETA*F(6, 25)*q)/2
        +rho*rho*(q*q+ETA*q)/4
        +(F(3, 20)*q*q+ETA*F(47, 200)*q)/2*ratio**3/(1-ratio)
    )


@lru_cache(maxsize=None)
def cap(w, fraction):
    q, time = w/WMAX, F(75, 134)*w*fraction
    # Floating point chooses a candidate only; exact arithmetic certifies it.
    start = ceil(float(psi_strong(F(0), q))*1000)
    for index in range(start, 701):
        M = F(index, 1000)
        rho = ETA*time*M+F(1, 10**7)
        if rho < F(159, 500) and psi_strong(rho, q) < M:
            return M
    raise AssertionError((str(w), str(fraction)))


@lru_cache(maxsize=None)
def averaged_cap(w):
    return sum(cap(w, F(j, 16)) for j in range(1, 17))/16


def verify_large():
    pending = [(WEAK, WMAX)]
    intervals, worst = 0, F(0)
    while pending:
        lo, hi = pending.pop()
        # f is only bounded above. If the early coefficient is smaller
        # than the mature one, f=0 gives the largest convex combination.
        early = max(F(9, 5), (F(1489, 500)*averaged_cap(hi)-1)/lo)
        total = F(28, 53)*F(9, 5)+F(25, 53)*early
        if total < F(12, 5):
            intervals += 1
            worst = max(worst, total)
        else:
            assert hi-lo > F(1, 2**20), (float(lo), float(hi), float(total))
            middle = (lo+hi)/2
            pending.extend(((lo, middle), (middle, hi)))
    print('PASS: large-prize remaining output C<2.4; intervals:', intervals)
    print('Largest rational upper:', float(worst))


if __name__ == '__main__':
    verify_weak()
    verify_mature()
    verify_large()
    print('Scope: payoff inequalities after the analytic comparisons, not optimization numerics.')
