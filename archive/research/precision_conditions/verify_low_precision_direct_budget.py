"""Rational scalar comparison; no Nash/HJB coefficient truncation.

Certifies the explicit h upper envelope and the ideal-budget output
comparison described in LOW_PRECISION_DIRECT_BUDGET.txt.
"""
from fractions import Fraction as F
from math import comb
from verify_low_precision_payoff_envelope import sqrt_upper


def cover(test, interval):
    pending, count = [interval], 0
    while pending:
        lo, hi = pending.pop()
        if test(lo, hi):
            count += 1
        else:
            assert hi-lo > F(1, 2**24), (lo, hi)
            mid = (lo+hi)/2
            pending.extend(((lo, mid), (mid, hi)))
    return count


def add(a, b):
    return [(a[i] if i<len(a) else 0)+(b[i] if i<len(b) else 0)
            for i in range(max(len(a), len(b)))]


def scale(a, c):
    return [c*x for x in a]


def mul(a, b):
    out = [F(0)]*(len(a)+len(b)-1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i+j] += x*y
    return out


def bernstein(a, lo, hi):
    n = len(a)-1
    c = [sum(a[j]*comb(j,k)*lo**(j-k)*(hi-lo)**k
             for j in range(k,n+1)) for k in range(n+1)]
    return [sum(c[j]*F(comb(k,j),comb(n,j)) for j in range(k+1))
            for k in range(n+1)]


def verify():
    h0, b, eta, k, a = F(3991, 10000), F(50, 157), F(7, 4), F(6, 5), F(9, 50)
    end = F(18, 35)

    def cap(e):
        return h0+a*e/(1-k*e)

    def envelope_test(lo, hi):
        h = cap(hi)
        radius_square = (eta*hi*h)**2
        assert radius_square < 1
        upper = (b-F(3, 40))*eta*h
        upper += h0*eta**2*hi*h*h/(2*(1-radius_square/4))
        upper += b*eta**3*hi**2*h**3/(3*(1-radius_square/5))
        return upper < a/(1-k*lo)

    count_h = cover(envelope_test, (F(0), end))
    denominator = [F(1), -k]
    hnum = add(scale(denominator,h0), [F(0),a])
    radius_numerator = scale(mul([0,0,1],mul(hnum,hnum)),eta**2)
    a2 = add(mul(denominator,denominator), scale(radius_numerator,-F(1,4)))
    a3 = add(mul(denominator,denominator), scale(radius_numerator,-F(1,5)))
    hp = mul(add([a],scale(hnum,-(b-F(3,40))*eta)),mul(a2,a3))
    hp = add(hp,scale(mul(mul([0,1],mul(hnum,hnum)),mul(denominator,a3)), -h0*eta**2/2))
    hp = add(hp,scale(mul(mul([0,0,1],mul(mul(hnum,hnum),hnum)),a2), -b*eta**3/3))
    h_lower = min(min(bernstein(hp,lo,hi)) for lo,hi in
                  [(F(0),end/2),(end/2,end)])
    assert h_lower > 0
    # The ideal budget has x=t^2 and w=(1-t^4)/4.
    # R^(3/2)=(1+21t^4)^(3/4), bounded by two square roots.
    def budget_test(lo, hi):
        r_upper = sqrt_upper(1+21*hi**4)
        r_three_half = r_upper*sqrt_upper(r_upper)
        # Exact maximum of (1/4-x^2/4)^2(1+21x^2).
        vertex = F(1, 3)-F(2, 63)
        def e_square(t):
            return ((1-t**4)/4)**2*(1+21*t**4)
        e_upper = sqrt_upper(max(e_square(lo), e_square(hi)))
        if lo**4 <= vertex <= hi**4:
            e_upper = max(e_upper, sqrt_upper(((1-vertex)/4)**2*(1+21*vertex)))
        assert e_upper < end
        coefficient = F(251, 100)*r_three_half*a*(F(1, 2)+F(81, 500)*e_upper)/(1-k*e_upper)
        # Allowed coefficient is 4(t^3+2t^2+3t+4)/(t(1+t+t^2+t^3)^2).
        allowed_lower = 4*(lo**3+2*lo*lo+3*lo+4)/(hi*(1+hi+hi*hi+hi**3)**2)
        return coefficient < allowed_lower

    count_b = cover(budget_test, (F(0), F(1)))
    # An alternative polynomial certificate removes the adaptive covering.
    # Convex chords in t^2 and t^3 give R<=1+3.7t^2,
    # R^(3/2)<=1+9.2t^3 (the latter ratio is increasing).
    assert F(47, 10)**2 > 22
    assert F(51, 5)**4 > 22**3
    ep = scale(mul([1,0,0,0,-1], [1,0,F(37,10)]), F(1,4))
    positive_denominator = add([1],scale(ep,-k))
    assert min(min(bernstein(positive_denominator,lo,hi)) for lo,hi in
               [(F(0),F(1,2)),(F(1,2),F(3,4)),(F(3,4),F(1))]) > 0
    numerator = list(map(F, [4,3,2,1]))
    left = scale(mul(numerator, add([1], scale(ep, -k))), 4)
    right = mul([1,0,0,F(46,5)], add([F(1,2)], scale(ep,F(81,500))))
    right = scale(mul(right, mul([0,1], mul([1,1,1,1],[1,1,1,1]))), F(251,100)*a)
    polynomial = add(left, scale(right,-1))
    lower = min(min(bernstein(polynomial, lo, hi)) for lo,hi in
                [(F(0),F(1,2)), (F(1,2),F(3,4)), (F(3,4),F(1))])
    assert lower > 0
    # Strict local coefficient <2.4 near the sole zero (x,w)=(1,0).
    r0 = sqrt_upper(F(22))
    assert F(251, 100)*r0*sqrt_upper(r0)*a/2 < F(12, 5)
    print("PASS: h(e)<=.3991+.18e/(1-1.2e), 0<=e<=18/35.")
    print("PASS: ideal-budget output <reference output for 0<x<1.")
    print("Rational covering counts:", count_h, count_b)
    print("Three-interval Bernstein lower bound:", float(lower))
    print("Envelope polynomial Bernstein lower bound:", float(h_lower))
    print("Scope: scalar inequalities, with analytic domination as a separate input.")


if __name__ == "__main__":
    verify()
