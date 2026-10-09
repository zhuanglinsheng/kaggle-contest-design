"""Exact polynomial identities in ledger 113; full boundary, not joint precision."""
from fractions import Fraction as F


def add(*polys):
    result = {}
    for p in polys:
        for key, val in p.items():
            result[key] = result.get(key, F(0))+val
    return {k: v for k, v in result.items() if v}


def scale(p, c):
    return {k: c*v for k, v in p.items() if c*v}


def mul(p, q):
    result = {}
    for k, v in p.items():
        for l, u in q.items():
            key = tuple(a+b for a, b in zip(k, l))
            result[key] = result.get(key, F(0))+v*u
    return {k: v for k, v in result.items() if v}


def verify():
    one, a, x, w = [{k: F(1)} for k in [(0,0,0),(1,0,0),(0,1,0),(0,0,1)]]
    a2, x2, x3 = mul(a,a), mul(x,x), mul(mul(x,x),x)
    m = add(a,x)
    n = add(m, mul(x,w), scale(mul(x,mul(w,w)),F(1,3)))
    actual = add(mul(add(scale(one,2),a,scale(x,-1),scale(w,-2)),mul(n,n)),
                 scale(mul(mul(add(a,one),add(a,one)),m),-1))
    c0 = scale(mul(m,mul(add(x,scale(one,-1)),add(x,scale(one,-1)))),-1)
    c1 = scale(mul(m,mul(add(x,scale(one,-1)),add(x,scale(a,-1)))),-2)
    c2 = scale(mul(x,add(scale(a2,2),scale(a,-8),mul(add(scale(a,3),scale(one,-2)),x),scale(x2,-5))),F(1,3))
    c3 = add(scale(x3,F(-2,3)),mul(add(scale(a,F(2,3)),scale(one,-2)),x2),scale(mul(a,x),F(-4,3)))
    c4 = scale(mul(x2,add(a,scale(x,-1),scale(one,-10))),F(1,9))
    c5 = scale(x2,F(-2,9))
    expected, wp = {}, one
    for c in [c0,c1,c2,c3,c4,c5]:
        expected = add(expected,mul(c,wp))
        wp = mul(wp,w)
    assert actual == expected
    remainder = add(c2,mul(m,mul(add(x,scale(a,-1)),add(x,scale(a,-1)))))
    assert remainder == scale(add(scale(mul(a2,a),3),scale(x3,-2),scale(x2,-2),scale(mul(a2,x),-1),scale(mul(a,x),-8)),F(1,3))
    assert F(1,6)+F(1,11)+F(1,30) == F(16,55) < F(1,3)
    assert F(5,57) < F(1,11)
    assert F(23,121) < F(1,5)
    assert F(7,8)/F(9,10) == F(35,36) < 1
    print('Verified full-boundary payoff polynomial, completed-square remainder and constants.')
    print('Continuous profile and sign arguments are in ledger 113.')
    print('This script checks full-disclosure payoff algebra only, not the ranking of finite-precision competitors.')


if __name__ == '__main__':
    verify()
