"""Conditional reduction in ledger 110, not a proof of its output conjecture."""
from fractions import Fraction as F


def multiply(a, b):
    c = [F(0)]*(len(a)+len(b)-1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            c[i+j] += ai*bj
    return c


def verify():
    # At r=9/10: 64 times the exposure discriminant, as a polynomial in a.
    # (a+5-2r)^2 - 32*r*a.
    r = F(9, 10)
    c = 5-2*r
    coefficients = [c*c, 2*c-32*r, F(1)]
    assert coefficients == [F(256, 25), F(-112, 5), F(1)]
    a = F(1, 2)
    assert sum(v*a**i for i, v in enumerate(coefficients)) == F(-71, 100)
    assert 2*2+coefficients[1] < 0
    # At entry cost, gamma=3 utility envelope numerator.
    # Constant-in-e numerator -(x-1)^2; linear coefficient is
    # 2*(1-x)*(x-a^2)/x; e^2 coefficient is -(1+a)^2.
    # Its x(a+x)-cleared linear identity, separately for each a power.
    assert multiply([1, -1], [0, 1]) == [0, 1, -1]
    assert multiply([1, -1], [-1]) == [-1, 1]
    # Ledger 111: rigorous failure of the global output bound at q=128.
    assert (128-F(3, 4))*7/880 > 1
    print('Verified quarter-exposure discriminant and conditional payoff identities.')
    print('Verified the rational margin in the analytic q=128 counterexample.')
    print('The GLOBAL output upper bound is FALSE; its restricted joint-domain version is unproved.')


if __name__ == '__main__':
    verify()
