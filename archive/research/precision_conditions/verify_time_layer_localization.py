"""Exact algebra for the Section 29 maturity buffer.

These signs verify the continuous argument in the research note; they do not
certify equilibrium derivative bounds or the remaining joint-optimum gap.
"""
from fractions import Fraction as F
from math import factorial


def verify():
    h=F(63,250)
    r=F(18,25)
    # The established atanh(sqrt(r))^2<8/5 implies delta/T0<.31.
    assert F(8,5)/17<F(31,100)**2
    assert F(3,2)**2*F(8,5)<F(19,10)**2
    # z*coth(z)<=2 at z=1.9, by exp(3.8)>39 and monotonicity.
    assert sum(F(19,5)**k/factorial(k) for k in range(12))>39
    assert 2*h<F(18,35)
    C=F(5,8)*F(251,100)*F(283,200)
    xmax=F(93,200)
    assert F(283,200)**2>2
    assert F(683,1000)**2>F(1,1000)+xmax
    assert F(683,1000)*(1+C*(h-xmax*xmax/4))<F(99,100)
    # Uniform positivity of the upper-bound derivative on [0,xmax].
    assert 1+C*h-C*(5*xmax*xmax/4+F(1,1000)*xmax)>0
    # All competitive deadlines are greater than T0/8, independently of ell.
    assert F(9,25)**2>F(1,1000)+F(1,8)
    assert F(63,100)**2>4*F(22,7)*h/8
    assert F(9,25)+F(63,100)==F(99,100)
    # Early exposure alpha*delta/Q_delta < (6/7)w < .216.
    assert F(6,7)*h==F(27,125)
    # For ell<=L use L>1000: b0<.032, atanh(b0)<.04.
    assert F(4,125)**2>F(1,1000)
    assert F(4,125)/(1-F(4,125)**2)<F(1,25)
    # atanh(sqrt(r))>6/5, via a finite positive series and a rational root lower.
    assert F(106,125)**2<r
    assert F(106,125)*sum(r**k/F(2*k+1) for k in range(10))>F(6,5)
    assert (F(29,25))**2>F(16,9)*r
    # Hence S_delta/(sigma^2*(T-delta)) < 3/2.
    # For ell>=L use T-delta>T/3>T0/24 instead.
    assert 24**2*r/F(1000)<F(3,2)**2
    assert r/F(5,2)==F(36,125)
    print('PASS: every complementary competitive design satisfies T>1.5 delta.')
    print('PASS: early exposure < .216; mature innovation variance > .288 Q_delta.')
    print('Scope: parameter localization, not the missing derivative propagation theorem.')


if __name__=='__main__':
    verify()
