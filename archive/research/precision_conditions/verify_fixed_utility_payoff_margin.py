"""Exact constants in Section 30's stronger FIXED-utility payoff argument.

The differentiation and exclusion of stationary points are continuous
arguments in the note. This script checks their rational sign margins.
"""
from fractions import Fraction as F


def verify():
    C=F(12,5); m=100; a=F(1,1000)
    # H-(3/5)L = 2/5 + (1-6C/5)w + Cw^2 is positive for all real w.
    assert (1-F(6,5)*C)**2-4*C*F(2,5)<0
    assert F(3,5)>F(3,4)**2
    kmin=F(3,4)/(F(3,4)+a)
    assert F(m,4)+(m+1)*kmin>2*C*(1+F(m,4))
    assert (m+1)/(1+a)>2*C
    assert F(m+2,2)/(1+a)>1
    assert 2*F(99,100)*251>m+1
    # Far boundaries of the auxiliary payoff function are negative.
    assert F(1,m)+F(1,4)-F(1,2)/(1+a)<0
    assert F(1,m)+F(1,4)-1/(1+a)<0
    # Abstract tail/local-delay assumptions used in the generalized proof.
    assert F(100001,100000)*(F(1,4)+F(1001,1000)/502)<F(63,250)
    assert F(100,99)**502>1+F(502,4)
    assert 503>m+1
    print('PASS: payoff algebra and generalized utility-tail/local-delay margins.')
    print('Scope: global payoff algebra only; the actual output envelope is checked separately.')


if __name__=='__main__':
    verify()
