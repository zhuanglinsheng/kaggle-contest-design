"""Exact scalar cover for Section 32's improved Gaussian heat operator.

The Gaussian moment calculation in the note reduces the operator constant
to one scalar function. This is not a grid of Nash policies or payoffs.
"""
from fractions import Fraction as F
from math import isqrt


def sqrt_upper(q):
    scale=10**9
    return F(isqrt(q.numerator*scale*scale//q.denominator)+1,scale)


def atanh_upper(u):
    n=22
    return sum(u**(2*k+1)/F(2*k+1) for k in range(n))+u**(2*n+1)/(F(2*n+1)*(1-u*u))


def log_upper(q):
    m=0
    while q>2:
        q/=2; m+=1
    assert q>=1
    return 2*m*atanh_upper(F(1,3))+2*atanh_upper((q-1)/(q+1))


def exp_lower(q):
    terms=[F(1)]
    for k in range(1,24):
        terms.append(terms[-1]*q/k)
    return sum(terms)


def verify():
    target=F(7,3)
    # Uniform absolute-normal moment majorant coefficients.
    assert F(4,5)**2*F(157,50)>2
    assert F(2,5)**2*2*F(157,50)>1
    assert F(4,5)*(2+F(5,2)*F(1,10))<target
    assert F(48,5)/exp_lower(F(4))<target
    pending=[(F(1,10),F(4))]
    count=0; largest=F(0); smallest=F(1)
    while pending:
        lo,hi=pending.pop()
        z=F(5,2)/lo
        arsinh=log_upper(z+sqrt_upper(1+z*z))
        bound=F(4,5)*(2+hi*hi*arsinh)/exp_lower(lo*lo/4)
        if bound<target:
            count+=1; largest=max(largest,bound); smallest=min(smallest,hi-lo)
        else:
            assert hi-lo>F(1,2**18),(float(lo),float(hi),float(bound))
            mid=(lo+hi)/2
            pending.extend(((lo,mid),(mid,hi)))
    print('PASS: Gaussian bilinear heat constant C_G<7/3.')
    print('Cover intervals:',count,'largest upper:',float(largest),'smallest width:',str(smallest))
    print('Scope: a mature subgame with a positive proved diffusion minimum.')


if __name__=='__main__':
    verify()
