"""Exact initial analytic-norm bound keeping the second Hermite cancellation."""
from fractions import Fraction as F


def exp_lower(z):
    terms=[F(1)]
    for k in range(1,52):
        terms.append(terms[-1]*z/k)
    return sum(terms)


def exp_upper(z):
    assert z>=0
    terms=[F(1)]
    for k in range(1,52):
        terms.append(terms[-1]*z/k)
    return sum(terms)+terms[-1]*z/52/(1-z/53)


def verify():
    r=F(4,5); M=F(17,25); E=F(67,100); eta=F(7,4)
    assert r-eta*E*M>0
    # y>=4: the uncorrected Gaussian translation majorant decreases.
    assert F(2,5)*exp_upper(r*r)/exp_lower((4-r)**2/2)<M
    pending=[(F(0),F(4))]; count=0; worst=F(0)
    while pending:
        lo,hi=pending.pop()
        nearest=min(max(r,lo),hi)
        exponent=r*r-(nearest-r)**2/2
        positive=exp_upper(exponent) if exponent>=0 else 1/exp_lower(-exponent)
        cancellation=r*r*min(lo*lo,F(1))/exp_upper(hi*hi/2)
        upper=F(2,5)*(positive-cancellation)
        if upper<M:
            count+=1; worst=max(worst,upper)
        else:
            assert hi-lo>F(1,2**20)
            mid=(lo+hi)/2; pending.extend(((lo,mid),(mid,hi)))
    print('PASS: Gaussian derivative-sum initial norm < .68 at radius .8.')
    print('Intervals:',count,'largest upper:',float(worst))
    print('PASS: analytic Nash continuation for |alpha|T/S_T<=.67.')
    print('Scope: initial derivative-norm estimate; no designer optimality claim.')


if __name__=='__main__':
    verify()
