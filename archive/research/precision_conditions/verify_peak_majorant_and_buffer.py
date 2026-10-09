"""Exact signs for the pointwise analytic-jet estimate in Section 30.

No mesh of HJB solutions is used. The finite interval cover checks a scalar
upper bound after the analytic comparison and exact output identity.
"""
from fractions import Fraction as F
from math import factorial, isqrt


def exp_upper(z):
    n=16
    terms=[F(1)]
    for k in range(1,n+1):
        terms.append(terms[-1]*z/k)
    return sum(terms)+terms[-1]*z/(n+1)/(1-z/(n+2))


def sqrt_upper(x):
    scale=10**8
    return F(isqrt(x.numerator*scale*scale//x.denominator)+1,scale)


def cap(e):
    for m in range(401,661):
        M=F(m,1000)
        if F(2,5)*exp_upper((F(7,4)*e*M)**2)<M:
            return M
    raise AssertionError(('No certified analytic cap',str(e)))


def verify():
    eta=F(7,4); E=F(61,100); M=F(33,50); radius=F(707,1000)
    assert eta**2>3
    assert F(2,5)*exp_upper(radius**2)<M
    assert radius-eta*E*M>0
    # y <=1.9 atanh(sqrt(.72)) <2.41 and y*coth(y)<2.45.
    assert F(19,10)**2*F(8,5)<F(241,100)**2
    assert sum(F(241,50)**k/factorial(k) for k in range(24))>F(243,2)
    assert F(8,5)/17<F(307,1000)**2
    xmax=F(19,10)*F(307,1000)
    h=F(63,250); R=F(49,20)
    assert (h-F(1,8)**2/4)*R<E
    assert F(783,500)**2>R
    pending=[(F(1,8),xmax)]
    count=0; worst=F(0); smallest=F(1)
    while pending:
        lo,hi=pending.pop()
        w=h-lo*lo/4
        mc=cap(w*R)
        upper=sqrt_upper(F(1,1000)+hi)*(1+mc*F(251,100)*F(783,500)*w)
        if upper<F(99,100):
            count+=1; worst=max(worst,upper); smallest=min(smallest,hi-lo)
        else:
            assert hi-lo>F(1,2**22),(str(lo),str(hi),float(upper))
            mid=(lo+hi)/2
            pending.extend(((lo,mid),(mid,hi)))
    # New early-layer exposure, using S_delta>(2/3)sigma^2 delta.
    assert F(19,10)-F(1,3)==F(47,30)
    assert F(30,47)*h<F(161,1000)
    # Mature variance lower bound: tau>.9 delta.
    assert F(3,4)/F(9,10)==F(5,6)
    assert F(18,25)/F(11,6)==F(108,275)
    assert F(108,275)>F(39,100)
    assert 24**2*F(18,25)/1000<F(5,6)**2
    # The joint-budget analytic construction now covers ell<=36.
    h=F(63,250); ell=F(36)
    vertex=4*h/3-F(2,3)/ell
    assert (h-vertex/4)**2*(1+ell*vertex)<E**2
    print('PASS: analytic gradient cap for |alpha|T/S_T<=.61, positive radius margin.')
    print('PASS: T<=1.9 delta excluded on',count,'rational covering intervals.')
    print('Largest output upper:',float(worst),'smallest width:',str(smallest))
    print('PASS: early exposure < .161; future innovation variance > .39 Q_delta.')
    print('Scope: analytic construction and time localization, not full joint optimality.')


if __name__=='__main__':
    verify()
