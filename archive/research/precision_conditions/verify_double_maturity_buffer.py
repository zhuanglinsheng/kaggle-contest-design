"""Exact payoff exclusion for T<=2 delta in Section 33."""
from fractions import Fraction as F
from math import factorial
from verify_peak_majorant_and_buffer import cap, sqrt_upper


def verify():
    h=F(63,250); R=F(64,25); xmax=F(11,20)
    assert F(63,50)**2/21<F(11,40)**2
    assert sum(F(126,25)**k/factorial(k) for k in range(24))>127
    assert (h-F(1,8)**2/4)*R<F(67,100)
    pending=[(F(1,8),xmax)]; count=0; worst=F(0)
    while pending:
        lo,hi=pending.pop()
        w=h-lo*lo/4; exposure=w*R
        M=cap(exposure) if exposure<=F(61,100) else F(17,25)
        bound=sqrt_upper(F(1,1000)+hi)*(1+M*F(251,100)*F(8,5)*w)
        if bound<F(99,100):
            count+=1; worst=max(worst,bound)
        else:
            assert hi-lo>F(1,2**22)
            mid=(lo+hi)/2; pending.extend(((lo,mid),(mid,hi)))
    # Exact real principal-symbol eigenvalue bound, evaluated at maturity only.
    K=F(7,20); H=F(23,25)
    speed_squared=H*H/4*(1+3*K*K/4)
    assert speed_squared<F(49,100)**2
    exposure=F(3,5)*h
    assert (F(49,100)*exposure)**2<F(1,128)
    print('PASS: competitive deadlines satisfy T>2 delta; intervals:',count)
    print('Largest output upper:',float(worst))
    print('PASS: early exposure < .1512; FROZEN real-symbol travel is less than half the mature strip.')
    print('Scope: the frozen-symbol observation does not prove nonlinear propagation.')


if __name__=='__main__':
    verify()
