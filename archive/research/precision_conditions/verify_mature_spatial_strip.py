"""Exact moving-complex-shift contraction constants for Section 32."""
from fractions import Fraction as F


def exp_upper(z):
    terms=[F(1)]
    for k in range(1,30):
        terms.append(terms[-1]*z/k)
    return sum(terms)+terms[-1]*z/30/(1-z/31)


def atanh_upper(u):
    n=80
    return sum(u**(2*k+1)/F(2*k+1) for k in range(n))+u**(2*n+1)/(F(2*n+1)*(1-u*u))


def verify():
    K=F(7,20); R=F(1,4); d=F(1,10000)
    exponential=exp_upper((R*R+d*d)/2)
    # The sharp unshifted constant cannot just be reused after a real shift.
    # Control that perturbation by an explicit Gaussian-weight remainder.
    C=exp_upper(R*R/2)*(F(7,3)+8*d*exp_upper(2*d*d)+2*(R+d)*exp_upper(d*d))
    H=F(103,100); A=F(39,100); weight=F(29,20)
    base=F(3991,5000)*exponential
    h_rhs=base+K*C*A*H/2
    a_rhs=K*C*(3*A*A+H*H)/4
    lh=K*C*(A+H*weight)/2
    la=K*C*(H/weight+3*A)/2
    assert h_rhs<H and a_rhs<A
    assert max(lh,la)<1
    # An improved diffusion fraction for the fixed b=251 reference.
    r=F(18,25)
    assert 6*251**2-1>600**2
    assert F(106,125)**2<r
    assert F(106,125)*sum(r**k/F(2*k+1) for k in range(45))>F(5,4)
    assert atanh_upper(F(849,1000))<F(63,50)
    assert F(849,1000)**2>r
    assert F(1,600)/(1-F(1,600)**2)<F(1,599)
    assert exp_upper(F(2394,500))<124
    assert F(106,125)+F(9,10)*(F(5,4)-F(1,599))>2*F(123,125)
    assert F(9,152)*600>2
    print('PASS: actual mature spatial strip width (1/4)*sqrt(Q_t-S_T).')
    print('Complex cap RHS:',float(h_rhs),float(a_rhs),'contraction:',float(max(lh,la)))
    print('PASS: Q_delta-S_T > Q_delta/2 at the fixed external reference.')
    print('Scope: mature spatial analyticity, not early-layer payoff exclusion.')


if __name__=='__main__':
    verify()
