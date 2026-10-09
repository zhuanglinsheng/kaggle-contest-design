"""Exact polynomial certificate for Section 28's actual output envelope.

No HJB grid or floating-point boundary estimate is used. Analytic domination
and the stochastic output identity remain the mathematical inputs in the note.
"""
from fractions import Fraction as F
from math import factorial, isqrt

ETA = F(7, 4)


def sqrt_upper(x):
    scale = 10**10
    bound = F(isqrt(x.numerator*scale*scale//x.denominator)+1, scale)
    assert bound*bound >= x
    return bound


def conv(a, b, n):
    out = [F(0)]*(n+1)
    for i, x in enumerate(a):
        for j, y in enumerate(b[:n-i+1]):
            out[i+j] += x*y
    return out


def evaluate(coefficients, x):
    result = F(0)
    for c in reversed(coefficients):
        result = result*x+c
    return result


def psi_upper(u):
    z=u*u/2
    even=[F(1)]
    for n in range(1, 13):
        even.append(even[-1]*z/n)
    e=sum(even)+even[-1]*z/13/(1-z/14)
    odd=[u]
    for m in range(1, 13):
        odd.append(odd[-1]*u*u/(2*m+1))
    o=sum(odd)+odd[-1]*u*u/27/(1-u*u/29)
    return F(3991,10000)*e+F(50,157)*o-F(3,40)*u


def coefficients(n):
    a=F(3991,10000)
    b=F(50,157)
    psi=[]
    for k in range(n+1):
        if k%2==0:
            m=k//2
            psi.append(a/F(2**m*factorial(m)))
        else:
            m=(k-1)//2
            psi.append(b*F(2**m*factorial(m), factorial(k)))
    psi[1]-=F(3,40)
    power=[F(1)]
    hs=[]
    for k in range(n+1):
        power=conv(power,psi,n)
        hs.append(ETA**k/F(k+1)*power[k])
    return hs


def verify(n=24, ell=F(17), target=F(207,100)):
    # Elementary exact bounds behind the Gaussian derivative majorant.
    assert F(3991,10000)**2*2*F(157,50)>1
    assert F(243,1000)**2*2*F(157,50)*F(163,60)>1
    assert F(7,22)-F(243,1000)>F(3,40)
    # Flux domination for H+ETA*A: cross and A^2 coefficients are larger.
    assert ETA**2 >= 3
    assert ETA**2/2 >= F(1,2)
    E=F(18,35); M=F(5,8); h=F(63,250)
    assert psi_upper(ETA*E*M)<M
    hs=coefficients(n)
    remainder=M-evaluate(hs,E)
    assert remainder>0
    # P bounds gain(e)/e, retaining time-integration denominators.
    p=[F(0)]*(n+2)
    for k in range(1,n+1):
        p[k-1]+=hs[k]/(k+1)
        p[k]+=2*F(243,1000)*hs[k]/(k+2)
    p[n]+=remainder/(E**(n+1)*(n+2))
    p[n+1]+=2*F(243,1000)*remainder/(E**(n+1)*(n+3))

    def exposure_square(x):
        return (h-x*x/4)**2*(1+ell*x*x)

    # Global maximum in x^2 for the joint-budget exposure polynomial.
    vertex_square=4*h/3-F(2,3)/ell
    e_max=sqrt_upper((h-vertex_square/4)**2*(1+ell*vertex_square))
    assert e_max<E
    pending=[(F(0),F(1)),(F(1),F(251,250))]
    certified=0
    finest=F(1)
    largest=F(0)
    while pending:
        lo,hi=pending.pop()
        r=sqrt_upper(1+ell*hi*hi)
        rq=sqrt_upper(1+ell*(hi+F(1,1000))**2)
        em=sqrt_upper(max(exposure_square(lo),exposure_square(hi)))
        if lo*lo<=vertex_square<=hi*hi:
            em=e_max
        # Above x=1 keep the positive exposure bound; competitive x<=sqrt(4h).
        if lo>=1:
            em=max(em,(h-F(1,4))*r)
        bound=F(251,100)**2*rq*r*r*evaluate(p,em)**2
        if bound<target**2:
            certified+=1
            finest=min(finest,hi-lo)
            largest=max(largest,bound)
        else:
            if hi-lo<F(1,2**20):
                raise AssertionError((n,lo,hi,float(bound)))
            mid=(lo+hi)/2
            pending.extend(((lo,mid),(mid,hi)))
    print('PASS: exact low-precision output coefficient <',str(target))
    print('Taylor order:',n,'rational covering intervals:',certified,
          'finest width:',str(finest),'largest squared upper:',float(largest))
    print('Scope: ell<=',str(ell),'at the Section 27 external reference, not all precisions.')
    # A competitive output must exceed .99 of the reference output.
    assert sum(F(5**k,factorial(k)) for k in range(12))>F(253,2)
    assert 1-F(5,502)>F(99,100)
    # Empty mature subgame: x<=1/2, Q0/ST<3/2, e<18/35.
    constant=M*F(251,100)*F(49,40)
    assert h*F(3,2)<E
    assert F(49,40)**2>F(3,2)
    assert F(71,100)**2>F(501,1000)
    assert F(71,100)*(1+constant*(h-F(1,16)))<F(99,100)
    # d/dx sqrt(a+x)*(1+constant*(h-x^2/4)) is positive on [0,1/2].
    assert 1+constant*h-constant*(F(5,16)+F(1,2000))>0
    print('PASS: competitive deadlines T<=delta are excluded.')
    return certified, finest, largest


if __name__=='__main__':
    verify()
    verify(ell=F(21),target=F(12,5))
