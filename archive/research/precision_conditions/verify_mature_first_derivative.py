"""Exact constants for Section 31's actual mature-gradient derivative bounds.

The proof differentiates the mild equation and splits its heat integrals;
it never differentiates the real Gaussian envelope itself.
"""
from fractions import Fraction as F


def verify():
    # Near interval: log(10/9)<.106 and asin(sqrt(.1))<.323.
    z=F(1,9)
    assert z-z*z/2+z**3/3<F(53,500)
    assert F(317,1000)**2>F(1,10)
    assert F(317,1000)*(1+F(1,60)+F(3,4000)+F(1,900))<F(323,1000)
    assert F(43,50)*F(53,500)+F(4,5)*2*F(323,1000)<F(61,100)
    # Far interval: (8/exp(1)-2)*sqrt(.9)+4*atanh(sqrt(.9))<8.2.
    u=F(949,1000)
    assert u*u>F(9,10)
    N=80
    atanh_upper=sum(u**(2*k+1)/F(2*k+1) for k in range(N))
    atanh_upper+=u**(2*N+1)/(F(2*N+1)*(1-u*u))
    assert (F(480,163)-2)*u+4*atanh_upper<F(41,5)
    # Existing mature real-gradient bootstrap gives H<=.92 E_Q, |A|<=K H/2.
    K=F(7,20); H=F(23,25); An=K*H/2
    near=F(61,100); far=F(41,5)
    Dh=F(11,10); Da=F(5,2)*K
    rhs_h=F(7,10)+K*far*H*An/2+K*near*(An*Dh+H*Da)/2
    rhs_a=K*far*(3*An*An+H*H)/4+K*near*(3*An*Da+H*Dh)/2
    assert rhs_h<Dh
    assert rhs_a<Da
    # Keep the explicit first prize coefficient A^(1)=-alpha*T(G,G).
    # Its first derivative has Gaussian norm <=(2/pi)K<(2/3)K.
    assert F(163,60)>F(8,3)
    B=F(5,6)*H*H
    source=3*H*H/4+B*(H+F(4,5))
    U=F(23,10); L=F(7,5)
    rh=far*H*H/4+near*(H/2*(F(7,10)+U*K*K)+H*L)/2
    ra=F(2,3)+K*K/4*(far*source+near*(3*H*L+U*(H+F(4,5))+B*(F(7,5)+U*K*K)))
    assert rh<U
    assert ra<L
    # A shorter independent audit of the full mature linearization.
    scale=K*F(10,3)/2
    assert scale*(An+H)<1
    assert scale*(H+3*An)<1
    # Uniform holomorphic policy neighborhood about every real mature branch.
    eps=F(1,1000); radius=F(1,50)
    lip=(K+eps)*F(10,3)/2*(H+3*An+4*radius)
    residual=eps*F(10,3)*max(H*An/2,(3*An*An+H*H)/4)
    assert lip<1
    assert residual<(1-lip)*radius
    print('PASS: Q-weighted |H_y|<11/10 and |A_y|<(5/2)K.')
    print('Peak bootstrap upper bounds:',float(rhs_h),float(rhs_a))
    print('PASS: Gaussian-norm linearization has row norm <',float(scale*(H+3*An)))
    print('PASS: complex effective-prize radius .001, policy radius .02; contraction <',float(lip))
    print('PASS: sharper |(H-2G)_y|<(23/10)K^2, |A_y|<(7/5)K.')
    print('Normalized residual bootstrap bounds:',float(rh),float(ra))
    print('Scope: real first derivatives on the mature subgame, not early-layer optimality.')


if __name__=='__main__':
    verify()
