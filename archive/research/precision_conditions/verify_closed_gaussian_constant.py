"""Rational signs after Section 37's closed Gaussian moment identity.

The integral identity and Jensen step are proved in the note. This script
checks only their numerical rational bound and the mature contraction caps.
It does not certify early continuation or a joint optimum.
"""
from fractions import Fraction as F
from verify_mature_spatial_strip import exp_upper


def verify():
    # 2*sqrt(2)*exp(-1/4) < 221/100, after squaring.
    lower_exp_half = sum(F(1, 2)**j / factorial(j) for j in range(8))
    gaussian = F(221, 100)
    assert gaussian*gaussian*lower_exp_half > 8
    K, R, d = F(7, 20), F(7, 25), F(1, 10000)
    H, A, weight = F(26, 25), F(2, 5), F(29, 20)
    C = exp_upper(R*R/2) * (
        gaussian+8*d*exp_upper(2*d*d)+2*(R+d)*exp_upper(d*d)
    )
    base = F(3991, 5000)*exp_upper((R*R+d*d)/2)
    assert base+K*C*A*H/2 < H
    assert K*C*(3*A*A+H*H)/4 < A
    lipschitz = max(K*C*(A+H*weight)/2,
                    K*C*(H/weight+3*A)/2)
    assert lipschitz < 1
    beta = F(479, 500)
    assert 4*(beta-A)**2 > A*A+H*H
    # Only initial-cap feasibility for Section 36's conditional energy bound.
    assert R*R/2 > (beta*F(71, 500)+F(61, 1000))**2
    print("PASS: C_G < 2*sqrt(2)*exp(-1/4) < 2.21.")
    print("PASS: mature strip R=.28; contraction:", float(lipschitz))
    print("PASS: initial-cap strip energy margin exceeds .061*sqrt(Q_delta).")
    print("Scope: no evolving pointwise-cap or joint-optimum certification.")


def factorial(n):
    result = 1
    for j in range(2, n+1):
        result *= j
    return result


if __name__ == "__main__":
    verify()
