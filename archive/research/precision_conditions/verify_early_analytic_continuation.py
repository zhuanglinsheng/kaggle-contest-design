"""Exact signs for Section 39's real jets and early continuation.

This certificate checks the constants AFTER the kernel and comparison
arguments in the note. It does not certify the missing global payoff bound.
"""
from fractions import Fraction as F
from math import factorial
from verify_mature_spatial_strip import exp_upper, atanh_upper


def exp_lower(q):
    return sum(q**j/factorial(j) for j in range(25))


def verify():
    # Gaussian weighted derivative factors used for the explicit first term.
    assert exp_lower(F(5, 4)) > F(80, 23)
    assert F(377, 50)**2 < 57 < F(189, 25)**2
    assert F(73, 100)*F(57, 25)**2 < 4
    assert exp_lower(F(827, 200)) > F(207, 25)*F(132, 25)**2/4
    assert F(367, 1000)*F(157, 50) > F(23, 20)
    assert F(147, 200)*4*F(157, 50)*F(707, 500) > 13

    # General near bound for n=2 and a direct third-kernel far bound.
    assert F(4743, 5000)**2 < F(9, 10)
    assert F(3163, 10000)**2 > F(1, 10)
    assert F(799, 1000)**2*F(157, 50) > 2
    assert F(429, 500)**2*exp_lower(F(1)) > 2
    near = (
        2*F(429, 500)*(1-F(4743, 5000))
        + 2*F(799, 1000)*F(3163, 10000)
    )/F(9, 10)
    assert near < F(33, 50)
    assert F(9487, 10000)**2 > F(9, 10)
    assert atanh_upper(F(9487, 10000)) < F(91, 50)
    assert F(633, 200)**2 > 10
    assert F(79, 250)**2 < F(1, 10)
    assert F(317, 1000)**2 > F(1, 10)
    assert F(19, 20)**2 > F(9, 10)
    assert F(237, 250)**2 < F(9, 10)
    # Three elementary weighted monomial maxima are <3.3,1.5,.86.
    assert exp_lower(F(1)) > F(271, 100)
    assert F(271, 100)**3*F(33, 10)**2 > 216
    assert F(271, 100)*F(3, 2) > 4
    assert F(271, 100)*F(43, 50)**2 > 2
    far = (
        F(33, 10)*2*F(19, 20)
        + 3*F(4, 5)*F(3, 2)*2*(1-F(79, 250))
        + 3*F(43, 50)*(2*F(91, 50)-2*F(237, 250))
        + 2*F(4, 5)*(2*F(633, 200)+2*F(317, 1000)-4)
        + 3*F(43, 50)*2*F(91, 50)
        + 3*F(4, 5)*(2*F(633, 200)-2)
    )
    assert far < 41

    K = F(7, 20)
    H0, H, EH, A = F(4, 5), F(21, 25), F(1, 25), F(11, 100)
    # The real centered map sits inside the complex region of Section 38.
    leading = K*F(97, 500)*F(4633, 4000)
    assert K*F(221, 100)*A*H/2 < EH
    assert leading+K*F(221, 100)*(3*A*A+EH*(H+H0))/4 < A
    assert leading < F(2, 25)
    dg1, dg2 = F(7, 10), F(23, 25)
    de1, da1 = F(17, 100), F(6, 25)
    first_h = K/2*(F(41, 5)*A*H+F(61, 100)*(A*(dg1+de1)+H*da1))
    first_a = F(367, 1000)*K+K/4*(
        F(41, 5)*(3*A*A+EH*(H+H0))
        + F(61, 100)*(6*A*da1+2*EH*dg1+2*H*de1)
    )
    assert first_h < de1 and first_a < da1
    de2, da2 = F(1), F(1)
    second_h = K/2*(
        41*A*H+F(33, 50)*(A*(dg2+de2)+2*da1*(dg1+de1)+H*da2)
    )
    second_a = F(147, 200)*K+K/4*(
        41*(3*A*A+EH*(H+H0))
        + F(33, 50)*(6*A*da2+6*da1*da1+2*EH*dg2
                      +4*dg1*de1+2*de1*de1+2*H*de2)
    )
    assert second_h < de2 and second_a < da2

    radius, rho, M = F(159, 500), F(173, 1000), F(7, 10)
    exposure, eta = F(1411, 10000), F(7, 4)
    assert radius*radius < F(9, 20)**2/2
    assert F(63, 250)/(F(53, 25)-F(1, 3)) < exposure
    assert eta*exposure*M < rho < radius
    ratio = rho/radius
    psi = (
        F(2, 5)*exp_upper(rho*rho)
        +(EH+eta*A)/2+rho*(de1+eta*da1)/2
        +rho*rho*(de2+eta*da2)/4
        +(F(3, 20)+eta*F(47, 200))/2*ratio**3/(1-ratio)
    )
    assert psi < M
    print("PASS: first residual jets:", float(first_h), float(first_a))
    print("PASS: second residual jets:", float(second_h), float(second_a))
    print("PASS: initial early analytic jet:", float(psi), "< .7")
    print("PASS: positive final radius:", float(rho-eta*exposure*M))
    print("Scope: early analytic continuation; not the global payoff screen.")


if __name__ == "__main__":
    verify()
