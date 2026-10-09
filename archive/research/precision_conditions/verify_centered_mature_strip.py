"""Exact constants for Section 38's Gaussian-centered mature fixed point.

No finite difference solution or numerical integral is a proof input.
The analytic leading-term inequality is derived in the research note.
"""
from fractions import Fraction as F
from verify_mature_spatial_strip import exp_upper


def verify():
    # Rational chord above (2-v)^(-1/2), 1/2 <= v <= 1.
    a, b = F(633, 1000), F(367, 1000)
    assert a+b == 1 and (a+b/2)**2 > F(2, 3)
    # J1 = sum 2^(-j)/(2j+1), with a geometric tail.
    n = 24
    j1 = sum(F(1, 2)**j/F(2*j+1) for j in range(n))
    j1 += 2*F(1, 2)**n/F(2*n+1)
    assert j1 < F(5, 4)
    assert (3*a/2-b)*F(5, 4)+2*b-a == F(6633, 8000)
    assert a-b+(b-a/2)*F(5, 4) == F(2633, 8000)
    assert (F(157, 50)**2)*F(27, 10)*F(97, 500)**2 > 1

    R, d, K = F(9, 20), F(1, 10000), F(7, 20)
    assert exp_upper((d*d+d)/2) < F(10001, 10000)**2*(1-d)
    initial = F(3991, 5000)*exp_upper((R*R+d*d)/2)
    constant = exp_upper(R*R/2)*(
        F(221, 100)+8*d*exp_upper(2*d*d)+2*(R+d)*exp_upper(d*d)
    )
    leading = K*F(97, 500)*F(10001, 10000)*(
        F(6633, 8000)*exp_upper(3*R*R/4)
        + F(2633, 8000)*exp_upper(7*R*R/4)
    )
    remainder, difference = F(3, 20), F(47, 200)
    total = initial+remainder
    h_rhs = K*constant*difference*total/2
    a_rhs = leading+K*constant*(
        3*difference*difference+remainder*(2*initial+remainder)
    )/4
    assert h_rhs < remainder and a_rhs < difference
    weight = F(5, 4)
    contraction = max(
        K*constant*(difference+total*weight)/2,
        K*constant*(total/weight+3*difference)/2,
    )
    assert contraction < 1
    beta = F(767, 1000)
    assert 4*(beta-difference)**2 > total*total+difference*difference
    # Initial-cap energy comparison only; not persistence of these caps.
    assert R*R/2 > (beta*F(71, 500)+F(209, 1000))**2
    print("PASS: actual mature strip R=.45 via Gaussian-centered contraction.")
    print("Leading difference cap:", float(leading))
    print("Map RHS:", float(h_rhs), float(a_rhs),
          "contraction:", float(contraction))
    print("PASS: initial-cap energy radius margin exceeds .209*sqrt(Q_delta).")
    print("Scope: no early pointwise-cap persistence or joint-optimum proof.")


if __name__ == "__main__":
    verify()
