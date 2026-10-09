"""Rational certificate for the Gaussian analytic jet at radius 4/5.

This checks the scalar inequalities in GAUSSIAN_JET_CAP.txt, not an HJB
solution or the full joint-design theorem.
"""
from fractions import Fraction as F
from math import factorial


def exp_upper(x, n=16):
    return sum(x**j / factorial(j) for j in range(n)) + (
        x**n / factorial(n) / (1-x/F(n+1))
    )


def verify():
    r, m, eta = F(4, 5), F(17, 25), F(7, 4)
    assert exp_upper(F(8, 25)) < F(689, 500)
    # Taylor upper bound for exp(4v/5), 0<=v<=1.
    tail = r**5 / factorial(5) / (1-r/6)
    assert tail == F(128, 40625)
    a = F(689, 500)
    c0, c1 = F(17, 10)-a, -a*r
    c2 = F(17, 20)+r*r-a*r*r/2
    c3, c4 = -a*r**3/6, F(17, 80)-a*r**4/24
    c5 = -a*tail
    # v^3 <= (v^2+v^4)/2 and v^5 <= v^4.
    assert c2+c3/2 > F(4951, 5000)
    assert c4+c3/2+c5 > 0
    assert c1*c1 < 4*c0*F(4951, 5000)
    # The remaining two intervals: [1,1.3] and [1.3,infinity).
    assert exp_upper(F(31, 50)) < F(93, 50)
    assert exp_upper(F(169, 200)) < F(50, 21)
    assert F(2, 5)*(F(93, 50)-r*r*F(21, 50)) < m
    assert exp_upper(F(103, 200)) < F(42, 25)
    assert F(2, 5)*F(42, 25) < m
    assert eta*F(67, 100)*m < r
    print("PASS: Gaussian jet <17/25 at radius 4/5, uniformly in y.")
    print("PASS: characteristic travel at exposure .67 is below 4/5.")


if __name__ == "__main__":
    verify()
