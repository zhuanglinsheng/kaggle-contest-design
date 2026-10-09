"""Exact checks for Section 27's joint-budget and maturity overlap.

These are equilibrium-construction bounds, not a global payoff certificate.
"""
from fractions import Fraction as F


def verify():
    h, ell = F(63, 250), F(32, 5)
    # b=251, L=2*pi*b^2-1>1000, q=1+1/L<1001/1000.
    assert F(1, 4)+F(1001, 1000)/(2*251) < h
    linear = ell*h/3-F(1, 4)
    maximum = h+3*linear*linear/ell
    assert maximum == F(11630883, 40000000)
    assert maximum < F(291, 1000)

    # (1+z^2/3)sinh(z)-z*cosh(z): scaled coefficient of z^(2n+1).
    for n in range(1, 100):
        coefficient = 1+F(2*n*(2*n+1), 3)-(2*n+1)
        assert coefficient == F(4*n*(n-1), 3)
        assert coefficient >= 0

    # Gaussian analytic norm bound and decreasing-strip margin.
    assert F(1, 2)+F(2, 5)*F(103, 120) < F(17, 20)
    z = F(1, 2)
    terms = [F(1)]
    for k in range(1, 10):
        terms.append(terms[-1]*z/k)
    remainder = terms[-1]*z/10/(1-z/11)
    assert sum(terms)+remainder < F(33, 20)
    assert 4*F(33, 20)*F(17, 20)*F(2, 5) < F(9, 4)
    assert -F(99, 100)+F(3, 2)*F(291, 1000)*F(9, 4) == -F(63, 8000)

    # atanh(sqrt(r))^2 < 8/5 from a rational tail bound.
    r, n = F(18, 25), 8
    series_upper = sum((r**k/F(2*k+1) for k in range(n+1)), F(0))
    series_upper += r**(n+1)/(F(2*n+3)*(1-r))
    assert r*series_upper*series_upper < F(8, 5)

    print("PASS: exposure < .291 on ell <= 6.4 and maturity by T0/2 above that cutoff.")
    print("Uniform analytic construction uses reflection symmetry, not positivity of a vector semigroup.")
    print("Early-layer propagation and the global payoff comparison remain uncertified.")


if __name__ == "__main__":
    verify()
