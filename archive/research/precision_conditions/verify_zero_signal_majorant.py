"""Exact constants for note Section 22; no intermediate-precision certificate."""

from fractions import Fraction as F

from results.theory.precision_conditions.verify_original_propagation_bound import (
    add, mul, scale,
)


def exp_upper(t):
    return 1 + t + t*t / (2*(1-t/3))


def main():
    t = F(529, 2500)
    assert exp_upper(t) < F(5, 4)
    assert F(9, 10)**2 < F(8281, 10000)
    assert F(8281, 10000) > F(3, 4)
    assert F(29, 100) < 2*(1-F(29, 100)**2)
    # b^2=1001/(2*pi), pi<22/7 => q/(2b)<1/25.
    q = F(1001, 1000)
    b_square_lower = F(1001*7, 44)
    assert q*q / (4*b_square_lower) < F(1, 25)**2
    assert F(51, 1000) < F(23, 100)**2
    assert 4*F(22, 7)*F(29, 100)/20 < F(43, 100)**2
    assert F(66, 100) < F(3, 4)
    assert -F(99, 100) + F(3, 2)*F(291, 1000)*F(9, 4) == -F(63, 8000)
    v = F(999, 1000)
    # exp(500/999)<33/20: sharper finite Taylor upper remainder.
    z = F(500, 999)
    terms = [F(1)]
    for n in range(1, 13):
        terms.append(terms[-1]*z/n)
    remainder = terms[-1]*z/13/(1-z/14)
    assert sum(terms)+remainder < F(33, 20)
    cdf_upper = F(1, 2)+F(2, 5)*(F(103, 120)+F(1, 1000))
    assert cdf_upper < F(169, 200)
    m_numerator = 4*F(33, 20)*F(169, 200)
    assert m_numerator**2/(2*F(157, 50)*v) < F(9, 4)**2
    one = {(0, 0): F(1)}
    Q = {(1, 0): F(1)}
    eta = {(0, 1): F(1)}
    denominator = add(eta, one, scale(Q, 2))
    cross = add(scale(Q, 2), scale(eta, -1), scale(one, -1))
    numerator = add(scale(mul(eta, denominator), -1), mul(cross, cross))
    expected = add(mul(add(scale(Q, 2), scale(one, -1)),
                       add(scale(Q, 2), scale(one, -1))),
                   scale(mul(add(scale(Q, 6), scale(one, -1)), eta), -1))
    assert numerator == expected
    assert (2*q-1)**2 - (6*q-1)*24 < 0
    print("PASS: exact majorant, zero-output, Fourier-margin and Hessian checks.")
    print("Scope: zero precision and a small positive neighborhood only.")
    print("No intermediate-precision or unrestricted joint optimum certified.")


if __name__ == "__main__":
    main()
