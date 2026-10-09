"""Algebra audit for ledger 101; not a global information-regime proof."""

from fractions import Fraction as F


def verify():
    for k in [F(0), F(1, 5), F(1, 2), F(4, 5)]:
        # G*(g-g(k*z)) at zero: third derivative by product rule.
        # Divide by G(0)^2; g'(0)=2G(0), g'''(0)=-2G(0).
        product = -2*(1-k**3)-6*(1-k)
        assert product == -2*(4-3*k-k**3)
        assert product < 0
    assert 33 > 25  # 1/sqrt(33) < 1/5
    assert 4-3*F(1, 5)-F(1, 5)**3 == F(424, 125)
    # The finite Gaussian initial curvature formula is a diagnostic identity.
    for P, bp, bppp in [(F(10), F(1, 2), -F(1, 100)),
                        (F(100), F(3, 5), -F(1, 500))]:
        initial_time_derivative = F(3, 2)/P**2+3*bp/P-bppp
        assert initial_time_derivative > 0
    print("Verified central product derivative and finite Gaussian curvature algebra.")
    print("The density counterexample additionally uses continuous semigroup dependence.")
    print("No global optimum or exact precision sorting is certified.")


if __name__ == "__main__":
    verify()
