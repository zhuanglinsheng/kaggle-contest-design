"""Rational bounds for ledger section 95, not global design classification."""

from fractions import Fraction as F


def verify():
    stable = lambda q: 1-F(11, 3)*q+F(4, 3)*q*q
    assert stable(F(1)) == -F(4, 3)
    assert stable(F(12, 5)) == -F(3, 25)
    # Section 98 strengthens the stable range to sigma^2*T0 >= 7*S0/10.
    assert 1+F(10, 7) == F(17, 7)
    assert stable(F(17, 7)) == -F(2, 49)
    assert 1-4*F(8, 3)+F(4, 3)*F(8, 3)**2 == -F(5, 27)
    q, eta, xi = F(5, 2), F(401, 200), F(1, 200)
    # pi < 22/7 and the concave chord bound on (q-y)/(2q-y).
    lower_h = q*(4*q-3)/(2*F(22, 7)*(2*q-1))
    assert lower_h == F(245, 352)
    margin = -eta+lower_h+(2*q-eta-1)**2/(eta+1+2*q*xi)
    assert margin == F(811, 177760) > 0
    # The explicit nonlinear delay example has elasticity < 1/200.
    epsilon = F(1, 1000)
    assert (2*epsilon*F(2, 3))/(1+2*epsilon*F(2, 3)) < F(1, 200)
    assert 2+F(1, 400) < F(401, 200)
    print("Verified stable-entry and precision-sensitive saddle-entry margins.")
    print("The statements concern exact local derivatives and profitable designs,")
    print("not the precision of a global joint optimum.")


if __name__ == "__main__":
    verify()
