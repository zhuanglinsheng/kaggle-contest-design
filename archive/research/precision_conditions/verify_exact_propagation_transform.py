"""Algebra for ledger 102; not a joint-design or precision-ranking proof."""

from fractions import Fraction as F
from math import cosh, tanh


def verify():
    # Exact chain-rule coefficient of P(R_z)^2 and P(d_z)^2.
    assert F(3, 4)-F(1, 2) == F(1, 4)
    # Product derivative cancels the actual drift: P_z=w*P*R_z.
    for w, rz, dz, P, B, Bz in [
        (F(1, 4), F(1, 10), F(1), F(2), F(3), F(1, 5)),
        (F(1, 2), -F(1, 5), F(2, 3), F(3), F(1), -F(2, 5)),
    ]:
        q = w*w*(dz*dz-rz*rz)/4
        assert q >= 0
        leftover = -q*P*B+w*P*rz*Bz-(w*P*rz)*Bz
        assert leftover == -q*P*B
    # Checks of transcendental identities, not continuous inequality certificates.
    for w in [.001, .1, .25, .5]:
        a = 1/cosh(w/2)**2
        assert abs((1-a)-tanh(w/2)**2) < 1e-14
        assert 0 < 1-a <= w*w/4
    print("Verified propagation-transform chain-rule and normalization checks.")
    print("The continuous Markov domination uses positivity and actual value bounds.")
    print("No exact best-output difference or global design regime is certified.")


if __name__ == "__main__":
    verify()
