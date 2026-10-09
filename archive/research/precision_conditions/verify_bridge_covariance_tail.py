"""Gaussian algebra for ledger section 97; not an output-ordering proof."""

from fractions import Fraction as F


def verify():
    for delta in (F(1, 100), F(1, 10), F(1), F(10)):
        assert 1/(1+delta)-1/(1+2*delta) == delta/((1+delta)*(1+2*delta)) > 0
        assert 1/(2*delta) > 1/(1+2*delta)
        assert F(1, 2)/(delta+F(1, 2)) == 1/(1+2*delta)
    # The background equation for h=G*v uses L0(G)=-G/2.
    for v, vz, vzz, z in ((F(2), F(3), F(4), F(5)),):
        transformed_l0 = vzz/2-z*vz/2-v/2
        assert v/2-transformed_l0 == v-vzz/2+z*vz/2
    print("Verified Gaussian tilt, covariance-tail exponent gap, and background conjugation.")
    print("Continuous tail signs require the ODE asymptotics in the ledger.")
    print("No exact best-output or joint-design ordering is certified.")


if __name__ == "__main__":
    verify()
