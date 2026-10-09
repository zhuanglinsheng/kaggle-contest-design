"""Rational constants in ledger 103, not a global information-regime certificate."""

from fractions import Fraction as F


def verify():
    assert F(1, 4)*F(4, 3)**2 == F(4, 9)
    assert F(441, 1600) < F(1, 3)
    source = F(2, 5)*(F(441, 800)+F(1, 6))+F(1, 3)+F(4, 9)
    assert source == F(19169, 18000) < F(11, 10)
    assert F(2, 23) < F(1, 3) and F(12, 47) < F(1, 3)
    # Integral exp(-t)*(1/3+4*t/9) over t>=0 is 7/9.
    assert F(11, 10)*(F(1, 3)+F(4, 9)) == F(77, 90)
    for growth, P in [(F(1), F(0)), (F(2), F(3)), (F(10), F(20))]:
        V = growth*(P+1)-1
        assert 1+V/2 == (1+growth*(P+1))/2
    print("Verified weighted source, potential, Gaussian, and integral constants.")
    print("The exact propagation bound additionally uses the ledger's kernel comparison.")
    print("No sign of the complete output difference or joint regime is certified.")


if __name__ == "__main__":
    verify()
