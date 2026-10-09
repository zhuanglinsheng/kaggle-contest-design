"""Rational constants in ledger 107, not an all-state or global design proof."""

from fractions import Fraction as F


def verify():
    m, k = F(63, 160), F(21, 20)
    assert m/k == F(3, 8)
    bracket = F(3, 8)-F(6, 23)
    assert bracket == F(21, 184) > F(1, 9)
    coefficient = F(15, 16)*F(47, 96)*F(1, 9)
    assert coefficient == F(235, 4608) > F(1, 20)
    assert F(1, 20)*F(2) == F(1, 10)
    print("Verified complete-background center pairing and Gaussian coefficient.")
    print("The center sign also uses the actual value-difference boundary comparison.")
    print("No middle-state source sign or global joint-design regime is certified.")


if __name__ == "__main__":
    verify()
