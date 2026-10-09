"""Exact containment constants for ledger section 96.

The global optimum and precision conclusions use the established finite
rectangle theorem; this script checks only its new interval containment.
"""

from fractions import Fraction as F


def verify():
    assert 3*F(2, 3)**2 == F(4, 3)
    assert F(4, 3)/2 == F(2, 3)
    assert F(2, 3)*F(1, 2) == F(1, 3)
    assert F(2, 3)**2*F(1, 2)*F(1, 2) == F(1, 9)
    assert 3*F(1, 9) == F(1, 3)
    assert F(1, 3)/F(1, 4) == F(4, 3)
    # Nonempty example: S0 = 2*pi, initial utility slope 31, d = 1.
    # Multiplying costs and risk by the stated powers of pi removes pi.
    assert F(1, 4) < F(7, 24) < F(1, 3)
    assert F(1, 100) < F(4, 11*31)
    assert 32 < F(3*32-1, 2)
    print("Verified primitive cost-band containment and nonempty example.")
    print("No new information-regime ranking is inferred from this script.")


if __name__ == "__main__":
    verify()
