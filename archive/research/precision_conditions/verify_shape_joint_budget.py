"""Exact constants for ledger 120; no new HJB or precision-ordering proof."""
from fractions import Fraction as F


def verify():
    # rho <= 1/4, eta-1 >= 8/rho, p >= 2/rho, r >= 9/10.
    assert (F(3, 8) + F(1, 4)) / F(9, 10) == F(25, 36) < 1
    assert 1 + 6 * F(1, 4) / 8 == F(19, 16)
    assert F(1, 2) / F(19, 16) == F(8, 19) > F(1, 4)
    # Ordinary quadratic delay and relative utility curvature >= 21.
    assert (F(3, 20) + F(1, 4)) / F(9, 10) == F(4, 9) < F(9, 20)
    assert 1 + F(6, 20) == F(13, 10)
    assert F(1, 2) / F(13, 10) == F(5, 13) > F(1, 4)
    # Optional smaller domain from relative curvature >= 64.
    assert (F(3, 63) + F(1, 4)) / F(9, 10) == F(125, 378) < F(1, 3)
    assert F(1, 2) / (1 + F(6, 63)) == F(21, 46)
    # Tail budget divided by c*sigma^2 = 2*r*d*T0.
    for a in [F(1, 2), F(1), F(2)]:
        for eta in [F(21), F(64), F(1000)]:
            for p in [F(2), F(8), F(100)]:
                for r in [F(9, 10), F(1), F(3)]:
                    upper = (2 * (a + 1) / (eta - 1) + 1 / p) / (2 * r)
                    assert upper == ((a + 1) / (eta - 1) + 1 / (2 * p)) / r
    print("PASS: joint budget normalization and all exact constants in ledger 120.")
    print("Moderate-delay all-precision ranking remains unproved; strong-shape proof is in ledger 120-121.")


if __name__ == "__main__":
    verify()
