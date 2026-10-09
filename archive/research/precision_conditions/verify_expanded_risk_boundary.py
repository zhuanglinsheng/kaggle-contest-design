"""Exact algebra for ledger 126: full disclosure, a in [1/3, 3].

The payoff envelope is proved for full disclosure, not finite precision.
The paper separately sorts every relevant finite-precision design below
full disclosure. Continuous sign arguments are in the paper and ledger.
"""
from fractions import Fraction as F

from verify_relaxed_entry_polynomial import verify


def main():
    # Reuse exact degree-nine expansion and completed-square identities.
    verify(report=False)
    h = lambda a: 8*a**4 + 122*a**3 + 37*a**2 - 8*a - 4
    g = lambda a: 2*a**4 - a**3 - 12*a**2 - 13*a - 4
    q = lambda a: (9*a**2 - 36*a + 4)/8
    assert h(F(1, 3)) == F(167, 81) > 0
    assert g(F(2)) == -54 and g(F(3)) == -16
    assert q(F(1, 3)) == F(-7, 8) and q(F(3)) == F(-23, 8)
    assert (F(1, 2)+F(1, 16))/F(9, 10) == F(5, 8)
    # Ledger 127: same shape threshold permits the larger cost interval.
    assert (F(1, 2)+F(1, 16))/F(9, 16) == 1
    assert 2*F(9, 16) == F(9, 8)
    assert 1+F(1, 4) == F(5, 4)
    assert F(1, 3)/F(5, 4) == F(4, 15) > F(1, 4)
    print("PASS: exact boundary polynomial, square completion, sign endpoints, and joint budgets.")
    print("Continuous inequalities and precision sorting are proved in the appendix; no grid inference.")


if __name__ == '__main__':
    main()
