"""Exact constant checks for ledgers 131 and 134, not equilibrium experiments."""
from fractions import Fraction as F
from math import comb


def main():
    # Minimum curvature exponent a = gamma_0 - 1 is 32.
    lower_binomial = sum(F(comb(32, j), 128**j) for j in range(4))
    assert lower_binomial == F(84059, 65536)
    assert lower_binomial > F(50, 39)
    assert 1-F(39, 50) == F(11, 50)
    # Raising positive quantities to 32 gives an exact rational comparison.
    assert F(1, 5)**31 < F(43, 200)**32
    assert F(11, 50)-F(43, 200) == F(1, 200)
    # The fixed candidate deadline is within the old risk-certificate bounds.
    assert F(1, 7) < F(1, 5) < 1
    assert F(1, 20) < F(1, 4)
    assert F(32, 5) > 6 > 3
    # Ledger 134 extends the SAME candidate down to c_F/2.
    gain_at_cf = sum(F(comb(32, j), 20**j) for j in range(4))
    assert gain_at_cf == F(223, 50) > 4
    assert F(3, 4)-F(43, 200) == F(107, 200) > F(1, 2)
    assert F(1, 10) < F(1, 4)
    # For r <= 1 use concavity C(rE) >= r C(E), NOT a square-root lower bound.
    assert F(1, 2) > 1-F(107, 200)
    assert F(9, 8)/11 == F(9, 88) < 1
    # Ledger 135: overlap connects the precision exclusion to global prize entry.
    assert F(4, 11*32) == F(1, 88) < F(1, 3)
    assert F(1, 5) > F(1, 9)  # T5/tu >= 1/sqrt(5) > 1/3.
    assert F(43, 200) < F(107, 200)
    print("PASS: candidate gain, implementation cost, unchanged 1/200 margin,")
    print("      old deadline bracket, exposure, and expanded cost endpoint.")
    print("PASS: lower-cost candidate margin, concave-cost budget, and full-exit constants.")
print("Continuous-domain arguments are proved in ledgers 131, 133, 134, and 135.")


if __name__ == "__main__":
    main()
