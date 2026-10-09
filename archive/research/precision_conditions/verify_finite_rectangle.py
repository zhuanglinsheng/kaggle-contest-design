"""Rational constants in the curvature-32 finite-precision rectangle.

The HJB bound, implementation-cost scaling, and compactness argument are
proved in PROOF_LEDGER.md section 65. This verifies their finite constants,
not equilibrium existence or output sorting by numerical simulation.
"""

from fractions import Fraction as F
from math import comb

low = sum(F(comb(31, j), 14**j) for j in range(6))
high = sum(F(comb(31, j), 25**j) for j in range(5))
assert low == F(642303, 76832) > F(25, 3)
assert high == F(261893, 78125) > F(10, 3)

# Low-cost branch, 1/2 <= r <= 1: cost <= sqrt(2r/7) R_0.
assert F(2, 7) < F(3, 5)**2
assert F(22, 25) - F(3, 5) == F(7, 25) > F(1, 200)
# r - sqrt(2r/7) - 3/25 is increasing because 14r >= 7 > 1.
assert F(1, 7) < F(189, 500)**2
assert F(1, 2) - F(189, 500) - F(3, 25) == F(1, 500)

# High-cost branch, 1 <= r <= 3: cost <= sqrt(12/25) R_0.
assert F(12, 25) < F(139, 200)**2
assert F(7, 10) - F(139, 200) == F(1, 200) > F(1, 500)

# Both candidate exposures are strictly inside the selected zero branch.
assert F(1, 14) < F(1, 4)
assert F(1, 25) < F(1, 4)
# Candidate implementation energies share one external deadline bracket.
assert F(1, 7) <= F(4, 25) <= F(12, 25) < 1
assert F(1, 7) <= F(2, 7) <= F(12, 25)

print("Curvature 32; cost rectangle [c_F/2, 3c_F].")
print("Uniform candidate payoff margin > R_0/200.")
print("Uniform quarter-exposure budget margin > R_0/500.")
print("All finite comparisons passed with exact rational arithmetic.")
