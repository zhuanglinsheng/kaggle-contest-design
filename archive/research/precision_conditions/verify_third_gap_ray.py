"""Exact Bernstein certificate for d_rho(L_q + 2 K_q) >= 0.

The three cube coordinates are tau, k, rho; q = 5/4 + S, S >= 0,
a = q - tau, and b = q - tau*k. All arithmetic is rational.
This certifies a third-order coefficient bound, not exact output sorting.
"""

from fractions import Fraction as F
from itertools import product
from math import comb


class Polynomial(dict):
    def __add__(self, other):
        if not isinstance(other, Polynomial):
            other = Polynomial({(0, 0, 0, 0): F(other)})
        result = Polynomial(self)
        for key, value in other.items():
            result[key] = result.get(key, 0) + value
        return Polynomial({k: v for k, v in result.items() if v})

    __radd__ = __add__

    def __neg__(self):
        return Polynomial({k: -v for k, v in self.items()})

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        if not isinstance(other, Polynomial):
            return Polynomial({k: v * F(other) for k, v in self.items() if v * other})
        result = Polynomial()
        for key, value in self.items():
            for key2, value2 in other.items():
                key3 = tuple(x + y for x, y in zip(key, key2))
                result[key3] = result.get(key3, 0) + value * value2
        return Polynomial({k: v for k, v in result.items() if v})

    __rmul__ = __mul__

    def __pow__(self, n):
        result = Polynomial({(0, 0, 0, 0): F(1)})
        for _ in range(n):
            result = result * self
        return result


S, tau, k, rho = [
    Polynomial({tuple(int(i == j) for i in range(4)): F(1)})
    for j in range(4)
]
q = S + F(5, 4)
a, b = q - tau, q - tau * k
A, B = rho * a, rho * b
D, N = q * q + q * B - 2 * A * A, (1 - tau) * q - (1 + tau) * A
PK = (
    4 * a * (q - B) * (q + B) * D * D
    + 2 * rho * a * b * (q + B) * D * D
    - 4 * rho * a * b * (q - B) * D * D
    - 2 * rho * a * (q - B) * (q + B) * D * (q * b - 4 * rho * a * a)
)
PL = (
    -2 * (1 + tau) * a * (q - B) * (q + B)**2 * D
    + N * b * (q + B)**2 * D
    - 3 * N * (q - B) * (q + B)**2 * (q * b - 4 * rho * a * a)
)
polynomial = 2 * PK + PL
degrees = tuple(max(key[i] for key in polynomial) for i in range(4))
assert degrees == (7, 7, 4, 6)
total_checked = 0
for power in range(8):
    sub = {key[1:]: value for key, value in polynomial.items() if key[0] == power}
    coefficients = []
    for index in product(range(8), range(5), range(7)):
        beta = sum(
            value * F(comb(index[0], key[0]), comb(7, key[0]))
            * F(comb(index[1], key[1]), comb(4, key[1]))
            * F(comb(index[2], key[2]), comb(6, key[2]))
            for key, value in sub.items()
            if all(j <= i for j, i in zip(key, index))
        )
        assert beta >= 0, (power, index, beta)
        coefficients.append(beta)
    positives = [beta for beta in coefficients if beta > 0]
    total_checked += len(coefficients)
    print("S power", power, ": positive", len(positives),
          "; zero", len(coefficients) - len(positives),
          "; smallest positive", min(positives))
assert total_checked == 2240
print("All 2240 Bernstein coefficients are nonnegative exactly.")
