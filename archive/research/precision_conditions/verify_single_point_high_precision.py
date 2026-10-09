"""Exact algebra for note Section 21; not a low-precision equilibrium certificate."""

from fractions import Fraction as F

from results.theory.precision_conditions.verify_original_propagation_bound import (
    add, mul, scale,
)
from results.theory.precision_conditions.verify_moderate_full_exit import product


def main():
    one = {(0, 0): F(1)}
    w = {(1, 0): F(1)}
    x = {(0, 1): F(1)}
    a = F(1, 1000)
    q = 1 + a
    C = add(one, w, scale(mul(w, w), 2))
    L = add(one, scale(w, 4))
    assert add(scale(C, 25), scale(L, -16)) == {
        (0, 0): F(9), (1, 0): F(-39), (2, 0): F(50),
    }
    assert 39**2 - 4 * 50 * 9 == -279
    assert F(4, 5) / (F(4, 5) + F(1, 20)) == F(16, 17)
    assert q < F(21, 20)
    assert F(1, 16) + F(1, 4) / q - F(1, 2) / q < 0
    assert F(1, 16) - F(3, 4) / q < 0
    gap = add(product(C, C, C, C), scale(L, -1))
    assert all(coefficient > 0 for coefficient in gap.values())
    assert all(power[0] >= 2 for power in gap)

    # N=(a+x)+x(w+2w^2); r=N/(sqrt(q)*sqrt(a+x)).
    m = add(scale(one, a), x)
    B = add(w, scale(mul(w, w), 2))
    N = add(m, mul(x, B))
    numerator_rx = add(scale(product(m, add(one, B)), 2), scale(N, -1))
    expected_rx = add(m, mul(add(scale(one, 2*a), x), B))
    assert numerator_rx == expected_rx
    # Eliminating r^(-17) between F_w=0 and F_x=0:
    deadline = add(expected_rx, scale(product(m, x, x, L), -1))
    expected_deadline = add(m, mul(add(scale(one, 2*a), x), B),
                            scale(product(m, x, x, L), -1))
    assert deadline == expected_deadline
    print("PASS: exact derivative, barrier, and polynomial checks for Section 21.")
    print("Scope: fixed exponential utility/quadratic delay, high precision only.")
    print("No unrestricted joint-optimum point or low-precision envelope certified.")


if __name__ == "__main__":
    main()
