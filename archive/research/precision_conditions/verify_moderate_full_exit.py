"""Exact algebra for note Section 14; NOT a joint-disclosure certificate."""
from fractions import Fraction as F

from results.theory.precision_conditions.verify_original_propagation_bound import (
    add, mul, scale,
)


def product(*polys):
    result = {(0, 0): F(1)}
    for poly in polys:
        result = mul(result, poly)
    return result


def verify():
    one = {(0, 0): F(1)}
    a = {(1, 0): F(1)}
    z = {(0, 1): F(1)}
    q = add(a, one)
    x = add(product(q, z, z), scale(a, -1))
    m = product(q, z, z)
    k = add(scale(q, 4), one, scale(product(x, x), -1))
    c0 = add(product(k, m), scale(product(q, q, z), -4))
    c1 = add(product(x, k), scale(m, -4))
    c2 = add(scale(product(x, k), F(1, 2)), scale(x, -4))
    gap = add(product(q, z, z, z), scale(product(q, z, z), 2),
              product(add(q, scale(one, 2)), z), scale(one, 4))
    zminus = add(z, scale(one, -1))
    assert c0 == scale(product(q, q, z, zminus, zminus, gap), -1)
    assert c1 == product(add(x, scale(one, -1)),
                         add(scale(a, 4), scale(product(x, x), -1), scale(x, -1)))
    curv = add(scale(one, 3), scale(a, -4), product(x, x))
    assert c2 == scale(product(x, curv), F(-1, 2))
    discriminant = add(scale(product(c0, c2), 4), scale(product(c1, c1), -1))
    cross = add(product(x, x), x, scale(a, -4))
    bracket = add(scale(product(z, x, curv, gap), 2),
                  scale(product(add(z, one), add(z, one), cross, cross), -1))
    assert discriminant == product(q, q, zminus, zminus, bracket)
    print("PASS: exact zero-prize gap factorization, incentive coefficients, and discriminant.")
    print("The interval signs follow from the inequalities in note Section 14.")
    print("This verifies a full-disclosure benchmark proof, not finite-precision dominance.")


if __name__ == "__main__":
    verify()
