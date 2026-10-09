"""Exact polynomial identities for note Section 18.

The continuous sign arguments are in the note. The finite-precision output
envelope with coefficient 2/3 is an unproved input, not checked here.
"""

from fractions import Fraction as F

from results.theory.precision_conditions.verify_moderate_full_exit import product
from results.theory.precision_conditions.verify_original_propagation_bound import (
    add, scale,
)


def main():
    one = {(0, 0): F(1)}
    a = {(1, 0): F(1)}
    z = {(0, 1): F(1)}
    q = add(one, a)
    x = add(product(q, z, z), scale(a, -1))
    m = product(q, z, z)
    k = add(scale(q, 4), one, scale(product(x, x), -1))
    c0 = add(product(k, m), scale(product(q, q, z), -4))
    c1 = add(product(x, k), scale(m, -4))
    c2 = add(scale(product(x, k), F(2, 3)), scale(x, -4))
    curv = add(one, scale(a, -4), product(x, x))
    t = add(product(x, x), x, scale(a, -4))
    gap = add(product(q, z, z, z), scale(product(q, z, z), 2),
              product(add(q, scale(one, 2)), z), scale(one, 4))
    zminus = add(z, scale(one, -1))
    assert c0 == scale(product(q, q, z, zminus, zminus, gap), -1)
    assert c1 == product(add(x, scale(one, -1)), scale(t, -1))
    assert c2 == scale(product(x, curv), F(-2, 3))
    discriminant = add(scale(product(c0, c2), 4),
                       scale(product(c1, c1), -1))
    bracket = add(scale(product(z, x, curv, gap), F(8, 3)),
                  scale(product(add(z, one), add(z, one), t, t), -1))
    assert discriminant == product(q, q, zminus, zminus, bracket)
    print("Exact coefficient and discriminant identities passed.")
    print("This is not a finite-precision equilibrium-envelope certificate.")


if __name__ == "__main__":
    main()
