"""Exact algebra for ledger 117, conditional on the unproved finite envelope.

This audit does not prove a finite-precision output bound or a design regime.
"""
from fractions import Fraction as F

from verify_full_entry_boundary import add, mul, scale


def power(p, n):
    result = {(0, 0, 0): F(1)}
    for _ in range(n):
        result = mul(result, p)
    return result


def verify(report=True):
    one, a, x, w = [{k: F(1)} for k in
                    [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]]
    m, A = add(a, x), add(a, one)
    B = add(a, scale(one, 3), scale(x, -2))
    K = add(scale(a, 3), one, scale(x, 2))
    N = add(m, mul(x, w), scale(mul(x, power(w, 2)), F(1, 2)))
    actual = add(mul(add(B, scale(w, -4)), power(N, 4)),
                 scale(mul(power(A, 3), power(m, 2)), -1))
    coefficients = [
        scale(mul(power(m, 2), mul(power(add(x, scale(one, -1)), 2), K)), -1),
        scale(mul(power(m, 3), mul(add(x, scale(one, -1)),
                                   add(scale(x, 2), scale(a, -1)))), -4),
        scale(mul(mul(power(m, 2), x),
                  add(power(a, 2), scale(a, -5), scale(mul(a, x), 2),
                      scale(x, 4), scale(power(x, 2), -8))), 2),
        scale(mul(mul(m, x), add(scale(power(x, 3), -10),
              scale(mul(add(a, one), power(x, 2)), -1),
              mul(mul(a, add(scale(a, 3), scale(one, -11))), x),
              scale(power(a, 2), -4))), 2),
        scale(mul(power(x, 2), add(scale(power(x, 3), -34),
              scale(mul(add(scale(a, 19), scale(one, 29)), power(x, 2)), -1),
              mul(mul(a, add(scale(a, 12), scale(one, -74))), x),
              scale(mul(power(a, 2), add(a, scale(one, -13))), 3))), F(1, 2)),
        mul(power(x, 2), add(mul(add(scale(B, 2), scale(one, -4)), power(x, 2)),
              mul(mul(add(scale(B, 3), scale(one, -24)), m), x),
              scale(power(m, 2), -6))),
        mul(power(x, 3), add(mul(add(scale(B, F(3, 2)), scale(one, -8)), x),
              mul(add(scale(B, F(1, 2)), scale(one, -12)), m))),
        mul(power(x, 3), add(mul(add(scale(B, F(1, 2)), scale(one, -6)), x),
              scale(m, -2))),
        mul(power(x, 4), add(scale(B, F(1, 16)), scale(one, -2))),
        scale(power(x, 4), F(-1, 4)),
    ]
    assert actual == add(*(mul(c, power(w, j)) for j, c in enumerate(coefficients)))
    E = add(mul(coefficients[2], K),
            scale(mul(power(m, 4), power(add(scale(x, 2), scale(a, -1)), 2)), 4))
    f = add(scale(power(a, 4), 2),
            scale(mul(add(power(a, 3), scale(power(a, 2), 14), scale(a, 5)), x), -1),
            mul(add(scale(power(a, 2), 2), scale(a, 4), scale(one, 4)), power(x, 2)),
            scale(mul(a, power(x, 3)), -12), scale(power(x, 4), -8))
    assert E == scale(mul(power(m, 2), f), 2)
    at_half = {}
    for (i, j, _), v in f.items():
        key = (i+j, 0, 0)
        at_half[key] = at_half.get(key, F(0))+v/F(2)**j
    at_half = {k: v for k, v in at_half.items() if v}
    assert at_half == scale(mul(power(a, 2), add(scale(a, 4), one)), F(-3, 2))
    derivative_margin = add(
        scale(mul(power(a, 2), add(power(a, 2), scale(a, 14), scale(one, 5))), 9),
        scale(power(add(power(a, 2), scale(a, 2), scale(one, 2)), 2), -1))
    assert derivative_margin == add(scale(power(a, 4), 8), scale(power(a, 3), 122),
                                    scale(power(a, 2), 37), scale(a, -8), scale(one, -4))
    assert 37*F(1, 2)**2-8*F(1, 2)-4 == F(5, 4) > 0
    if report:
        print("PASS: exact payoff expansion, completed-square remainder, derivative margin.")
        print("The original finite-precision envelope remains unproved; no new regime certified.")


if __name__ == "__main__":
    verify()
