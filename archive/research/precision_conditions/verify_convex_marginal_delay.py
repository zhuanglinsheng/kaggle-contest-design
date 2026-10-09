"""Exact algebra for ledger 122; not a proof of precision ordering."""
from fractions import Fraction as F


def product(left, right):
    result = [F(0)] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            result[i + j] += a * b
    return result


def verify():
    # Increasing powers of x in the exact square decompositions.
    first = product([F(9, 25), F(-6, 5), F(1)], [F(24), F(20)])
    first[0] += F(9, 25)
    first[1] -= F(2, 5)
    assert first == [F(9), F(-22), F(0), F(20)]
    second = product([F(1, 4), F(-1), F(1)], [F(100), F(100)])
    second[0] += 2
    second[1] -= 1
    assert second == [F(27), F(-76), F(0), F(100)]
    x = F(4, 5)
    assert sum(coef * x**i for i, coef in enumerate(first)) == F(41, 25)
    assert 60 * x**2 - 22 > 0
    assert (9 - 10 * x) / 25 == F(1, 25)
    # Uniform gap from the quarter-exposure boundary on 0 <= x <= 1.
    assert F(1, 25) / 108 == F(1, 2700)
    assert F(1, 4) - F(1, 2700) == F(337, 1350)
    assert F(1) / (F(9, 5) * 3) == F(5, 27) < F(1, 4)
    assert F(3, 10) / (F(9, 5) * 3) == F(1, 18) < F(3, 20)
    assert (1 + 6 / F(20)) / (8 * F(9, 10)) == F(13, 72)
    assert F(3, 20) < F(13, 72) < F(1, 4)
    # The general-convex-delay triangle remains valid for all r >= 9/10.
    # Its discriminant decreases throughout 1/2 <= a <= 2.
    a = F(1, 2)
    assert 25 * a**2 - 560 * a + 256 == -F(71, 4)
    assert 50 * F(2) - 560 < 0
    print("PASS: exact exposure certificates for curvature 7 and 21.")
    print("All-precision ranking for ordinary quadratic delay remains unproved.")


if __name__ == "__main__":
    verify()
