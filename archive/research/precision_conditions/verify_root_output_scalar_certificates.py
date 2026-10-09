"""Exact scalar algebra only; does not certify the complex Nash-output bounds."""
from fractions import Fraction as F
from math import comb


def multiply(a, b):
    c = [F(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            c[i + j] += x * y
    return c


def power(a, n):
    c = [F(1)]
    for _ in range(n):
        c = multiply(c, a)
    return c


def margin_polynomial(radius, second, tail):
    denominator = [radius, F(-1)]
    numerator = multiply(denominator, [F(1), F(1), second])
    numerator[3] += tail
    left = power(denominator, 4)
    right = multiply([F(1), F(-4)], power(numerator, 4))
    difference = [
        (left[i] if i < len(left) else 0) - value
        for i, value in enumerate(right)
    ]
    assert difference[0] == difference[1] == 0
    return difference[2:]


def bernstein(polynomial, lo, hi):
    n = len(polynomial) - 1
    shifted = [
        sum(polynomial[k] * comb(k, j) * lo ** (k - j) * (hi - lo) ** j
            for k in range(j, n + 1))
        for j in range(n + 1)
    ]
    return [
        sum(shifted[j] * F(comb(k, j), comb(n, j))
            for j in range(k + 1))
        for k in range(n + 1)
    ]


def main():
    # Whole-function bound |f|<=2 on radius 1/2:
    # U=1+w+w^2/2+16w^3/(1-2w).
    p = margin_polynomial(F(1, 2), F(1, 2), F(8))
    assert min(bernstein(p, F(0), F(1, 4))) > 0
    print("PASS: whole-function conditional scalar certificate.")

    # Centered bound |(f-1-k*w)/w^2|<=2 on radius 27/100:
    # U=1+w+w^2/8+2w^3/(27/100-w).
    p = margin_polynomial(F(27, 100), F(1, 8), F(2))
    minima = [min(bernstein(p, F(j, 16), F(j + 1, 16))) for j in range(4)]
    assert min(minima) == F(281291929387, 442918502400000000) > 0
    print("PASS: centered conditional scalar certificate on four intervals.")

    # sqrt(3)>12/7 and pi<22/7 suffice for both strict coefficient signs.
    assert F(12, 7)**2 < 3
    assert 24 * (F(12, 7) - 1) > 5 * F(22, 7)
    assert F(12, 7) - 1 > F(22, 7) / 6
    print("PASS: exact full-disclosure f2<1/8 and f3<0 signs.")
    print("The required complex analyticity and root-output bounds remain unproved.")


if __name__ == "__main__":
    main()
