"""Exact rational constants for the averaged Gaussian bridge comparison."""

from fractions import Fraction as F


def verify():
    assert 40 * F(22, 7) * F(9, 5) < 240
    assert F(1, 10**13) < F(1, 3 * 10**6) ** 2
    assert F(3000, 3 * 10**6) < F(1, 480)
    moderate_gap = F(1, 72000 * 10**13)
    terminal_gap = F(1, 480 * 10**90)
    assert moderate_gap > F(1, 10**93)
    assert terminal_gap > F(1, 10**93)
    w = F(1, 10**51)
    lower = F(1, 10**93)
    assert lower * (1 - 2 * w) - 5 * 10**7 * w**2 > lower / 2
    assert w < F(1, 10**6)
    assert F(5, 10**52 - 1) < w
    print("Verified constants for the original-path second-order tail gap.")
    print("Existing exact remainder bounds then give sorting for w <= 10^-51.")


if __name__ == "__main__":
    verify()
