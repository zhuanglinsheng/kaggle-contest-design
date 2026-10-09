"""Rational constants in the uniform second-order disclosure gap proof."""

from fractions import Fraction as F


def verify():
    q_min, m_max = F(5, 4), F(1, 4)
    u_max, v_max = F(3, 4) + m_max / 2, F(1, 2) + m_max / 2
    assert u_max == F(7, 8) and v_max == F(5, 8)
    assert 1 - u_max / q_min == F(3, 10)
    assert v_max < F(4, 5) ** 2
    assert 4 * u_max < F(15, 8) ** 2
    assert 2 * F(22, 7) ** 3 < 64  # (pi*sqrt(2*pi))**2 < 64
    kernel_lower = F(3, 10) / (8 * F(4, 5) * 2 * F(15, 8))
    assert kernel_lower == F(1, 80)
    assert 2 * 4 * u_max == 7
    integrated_lower = kernel_lower / 7 / 2 / 64
    assert integrated_lower == F(1, 71680) > F(1, 72000)

    # Existing fourth-order absolute remainder and the existing tail cutoff.
    lower = F(1, 72000 * 10**180)
    w = F(1, 10**97)
    assert lower * (1 - 2 * w) - 5 * 10**7 * w * w > lower / 2
    assert w < F(1, 200) and w < F(1, 10**6)
    assert F(5, 10**98 - 1) < w
    print("Verified: scaled second-order gap > 1/(72000*sqrt(L)).")
    print("Verified: existing exact estimates imply sorting for w <= 10^-97.")


if __name__ == "__main__":
    verify()
