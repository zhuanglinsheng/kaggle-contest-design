"""Exact constants for Section 23; no remaining-interval certification."""

from fractions import Fraction as F


def main():
    r0, K, M, d, C = F(18, 25), F(29, 72), F(23, 25), F(12, 47), F(207, 100)
    assert K < F(1, 2)
    assert F(4, 5)+F(5, 6)*K*K*M*M < M
    coefficient = (d*K+(1+d*K*K)*F(71, 20)*(M/4+F(5, 12)*M*M*K))/r0
    assert coefficient < C
    discriminant = (25-32*C)**2-4*9*25*C
    assert discriminant == -F(101414, 625)
    assert F(800, 801) > F(167, 170)
    assert 4+17*F(167, 170) == 10*C
    assert 17/F(1001, 1000) > 2*C
    print("PASS: enlarged actual-output envelope and global-payoff margins.")
    print("Covered precision: lambda >= (18/25)*sigma^2.")
    print("No full coverage of intermediate precision or joint-optimum certification.")


if __name__ == "__main__":
    main()
