"""Exact Gaussian identities for ledger section 89, not precision ranking."""

from fractions import Fraction as F
from math import comb


def moment(n):
    if n % 2:
        return 0
    result = 1
    for j in range(1, n, 2):
        result *= j
    return result


def convolved_power(n, z, eps):
    return sum(F(comb(n, j)*moment(j))*z**(n-j)*eps**(j//2)
               for j in range(0, n+1, 2))


def verify():
    for root in [F(1, 2), F(2, 3), F(3, 4)]:
        a, eps = root**2, 1-root**2
        for z in [F(0), F(1, 3), F(-2), F(7, 5)]:
            for n in range(9):
                h = convolved_power(n, z, eps)/root**n
                hz = n*convolved_power(n-1, z, eps)/root**n if n else F(0)
                hzz = n*(n-1)*convolved_power(n-2, z, eps)/root**n if n >= 2 else F(0)
                hv = eps*(-F(n, 2)*h/a-hzz/2)
                l0h = (hzz+z*hz)/2
                convolved_l0 = F(n, 2)*h+a*hzz/2
                assert hv-l0h == -convolved_l0/a
    assert moment(2)-moment(1)**2 == 1
    assert moment(4)-moment(2)**2 == 2
    assert F(1, 4)*(moment(4)-moment(2)**2) == F(1, 2)
    print("Verified Gaussian bridge operator identity on polynomial test functions.")
    print("Verified the exact moments for the covariance sign reversal.")
    print("No global comparison or joint-design region is asserted.")


if __name__ == "__main__":
    verify()
