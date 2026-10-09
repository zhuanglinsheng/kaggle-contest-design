"""Exact algebra for ledger section 90's real-gradient comparison.

This verifies the comparison equation and bootstrap constants, not the
parabolic existence proof or full-versus-finite output ranking.
"""

from fractions import Fraction as F
from itertools import product


def verify():
    # Both sides have degree at most three in each variable separately.
    # Four distinct values on the entire tensor grid certify this polynomial
    # identity by repeated univariate interpolation, not floating-point tests.
    numbers = [F(-1), F(0), F(1, 2), F(2)]
    for p, n, py, ny, k, m, source in product(numbers, repeat=7):
        h, a = p+n, p-n
        value = p-m*n
        gradient = py-m*ny-m*k*h*n
        direct = -k*ny*p-m*k*py*n-m*n*(source-k*k*a*h/2-k*k*h*h/2)+m*k*h*ny
        factored = -m*k*n*gradient+(k*(m-1)*ny+m*n*k*k*h)*value-m*n*source
        assert direct == factored
    assert F(4, 5)+F(1, 4)*F(4, 3)**2 == F(56, 45) < F(4, 3)
    assert F(3, 4)*F(1, 2)**2 == F(3, 16)
    assert (1-F(3, 16))/4 == F(13, 64)
    # These bounds use C_G < 4 and C_H < 5, not Taylor coefficients.
    assert F(1, 3)+F(8, 9)*F(1, 2) == F(7, 9)
    assert F(2, 3)*5*F(1, 2)**2 == F(5, 6) < 1
    assert (F(4, 3)*4)/(1-F(5, 6)) == 32
    print("Verified the nonsingular gradient-ratio comparison equation.")
    print("Verified the high-precision Gaussian bootstrap and Hamiltonian constants.")
    print("Verified the real organizer-gradient resolvent constants.")
    print("No output ranking or external joint-design region is asserted.")


if __name__ == "__main__":
    verify()
