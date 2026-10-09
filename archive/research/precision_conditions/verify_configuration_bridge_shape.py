"""Algebra for ledger 105; not finite-prize or global precision sorting."""

from fractions import Fraction as F


def verify():
    for a, w, k in [(F(1, 10), F(1, 4), F(1)),
                    (F(1, 5), F(1, 2), F(21, 20))]:
        da = w*k*(1-a*a)**2/2
        first = w*k*k*(1-3*a*a)*(1-a*a)**2
        derivative_a = w*k*k*(-6*a*(1-a*a)**2
                              -4*a*(1-3*a*a)*(1-a*a))
        second = -w*w*k**3*a*(1-a*a)**3*(5-9*a*a)
        assert derivative_a*da == second < 0 and first > 0
    for eps in [F(1, 100), F(1, 5), F(1, 2), F(4, 5), F(99, 100)]:
        den = (1+eps)*(2-eps)
        r = eps*(1-eps)/den
        assert 0 < r <= F(1, 9)
        assert den <= F(9, 4)
        assert (1-eps)/(1+eps)+eps/(2-eps)-1 == -3*r
    assert F(4, 3) < F(4, 3)**2  # sqrt(4/3) < 4/3
    print("Verified configuration-shape derivatives and bridge parameter identities.")
    print("Global interval signs follow from the ledger's rational inequalities.")
    print("Finite-prize actual gradients and global design regimes are not certified.")


if __name__ == "__main__":
    verify()
