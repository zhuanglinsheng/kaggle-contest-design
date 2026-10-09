"""Rational algebra checks for ledger 100; not a joint-design certificate."""

from fractions import Fraction as F


def verify():
    w, k = F(1, 4), F(21, 20)
    lower = 1-F(11, 10)*w-F(441, 400)*w*w
    upper = 2*w*k*(1+w*w*k*k/F(3))*(1+w/2)
    assert lower == F(4199, 6400)
    assert upper == F(1237383, 2048000)
    assert lower-upper == F(106297, 2048000) > F(1, 20)
    # Verify the substitution for the differentiated background ODE.
    for z, a, p, h, y in [
        (F(1), F(1, 8), F(2, 3), F(1, 4), F(1, 3)),
        (F(3), F(1, 10), F(1, 20), F(2, 5), F(1, 100)),
    ]:
        r = a*p
        rp = w*(1-a*a)*p*p/2-(z+w*r)*r
        pp = -(z+w*r)*p
        original = -(z+2*w*r)*y-2*w*rp*h-w*pp
        rewritten = (-(z+2*w*r)*y+w*(z+w*r)*p*(1+2*a*h)
                     -w*w*(1-a*a)*p*p*h)
        assert original == rewritten
    print("Verified shape-bound margin and background ODE substitution checks.")
    print("Continuous covariance positivity requires the ledger's heat and pairing proofs.")
    print("Actual propagation unimodality and global precision sorting are not certified.")


if __name__ == "__main__":
    verify()
