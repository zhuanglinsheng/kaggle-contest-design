"""Tail algebra in ledger 106; not a full-source or joint-design certificate."""

from fractions import Fraction as F


def verify():
    w = F(1, 2)
    assert F(1, 2)-w/3-5*w*w/24 == F(9, 32)
    # Expand the two lower bounds without discarding the favorable fifth order.
    for w in [F(1, 100), F(1, 4), F(1, 2)]:
        product = (1-w*w/3)*(1+w/2-w**3/24)-1
        expanded = w*(F(1, 2)-w/3-5*w*w/24+w**4/72)
        assert product == expanded >= F(9, 32)*w
    # Rescaled Gaussian variance after an outward-OU convolution.
    for growth in [F(1), F(2), F(10)]:
        r = F(3, 4)
        target = (r+growth-1)/growth
        assert target == 1-F(1, 4)/growth < 1
    print("Verified complete-tail lower bound and OU variance identities.")
    print("Actual gradient tails require the ledger's two mild-bootstrap steps.")
    print("No all-state source sign or global design regime is certified.")


if __name__ == "__main__":
    verify()
