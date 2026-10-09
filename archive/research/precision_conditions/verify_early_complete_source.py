"""Constants in ledger 108; not an original-Riccati or joint-design certificate."""

from fractions import Fraction as F


def verify():
    r = F(1, 16)
    assert (75*r*r+9)/2 < 5
    assert 2+15*r*r < 3
    margin = F(7, 660)-F(1, 160)-F(3, 128000)
    assert margin > F(1, 240)
    assert F(10, 9)/160000 < F(1, 1000)
    assert F(15, 16)*F(999, 1000)/240 > F(1, 260)
    assert F(1, 160000) < F(1, 64)
    print("Verified early strong-gradient, complete-source, and output margins.")
    print("Continuous auxiliary comparison requires the ledger's mild and kernel proofs.")
    print("The original Riccati family and global joint-design regimes are not certified.")


if __name__ == "__main__":
    verify()
