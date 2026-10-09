"""Scale algebra for ledger 104; not a complete-output sign certificate."""

from fractions import Fraction as F


def verify():
    alpha = F(9, 20)
    integral = F(1, 3)/alpha+F(4, 9)/alpha**2
    assert integral == F(2140, 729)
    for eta, s in [(F(1, 16), F(1)), (F(1, 8), F(4)),
                   (F(1, 100), F(1, 4))]:
        h = 1+eta
        P = (s-eta)/h
        assert P >= 0 and 1+P == (1+s)/h
        # sqrt(h)/sqrt(1+P) = h/sqrt(1+s).
        assert h/(1+P) == h*h/(1+s)
    print("Verified matched time-convolution constant and auxiliary scaling.")
    print("Continuous decay additionally requires actual players and bridge cancellation.")
    print("No middle-precision sign or global joint-design regime is certified.")


if __name__ == "__main__":
    verify()
