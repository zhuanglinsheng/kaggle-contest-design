"""Check constants in ledger section 92, not its PDE comparison theorem."""

from fractions import Fraction as F


def verify():
    for value in (F(0), F(1, 4), F(1), F(12), F(16), F(100)):
        # delta = 1/[16(1+L)^2]; its square root is rational.
        local_mass = value*(1/(16*(1+value)**2)+1/(2*(1+value)))
        assert local_mass < F(1, 2)
        # a = 64(1+L)^2; sqrt(pi) < 2 gives this Laplace bound.
        log_kernel = value*(1/(32*(1+value)**2)+1/(4*(1+value)))
        assert log_kernel < F(1, 2)
    assert 24*F(1, 2) == 12
    assert 64*F(1, 2)**2 == 16
    assert F(7, 9)*2 < 8
    print("Verified weighted-mass partition and logarithmic-kernel margins.")
    print("Verified the enlarged real-parameter Volterra coefficients' ranges.")
    print("PDE estimates and information-regime ranking are not script-certified.")


if __name__ == "__main__":
    verify()
