"""Exact algebra checks for the Riccati information-elasticity proof.

No simulation or finite grid is used as a sign certificate.
"""
from fractions import Fraction as F


def verify():
    for n in range(1, 35):
        lam, s, a = F(n, 37), F(n+3, 19), F(n+1, 23)
        z = a*s/(lam*(s+2*a))
        sprime = 1-lam*s*s
        aprime = lam*s*s
        zprime = s*s-2*lam*s*z
        nprime = zprime*(1+2*a/s)+z*(2*aprime/s-2*a*sprime/(s*s))-aprime/lam
        assert nprime == 2*a*(s-z*(lam+1/(s*s)))
        assert nprime == 2*a*(lam*s**3-a*sprime)/(lam*s*(s+2*a))
    # (1-x)(3*x^3+x^2+x+1) = 1+2*x^3-3*x^4.
    coefficients = [F(1), F(1), F(1), F(3)]
    expanded = [coefficients[0]]+[coefficients[i]-coefficients[i-1] for i in range(1, 4)]+[-coefficients[-1]]
    assert expanded == [1, 0, 0, 2, -3]
    print("Verified Riccati crossing algebra and radial-kernel factorization.")


if __name__ == "__main__":
    verify()
