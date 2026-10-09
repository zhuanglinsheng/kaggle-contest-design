"""Exact integer Bernstein certificate for the third-order precision bound.

It proves the stated polynomial sign, not an all-order output ranking.
"""
from math import comb
from fractions import Fraction as F
import numpy as np


class Poly(dict):
    def __add__(self, other):
        if not isinstance(other, Poly):
            other = Poly({(0,)*5: other})
        out = Poly(self)
        for key, value in other.items():
            out[key] = out.get(key, 0)+value
        return Poly({k: v for k, v in out.items() if v})

    __radd__ = __add__

    def __neg__(self):
        return Poly({k: -v for k, v in self.items()})

    def __sub__(self, other):
        return self+(-other)

    def __rsub__(self, other):
        return -self+other

    def __mul__(self, other):
        if not isinstance(other, Poly):
            return Poly({k: v*other for k, v in self.items() if v*other})
        out = {}
        for key, value in self.items():
            for key2, value2 in other.items():
                key3 = tuple(a+b for a, b in zip(key, key2))
                out[key3] = out.get(key3, 0)+value*value2
        return Poly({k: v for k, v in out.items() if v})

    __rmul__ = __mul__

    def __pow__(self, n):
        out = Poly({(0,)*5: 1})
        for _ in range(n):
            out = out*self
        return out


def verify():
    # Independently check the cleared radial derivative against the
    # separately differentiated K and L expressions.
    for n in range(1, 15):
        q0, a0, b0, t0 = F(8+n, 5), F(n, 19), F(n+1, 13), F(2, 3)
        d0, n0 = q0*q0+q0*b0-2*a0*a0, t0*q0-(2-t0)*a0
        ka = 2*q0*d0
        kb = 2*a0*d0*d0/(q0+b0)*(1/(2*(q0-b0))-1/(q0+b0)-q0/(2*d0))
        la = (-(2-t0)*d0+6*a0*n0)/2
        lb = n0*d0/2*(1/(2*(q0-b0))-3*q0/(2*d0))
        direct = 4*(q0-b0)*(q0+b0)**2*(a0*(ka+la)+b0*(kb+lb))
        cleared = (4*a0*d0*((2*q0*q0-q0*b0+b0*b0)*d0-(q0*q0-b0*b0)*(q0*b0-4*a0*a0))
                   +(q0+b0)**2*(-2*(2-t0)*a0*(q0-b0)*d0
                                  +n0*(b0*d0-3*(q0-b0)*(q0*b0-4*a0*a0))))
        assert direct == cleared
    assert 4*F(3, 5)**3+F(3, 5)**2-F(3, 5)-F(1, 3) == F(109, 375)
    S, tau, j, rho, k = [
        Poly({tuple(int(i == m) for i in range(5)): 1}) for m in range(5)
    ]
    # Actual q=Q/5; common denominator for actual A,B,q is scale.
    Q, D0 = 8+5*S, 22+20*S
    t, nu = 1-tau, j*tau
    scale = 5*D0
    A = rho*(5*t*(Q-5*nu)+k*(D0*(Q-5*tau)-5*t*(Q-5*nu)))
    B, q = D0*rho*(Q-5*nu), D0*Q
    D, N = q*q+q*B-2*A*A, t*q-(2-t)*A
    polynomial = (
        4*A*D*((2*q*q-q*B+B*B)*D-(q*q-B*B)*(q*B-4*A*A))
        +scale*(q+B)**2*(-2*(2-t)*A*(q-B)*D
                            +N*(B*D-3*(q-B)*(q*B-4*A*A)))
    )
    degrees = tuple(max(key[i] for key in polynomial) for i in range(5))
    assert degrees == (14, 12, 7, 7, 5)
    print("power degrees", degrees, "monomials", len(polynomial), flush=True)
    shape = tuple(d+1 for d in degrees[1:])
    all_nonnegative = True
    for power in range(degrees[0]+1):
        array = np.zeros(shape, dtype=object)
        for key, value in polynomial.items():
            if key[0] == power:
                array[key[1:]] = value
        for axis, degree in enumerate(degrees[1:]):
            # Each transformed coefficient is Bernstein coefficient times
            # the positive binomial denominator.
            matrix = np.zeros((degree+1, degree+1), dtype=object)
            for i in range(degree+1):
                for h in range(i+1):
                    matrix[i, h] = comb(degree-h, i-h)
            array = np.moveaxis(np.tensordot(matrix, array, axes=(1, axis)), 0, axis)
        negative = sum(value < 0 for value in array.flat)
        all_nonnegative &= negative == 0
        print("S power", power, "negative coefficients", negative, flush=True)
    assert all_nonnegative
    print("Verified all 74880 scaled Bernstein coefficients are nonnegative.", flush=True)


if __name__ == "__main__":
    verify()
