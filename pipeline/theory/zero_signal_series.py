"""Exact rational time-series audit; not a numerical equilibrium proof.

At zero signal, scale the terminal variance to one and absorb cumulative
incentives into reverse time z. If g is the standard normal density, write
p=sum P_n(x) g(x)^(n+1) z^n, r(x,z)=p(-x,z).
"""

from fractions import Fraction as F
import math


def multiply(a, b):
    result = [F(0)] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            result[i + j] += ai * bj
    return result


def coefficients(order):
    polynomials = [[F(1)]]
    for n in range(order):
        flux = [F(0)] * (n + 1)
        for j in range(n + 1):
            product = multiply(polynomials[n - j], polynomials[j])
            weight = F(1, 2) - (-1) ** j
            for k, value in enumerate(product):
                flux[k] += weight * value
        derivative = [F(0)] * (n + 2)
        for k, value in enumerate(flux):
            if k:
                derivative[k - 1] += k * value
            derivative[k + 1] -= (n + 2) * value
        polynomials.append([value / (n + 1) for value in derivative])
    return polynomials


def center_energy_certificate(order=80):
    """Finite rational lower bound, conditional on analytic-branch continuation.

    199/500 < 1/sqrt(2*pi), using pi < 22/7. Positive Taylor coefficients
    permit replacing the latter constant by the former in a lower bound.
    This is not an existence/nonexistence test for arbitrary weak equilibria.
    """
    a, horizon = F(199, 500), F(3, 2)
    assert 2 * F(22, 7) * a*a < 1
    polynomials = coefficients(order)
    square = multiply([p[0] for p in polynomials],
                      [p[0] for p in polynomials])
    bound = sum(value * a**(n+2) * horizon**(n+1)/(n+1)
                for n, value in enumerate(square))
    return bound


def main():
    polynomials = coefficients(40)
    assert polynomials[1] == [F(0), F(1)]
    assert polynomials[2] == [F(1, 2), F(0), F(-3, 2)]
    # In the variable s=g(0) z, compare to K'=1+3K^2, H'=HK.
    lower_k, lower_h = [F(0)], [F(1)]
    for n in range(40):
        kk = sum(lower_k[j] * lower_k[n-j] for j in range(n+1))
        hk = sum(lower_h[j] * lower_k[n-j] for j in range(n+1))
        lower_k.append((F(n == 0) + 3*kk)/(n+1))
        lower_h.append(hk/(n+1))
    density = 1 / math.sqrt(2 * math.pi)
    print("n, P_n(0), coefficient of p(0,z), empirical root scale")
    for n, polynomial in enumerate(polynomials):
        assert all(not value or k % 2 == n % 2
                   for k, value in enumerate(polynomial))
        assert all(value * (-1)**(k//2) >= 0
                   for k, value in enumerate(polynomial))
        assert polynomial[0] >= lower_h[n]
        if n % 2:
            assert polynomial[1] >= lower_k[n]
        if n % 2 == 0:
            value = polynomial[0]
            central = float(value) * density ** (n + 1)
            root = abs(central) ** (-1 / n) if n and central else math.inf
            print(n, str(value), central, root)
    print("Finite coefficients do not prove convergence, blow-up, or existence.")
    print("Analytic-continuation upper bound:",
          math.pi/(2*math.sqrt(3)*density))
    certificate = center_energy_certificate()
    assert certificate > 4
    print("80-order exact energy lower bound at exposure 3/2:",
          float(certificate), "> 4 > 1")


if __name__ == "__main__":
    main()
