"""Exact constants for the diffused-subgame certificate, not the full goal."""
from fractions import Fraction as F
from math import factorial


def main():
    mu, r, eh, imbalance = F(18, 25), F(27, 100), F(9, 50), F(43, 100)
    k = r / mu
    assert k == F(3, 8)
    assert k * imbalance * (2 + eh) / 2 < eh
    assert r + k * (3 * imbalance**2 + eh * (4 + eh)) / 4 < imbalance
    row1 = k * imbalance / 2 + k * (2 + eh) / 2
    row2 = k * (4 + 2 * eh) / 4 + 3 * k * imbalance / 2
    assert max(row1, row2) < 1

    value_bound = r * (3 * imbalance**2 + (2 + eh)**2) / (16 * mu)
    potential_bound = r**2 * ((2 + eh)**2 + imbalance**2) / (16 * mu**2)
    assert value_bound < F(311, 2500)  # .1244
    assert potential_bound < F(217, 5000)  # .0434
    assert 3 * k * value_bound < F(7, 50)

    # Elementary lower bounds pi>3.14 and Euler's e>2.718 give this cap.
    gaussian_derivative_cap = F(243, 1000)
    assert 2 * F(157, 50) * F(1359, 500) * gaussian_derivative_cap**2 > 1
    source = imbalance / (2 * r) + eh / (2 * r) + imbalance * gaussian_derivative_cap
    assert source < F(247, 200)  # 1.235

    # The real conditional-output proof uses the sharper existing kernel bound.
    x_cosh = k
    cosh_upper = sum(x_cosh**(2*j) / factorial(2*j) for j in range(5))
    cosh_upper += x_cosh**10 / factorial(10) / (1-x_cosh**2/(11*12))
    cosh_square_upper = (1+cosh_upper)/2
    assert cosh_square_upper < F(259, 250)  # 1.036
    assert F(247, 200) * F(259, 250) < F(32, 25)  # 1.28

    # Positive Taylor terms and a geometric bound on the exponential tail.
    x = F(7, 50)
    exp_upper = sum(x**j / factorial(j) for j in range(7))
    exp_upper += x**7 / factorial(7) / (1 - x / 8)
    assert exp_upper < F(1151, 1000)
    sqrt_lower = F(2389, 2500)
    assert sqrt_lower**2 < 1 - 2 * potential_bound
    assert F(1151, 1000) / sqrt_lower < F(241, 200)  # 1.205
    assert F(247, 200) * F(241, 200) * F(4, 3) < 2

    print("PASS: invariant ball and complex-prize contraction.")
    print("PASS: value, heat-potential, source and centered-output constants.")
    print("PASS: real conditional-output coefficient 32/25, at every state.")
    print("Scope: a_t >= .72 sigma^2; the low-diffusion initial layer is not certified.")


if __name__ == "__main__":
    main()
