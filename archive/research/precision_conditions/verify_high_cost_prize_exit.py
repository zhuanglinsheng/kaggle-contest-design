"""Check algebra/constants in the high-cost proof, not full-disclosure existence."""
from fractions import Fraction as F
from itertools import product

from verify_original_propagation_bound import add, mul, scale


def verify():
    # e=E_y, h=d_y, f=f'_w, a=diffusion, w=alpha/sigma^2.
    one = {(0,) * 6: F(1)}
    variables = []
    for i in range(6):
        exponent = [0] * 6
        exponent[i] = 1
        variables.append({tuple(exponent): F(1)})
    e, h, f, a, w, alpha = variables
    r_y = add(e, mul(f, h))
    h2 = mul(h, h)
    one_minus_f2 = add(one, scale(mul(f, f), -1))
    chain_rule = add(
        scale(mul(alpha, add(scale(mul(r_y, r_y), 3), scale(h2, -1))), F(1, 4)),
        scale(mul(mul(alpha, f), mul(r_y, h)), F(-1, 2)),
        scale(mul(mul(a, w), mul(one_minus_f2, h2)), F(1, 4)),
    )
    residual = add(
        scale(mul(alpha, mul(e, e)), F(3, 4)),
        mul(mul(alpha, f), mul(h, e)),
        scale(mul(add(mul(a, w), scale(alpha, -1)), mul(one_minus_f2, h2)), F(1, 4)),
    )
    assert chain_rule == residual

    # The total nonlinear Fourier norm is bounded by 2 M N.
    for p, n, py, ny in product((F(0), F(1, 7), F(2), F(11)), repeat=4):
        actual = (p + n) * (py + ny) + p * ny + n * py
        assert actual <= 2 * (p + n) * (py + ny)

    # At the chosen alpha cap, radius decay retains a strict margin.
    radius_rate = F(-1, 2)
    nonlinear_rate = 2 * F(1, 8)
    assert radius_rate + nonlinear_rate == F(-1, 4)

    # Complementary terminal-win weights have squared norm at most one.
    for k in range(101):
        probability = F(k, 100)
        assert probability**2 + (1-probability)**2 <= 1

    print("PASS: exact barrier residual, analytic-norm bootstrap, and effort weights.")
    print("Analytic convergence and Nash verification are arguments in the proof, not numerical tests.")
    print("No positive-prize full-disclosure joint optimum is certified by this check.")


if __name__ == "__main__":
    verify()
