"""Independent finite-difference audit of the feedback mixed variation.

This checks identities (239)--(240) in the research note, not convergence of
exact positive-precision Nash equilibria. A short-exposure Gaussian series
represents the already verified zero-precision reference branch.
"""

import math

from .zero_signal_series import coefficients


POLYNOMIALS = [list(map(float, row)) for row in coefficients(18)]
DENSITY = 1 / math.sqrt(2 * math.pi)
HORIZON = 0.1


def polynomial(row, x):
    result = 0.0
    for coefficient in reversed(row):
        result = result * x + coefficient
    return result


def gradient(x, z):
    g = DENSITY * math.exp(-x * x / 2)
    return g * polynomial([polynomial(row, x) for row in POLYNOMIALS], g * z)


def center(z):
    a = DENSITY * polynomial([row[0] for row in POLYNOMIALS], DENSITY * z)
    b = DENSITY * polynomial(
        [row[1] if len(row) > 1 else 0.0 for row in POLYNOMIALS],
        DENSITY * z,
    )
    return a, b


def integrate(rhs, initial, steps):
    state = list(initial)
    dt = HORIZON / steps
    for index in range(steps):
        t = index * dt
        k1 = rhs(t, state)
        k2 = rhs(t + dt / 2, [s + dt * k / 2 for s, k in zip(state, k1)])
        k3 = rhs(t + dt / 2, [s + dt * k / 2 for s, k in zip(state, k2)])
        k4 = rhs(t + dt, [s + dt * k for s, k in zip(state, k3)])
        state = [s + dt * (v1 + 2*v2 + 2*v3 + v4) / 6
                 for s, v1, v2, v3, v4 in zip(state, k1, k2, k3, k4)]
    return state


def reference_integral(steps):
    def rhs(t, state):
        a, b = center(HORIZON - t)
        return [2 * b * state[0] + 1, a * state[0]]

    return integrate(rhs, [0.0, 0.0], steps)[1]


def payoffs(epsilon_i, epsilon_j, frequency, steps):
    def rhs(t, state):
        y = state[0]
        # Smooth, compactly supported perturbations equal these functions
        # on all sampled states, which remain within |y| < 0.01.
        assert abs(y) < 0.01
        qi = gradient(y, HORIZON - t) + epsilon_i
        qj = gradient(-y, HORIZON - t) + epsilon_j * math.sin(frequency * y)
        assert qi > 0 and qj > 0
        return [qi - qj, qi * qi / 2, qj * qj / 2]

    y, cost_i, cost_j = integrate(rhs, [0.0, 0.0, 0.0], steps)
    probability = (1 + math.erf(y / math.sqrt(2))) / 2
    return probability - cost_i, 1 - probability - cost_j


def mixed_difference(delta, frequency, steps):
    samples = [payoffs(si * delta, sj * delta, frequency, steps)
               for si, sj in [(1, 1), (1, -1), (-1, 1), (-1, -1)]]
    return tuple((samples[0][k] - samples[1][k] - samples[2][k] + samples[3][k])
                 / (4 * delta * delta) for k in range(2))


def main():
    integral = reference_integral(400)
    assert abs(reference_integral(200) - integral) < 1e-12
    for frequency in (10, 30, 100):
        predicted = -frequency * integral
        coarse = mixed_difference(2e-4, frequency, 200)
        fine = mixed_difference(1e-4, frequency, 400)
        assert abs(fine[0] - predicted) < 2e-7
        assert abs(fine[1]) < 2e-7
        assert abs(fine[0] - coarse[0]) < 2e-7
        print(f"n={frequency}: mixed_i={fine[0]:.9g}, "
              f"predicted={predicted:.9g}, mixed_j={fine[1]:.3g}")
    print("PASS: feedback derivative appears in the mixed payoff variation.")
    print("This is not a proof of exact Nash zero-precision convergence.")


if __name__ == "__main__":
    main()
