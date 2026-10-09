"""Independent quadrature checks for the second-order design coefficient."""
import math
import unittest

import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss
from scipy.special import ndtr


def coefficient(mu, horizon, sigma, initial, precision):
    nodes, weights = leggauss(36)
    nodes, weights = (nodes + 1) / 2, weights / 2
    normals, normal_weights = hermgauss(56)
    normals, normal_weights = normals * np.sqrt(2), normal_weights / np.sqrt(np.pi)

    def q(time):
        if precision == 0:
            variance = initial + sigma**2 * time
        else:
            argument = sigma * np.sqrt(precision) * time
            tanhc = np.tanh(argument) / argument
            variance = (initial + sigma**2 * time * tanhc) / (
                1 + precision * initial * time * tanhc)
        return variance + sigma**2 * (horizon - time)

    total = 0.
    q0 = initial + sigma**2 * horizon
    for t, weight in zip(horizon * nodes, horizon * weights):
        qt = q(t)
        s = t + (horizon - t) * nodes
        qs = q(s)
        x = mu + np.sqrt(max(q0 - qt, 0)) * normals
        d = 2 * qt - qs
        integrand = (
            x[:, None] * (2 * ndtr(x[:, None] / np.sqrt(qt)) - 1)
            * np.exp(-x[:, None]**2 / d)
            / (2 * np.pi * np.sqrt(qs) * d**1.5))
        total += weight * (horizon - t) * (normal_weights @ integrand @ weights)
    return float(total)


def zero_coefficient(mu, horizon, sigma, initial):
    q = initial + sigma**2 * horizon
    return (horizon**2 * mu * (2 * ndtr(mu / np.sqrt(q)) - 1)
            * np.exp(-mu**2 / q) / (4 * np.pi * q**2))


def zero_derivative(mu, horizon, sigma, initial):
    q = initial + sigma**2 * horizon
    z = abs(mu) / np.sqrt(q)
    l1 = initial**2 * horizon**3 / 6 + initial * sigma**2 * horizon**4 / 12 + sigma**4 * horizon**5 / 60
    l2 = initial**2 * horizon**3 / 3 + initial * sigma**2 * horizon**4 / 4 + sigma**4 * horizon**5 / 15
    phi = np.exp(-z*z / 2) / np.sqrt(2 * np.pi)
    return (np.exp(-z*z) / (2 * np.pi * q**2.5)
            * (2 * phi * (1 - 2*z*z) * l1
               - z * (2 * ndtr(z) - 1) * (1 - z*z) * l2))


class JointDesignChecks(unittest.TestCase):
    def test_zero_precision_and_right_derivative(self):
        for initial, sigma, horizon in [(1., 1., 1.), (4., 2., 3.), (.5, 1.3, .7)]:
            for mu in [0., math.sqrt(initial / 6), -.8, 2.]:
                args = (mu, horizon, sigma, initial)
                zero = zero_coefficient(*args)
                self.assertAlmostEqual(coefficient(*args, 0.), zero, places=12)
                step = 1e-5 / (1 + initial * horizon + sigma**2 * horizon**2)
                numeric = (4 * coefficient(*args, step) - coefficient(*args, 2 * step) - 3 * zero) / (2 * step)
                expected = zero_derivative(*args)
                self.assertAlmostEqual(numeric, expected, delta=2e-7 * max(1., abs(expected)))
                if mu**2 <= initial / 6 + 1e-12:
                    self.assertGreater(expected, 0.)

    def test_uniform_bound_and_nonnegativity(self):
        for initial, sigma, horizon in [(1., 1., 1.), (4., 2., 3.), (.5, 1.3, .7)]:
            bound = 2 * math.sqrt(horizon) / (math.pi * math.sqrt(2 * math.e) * sigma**3)
            for mu in [0., -.8, 2.]:
                for precision in [0., .01, .5, 10., 1e4]:
                    value = coefficient(mu, horizon, sigma, initial, precision)
                    self.assertGreaterEqual(value, -1e-14)
                    self.assertLessEqual(value, bound + 1e-12)


if __name__ == '__main__':
    unittest.main()
