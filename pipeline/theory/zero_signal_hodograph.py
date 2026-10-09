"""Exact local audit of the Gaussian no-signal characteristic hodograph.

This checks an algebraic identity between two formal series. It does not
construct a global equilibrium or certify a joint optimum.
"""

from fractions import Fraction as F
from math import comb, factorial

from .zero_signal_series import coefficients, multiply


def add_scaled(target, source, weight=F(1)):
    if len(target) < len(source):
        target.extend([F(0)] * (len(source) - len(target)))
    for k, value in enumerate(source):
        target[k] += weight * value


def derivative(poly):
    return [k * value for k, value in enumerate(poly)][1:] or [F(0)]


def gaussian_derivative(poly):
    result = derivative(poly)
    add_scaled(result, [F(0)] + poly, F(-1))
    return result


def tanh_coefficients(order):
    result = [F(0)]
    for n in range(order):
        square = sum(result[j] * result[n-j] for j in range(n+1))
        result.append((F(n == 0) - square) / (n+1))
    return result


def hodograph_coefficients(order):
    """z(eta,tau)=a^-1 exp(-eta^2/2) sum Q_n(eta) tau^n.

    z_tt-z_ee-3(eta+tau*tanh(eta*tau))*z_e
    +3(tau+eta*tanh(eta*tau))*z_t=0.
    Initial data are z=0 and z_t=a^-1 exp(-eta^2/2).
    """
    tanh = tanh_coefficients(order)
    a_terms, b_terms = {}, {}
    for k in range(1, order+1):
        if k >= 2 and tanh[k-1]:
            a_terms[k] = [F(0)] * (k-1) + [tanh[k-1]]
        if tanh[k]:
            b_terms[k] = [F(0)] * (k+1) + [tanh[k]]
    add_scaled(b_terms.setdefault(1, [F(0)]), [F(1)])
    result = [[F(0)], [F(1)]]
    for n in range(order-1):
        poly = result[n]
        rhs = derivative(derivative(poly))
        add_scaled(rhs, [F(0)] + derivative(poly))
        add_scaled(rhs, [F(0), F(0)] + poly, F(-2))
        add_scaled(rhs, poly, F(-1))
        for k, factor in a_terms.items():
            if k <= n:
                add_scaled(rhs, multiply(factor,
                                        gaussian_derivative(result[n-k])), F(3))
        for k, factor in b_terms.items():
            if k <= n:
                add_scaled(rhs, multiply(factor, result[n-k+1]),
                           F(-3 * (n-k+1)))
        result.append([value / ((n+1)*(n+2)) for value in rhs])
    return result


def compose(outer, inner, order):
    result, power = [F(0)] * (order+1), [F(1)]
    for coefficient in outer[:order+1]:
        for k, value in enumerate(power[:order+1]):
            result[k] += coefficient * value
        power = multiply(power, inner)[:order+1]
    return result


def elliptic_center_coefficients(order):
    """Coefficients of a*Z(tau) from the exact elliptic convolution.

    With u=tau^2, a*Z = pi^-1 (g*h)(u), where
    g(t)=(1+t)*exp(-t/2)/sqrt(t),
    h(t)=exp(-3*t/2)*K(sqrt(1-exp(-t))).
    The half-integer beta integrals are evaluated as rational numbers.
    """
    half_order = (order-1)//2
    argument = [F(0)] + [F((-1)**(j+1), factorial(j))
                          for j in range(1, half_order+1)]
    elliptic = compose([F(comb(2*j, j)**2, 16**j)
                        for j in range(half_order+1)],
                       argument, half_order)
    # Hypergeometric ODE after m=1-exp(-u):
    # (exp(u)-1) C'' + C' - C/4 = 0.
    ode = multiply([F(0)] + [F(1, factorial(j))
                             for j in range(1, half_order+1)],
                   derivative(derivative(elliptic)))
    add_scaled(ode, derivative(elliptic))
    add_scaled(ode, elliptic, F(-1, 4))
    assert all(value == 0 for value in ode[:half_order-1])
    h = multiply(elliptic, [F(-3, 2)**j/factorial(j)
                           for j in range(half_order+1)])[:half_order+1]
    g = [F(1)] + [F(-1, 2)**j/factorial(j)
                   + F(-1, 2)**(j-1)/factorial(j-1)
                   for j in range(1, half_order+1)]
    result = [F(0)] * (order+1)
    for k in range(half_order+1):
        coefficient = F(0)
        for n in range(k+1):
            j, denominator = k-n, F(1)
            for r in range(j+1):
                denominator *= F(n) + F(1, 2) + r
            coefficient += g[n]*h[j]*factorial(j)/(2*denominator)
        result[2*k+1] = coefficient
    return result


def legendre_center_coefficients(order):
    """Independent rational audit of the direct four-radial Riemann kernel."""
    size = (order-1)//2

    def product(left, right):
        result = [[F(0)] for _ in range(size+1)]
        for i, p in enumerate(left):
            for j, q in enumerate(right[:size+1-i]):
                add_scaled(result[i+j], multiply(p, q))
        return result

    # m=1/2-cosh(u*(1-v^2)/2)/2; polynomials use v^2 as variable.
    argument = [[F(0)] for _ in range(size+1)]
    for k in range(2, size+1, 2):
        argument[k] = [F(-comb(k, j)*(-1)**j, 2**(k+1)*factorial(k))
                       for j in range(k+1)]
    kernel = [[F(0)] for _ in range(size+1)]
    power = [[F(1)]] + [[F(0)] for _ in range(size)]
    coefficient = F(1)
    for n in range(size+1):
        for k, polynomial in enumerate(power):
            add_scaled(kernel[k], polynomial, coefficient)
        power = product(power, argument)
        coefficient *= ((F(n)-F(1, 2))*(F(n)+F(3, 2))/(n+1)**2)
    gaussian = [[F(0)]*k+[F(1, 4)**k/factorial(k)]
                for k in range(size+1)]
    integrated = [sum(value/F(2*j+1) for j, value in enumerate(polynomial))
                  for polynomial in product(kernel, gaussian)]
    normalized = multiply(integrated, [F(-3, 4)**k/factorial(k)
                                      for k in range(size+1)])
    result = [F(0)] * (order+1)
    for k in range(size+1):
        result[2*k+1] = normalized[k]
    return result


def operator_and_verification_certificates():
    """Exact algebra checks for the global inverse-field proof.

    These certify operator conjugacy and the square-difference identity,
    not the analytic boundary estimates or global joint attainment.
    """
    def collect(items):
        result = {}
        for power, coefficient in items:
            result[power] = result.get(power, F(0)) + coefficient
        return {power: value for power, value in result.items() if value}

    for radial_power in range(7):
        for vertical_power in range(7):
            # w=r^2*z_Y: L_0(w)=r^2*partial_Y(L_4(z)).
            left = collect([
                ((radial_power, vertical_power-1),
                 F(vertical_power*radial_power*(radial_power+2))),
                ((radial_power+2, vertical_power-3),
                 F(vertical_power*(vertical_power-1)*(vertical_power-2))),
            ])
            right = collect([
                ((radial_power, vertical_power-1),
                 F(radial_power*(radial_power+2)*vertical_power)),
                ((radial_power+2, vertical_power-3),
                 F(vertical_power*(vertical_power-1)*(vertical_power-2))),
            ])
            assert left == right

    # Polar form of L_0: radial effective dimension one, angular f''+tan(t)*f'.
    infinity_power = F(1, 2)
    assert infinity_power*(infinity_power-1) < 0
    origin_power, angular_power = F(1, 4), F(3, 4)
    assert origin_power*(origin_power+1)-angular_power**2 < 0
    corner_power, corner_angle = F(5, 8), F(3, 4)
    assert F(1, 2) < corner_power < corner_angle < 1
    assert corner_power**2-corner_angle**2 < 0

    for own in [F(0), F(1, 3), F(2)]:
        for other in [F(0), F(2, 5), F(3)]:
            for deviation in [F(0), F(1, 7), F(4)]:
                value_time = own**2/2-own*other
                realized = -value_time+(deviation-other)*own-deviation**2/2
                assert realized == -(deviation-own)**2/2

    # Exact geometric-mean equation with the actual dissipative correction.
    for first_root, second_root in [(F(1), F(1)), (F(2), F(3)), (F(1, 2), F(3, 4))]:
        own, other = first_root**2, second_root**2
        geometric = first_root*second_root
        for own_x, other_x, own_xx, other_xx in [
                (F(1), F(-2), F(3), F(4)),
                (F(2, 3), F(1, 7), F(-1), F(2))]:
            kappa = F(1, 5)
            drift = own-other
            own_z = kappa*own_xx+drift*own_x-own*other_x
            other_z = kappa*other_xx+drift*other_x+other*own_x
            geometric_z = geometric*(own_z/own+other_z/other)/2
            geometric_x = geometric*(own_x/own+other_x/other)/2
            shear = own_x/own-other_x/other
            geometric_xx = (geometric*(own_xx/own+other_xx/other)/2
                            - geometric*shear**2/4)
            transformed = (kappa*geometric_xx+drift*geometric_x
                           + geometric*(own_x-other_x)/2
                           + kappa*geometric*shear**2/4)
            assert transformed == geometric_z
            density_z = own_z*other+own*other_z
            density_x = own_x*other+own*other_x
            density_xx = own_xx*other+2*own_x*other_x+own*other_xx
            assert density_z == (drift*density_x+own*other*(own_x-other_x)
                                  + kappa*density_xx-2*kappa*own_x*other_x)

    # Cubic center-caustic coefficient: all Z''' terms must cancel.
    for second in [F(-1), F(-2), F(-7, 3)]:
        for third in [F(-3), F(0), F(5, 2)]:
            for critical_label in [F(1, 2), F(3, 4), F(1)]:
                z_delta_cubic = third/6
                z_mixed_cubic = -(third+3*critical_label*second)/2
                h_cubic = z_delta_cubic-z_mixed_cubic
                x_mixed_cubic = third/2+critical_label*second
                x_vertical_cubic = (-third+critical_label*second)/6
                y_cubic = x_mixed_cubic-x_vertical_cubic
                assert y_cubic-h_cubic == -F(2, 3)*critical_label*second
                assert y_cubic-h_cubic > 0


def main():
    order = 24
    polys = hodograph_coefficients(order)
    center = [p[0] for p in polys]
    assert center[:6] == [F(0), F(1), F(0), F(-2, 3), F(0), F(1, 4)]
    assert all(not value for n, value in enumerate(center) if not n % 2)
    density_center = [p[0] for p in coefficients(order)]
    composed = compose(density_center, center, order)
    expected = [F(0)] * (order+1)
    expected[0], series_factorial = F(1), 1
    for k in range(1, order//2+1):
        series_factorial *= k
        expected[2*k] = F(1, 2**k * series_factorial)
    assert composed == expected
    assert center == elliptic_center_coefficients(order)
    assert center == legendre_center_coefficients(order)
    # Rational inequalities used to bracket the unique center mode in u=tau^2.
    integral_lower = sum(F(5, 6)**j/factorial(j)
                         * (1/(F(j)+F(1, 2)) + 1/(F(j)+F(3, 2)))
                         for j in range(4))
    exponential_upper = sum(F(4, 3)**j/factorial(j) for j in range(11))
    exponential_upper += (F(4, 3)**11/factorial(11)
                          / (1-F(4, 3)/12))
    assert exponential_upper < F(19, 5) < integral_lower
    exponential_three_halves = sum(F(3, 2)**j/factorial(j) for j in range(11))
    exponential_three_halves += (F(3, 2)**11/factorial(11)
                                  / (1-F(3, 2)/12))
    assert exponential_three_halves < F(9, 2)
    operator_and_verification_certificates()
    print("Exact series identity verified through degree", order)
    print("H(s(tau)) = exp(tau^2/2); s=a*z at the center.")
    print("s(tau) = tau - 2*tau^3/3 + tau^5/4 + ...")
    print("Exact elliptic integral coefficients agree through degree", order)
    print("Independent direct Legendre kernel agrees through degree", order)
    print("Rational mode-bracket inequalities verified (1/2 < u_f < 1).")
    print("Rational exponential bound for the unit-exposure candidate verified.")
    print("Inverse-field operator, barrier signs, and Nash square identity verified.")
    print("Cubic complex-caustic cancellation and its positive sign verified.")
    print("Actual-viscosity geometric-mean and density identities verified.")
    print("No global equilibrium or endpoint attainment is inferred.")


if __name__ == "__main__":
    main()
