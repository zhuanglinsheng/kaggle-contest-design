"""Independent numerical falsification checks for the exact inverse field.

Only the Python standard library is used. These quadratures do not prove
global Nash existence, vanishing-noise convergence, or joint attainment.
The analytic statements and their limits are recorded in the research note.
"""

import cmath
import math


A = 1 / math.sqrt(2 * math.pi)


def simpson(function, left, right, intervals):
    if intervals <= 0 or intervals % 2:
        raise ValueError("An even, positive interval count is required")
    step = (right-left) / intervals
    total = function(left) + function(right)
    total += sum((4 if index % 2 else 2) * function(left+index*step)
                 for index in range(1, intervals))
    return step * total / 3


def legendre_degree(argument, degree, intervals):
    """Laplace integral, using a stable exponential form of its base."""
    coordinate = cmath.acosh(argument)
    high, low = cmath.exp(coordinate), cmath.exp(-coordinate)

    def integrand(angle):
        cosine = math.cos(angle)
        base = ((1+cosine)*high + (1-cosine)*low) / 2
        return base**degree

    return simpson(integrand, 0, math.pi, intervals) / math.pi


def legendre_half(argument, intervals):
    return legendre_degree(argument, 0.5, intervals)


def labels(radial, vertical):
    if radial <= 0 or vertical <= 0:
        raise ValueError("The check is restricted to the open positive quadrant")
    logarithm = math.log(math.hypot(radial, vertical))
    angle = math.atan2(vertical, radial)
    norm = math.hypot(logarithm, angle)
    square = (logarithm+norm if logarithm >= 0
              else angle**2/(norm-logarithm))
    tau = math.sqrt(square)
    return angle/tau, tau, angle


def inverse_time(radial, vertical, intervals=128):
    eta, tau, angle = labels(radial, vertical)

    def integrand(t):
        argument = cmath.cosh((tau*tau-t*t)/2+1j*eta*t)/math.cos(angle)
        factor = cmath.exp(-(eta+1j*t)**2/4)
        return (factor*legendre_half(argument, intervals)).real

    return simpson(integrand, 0, tau, intervals)/(A*radial**1.5)


def axis_source(vertical, intervals=256):
    """rho and rho' with the endpoint square root removed by substitution."""
    logarithm = math.log(vertical)
    norm = math.hypot(logarithm, math.pi/2)
    tau_squared = (logarithm+norm if logarithm >= 0
                   else (math.pi/2)**2/(norm-logarithm))
    tau = math.sqrt(tau_squared)
    eta = math.pi/(2*tau)

    def integrand(u, derivative):
        if not u:
            if derivative:
                return (2*tau/cmath.sqrt(2*tau*(tau-1j*eta))).real
            return 0.0
        distance = tau*u*u
        t = tau-distance
        root = cmath.sqrt(1-cmath.exp(t*t-tau*tau+2j*eta*distance))
        term = 2*tau*u/root if derivative else 2*tau*u*vertical*root
        return term.real

    return tuple(2*simpson(lambda u: integrand(u, derivative), 0, 1, intervals)
                 /(A*math.pi) for derivative in (False, True))


def imaginary_label_time(vertical_label, tau, intervals=96):
    """Analytic continuation eta=i*v, not a real-state equilibrium."""
    v = vertical_label
    radial = math.exp((tau*tau+v*v)/2)*math.cosh(v*tau)

    def integrand(t):
        argument = math.cosh((tau*tau-t*t)/2-v*t)/math.cosh(v*tau)
        return (math.exp((v+t)**2/4)
                * legendre_degree(argument, 0.5, intervals).real)

    return simpson(integrand, -tau, tau, intervals)/(2*A*radial**1.5)


def imaginary_label_state(v, tau, intervals=96):
    radial = math.exp((tau*tau+v*v)/2)*math.cosh(v*tau)
    vertical = math.exp((tau*tau+v*v)/2)*math.sinh(v*tau)
    endpoint = ((v+tau)*math.sqrt(radial+vertical)
                + (v-tau)*math.sqrt(radial-vertical))/(2*math.sqrt(radial))

    def integrand(t):
        argument = math.cosh((tau*tau-t*t)/2-v*t)/math.cosh(v*tau)
        delta = argument-1
        if abs(delta) < 1e-5:
            derivative = -0.125+0.0703125*delta
        else:
            lower = legendre_degree(argument, -0.5, intervals).real
            upper = legendre_degree(argument, 0.5, intervals).real
            derivative = -0.5*(argument*lower-upper)/(argument**2-1)
        return -(v+t)**2*math.exp((v+t)**2/4)*derivative

    value = endpoint + vertical*simpson(integrand, -tau, tau, intervals)/(2*radial**1.5)
    return value-2*A*vertical*imaginary_label_time(v, tau, intervals)


def local_complex_caustic_check():
    def partials(v, tau):
        step = 1e-4
        vertical = (imaginary_label_time(v+step, tau)
                    - imaginary_label_time(v-step, tau))/(2*step)
        temporal = (imaginary_label_time(v, tau+step)
                    - imaginary_label_time(v, tau-step))/(2*step)
        return vertical, temporal

    left, right = 0.7, 1.0
    for _ in range(35):
        middle = (left+right)/2
        if partials(0, middle)[1] > 0:
            left = middle
        else:
            right = middle
    fold = (left+right)/2
    maximum = imaginary_label_time(0, fold)
    step = 0.0005
    second = (imaginary_label_time(0, fold+step)-2*maximum
              + imaginary_label_time(0, fold-step))/step**2
    amplitude = A*math.exp(fold*fold/2)
    predicted = 2*amplitude*fold/(3*math.sqrt(-second))
    errors = []
    print(f"Quadrature-only fold estimate: tau_f={fold:.10g}, z_f={maximum:.10g}")

    for remaining in [0.01, 0.003, 0.001]:
        v = math.sqrt(remaining/(-second))
        tau = fold-v

        def equations(v, tau):
            vertical, temporal = partials(v, tau)
            return imaginary_label_time(v, tau)-maximum+remaining, vertical+temporal

        for _ in range(7):
            f1, f2 = equations(v, tau)
            step = 2e-4
            a, b = [(q-p)/(2*step) for q, p in
                    zip(equations(v+step, tau), equations(v-step, tau))]
            c, d = [(q-p)/(2*step) for q, p in
                    zip(equations(v, tau+step), equations(v, tau-step))]
            determinant = a*d-b*c
            v -= (d*f1-c*f2)/determinant
            tau -= (-b*f1+a*f2)/determinant

        distance = imaginary_label_state(v, tau)
        observed = (distance-amplitude*remaining)/remaining**1.5
        errors.append(abs(observed-predicted))
        assert distance > amplitude*remaining
        print(f"h={remaining:g}, observed cubic={observed:.10g}, predicted={predicted:.10g}")

    assert all(new < old for old, new in zip(errors, errors[1:]))
    assert errors[-1] < 0.002


def main():
    locations = [(0.2, 0.02), (1.1, 0.1), (1.5, 0.2), (3.0, 1.0), (0.1, 2.0)]
    for radial, vertical in locations:
        coarse = inverse_time(radial, vertical, 128)
        fine = inverse_time(radial, vertical, 256)
        step = 1e-3*min(radial, vertical)
        derivative = (inverse_time(radial, vertical+step, 256)
                      - inverse_time(radial, vertical-step, 256))/(2*step)
        print(f"r={radial:g}, Y={vertical:g}, z={fine:.10g}, "
              f"z_Y={derivative:.10g}, refinement={abs(fine-coarse):.3g}")
        assert fine > 0 and derivative > 0
        assert abs(fine-coarse) < 1e-4*max(1.0, abs(fine))

    for vertical in [0.02, 0.2, 2.0]:
        source, source_derivative = axis_source(vertical)
        radial = 0.002*min(1, vertical)
        observed = radial**2*inverse_time(radial, vertical, 256)
        print(f"Y={vertical:g}, rho={source:.10g}, rho'={source_derivative:.10g}, "
              f"r^2*z={observed:.10g}")
        assert source > 0 and source_derivative > 0
        assert abs(observed-source) < 0.01*source

    local_complex_caustic_check()
    print("No vanishing-noise limit or joint maximum is certified by these checks.")


if __name__ == "__main__":
    main()
