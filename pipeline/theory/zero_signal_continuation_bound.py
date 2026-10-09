"""Diagnostic for ledger (F21), NOT a certified global payoff bound.

Both inner state maximization and the dual minimization are discretized.
The inner grid can underestimate a supremum, so reported numbers must not
be used as mathematical certificates. Uses the weak Gaussian Nash series.
"""

import math
from zero_signal_series import coefficients


def continuation_table(epsilon, order=20, steps=160, spacing=.05):
    polys = [[float(a) for a in p] for p in coefficients(order)]

    def effort(y, remaining):
        g = math.exp(-y*y/2) / math.sqrt(2*math.pi)
        total = 0.0
        power = 1.0
        for poly in polys:
            value = 0.0
            for a in reversed(poly):
                value = value*y+a
            total += value*power
            power *= g*remaining
        return g*total

    def rhs(state, remaining):
        y = state[0]
        p, n = effort(y, remaining), effort(-y, remaining)
        return p-n, (p+n)/2, (p*p+n*n)/2

    rows = []
    dt = epsilon/steps
    for j in range(round(7/spacing)+1):
        initial = j*spacing
        state = (initial, 0.0, 0.0)
        for k in range(steps):
            remaining = epsilon-k*dt
            a = rhs(state, remaining)
            b = rhs(tuple(x+dt*v/2 for x, v in zip(state, a)), remaining-dt/2)
            c = rhs(tuple(x+dt*v/2 for x, v in zip(state, b)), remaining-dt/2)
            d = rhs(tuple(x+dt*v for x, v in zip(state, c)), remaining-dt)
            state = tuple(x+dt*(aa+2*bb+2*cc+dd)/6
                          for x, aa, bb, cc, dd in zip(state, a, b, c, d))
        y, average, cost = state
        beta = math.exp(-y*y/2)/math.sqrt(2*math.pi) + y*math.erf(y/math.sqrt(2))/2
        rows.append((initial, average+beta, 1-cost))
    return rows


def zero_effort_rent(exposure):
    """Interior root of r=Phi(-sqrt(2*e*(1-2*r))); diagnostic only."""
    if exposure == 0:
        return .5
    lo, hi = 0.0, .5*(1-1e-10)
    for _ in range(80):
        mid = (lo+hi)/2
        residual = mid-.5*math.erfc(math.sqrt(exposure*(1-2*mid)))
        if residual > 0:
            hi = mid
        else:
            lo = mid
    return (lo+hi)/2


def dual_diagnostic(rows, exposure, nonnegative=False, rent_floor=0.0):
    # A finer local golden search follows a coarse logarithmic dual grid.
    # This remains a diagnostic: it does not certify the state supremum.
    def value(z):
        if nonnegative:
            inner, state = max((f+exposure*(r-rent_floor)/(2*z)-y*y/(8*z)
                                + (z/2 if y <= 2*z else y/2-y*y/(8*z)), y)
                               for y, f, r in rows)
            return inner, state
        inner, state = max((f+exposure*(r-rent_floor)/(2*z)-y*y/(8*z), y)
                           for y, f, r in rows)
        return z/2+inner, state

    zs = [math.exp(-6+j*.025) for j in range(481)]
    i = min(range(len(zs)), key=lambda j: value(zs[j])[0])
    lo, hi = zs[max(0, i-1)], zs[min(len(zs)-1, i+1)]
    ratio = (math.sqrt(5)-1)/2
    for _ in range(40):
        left, right = hi-ratio*(hi-lo), lo+ratio*(hi-lo)
        if value(left)[0] < value(right)[0]:
            hi = right
        else:
            lo = left
    z = (lo+hi)/2
    output, state = value(z)
    return output, z, state


def main():
    tables = {}
    for epsilon in (.3, .4, .5):
        rows = continuation_table(epsilon)
        tables[epsilon] = rows
        check = continuation_table(epsilon, order=26, steps=320, spacing=.1)
        error = max(max(abs(f-rows[2*j][1]), abs(r-rows[2*j][2]))
                    for j, (_, f, r) in enumerate(check))
        print(f"epsilon={epsilon}: centre F={rows[0][1]:.9f}, R={rows[0][2]:.9f}; refinement error={error:.3g}")
    print("Grid diagnostics only; no rigorous supremum or global optimum claim.")
    sampled = []
    for j in range(180):
        exposure = .515*math.exp(j*.023)
        floor = 2*zero_effort_rent(exposure)
        variants = [(dual_diagnostic(rows, exposure-eps, nonnegative=True,
                                    rent_floor=floor)[0], eps)
                    for eps, rows in tables.items()]
        output, eps = min(variants)
        sampled.append((exposure, output, eps))
    for cost in (.03, .05, .1, .2, .3, .4):
        best = (-math.inf, None, None, None)
        for exposure, output, eps in sampled:
            payoff = 3*(1-math.exp(-2*output))-1.5*(cost*exposure)**(2/3)
            if payoff > best[0]:
                best = payoff, exposure, output, eps
        print(f"c={cost}: outer sample maximum={best[0]:.9f}, exposure={best[1]:.5f}, output={best[2]:.6f}, epsilon={best[3]}")


if __name__ == '__main__':
    main()
