"""Counterexample search at the fixed reference; NOT a proof certificate.

Rescaling sigma=cost=1, S0=1/L, T=x, theta=w, lambda=ell preserves
the normalized HJB and output ratio at Section 27's reference.
Finite differences neither bound true output nor optimize all designs.
"""
import math

from pipeline.synthetic.nonlinear import Primitives
from pipeline.theory.joint_design_exact import (
    expected_best_output,
    solve_equilibrium_adaptive,
)


def audit():
    length = 2 * math.pi * 251**2 - 1
    for precision in (22, 40, 100, 400):
        for incentive in (0.08, 0.16, 0.24):
            p = Primitives(T=1, theta=incentive, c_i=1, c_j=1,
                           sigma=1, lamb=precision, S0=1 / length)
            eq = solve_equilibrium_adaptive(p, dy=0.01, base_steps=1500)
            ratio = expected_best_output(eq) / math.sqrt(
                (1 + 1 / length) / (2 * math.pi)
            )
            print(f"ell={precision:3d}, w={incentive:.2f}, B/b={ratio:.10f}, "
                  f"min raw effort={eq.diagnostics['min_raw_policy']:.3g}")
    print("Diagnostic only: no global payoff or existence certification.")


if __name__ == "__main__":
    audit()
