"""Exact-HJB output calculations for the joint-design analysis.

The equilibrium HJB is solved by ``pipeline.synthetic.nonlinear``.  Conditional
on its policies, expected best output follows from the exact drift identity in
the paper and a second, linear backward equation.  The routines here are for
theory diagnostics; they do not alter the synthetic-data pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded
from scipy.special import ndtr

from pipeline.synthetic.nonlinear import Equilibrium, Primitives, solve_equilibrium, variance


def _gradient(values: np.ndarray, spacing: float) -> np.ndarray:
    return np.gradient(values, spacing, axis=0, edge_order=2)


def solve_equilibrium_monotone(
    p: Primitives, *, dy: float, dt: float, extent: float = 6.0
) -> Equilibrium:
    """Solve the HJB with local Lax--Friedrichs numerical viscosity.

    The production solver is more accurate in smooth regions.  This monotone
    variant is used to audit strong-incentive, low-precision cases in which the
    nearly first-order HJB develops steep gradients.
    """
    steps = int(np.ceil(p.T / dt))
    time = np.linspace(0.0, p.T, steps + 1)
    dt = p.T / steps
    width = np.ceil(extent * np.sqrt(p.S0 + p.sigma**2 * p.T) / dy) * dy
    y = np.arange(-width, width + dy / 2, dy)
    filtering_variance = variance(time, p)
    values = np.column_stack((
        ndtr(y / np.sqrt(filtering_variance[-1])),
        ndtr(-y / np.sqrt(filtering_variance[-1])),
    ))
    policies = np.empty((steps + 1, len(y), 2), dtype=np.float32)
    band = np.zeros((3, len(y) - 2))
    minimum_policy = 0.0

    def nonlinear(candidate: np.ndarray) -> tuple[np.ndarray, float]:
        gradient = _gradient(candidate, dy)
        cross = gradient[:, 0] * gradient[:, 1]
        term = p.theta * np.column_stack((
            gradient[:, 0] ** 2 / (2 * p.c_i) + cross / p.c_j,
            gradient[:, 1] ** 2 / (2 * p.c_j) + cross / p.c_i,
        ))
        speed = p.theta * max(1 / p.c_i, 1 / p.c_j) * float(
            np.max(np.abs(gradient[:, 0]) + np.abs(gradient[:, 1]))
        )
        return term, speed

    for index in range(steps, -1, -1):
        gradient = _gradient(values, dy)
        effort = gradient * np.array([p.theta / p.c_i, -p.theta / p.c_j])
        minimum_policy = min(minimum_policy, float(effort.min()))
        policies[index] = np.maximum(effort, 0.0)
        if index == 0:
            break

        middle = (time[index] + time[index - 1]) / 2
        physical_diffusion = p.lamb * float(variance(middle, p)) ** 2
        first_term, speed = nonlinear(values)
        effective_diffusion = physical_diffusion + speed * dy
        ratio = dt * effective_diffusion / (4 * dy * dy)
        band[0, 1:] = -ratio
        band[1] = 1 + 2 * ratio
        band[2, :-1] = -ratio
        base = values[1:-1] + ratio * (
            values[:-2] - 2 * values[1:-1] + values[2:]
        )
        predictor = values.copy()
        predictor[1:-1] = solve_banded(
            (1, 1), band, base + dt * first_term[1:-1], check_finite=False
        )
        second_term, _ = nonlinear(predictor)
        values[1:-1] = solve_banded(
            (1, 1),
            band,
            base + dt * (first_term[1:-1] + second_term[1:-1]) / 2,
            check_finite=False,
        )

    diagnostics = dict(
        dy=dy,
        dt=dt,
        extent=extent,
        nodes_y=len(y),
        nodes_t=steps + 1,
        min_raw_policy=minimum_policy,
        max_policy=float(policies.max()),
        value_min=float(values.min()),
        value_max=float(values.max()),
        scheme="local Lax--Friedrichs IMEX",
    )
    return Equilibrium(p, time, y, policies, diagnostics)


def solve_equilibrium_adaptive(
    p: Primitives, *, dy: float, base_steps: int = 1000, extent: float = 7.0
) -> Equilibrium:
    """Solve the HJB on a time grid resolving the initial filtering layer.

    When precision is large, ``S`` moves from ``S0`` to its steady-state scale
    over time of order ``1 / (lambda * S0)``.  A uniform grid can miss that
    layer even though the remainder of the HJB is smooth.  The union grid below
    is uniform away from zero and geometric inside the layer.
    """
    uniform = np.linspace(0.0, p.T, base_steps + 1)
    rate = p.lamb * p.S0
    if rate > 0:
        layer_end = min(p.T, 30.0 / rate)
        smallest = min(layer_end / 5000.0, p.T / (base_steps * 100.0))
        if smallest > 0 and layer_end > smallest:
            layer = np.r_[0.0, np.geomspace(smallest, layer_end, 240)]
            time = np.unique(np.r_[uniform, layer])
        else:
            time = uniform
    else:
        time = uniform

    width = np.ceil(extent * np.sqrt(p.S0 + p.sigma**2 * p.T) / dy) * dy
    y = np.arange(-width, width + dy / 2, dy)
    filtering_variance = variance(time, p)
    values = np.column_stack((
        ndtr(y / np.sqrt(filtering_variance[-1])),
        ndtr(-y / np.sqrt(filtering_variance[-1])),
    ))
    policies = np.empty((len(time), len(y), 2), dtype=np.float32)
    band = np.zeros((3, len(y) - 2))
    minimum_policy = 0.0

    def nonlinear(candidate: np.ndarray) -> np.ndarray:
        gradient = _gradient(candidate, dy)
        cross = gradient[:, 0] * gradient[:, 1]
        return p.theta * np.column_stack((
            gradient[:, 0] ** 2 / (2 * p.c_i) + cross / p.c_j,
            gradient[:, 1] ** 2 / (2 * p.c_j) + cross / p.c_i,
        ))

    for index in range(len(time) - 1, -1, -1):
        gradient = _gradient(values, dy)
        effort = gradient * np.array([p.theta / p.c_i, -p.theta / p.c_j])
        minimum_policy = min(minimum_policy, float(effort.min()))
        if minimum_policy < -1e-7:
            raise RuntimeError(f"Non-monotone value gradient: {minimum_policy}")
        policies[index] = np.maximum(effort, 0.0)
        if index == 0:
            break

        step = float(time[index] - time[index - 1])
        middle = (float(time[index]) + float(time[index - 1])) / 2
        diffusion = p.lamb * float(variance(middle, p)) ** 2
        ratio = step * diffusion / (4 * dy * dy)
        band[0, 1:] = -ratio
        band[1] = 1 + 2 * ratio
        band[2, :-1] = -ratio
        base = values[1:-1] + ratio * (
            values[:-2] - 2 * values[1:-1] + values[2:]
        )
        base[0] += ratio * values[0]
        base[-1] += ratio * values[-1]
        first = nonlinear(values)
        predictor = values.copy()
        predictor[1:-1] = solve_banded(
            (1, 1), band, base + step * first[1:-1], check_finite=False
        )
        second = nonlinear(predictor)
        values[1:-1] = solve_banded(
            (1, 1), band, base + step * (first[1:-1] + second[1:-1]) / 2,
            check_finite=False,
        )

    diagnostics = dict(
        dy=dy,
        base_steps=base_steps,
        extent=extent,
        nodes_y=len(y),
        nodes_t=len(time),
        min_raw_policy=minimum_policy,
        max_policy=float(policies.max()),
        value_min=float(values.min()),
        value_max=float(values.max()),
        scheme="adaptive-time Crank--Nicolson/Heun",
    )
    return Equilibrium(p, time, y, policies, diagnostics)


def expected_best_output(eq: Equilibrium) -> float:
    """Return exact expected best output at the equilibrium's initial mean.

    This solves the Feynman--Kac equation associated with equation
    ``eq-best-output-drift-identity`` in the manuscript.  Boundary values are
    zero for the effort contribution; the state grid is already chosen far
    enough into the Gaussian tails by the equilibrium solver.
    """
    p = eq.p
    y, time = eq.y.astype(float), eq.t.astype(float)
    dy = float(y[1] - y[0])
    contribution = np.zeros_like(y)
    band = np.zeros((3, len(y) - 2))

    def flow(values: np.ndarray, policies: np.ndarray, q_value: float) -> np.ndarray:
        mi, mj = policies[:, 0], policies[:, 1]
        drift = mi - mj
        reward = (mi + mj) / 2 + drift * (ndtr(y / math.sqrt(q_value)) - 0.5)
        return drift * _gradient(values, dy) + reward

    for k in range(len(time) - 1, 0, -1):
        step = float(time[k] - time[k - 1])
        middle = (float(time[k]) + float(time[k - 1])) / 2
        s_middle = float(variance(middle, p))
        diffusion = p.lamb * s_middle * s_middle
        ratio = step * diffusion / (4 * dy * dy)
        band[0, 1:] = -ratio
        band[1] = 1 + 2 * ratio
        band[2, :-1] = -ratio

        policies = (eq.effort[k].astype(float) + eq.effort[k - 1].astype(float)) / 2
        q_middle = s_middle + p.sigma * p.sigma * (p.T - middle)
        base = contribution[1:-1] + ratio * (
            contribution[:-2] - 2 * contribution[1:-1] + contribution[2:]
        )
        current_flow = flow(contribution, policies, q_middle)
        predictor = contribution.copy()
        predictor[1:-1] = solve_banded(
            (1, 1), band, base + step * current_flow[1:-1], check_finite=False
        )
        corrected_flow = flow(predictor, policies, q_middle)
        contribution[1:-1] = solve_banded(
            (1, 1),
            band,
            base + step * (current_flow[1:-1] + corrected_flow[1:-1]) / 2,
            check_finite=False,
        )

    q0 = p.S0 + p.sigma * p.sigma * p.T
    baseline = math.sqrt(q0 / (2 * math.pi)) * math.exp(-p.mu0 * p.mu0 / (2 * q0))
    baseline += p.mu0 * (ndtr(p.mu0 / math.sqrt(q0)) - 0.5)
    return baseline + float(np.interp(p.mu0, y, contribution))


@dataclass(frozen=True)
class OutputResult:
    output: float
    baseline: float
    effort_gain: float
    max_effort: float
    nodes_y: int
    nodes_t: int


def full_disclosure_output(
    *,
    theta: float,
    horizon: float,
    initial_variance: float,
    sigma: float = 1.0,
    cost: float = 1.0,
    dy: float | None = None,
    dt: float | None = None,
    extent: float = 7.0,
) -> float:
    """Expected best output in the exact full-disclosure limit.

    The limiting symmetric value functions are characterized by equation
    ``eq-design-value-limit`` in the manuscript.  Their derivatives give the
    limiting policies, after which a linear backward equation evaluates the
    organizer's terminal-best-output criterion.
    """
    scale = math.sqrt(initial_variance + sigma * sigma * horizon)
    if dy is None:
        dy = 0.02 * scale
    if dt is None:
        dt = horizon / 800
    steps = int(np.ceil(horizon / dt))
    time = np.linspace(0.0, horizon, steps + 1)
    dt = horizon / steps
    width = np.ceil(extent * scale / dy) * dy
    y = np.arange(-width, width + dy / 2, dy)
    value = np.abs(y) / 2
    band = np.zeros((3, len(y) - 2))
    alpha = theta / cost
    w = alpha / (sigma * sigma)

    if w == 0:
        k_one = 1.0
        d_grid = np.linspace(-1.0, 1.0, 4001)
        k_grid = d_grid
    else:
        d_grid = np.linspace(-1.0, 1.0, 8001)
        k_grid = (d_grid + np.sinh(w * d_grid) / w) / 2
        k_one = (1 + math.sinh(w) / w) / 2

    for index in range(steps, 0, -1):
        middle = (time[index] + time[index - 1]) / 2
        remaining = horizon - middle
        standardized = y / (sigma * math.sqrt(remaining))
        target = k_one * (2 * ndtr(standardized) - 1)
        delta = np.interp(target, k_grid, d_grid)
        k_prime = (1 + np.cosh(w * delta)) / 2 if w else np.ones_like(delta)
        delta_y = (
            k_one * 2 * np.exp(-standardized * standardized / 2)
            / math.sqrt(2 * math.pi) / (sigma * math.sqrt(remaining)) / k_prime
        )
        f_prime = np.tanh(w * delta / 2) if w else np.zeros_like(delta)
        drift = alpha * f_prime * delta_y
        reward = alpha * delta_y / 2

        ratio = dt * sigma * sigma / (4 * dy * dy)
        band[0, 1:] = -ratio
        band[1] = 1 + 2 * ratio
        band[2, :-1] = -ratio
        base = value[1:-1] + ratio * (value[:-2] - 2 * value[1:-1] + value[2:])

        def flow(candidate: np.ndarray) -> np.ndarray:
            return drift * _gradient(candidate, dy) + reward

        first = flow(value)
        predictor = value.copy()
        predictor[1:-1] = solve_banded(
            (1, 1), band, base + dt * first[1:-1], check_finite=False
        )
        second = flow(predictor)
        value[1:-1] = solve_banded(
            (1, 1), band, base + dt * (first[1:-1] + second[1:-1]) / 2,
            check_finite=False,
        )
        value[0], value[-1] = abs(y[0]) / 2, abs(y[-1]) / 2

    if initial_variance == 0:
        return float(np.interp(0.0, y, value))
    density = np.exp(-y * y / (2 * initial_variance)) / math.sqrt(
        2 * math.pi * initial_variance
    )
    return float(np.trapezoid(value * density, y))


def solve_output(
    *,
    theta: float,
    horizon: float,
    precision: float,
    initial_variance: float,
    sigma: float = 1.0,
    cost: float = 1.0,
    mu0: float = 0.0,
    dy: float | None = None,
    dt: float | None = None,
    extent: float = 6.0,
    monotone: bool = False,
) -> OutputResult:
    """Solve the symmetric exact equilibrium and evaluate expected best output."""
    if precision <= 0:
        raise ValueError("precision must be positive; use a small positive limit value")
    scale = math.sqrt(initial_variance + sigma * sigma * horizon)
    terminal_scale = math.sqrt(float(variance(horizon, Primitives(
        T=horizon, theta=theta, c_i=cost, c_j=cost, sigma=sigma,
        lamb=precision, S0=initial_variance, mu0=mu0,
    ))))
    if dy is None:
        dy = min(0.035 * scale, 0.20 * terminal_scale)
    if dt is None:
        dt = horizon / 400
    primitives = Primitives(
        T=horizon,
        theta=theta,
        c_i=cost,
        c_j=cost,
        sigma=sigma,
        lamb=precision,
        S0=initial_variance,
        mu0=mu0,
    )
    equilibrium = (
        solve_equilibrium_monotone(primitives, dy=dy, dt=dt, extent=extent)
        if monotone else
        solve_equilibrium(primitives, dy=dy, dt=dt, extent=extent)
    )
    output = expected_best_output(equilibrium)
    q0 = initial_variance + sigma * sigma * horizon
    baseline = math.sqrt(q0 / (2 * math.pi)) * math.exp(-mu0 * mu0 / (2 * q0))
    baseline += mu0 * (ndtr(mu0 / math.sqrt(q0)) - 0.5)
    return OutputResult(
        output=output,
        baseline=baseline,
        effort_gain=output - baseline,
        max_effort=float(equilibrium.effort.max()),
        nodes_y=len(equilibrium.y),
        nodes_t=len(equilibrium.t),
    )


if __name__ == "__main__":
    for precision in np.logspace(-3, 3, 13):
        result = solve_output(
            theta=1.0,
            horizon=1.0,
            precision=float(precision),
            initial_variance=1.0,
        )
        print(f"{precision:10.4g}  {result.output:.10f}  {result.effort_gain:.10f}")
