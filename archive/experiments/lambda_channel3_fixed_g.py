#!/usr/bin/env python3
"""Isolate Channel 3 (fixed implementation): hold g = theta*kappa FIXED, sweep lambda.

Difference from lambda_y0_sweep.py / the first version of this script: the
filter error is maintained by the exact identity e_t = y_t - tilde{y}_t, i.e.
the (y, tilde) updates share the same Brownian increments (dW, dB):

    y   <- y   + dm*dt + sigma*dW
    til <- til + dm*dt + sigma*sqrt(lambda)*(y-til)*dt + sigma*dB

so that e_{k+1} = (1 - sigma*sqrt(lambda)*dt) e_k + sigma*(dW - dB) exactly on
the grid. In the previous version e was propagated with an *independent*
innovation, which breaks e == y - tilde, inflates the perceived-state noise
variance (asymptotically 3x the model's sigma^2), and makes that inflation
lambda-dependent -- contaminating any lambda-comparison at fixed g.

At fixed g the effort functions are frozen (w = g/(sigma^2 c)). Under the
innovation representation, with a consistent initial prior, the perceived-state
law is exactly lambda-free in continuous time, so scenario D should be flat up
to discretization error and MC noise.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def fn_gamma(u):
    u = np.clip(u, -1.0 + 1e-9, 1.0 - 1e-9)
    return u / (1.0 - u * u) + np.arctanh(u)


def inv_gamma(x):
    return 2.0 / np.pi * np.arctan(0.856 * x)


def kappa_of(lam, sigma, T):
    Sbar = sigma / np.sqrt(lam)
    rho = np.sqrt(sigma * sigma * T / (sigma * sigma * T + Sbar))
    return 2.0 / np.pi * np.arcsin(rho)


def effort_sym(til, t, T, sigma, c, g):
    rem = T - t
    s2 = sigma * sigma
    w = g / (s2 * c)
    gw = fn_gamma(np.tanh(w / 2.0))
    z = til / (sigma * np.sqrt(rem))
    rho_z = inv_gamma(stats.norm.cdf(z) * 2.0 * gw - gw)
    dens = stats.norm.pdf(z, 0.0, 1.0) / (sigma * np.sqrt(rem))
    K = s2 * gw * (1.0 - rho_z * rho_z)
    return dens * K * (1.0 + rho_z), dens * K * (1.0 - rho_z)


def simulate(lam, T, sigma, c, g, dt, N, rng, e0_mode, e0_val=0.0):
    kappa = kappa_of(lam, sigma, T)
    a = sigma * np.sqrt(lam)
    steps = int(round(T / dt))
    t_grid = np.arange(steps) * dt

    if e0_mode == "fixed":
        e0 = np.full(N, e0_val)
    else:  # consistent: draw initial error from the steady-state prior
        e0 = rng.normal(0.0, np.sqrt(sigma / np.sqrt(lam)), N)
    y = e0.copy()          # tilde_0 = 0, so y_0 = e_0
    til = np.zeros(N)
    M_acc = np.zeros(N)
    e_mid = til_mid_sd = None

    for k in range(steps):
        t = t_grid[k]
        m_i, m_j = effort_sym(til, t, T, sigma, c, g)
        dm = m_i - m_j
        dW = rng.normal(0.0, np.sqrt(dt), N)
        dB = rng.normal(0.0, np.sqrt(dt), N)
        e = y - til
        y = y + dm * dt + sigma * dW
        til = til + dm * dt + a * e * dt + sigma * dB
        M_acc = M_acc + (m_i + m_j) * dt
        if e_mid is None and t >= T / 2:
            e_mid = float(np.mean(np.abs(y - til)))
            til_mid_sd = float(np.std(til))

    M = float(np.mean(M_acc))
    sel = 0.5 * float(np.mean(np.abs(y)))
    return kappa, g / kappa, M, sel, 0.5 * M + sel, e_mid, til_mid_sd


def main():
    T, sigma, c, g = 1000.0, 1.0, 1.0, 0.5
    dt, N = 0.1, 4000
    lam_grid = [1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0]
    sd = sigma * np.sqrt(T)
    scenarios = [
        ("A: e0=0 (fixed)", "fixed", 0.0),
        ("B: e0=+0.5 sd (fixed)", "fixed", 0.5 * sd),
        ("C: e0=+1.0 sd (fixed)", "fixed", 1.0 * sd),
        ("D: consistent e0~N(0,Sbar)", "consistent", 0.0),
    ]

    rng = np.random.default_rng(20260919)
    print(f"fixed g={g} (w={g/(sigma*sigma*c)}), sigma={sigma}, T={T}, dt={dt}, "
          f"symmetric c={c}; identity-consistent filter update")
    for name, mode, val in scenarios:
        print(f"\n=== {name} ===")
        print(f"{'lambda':>9} {'kappa':>7} {'theta=g/k':>9} {'M':>8} {'E|yT|/2':>9} "
              f"{'B':>8} {'|e|@T/2':>8} {'sd(til)@T/2':>11}")
        for lam in lam_grid:
            r = simulate(lam, T, sigma, c, g, dt, N, rng, mode, val)
            print(f"{lam:9.0e} {r[0]:7.4f} {r[1]:9.4f} {r[2]:8.4f} {r[3]:9.4f} "
                  f"{r[4]:8.4f} {r[5]:8.4f} {r[6]:11.4f}")


if __name__ == "__main__":
    main()
