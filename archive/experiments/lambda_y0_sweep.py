#!/usr/bin/env python3
"""Sweep B(y0, theta, T, lambda) = M(tilde_y0=0, 0)/2 + E|y_T|/2 over lambda and y0.

Simulates the closed-loop equilibrium with the rank-alignment multiplier
kappa(lambda, sigma, T) (Kendall-tau form), asymmetric effort costs, and the
Kalman tracking of the true gap:
    dy   = (m_i - m_j) dt + sigma dW,
    dtil = (m_i - m_j) dt + sigma*sqrt(lambda)*(y - til) dt + sigma dB,
with til_0 = 0 and y_0 on a grid. Reports M, E|y_T|/2 and B for symmetric and
asymmetric cost pairs, so we can see whether B has a finite interior maximum
in lambda (the "noise sustains the weak player" channel).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Equilibrium effort (Ryvkin closed form, prize spread theta*kappa)
# ---------------------------------------------------------------------------


def fn_gamma(u):
    u = np.clip(u, -1.0 + 1e-9, 1.0 - 1e-9)
    return u / (1.0 - u * u) + np.arctanh(u)


def inv_gamma(x):
    return 2.0 / np.pi * np.arctan(0.856 * x)


def rho_z_fn(z, rho_i, rho_j):
    gi, gj = fn_gamma(rho_i), fn_gamma(rho_j)
    loc = stats.norm.cdf(z) * (gi + gj) - gj
    return inv_gamma(loc)


def kappa_of(lam, sigma, T):
    Sbar = sigma / np.sqrt(lam)
    rho = np.sqrt(sigma * sigma * T / (sigma * sigma * T + Sbar))
    return 2.0 / np.pi * np.arcsin(rho)


def effort(til, t, T, theta, sigma, c_i, c_j, kappa):
    rem = T - t
    s2 = sigma * sigma
    w_i = theta * kappa / (s2 * c_i)
    w_j = theta * kappa / (s2 * c_j)
    rho_i = (np.expm1(w_i) + np.expm1(-w_j)) / (np.expm1(w_i) - np.expm1(-w_j))
    rho_j = (np.expm1(w_j) + np.expm1(-w_i)) / (np.expm1(w_j) - np.expm1(-w_i))
    z = til / (sigma * np.sqrt(rem))
    rho_z = rho_z_fn(z, rho_i, rho_j)
    dens = stats.norm.pdf(z, 0.0, 1.0) / (sigma * np.sqrt(rem))
    K = s2 / 2.0 * (fn_gamma(rho_i) + fn_gamma(rho_j)) * (1.0 - rho_z * rho_z)
    m_i = dens * K * (1.0 + rho_z)
    m_j = dens * K * (1.0 - rho_z)
    return m_i, m_j


def simulate(lam, y0, T, theta, sigma, c_i, c_j, dt, N, rng):
    kappa = kappa_of(lam, sigma, T)
    a = sigma * np.sqrt(lam)
    steps = int(T / dt)
    t_grid = np.arange(steps) * dt

    # state arrays
    y = np.full(N, y0)
    til = np.zeros(N)
    e = np.full(N, y0 - 0.0)
    M_acc = np.zeros(N)

    ou_scale = sigma * np.sqrt((1.0 - np.exp(-2.0 * a * dt)) / a) if a > 0 else sigma * np.sqrt(2.0 * dt)

    for k in range(steps):
        t = t_grid[k]
        m_i, m_j = effort(til, t, T, theta, sigma, c_i, c_j, kappa)
        dm = m_i - m_j
        dW = rng.normal(0.0, np.sqrt(dt), N)
        dB = rng.normal(0.0, np.sqrt(dt), N)
        y = y + dm * dt + sigma * dW
        til = til + dm * dt + a * e * dt + sigma * dB
        e = e * np.exp(-a * dt) + ou_scale * rng.normal(0.0, 1.0, N)
        M_acc = M_acc + (m_i + m_j) * dt

    M = float(np.mean(M_acc))
    sel = 0.5 * float(np.mean(np.abs(y)))
    B = 0.5 * M + sel
    return kappa, M, sel, B


def main():
    T, theta, sigma = 1000.0, 1.0, 1.0
    dt, N = 0.5, 4000
    lam_grid = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]
    y0_grid = [0.0, 0.5 * sigma * np.sqrt(T), 1.0 * sigma * np.sqrt(T)]
    pairs = {"symmetric c=(1,1)": (1.0, 1.0),
             "asymmetric c=(1,2)": (1.0, 2.0),
             "asymmetric c=(1,4)": (1.0, 4.0)}

    rng = np.random.default_rng(20260918)
    for name, (c_i, c_j) in pairs.items():
        print(f"\n=== {name} ===")
        print(f"{'lambda':>9} {'kappa':>7} | " + " | ".join(f"y0={y0:8.2f}: B" for y0 in y0_grid))
        for lam in lam_grid:
            row = []
            for y0 in y0_grid:
                kappa, M, sel, B = simulate(lam, y0, T, theta, sigma, c_i, c_j, dt, N, rng)
                row.append((kappa, M, sel, B))
            Bs = [r[3] for r in row]
            kappa = row[0][0]
            cells = " | ".join(f"{B:16.4f}" for B in Bs)
            print(f"{lam:9.0e} {kappa:7.4f} | {cells}")
            # flag interior maxima
        for j, y0 in enumerate(y0_grid):
            Bs = [simulate(lam, y0, T, theta, sigma, c_i, c_j, dt, N, rng)[3] for lam in lam_grid]
            imax = int(np.argmax(Bs))
            print(f"  y0={y0:8.2f}: max B at lambda={lam_grid[imax]:.0e} (B={Bs[imax]:.4f})  B(1e-6)={Bs[0]:.4f}  B(100)={Bs[-1]:.4f}")

    # component breakdown for the asymmetric c=(1,4) case
    print("\n=== decomposition, c=(1,4), y0=0 ===")
    print(f"{'lambda':>9} {'kappa':>7} {'M':>10} {'E|yT|/2':>10} {'B':>10}")
    for lam in lam_grid:
        kappa, M, sel, B = simulate(lam, 0.0, T, theta, sigma, 1.0, 4.0, dt, N, rng)
        print(f"{lam:9.0e} {kappa:7.4f} {M:10.4f} {sel:10.4f} {B:10.4f}")


if __name__ == "__main__":
    main()
