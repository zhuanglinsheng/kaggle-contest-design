#!/usr/bin/env python3
"""Design preview v2: finite optimal precision without precision cost?

Same setup as v1 (true-measure objective, Theorem-1 efforts, contestants' prior
centered, organizer conditions on hidden true initial gap y0), but the expected
effort rate is computed with a numerically exact representation for ALL w:

    E[m_i+m_j](t) = 2 sigma^2 phi(mu_t; 0, srem^2+V_t) * I_t,
    I_t = E_z[1 - rho(z)^2],  z ~ N(nu_t, tau_t^2),
    nu_t = mu_t srem/(srem^2+V_t),  tau_t^2 = V_t/(srem^2+V_t),  srem = sigma sqrt(T-t),

and, substituting v = rho(z), s = atanh(v) (so s in (-w/2, w/2)),

    I_t = (1/gw) ∫_{-w/2}^{w/2} [phi_state(z_s) / phi_std(z_s)] ds,
    z_s = Phi^{-1}( (1 + gamma(tanh s)/gw)/2 ),  gamma(tanh s) = sinh(2s)/2 + s,
    gw = (sinh w + w)/2.

The gw factor cancels, so the formula is stable both for small w (leading order
recovered) and large w (spike width ~e^{-w} would break naive z-quadrature).

Designer: max over (theta >= 0, lambda > 0) of  A*log(B) - theta,  T fixed.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar

SIGMA = 1.0
COST = 1.0
T = 1.0

T_GRID = np.linspace(0.0, 1.0, 402)[1:-1] * T     # interior time points
N_S = 2001                                        # s-quadrature points (odd)


def kappa_of(lam, sigma, T):
    return 2.0 / np.pi * np.arctan(np.sqrt(sigma * T * np.sqrt(lam)))


def state_law(y0, lam, sigma, t):
    """True-measure mean and variance of tilde y_t (effort drift neglected)."""
    if np.isinf(lam):                              # instant tracking
        return np.full_like(t, y0), sigma**2 * t
    a = sigma * np.sqrt(lam)
    if a < 1e-10:
        return np.zeros_like(t), sigma**2 * t
    mu = y0 * (1.0 - np.exp(-a * t))
    V = sigma**2 * (t - 2.0 * (1.0 - np.exp(-a * t)) / a
                    + (1.0 - np.exp(-2.0 * a * t)) / a)
    return mu, V


def effort_rate(w, mu_t, V_t, sigma, rem):
    """E[m_i + m_j] on the t-grid via the s-space representation."""
    if w < 1e-12:
        return np.zeros_like(rem)
    rw = np.tanh(w / 2.0)
    gw = (np.sinh(min(w, 600.0)) + w) / 2.0
    s = np.linspace(-w / 2.0, w / 2.0, N_S)
    gamv = 0.5 * np.sinh(2.0 * s) + s              # gamma(tanh s)
    arg = np.clip(0.5 * (1.0 + gamv / gw), 1e-16, 1.0 - 1e-16)
    z_s = stats.norm.ppf(arg)                      # (N_S,)
    log_phi_std = stats.norm.logpdf(z_s)

    srem2 = sigma**2 * rem                         # kernel variance
    tot = srem2 + V_t
    nu = mu_t * np.sqrt(srem2) / tot               # mean of z = ytilde/srem
    tau2 = V_t / tot                               # var of z (< 1)
    tau = np.sqrt(tau2)
    # log ratio: log phi_state(z) - log phi_std(z), phi_state = N(nu, tau2)
    log_ratio = (stats.norm.logpdf(z_s[None, :], nu[:, None], tau[:, None])
                 - log_phi_std[None, :])
    ratio = np.exp(np.clip(log_ratio, -745, 50))
    I_t = np.trapezoid(ratio, s, axis=1) / gw
    phi_mu = stats.norm.pdf(mu_t, 0.0, np.sqrt(tot))
    return 2.0 * sigma**2 * gw * phi_mu * I_t


def B_of(y0, theta, lam, sigma=SIGMA, c=COST, T=T):
    t = T_GRID
    rem = T - t
    kap = 1.0 if np.isinf(lam) else kappa_of(lam, sigma, T)
    w = theta * kap / (sigma**2 * c)
    mu_t, V_t = state_law(y0, lam, sigma, t)
    M = np.trapezoid(effort_rate(w, mu_t, V_t, sigma, rem), t)
    vT = sigma**2 * T
    EyT = (abs(y0) * (1.0 - 2.0 * stats.norm.cdf(-abs(y0) / np.sqrt(vT)))
           + np.sqrt(2.0 * vT / np.pi) * np.exp(-y0**2 / (2.0 * vT)))
    return 0.5 * M + 0.5 * EyT


def optimal_value(y0, lam, A):
    def neg(theta):
        return -(A * np.log(B_of(y0, theta, lam)) - theta)
    res = minimize_scalar(neg, bounds=(0.0, 300.0), method="bounded",
                          options={"xatol": 1e-4})
    return -res.fun, res.x


def sanity_check():
    """Leading-order benchmark: at y0=0, lambda=inf, small w, M should equal
    2 w sigma sqrt(T) / sqrt(2 pi)."""
    for w_target in [0.01, 0.05, 0.2]:
        theta = w_target / 1.0                     # kappa = 1 at lam = inf
        t = T_GRID
        mu_t, V_t = state_law(0.0, np.inf, SIGMA, t)
        M = np.trapezoid(effort_rate(theta, mu_t, V_t, SIGMA, T - t), t)
        lo = 2.0 * w_target * SIGMA * np.sqrt(T) / np.sqrt(2.0 * np.pi)
        print(f"  w={w_target:5.2f}: M={M:.6f}, leading-order={lo:.6f}, "
              f"ratio={M/lo:.4f}")


def main():
    print("sanity check (y0=0, lam=inf):")
    sanity_check()

    lam_grid = np.concatenate(([1e-4, 1e-3], np.logspace(-2, 3, 18)))
    y0_grid = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    for A in [5.0, 20.0]:
        print(f"\nu(B) = {A}*log(B) - theta;  sigma={SIGMA}, c={COST}, T={T} (T fixed)")
        print(f"{'y0':>5} {'lam*':>10} {'theta*':>8} {'V*':>8} | "
              f"{'V(lam=inf)':>10} {'theta_inf':>9} | {'V(lam->0)':>9} | regime")
        for y0 in y0_grid:
            vals = [optimal_value(y0, lam, A) + (lam,) for lam in lam_grid]
            v_inf, th_inf = optimal_value(y0, np.inf, A)
            v_zero, _ = optimal_value(y0, 1e-12, A)
            v_best, th_best, lam_best = max(vals, key=lambda x: x[0])
            if th_best < 1e-3 and v_best <= v_inf + 1e-6:
                regime = "sampling/irrelevant"
            elif v_best > v_inf + 1e-6:
                regime = "FINITE lambda*"
            else:
                regime = "lambda*=inf"
            print(f"{y0:5.1f} {lam_best:10.2e} {th_best:8.3f} {v_best:8.3f} | "
                  f"{v_inf:10.3f} {th_inf:9.3f} | {v_zero:9.3f} | {regime}")


if __name__ == "__main__":
    main()
