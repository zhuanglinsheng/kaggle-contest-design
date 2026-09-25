"""Synthetic sensitivity check: estimate r jointly with structural parameters.

This script mirrors the likelihood in synthetic_data.stan for the baseline
90-day synthetic contest, but treats the effort-to-submission scale r as an
unknown parameter. It is intended as a lightweight robustness diagnostic for
the paper, not as a replacement for the CmdStan workflow.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import linalg, optimize


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "synthetic_data" / "revised_baseline_data.json"
OUT_PATH = ROOT / "synthetic_data" / "revised_baseline_estimate_r.csv"


PARAMS = ("c_i", "c_j", "sigma", "lambda", "mu_0", "r")
TRUE_VALUES = np.array([1.2, 1.5, 2.0, 1.0, 0.0, 15.0])
LOWER = np.array([0.1, 0.1, 0.5, 1e-6, -20.0, 1e-6])
UPPER = np.array([5.0, 5.0, 10.0, 100.0, 20.0, 1000.0])
PRIOR_MEAN = np.array([0.5, 0.5, 1.0, 1.0, 0.0, 15.0])
PRIOR_SD = np.array([5.0, 5.0, 5.0, 5.0, 1.0, 5.0])


def inv_logit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def to_constrained(z: np.ndarray) -> np.ndarray:
    return LOWER + (UPPER - LOWER) * inv_logit(z)


def log_jacobian(z: np.ndarray) -> float:
    s = inv_logit(z)
    return float(np.sum(np.log(UPPER - LOWER) + np.log(s) + np.log1p(-s)))


def efforts(
    y: float,
    t: float,
    T: float,
    theta: float,
    sigma: float,
    c_i: float,
    c_j: float,
    lamb: float,
) -> tuple[float, float]:
    """Leading-order effort rule used in PaperJK7."""
    if T <= t:
        return 0.0, 0.0
    bar_s = sigma / math.sqrt(lamb)
    q_t = sigma * sigma * (T - t) + bar_s
    kernel = math.exp(-0.5 * y * y / q_t) / math.sqrt(2.0 * math.pi * q_t)
    return theta * kernel / c_i, theta * kernel / c_j


class SyntheticLikelihood:
    def __init__(self, data: dict):
        self.theta = float(data["theta"])
        self.N_delta = int(data["N_Delta"])
        self.delta = float(data["Delta2f"])
        self.T = self.N_delta * self.delta
        self.hat_y = np.asarray(data["hat_y"], dtype=float)
        self.Ni = int(data["Ni"])
        self.Nj = int(data["Nj"])
        self.idx_i = np.ceil(np.asarray(data["hat_t_i"], dtype=float)).astype(int)
        self.idx_j = np.ceil(np.asarray(data["hat_t_j"], dtype=float)).astype(int)
        self.events_idx = np.unique(np.sort(np.concatenate([self.idx_i, self.idx_j]))) + 1
        n = len(self.events_idx)
        self.hat_y_events = self.hat_y[self.events_idx - 1]
        t_events = self.events_idx - 1
        self.obs_h = np.diff(np.concatenate(([0.0], t_events * self.delta)))
        self.unit_cov_y = np.minimum.outer(t_events, t_events) * self.delta

    def paths(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        c_i, c_j, sigma, lamb, mu_0, _r = x
        m_i = np.empty(self.N_delta)
        m_j = np.empty(self.N_delta)
        tilde = np.empty(self.N_delta + 1)
        tilde[0] = mu_0
        kalman_weight = -math.expm1(-math.sqrt(lamb) * sigma * self.delta)
        for i in range(self.N_delta):
            mi, mj = efforts(tilde[i], i * self.delta, self.T, self.theta, sigma, c_i, c_j, lamb)
            m_i[i] = mi
            m_j[i] = mj
            tilde[i + 1] = (
                tilde[i]
                + (mi - mj) * self.delta
                + kalman_weight * (self.hat_y[i] - tilde[i])
            )
        return m_i, m_j, tilde

    def logpost_x(self, x: np.ndarray) -> float:
        if np.any(x <= LOWER) or np.any(x >= UPPER):
            return -math.inf
        c_i, c_j, sigma, lamb, mu_0, r = x
        m_i, m_j, _tilde = self.paths(x)

        intensity_i = r * m_i / 24.0
        intensity_j = r * m_j / 24.0
        if np.any(intensity_i[self.idx_i - 1] <= 0.0) or np.any(intensity_j[self.idx_j - 1] <= 0.0):
            return -math.inf

        ll = 0.0
        if self.Ni > 1:
            ll += float(np.sum(np.log(intensity_i[self.idx_i - 1])) - np.sum(intensity_i))
        if self.Nj > 1:
            ll += float(np.sum(np.log(intensity_j[self.idx_j - 1])) - np.sum(intensity_j))

        effort_gap = m_i - m_j
        csum_gap = np.cumsum(effort_gap)
        mean_y = mu_0 + csum_gap[self.events_idx - 2] * self.delta
        bar_s = sigma / math.sqrt(lamb)
        cov = (
            bar_s * np.ones_like(self.unit_cov_y)
            + sigma * sigma * self.unit_cov_y
            + np.diag(1.0 / (lamb * self.obs_h))
        )
        try:
            cho = linalg.cho_factor(cov, lower=True, check_finite=False)
            resid = self.hat_y_events - mean_y
            quad = float(resid @ linalg.cho_solve(cho, resid, check_finite=False))
            logdet = 2.0 * float(np.sum(np.log(np.diag(cho[0]))))
        except linalg.LinAlgError:
            return -math.inf
        n = len(self.events_idx)
        ll += -0.5 * (n * math.log(2.0 * math.pi) + logdet + quad)

        lp = -0.5 * float(np.sum(((x - PRIOR_MEAN) / PRIOR_SD) ** 2))
        return ll + lp

    def logpost_z(self, z: np.ndarray) -> float:
        x = to_constrained(z)
        return self.logpost_x(x) + log_jacobian(z)


def summarize(draws: np.ndarray) -> list[list[str]]:
    rows = [["Name", "True Val.", "Posterior Mean", "Posterior Std.", "RMSE", "2.5% Quantile", "97.5% Quantile"]]
    for j, name in enumerate(PARAMS):
        vals = draws[:, j]
        mean = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1))
        rmse = math.sqrt((mean - TRUE_VALUES[j]) ** 2 + sd * sd)
        lo, hi = np.quantile(vals, [0.025, 0.975])
        rows.append([name, *[f"{v:.6g}" for v in [TRUE_VALUES[j], mean, sd, rmse, lo, hi]]])
    return rows


def write_csv(rows: list[list[str]], path: Path) -> None:
    path.write_text("\n".join(",".join(row) for row in rows) + "\n")


def hessian(f, z: np.ndarray, step: float = 2e-3) -> np.ndarray:
    n = len(z)
    H = np.empty((n, n))
    f0 = f(z)
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = step
        H[i, i] = (f(z + ei) - 2.0 * f0 + f(z - ei)) / (step * step)
        for j in range(i + 1, n):
            ej = np.zeros(n)
            ej[j] = step
            H[i, j] = (
                f(z + ei + ej)
                - f(z + ei - ej)
                - f(z - ei + ej)
                + f(z - ei - ej)
            ) / (4.0 * step * step)
            H[j, i] = H[i, j]
    return H


def laplace_draws(lik: SyntheticLikelihood, z_mode: np.ndarray, *, draws: int = 50000, seed: int = 20260630) -> np.ndarray:
    neg_hess = hessian(lambda z: lik.logpost_z(z), z_mode)
    precision = -neg_hess
    # Numerical finite differences can create tiny asymmetries.
    precision = 0.5 * (precision + precision.T)
    vals = np.linalg.eigvalsh(precision)
    if np.min(vals) <= 0:
        precision = precision + np.eye(len(z_mode)) * (abs(np.min(vals)) + 1e-6)
    cov = np.linalg.inv(precision)
    rng = np.random.default_rng(seed)
    z_draws = rng.multivariate_normal(z_mode, cov, size=draws)
    return np.asarray([to_constrained(z) for z in z_draws])


def main() -> None:
    with DATA_PATH.open() as f:
        data = json.load(f)
    lik = SyntheticLikelihood(data)

    # Start near the fixed-r posterior means reported in Table 1, with r at its
    # simulation value.
    x_start = np.array([1.229, 2.405, 2.252, 1.714, -0.334, 15.0])
    z_start = np.log((x_start - LOWER) / (UPPER - x_start))
    opt = optimize.minimize(lambda z: -lik.logpost_z(z), z_start, method="Nelder-Mead", options={"maxiter": 1200})
    z_mode = opt.x if opt.success else z_start
    x_mode = to_constrained(z_mode)
    print("mode", dict(zip(PARAMS, x_mode.round(6))), "success", opt.success)

    draws = laplace_draws(lik, z_mode)
    rows = summarize(draws)
    write_csv(rows, OUT_PATH)
    print(OUT_PATH)
    print("\n".join(",".join(row) for row in rows))
    corr = np.corrcoef(draws, rowvar=False)
    print("corr_r", {PARAMS[j]: round(float(corr[-1, j]), 3) for j in range(len(PARAMS) - 1)})


if __name__ == "__main__":
    main()
