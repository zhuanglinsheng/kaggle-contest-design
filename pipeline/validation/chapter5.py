#!/usr/bin/env python3
"""Regenerate Chapter 5 diagnostics from the accepted USD-100,000 fits."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from pipeline.config import (
    CACHE_DIR,
    MAIN_DATA_DIR,
    MAIN_ESTIMATION_DIR,
    MAIN_VALIDATION_DIR,
    MATPLOTLIB_CACHE_DIR,
    PAPER_DIR,
    PRIZE_DIVISOR_FROM_THOUSANDS,
)

DATA_DIR = MAIN_DATA_DIR
FIT_PATH = MAIN_ESTIMATION_DIR / "posterior_summary_final.csv"
RESULTS_DIR = MAIN_VALIDATION_DIR
PRIZE_DIVISOR = PRIZE_DIVISOR_FROM_THOUSANDS
SPLIT = 0.75
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

from matplotlib import pyplot as plt


def parameter_frame(mode: str) -> pd.DataFrame:
    frame = pd.read_csv(FIT_PATH)
    frame = frame.loc[frame["mode"] == mode].copy()
    keep = ["contest_id"]
    for name in ("c_i", "c_j", "sigma", "lambda", "mu_0", "r"):
        keep.extend([f"{name}_mean", f"{name}_sd", f"{name}_q025", f"{name}_q975"])
    frame = frame[keep]
    return frame.rename(columns={f"{name}_mean": name for name in ("c_i", "c_j", "sigma", "lambda", "mu_0", "r")})


def event_indices(data: dict) -> list[int]:
    n = int(data["N_Delta"])
    stan_indices = {
        max(1, min(n, int(math.ceil(float(value)))))
        for value in list(data["hat_t_i"]) + list(data["hat_t_j"])
    }
    return sorted(index - 1 for index in stan_indices)


def implied_paths(data: dict, params: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(data["N_Delta"])
    dt = float(data["Delta2f"])
    horizon = n * dt
    theta = float(data["theta"]) / PRIZE_DIVISOR
    sigma = float(params["sigma"])
    precision = float(params["lambda"])
    c_i = float(params["c_i"])
    c_j = float(params["c_j"])
    hat_y = np.asarray(data["hat_y"], dtype=float)
    tilde = np.empty(n + 1)
    effort_i = np.empty(n)
    effort_j = np.empty(n)
    tilde[0] = float(params["mu_0"])
    bar_s = sigma / math.sqrt(precision)
    weight = -math.expm1(-math.sqrt(precision) * sigma * dt)
    for index in range(n):
        remaining = horizon - index * dt
        q = sigma * sigma * remaining + bar_s
        kernel = math.exp(-(tilde[index] ** 2) / (2.0 * q)) / math.sqrt(2.0 * math.pi * q)
        effort_i[index] = theta * kernel / c_i
        effort_j[index] = theta * kernel / c_j
        tilde[index + 1] = (
            tilde[index]
            + (effort_i[index] - effort_j[index]) * dt
            + weight * (hat_y[index] - tilde[index])
        )
    return tilde, effort_i, effort_j


def benchmark_row(data: dict, params: pd.Series) -> dict[str, float]:
    indices = event_indices(data)
    dt = float(data["Delta2f"])
    public = np.asarray(data["hat_y"], dtype=float)
    private = np.asarray(data["real_y"], dtype=float)

    previous_stan_index = 0
    weighted_sse = 0.0
    for index in indices:
        stan_index = index + 1
        elapsed = (stan_index - previous_stan_index) * dt
        weighted_sse += elapsed * (private[index] - public[index]) ** 2
        previous_stan_index = stan_index
    lambda_bench = len(indices) / weighted_sse if weighted_sse > 0 else math.nan

    _, effort_i, effort_j = implied_paths(data, params)
    sigma_terms = []
    previous = 0
    for current in indices:
        if current <= previous:
            continue
        elapsed = (current - previous) * dt
        drift = float(np.sum(effort_i[previous:current] - effort_j[previous:current]) * dt)
        residual = private[current] - private[previous] - drift
        sigma_terms.append(residual * residual / elapsed)
        previous = current
    sigma_bench = math.sqrt(float(np.mean(sigma_terms))) if sigma_terms else math.nan

    diff = private - public
    increments = np.diff(private)
    cutoff = int(math.floor(SPLIT * len(private)))
    late_diff = diff[cutoff:]
    late_increments = np.diff(private[cutoff:])
    return {
        "lambda_bench": lambda_bench,
        "sigma_bench": sigma_bench,
        "D_path": float(np.mean(np.abs(diff))),
        "D_late": float(np.mean(np.abs(late_diff))),
        "V_path": float(np.std(increments, ddof=1)),
        "V_late": float(np.std(late_increments, ddof=1)),
    }


def fit_row(frame: pd.DataFrame, x: str, y: str, label: str) -> tuple[dict[str, float], object]:
    work = frame[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    design = sm.add_constant(work[x])
    fit = sm.OLS(work[y], design).fit(cov_type="HC1")
    ci_low, ci_high = fit.conf_int().loc[x].tolist()
    rank = stats.spearmanr(work[x], work[y])
    return {
        "specification": label,
        "constant": float(fit.params["const"]),
        "constant_se": float(fit.bse["const"]),
        "coefficient": float(fit.params[x]),
        "std_error": float(fit.bse[x]),
        "p_value": float(fit.pvalues[x]),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "r2": float(fit.rsquared),
        "adj_r2": float(fit.rsquared_adj),
        "spearman_rho": float(rank.statistic),
        "spearman_p": float(rank.pvalue),
        "n": int(fit.nobs),
    }, fit


def save_plot(frame: pd.DataFrame, x: str, y: str, fit, path: Path, xlabel: str, ylabel: str) -> None:
    work = frame[[x, y]].replace([np.inf, -np.inf], np.nan).dropna().sort_values(x)
    grid = np.linspace(float(work[x].min()), float(work[x].max()), 200)
    prediction = fit.predict(sm.add_constant(pd.Series(grid, name=x)))
    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.scatter(work[x], work[y], s=20, color="black")
    ax.plot(grid, prediction, color="black", linestyle="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    full = parameter_frame("full")
    early = parameter_frame("early")
    rows = []
    for _, params in full.iterrows():
        contest_id = int(params["contest_id"])
        data = json.loads((DATA_DIR / f"contest_{contest_id}.json").read_text())
        diagnostics = benchmark_row(data, params)
        rows.append({
            "contest_id": contest_id,
            "theta": float(data["theta"]) / PRIZE_DIVISOR,
            "duration": float(data["N_Delta"]) * float(data["Delta2f"]),
            "Ni": int(data["Ni"]),
            "Nj": int(data["Nj"]),
            "public_data_pct": float(data["percentage"]),
            **{name: float(params[name]) for name in ("c_i", "c_j", "sigma", "lambda", "mu_0", "r")},
            **diagnostics,
        })
    frame = pd.DataFrame(rows).merge(
        early[["contest_id", "sigma", "lambda"]].rename(
            columns={"sigma": "sigma_early", "lambda": "lambda_early"}
        ),
        on="contest_id",
        how="left",
    )
    for name in ("lambda", "lambda_bench", "sigma", "sigma_bench", "D_path", "D_late", "V_path", "V_late", "lambda_early", "sigma_early"):
        frame[f"log1p_{name}"] = np.log1p(frame[name])
    frame.to_csv(RESULTS_DIR / "chapter5_contest_level.csv", index=False)

    specifications = [
        ("log1p_lambda", "log1p_lambda_bench", "lambda_benchmark"),
        ("log1p_sigma", "log1p_sigma_bench", "sigma_benchmark"),
        ("log1p_lambda", "log1p_D_path", "signal_disagreement_full"),
        ("log1p_lambda_early", "log1p_D_late", "signal_disagreement_time_split"),
        ("log1p_sigma", "log1p_V_path", "private_volatility_full"),
        ("log1p_sigma_early", "log1p_V_late", "private_volatility_time_split"),
    ]
    regression_rows = []
    fits = {}
    for x, y, label in specifications:
        row, fit = fit_row(frame, x, y, label)
        regression_rows.append(row)
        fits[label] = fit
    pd.DataFrame(regression_rows).to_csv(RESULTS_DIR / "chapter5_regressions.csv", index=False)

    save_plot(
        frame,
        "log1p_lambda",
        "log1p_lambda_bench",
        fits["lambda_benchmark"],
        PAPER_DIR / "validate_lamb_revised_100k.pdf",
        r"$\log(1+\hat\lambda)$",
        r"$\log(1+\lambda^{bench})$",
    )
    save_plot(
        frame,
        "log1p_sigma",
        "log1p_sigma_bench",
        fits["sigma_benchmark"],
        PAPER_DIR / "validate_sigma_revised_100k.pdf",
        r"$\log(1+\hat\sigma)$",
        r"$\log(1+\sigma^{bench})$",
    )
    print(pd.DataFrame(regression_rows).to_string(index=False))


if __name__ == "__main__":
    main()
