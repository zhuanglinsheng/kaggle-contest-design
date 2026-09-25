#!/usr/bin/env python3
"""Regenerate Chapter 6 focal-pair robustness results."""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.stattools import durbin_watson, jarque_bera, omni_normtest

from pipeline.config import (
    CACHE_DIR,
    MATPLOTLIB_CACHE_DIR,
    PAPER_DIR,
    ROBUSTNESS_DATA_DIR,
    ROBUSTNESS_ESTIMATION_DIR,
    ROBUSTNESS_VALIDATION_DIR,
)
from pipeline.validation import chapter5

DATA_DIR = ROBUSTNESS_DATA_DIR
FIT_PATH = ROBUSTNESS_ESTIMATION_DIR / "posterior_summary_final.csv"
RESULTS_DIR = ROBUSTNESS_VALIDATION_DIR
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))


def parameter_frame() -> pd.DataFrame:
    frame = pd.read_csv(FIT_PATH)
    frame = frame.loc[frame["mode"] == "full"].copy()
    return frame.rename(
        columns={f"{name}_mean": name for name in ("c_i", "c_j", "sigma", "lambda", "mu_0", "r")}
    )


def regression_row(frame: pd.DataFrame, x: str, y: str, label: str):
    work = frame[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    design = sm.add_constant(work[x])
    fit = sm.OLS(work[y], design).fit(cov_type="HC1")
    ci_low, ci_high = fit.conf_int().loc[x].tolist()
    constant_ci_low, constant_ci_high = fit.conf_int().loc["const"].tolist()
    omnibus, omnibus_p = omni_normtest(fit.resid)
    jb, jb_p, skewness, kurtosis = jarque_bera(fit.resid)
    rank = stats.spearmanr(work[x], work[y])
    row = {
        "specification": label,
        "constant": float(fit.params["const"]),
        "constant_se": float(fit.bse["const"]),
        "constant_p": float(fit.pvalues["const"]),
        "constant_ci_low": float(constant_ci_low),
        "constant_ci_high": float(constant_ci_high),
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
        "omnibus": float(omnibus),
        "omnibus_p": float(omnibus_p),
        "jarque_bera": float(jb),
        "jarque_bera_p": float(jb_p),
        "skewness": float(skewness),
        "kurtosis": float(kurtosis),
        "condition_number": float(np.linalg.cond(fit.model.exog)),
        "durbin_watson": float(durbin_watson(fit.resid)),
    }
    return row, fit


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    parameters = parameter_frame()
    rows = []
    for _, params in parameters.iterrows():
        contest_id = int(params["contest_id"])
        raw = json.loads((DATA_DIR / f"contest_{contest_id}.json").read_text())
        benchmarks = chapter5.benchmark_row(raw, params)
        rows.append(
            {
                "contest_id": contest_id,
                **{name: float(params[name]) for name in ("c_i", "c_j", "sigma", "lambda", "mu_0", "r")},
                **benchmarks,
            }
        )
    frame = pd.DataFrame(rows)
    for name in ("lambda", "lambda_bench", "sigma", "sigma_bench"):
        frame[f"log1p_{name}"] = np.log1p(frame[name])
    frame.to_csv(RESULTS_DIR / "robust23_contest_level.csv", index=False)

    lambda_row, lambda_fit = regression_row(
        frame, "log1p_lambda", "log1p_lambda_bench", "lambda_benchmark"
    )
    sigma_row, sigma_fit = regression_row(
        frame, "log1p_sigma", "log1p_sigma_bench", "sigma_benchmark"
    )
    results = pd.DataFrame([lambda_row, sigma_row])
    results.to_csv(RESULTS_DIR / "robust23_regressions.csv", index=False)

    chapter5.save_plot(
        frame,
        "log1p_lambda",
        "log1p_lambda_bench",
        lambda_fit,
        PAPER_DIR / "validate_lamb_robust23.pdf",
        r"$\log(1+\hat\lambda)$",
        r"$\log(1+\lambda^{bench})$",
    )
    chapter5.save_plot(
        frame,
        "log1p_sigma",
        "log1p_sigma_bench",
        sigma_fit,
        PAPER_DIR / "validate_sigma_robust23.pdf",
        r"$\log(1+\hat\sigma)$",
        r"$\log(1+\sigma^{bench})$",
    )
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
