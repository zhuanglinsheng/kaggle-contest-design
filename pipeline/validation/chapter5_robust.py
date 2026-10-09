"""All-contest outlier sensitivity for the signal-disagreement diagnostics.

Keep the original OLS results unchanged. Huber's standard tuning constant is
fixed before fitting; neither observations nor settings are selected by p-value.
Pairs bootstrap intervals condition on the accepted structural point estimates.
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from pipeline.config import MAIN_VALIDATION_DIR


def huber_fit(x: np.ndarray, y: np.ndarray):
    design = np.column_stack((np.ones(len(x)), x))
    fit = sm.RLM(y, design, M=sm.robust.norms.HuberT(t=1.345)).fit(
        cov="H3", scale_est=sm.robust.scale.HuberScale(d=2.5),
        maxiter=2000, tol=1e-9, conv="coefs"
    )
    if fit.fit_history["iteration"] >= 2000:
        raise RuntimeError("Huber regression did not converge within 2000 iterations")
    return fit


def diagnostic(frame: pd.DataFrame, x_name: str, y_name: str, label: str,
               bootstrap: int, seed: int):
    work = frame[["contest_id", x_name, y_name]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    x, y = work[x_name].to_numpy(), work[y_name].to_numpy()
    design = np.column_stack((np.ones(len(x)), x))
    ols = sm.OLS(y, design).fit(cov_type="HC1")
    robust = huber_fit(x, y)
    rng = np.random.default_rng(seed)
    slopes = []
    for _ in range(bootstrap):
        indices = rng.integers(0, len(x), size=len(x))
        if np.ptp(x[indices]) == 0:
            raise RuntimeError("Degenerate bootstrap design; no draw silently discarded")
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            slopes.append(float(huber_fit(x[indices], y[indices]).params[1]))
    intervals = np.quantile(slopes, [0.025, 0.975])
    rows = []
    for method, fit in (("OLS_HC1", ols), ("Huber_H3", robust)):
        ci = fit.conf_int()[1]
        rows.append({
            "specification": label, "method": method, "n": len(x),
            "coefficient": float(fit.params[1]), "std_error": float(fit.bse[1]),
            "p_value": float(fit.pvalues[1]),
            "ci_low": float(ci[0]), "ci_high": float(ci[1]),
            "bootstrap_ci_low": float(intervals[0]) if method == "Huber_H3" else np.nan,
            "bootstrap_ci_high": float(intervals[1]) if method == "Huber_H3" else np.nan,
            "bootstrap_draws": bootstrap if method == "Huber_H3" else 0,
            "bootstrap_seed": seed if method == "Huber_H3" else np.nan,
            "huber_tuning": 1.345 if method == "Huber_H3" else np.nan,
            "scale_estimator": "Huber_proposal_2_d_2.5" if method == "Huber_H3" else "OLS",
        })
    weights = pd.DataFrame({
        "specification": label, "contest_id": work.contest_id.to_numpy(),
        "huber_weight": robust.weights, "huber_residual": robust.resid,
        "x": x, "y": y,
    })
    draws = pd.DataFrame({"specification": label,
                          "draw": np.arange(bootstrap), "coefficient": slopes})
    return rows, weights, draws


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261008)
    args = parser.parse_args()
    if args.bootstrap < 2:
        parser.error("--bootstrap must be at least 2")
    frame = pd.read_csv(MAIN_VALIDATION_DIR / "chapter5_contest_level.csv")
    rows, weights, draws = [], [], []
    specs = [
        ("log1p_lambda", "log1p_D_path", "signal_disagreement_full"),
        ("log1p_lambda_early", "log1p_D_late", "signal_disagreement_time_split"),
    ]
    for offset, (x, y, label) in enumerate(specs):
        result, weight, draw = diagnostic(frame, x, y, label,
                                           args.bootstrap, args.seed + offset)
        rows.extend(result)
        weights.append(weight)
        draws.append(draw)
    result = pd.DataFrame(rows)
    result.to_csv(MAIN_VALIDATION_DIR / "chapter5_signal_robust_regressions.csv", index=False)
    pd.concat(weights, ignore_index=True).to_csv(
        MAIN_VALIDATION_DIR / "chapter5_signal_robust_weights.csv", index=False)
    pd.concat(draws, ignore_index=True).to_csv(
        MAIN_VALIDATION_DIR / "chapter5_signal_robust_bootstrap.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
