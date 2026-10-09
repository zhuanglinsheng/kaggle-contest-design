"""One-pass Cook-distance sensitivity for Chapter 5 signal diagnostics.

Flag Cook D > 4/n using each complete regression, then refit OLS once with
HC1 covariance. This is an exploratory, outcome-dependent sensitivity check,
not evidence that the flagged contests contain errors. No source rows are
deleted and no primary regression output is overwritten.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

from pipeline.config import MAIN_VALIDATION_DIR


def diagnostic(frame: pd.DataFrame, x_name: str, y_name: str, label: str):
    work = frame[["contest_id", x_name, y_name]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna().reset_index(drop=True)
    if work.contest_id.duplicated().any():
        raise ValueError("Contests must be unique within each regression")
    x, y = work[x_name].to_numpy(), work[y_name].to_numpy()
    design = np.column_stack((np.ones(len(x)), x))
    if len(x) <= 3 or np.linalg.matrix_rank(design) != 2:
        raise ValueError("Insufficient observations or rank-deficient design")
    initial = sm.OLS(y, design).fit()
    influence = initial.get_influence()
    cooks = influence.cooks_distance[0]
    if not np.isfinite(cooks).all():
        raise ValueError("Undefined Cook distance; cannot silently filter")
    threshold = 4.0 / len(work)
    keep = cooks <= threshold
    if keep.sum() <= 3 or np.linalg.matrix_rank(design[keep]) != 2:
        raise ValueError("Insufficient retained sample")
    excluded = json.dumps(work.loc[~keep, "contest_id"].astype(int).tolist())
    rows = []
    for method, mask in (("OLS_all", np.ones(len(x), dtype=bool)),
                         ("OLS_Cook_screened", keep)):
        fit = sm.OLS(y[mask], design[mask]).fit(cov_type="HC1")
        ci = fit.conf_int()[1]
        rank = spearmanr(x[mask], y[mask])
        rows.append({
            "specification": label, "method": method,
            "n_original": len(work), "n": int(mask.sum()),
            "n_excluded": int((~mask).sum()), "cook_threshold": threshold,
            "excluded_contest_ids": excluded if method == "OLS_Cook_screened" else "[]",
            "coefficient": float(fit.params[1]), "std_error": float(fit.bse[1]),
            "p_value": float(fit.pvalues[1]), "ci_low": float(ci[0]),
            "ci_high": float(ci[1]), "r2": float(fit.rsquared),
            "adj_r2": float(fit.rsquared_adj),
            "spearman_rho": float(rank.statistic), "spearman_p": float(rank.pvalue),
            "standard_error_type": "HC1",
        })
    flags = pd.DataFrame({
        "specification": label, "contest_id": work.contest_id,
        "x": x, "y": y, "cook_distance": cooks,
        "threshold": threshold, "excluded": ~keep,
        "leverage": influence.hat_matrix_diag,
        "studentized_residual": influence.resid_studentized_internal,
    })
    return pd.DataFrame(rows), flags


def main() -> None:
    source = MAIN_VALIDATION_DIR / "chapter5_contest_level.csv"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    frame = pd.read_csv(source)
    results, flags = [], []
    for x, y, label in (
        ("log1p_lambda", "log1p_D_path", "signal_disagreement_full"),
        ("log1p_lambda_early", "log1p_D_late", "signal_disagreement_time_split"),
    ):
        result, flag = diagnostic(frame, x, y, label)
        results.append(result)
        flags.append(flag)
    result = pd.concat(results, ignore_index=True)
    result["input_sha256"] = source_hash
    result.to_csv(MAIN_VALIDATION_DIR / "chapter5_signal_cook_regressions.csv", index=False)
    pd.concat(flags, ignore_index=True).to_csv(
        MAIN_VALIDATION_DIR / "chapter5_signal_cook_flags.csv", index=False)
    print(result.drop(columns="input_sha256").to_string(index=False))


if __name__ == "__main__":
    main()
