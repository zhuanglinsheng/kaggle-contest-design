from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "metakaggle" / "__jsondata__"
RESULTS_DIR = ROOT / "validation" / "results"
PAPER_DIR = ROOT / "paper"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "validation" / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "validation" / ".cache"))

from matplotlib import pyplot as plt


EXCLUDE_CONTESTS = {37190, 7456, 4031, 8540}
MAX_N_DELTA = 2300
EPS = 1e-10


def contest_id_from_path(path: Path) -> int:
    return int(path.stem.split("_")[1])


def load_frame() -> pd.DataFrame:
    rows = []
    for json_path in sorted(DATA_DIR.glob("contest_*.json")):
        contest_id = contest_id_from_path(json_path)
        if contest_id in EXCLUDE_CONTESTS:
            continue
        posterior_path = json_path.with_suffix(".pkl")
        if not posterior_path.exists():
            continue
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if int(data["N_Delta"]) > MAX_N_DELTA:
            continue
        with posterior_path.open("rb") as f:
            posterior = pickle.load(f)

        private = np.asarray(data["real_y"], dtype=float)
        increments = np.diff(private)
        abs_increments = np.abs(increments)
        sigma_hat = float(np.asarray(posterior["sigma"], dtype=float).mean())

        rows.append(
            {
                "contest_id": contest_id,
                "sigma_hat": sigma_hat,
                "log1p_sigma_hat": float(np.log1p(sigma_hat)),
                "V_sd": float(np.std(increments, ddof=1)),
                "V_rms": float(np.sqrt(np.mean(increments**2))),
                "V_mad": float(np.median(np.abs(increments - np.median(increments)))),
                "V_abs_mean": float(np.mean(abs_increments)),
                "V_abs_q90": float(np.quantile(abs_increments, 0.90)),
                "V_abs_q95": float(np.quantile(abs_increments, 0.95)),
                "V_abs_max": float(np.max(abs_increments)),
                "N_Delta": int(data["N_Delta"]),
                "Ni": int(data["Ni"]),
                "Nj": int(data["Nj"]),
                "theta": float(data["theta"]),
                "public_split": float(data["percentage"]),
            }
        )
    frame = pd.DataFrame(rows).sort_values("contest_id").reset_index(drop=True)
    for col in ["V_sd", "V_rms", "V_mad", "V_abs_mean", "V_abs_q90", "V_abs_q95", "V_abs_max"]:
        frame[f"log1p_{col}"] = np.log1p(frame[col])
    frame["sigma_tercile"] = pd.qcut(
        frame["sigma_hat"], 3, labels=["Low sigma", "Middle sigma", "High sigma"]
    )
    return frame


def fit_ols(df: pd.DataFrame, outcome: str):
    x = sm.add_constant(df["log1p_sigma_hat"])
    return sm.OLS(df[outcome], x).fit(cov_type="HC1")


def summarize_regressions(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    outcomes = {
        "path_sd": "log1p_V_sd",
        "path_rms": "log1p_V_rms",
        "tail_q95": "log1p_V_abs_q95",
        "max_jump": "log1p_V_abs_max",
    }
    for label, outcome in outcomes.items():
        fit = fit_ols(df, outcome)
        ci_low, ci_high = fit.conf_int().loc["log1p_sigma_hat"].tolist()
        rows.append(
            {
                "outcome": label,
                "dependent_variable": outcome,
                "coef_log1p_sigma": float(fit.params["log1p_sigma_hat"]),
                "std_error": float(fit.bse["log1p_sigma_hat"]),
                "p_value": float(fit.pvalues["log1p_sigma_hat"]),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "r2": float(fit.rsquared),
                "n": int(fit.nobs),
            }
        )
    return pd.DataFrame(rows)


def summarize_rank(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in ["V_sd", "V_rms", "V_abs_q90", "V_abs_q95", "V_abs_max"]:
        spearman = stats.spearmanr(df["sigma_hat"], df[target], nan_policy="omit")
        kendall = stats.kendalltau(df["sigma_hat"], df[target], nan_policy="omit")
        rows.append(
            {
                "target": target,
                "spearman_rho": float(spearman.statistic),
                "spearman_p": float(spearman.pvalue),
                "kendall_tau": float(kendall.statistic),
                "kendall_p": float(kendall.pvalue),
            }
        )
    return pd.DataFrame(rows)


def summarize_groups(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in ["V_sd", "V_rms", "V_abs_q95", "V_abs_max"]:
        for group_name, group in df.groupby("sigma_tercile", observed=False):
            rows.append(
                {
                    "target": target,
                    "sigma_tercile": str(group_name),
                    "n": int(len(group)),
                    "mean": float(group[target].mean()),
                    "median": float(group[target].median()),
                    "q25": float(group[target].quantile(0.25)),
                    "q75": float(group[target].quantile(0.75)),
                }
            )
    return pd.DataFrame(rows)


def save_plot(df: pd.DataFrame, path: Path) -> None:
    fit = fit_ols(df, "log1p_V_sd")
    x = df["log1p_sigma_hat"].to_numpy(dtype=float)
    y = df["log1p_V_sd"].to_numpy(dtype=float)
    grid = np.linspace(x.min(), x.max(), 200)
    pred = fit.predict(sm.add_constant(pd.Series(grid, name="log1p_sigma_hat")))

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.scatter(x, y, s=30, alpha=0.78, color="#2b6cb0", edgecolor="white", linewidth=0.45)
    ax.plot(grid, pred, color="#b83232", linewidth=1.7)
    ax.set_xlabel(r"$\log(1+\hat\sigma)$")
    ax.set_ylabel(r"$\log(1+V^{path})$")
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_frame()
    regression = summarize_regressions(df)
    rank = summarize_rank(df)
    groups = summarize_groups(df)

    df.to_csv(RESULTS_DIR / "private_volatility_contest_level.csv", index=False)
    regression.to_csv(RESULTS_DIR / "private_volatility_regression_summary.csv", index=False)
    rank.to_csv(RESULTS_DIR / "private_volatility_rank_summary.csv", index=False)
    groups.to_csv(RESULTS_DIR / "private_volatility_group_summary.csv", index=False)
    save_plot(df, PAPER_DIR / "validate_private_volatility.pdf")

    print(regression.to_string(index=False))
    print(rank.to_string(index=False))
    print(groups.loc[groups["target"] == "V_sd"].to_string(index=False))


if __name__ == "__main__":
    main()
