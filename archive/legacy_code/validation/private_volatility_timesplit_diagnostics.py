from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "metakaggle" / "__jsondata__"
RESULTS_DIR = ROOT / "validation" / "results"
PAPER_DIR = ROOT / "paper"
EARLY_PARAMETER_PATH = RESULTS_DIR / "structural_time_split_holdout_parameters.csv"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "validation" / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "validation" / ".cache"))

from matplotlib import pyplot as plt


SPLIT = 0.75
EXCLUDE_CONTESTS = {37190, 7456, 4031, 8540}
MAX_N_DELTA = 2300
MIN_LATE_INCREMENTS = 10


def contest_id_from_path(path: Path) -> int:
    return int(path.stem.split("_")[1])


def eligible_contest_ids() -> set[int]:
    ids = set()
    for json_path in sorted(DATA_DIR.glob("contest_*.json")):
        contest_id = contest_id_from_path(json_path)
        if contest_id in EXCLUDE_CONTESTS:
            continue
        if not json_path.with_suffix(".pkl").exists():
            continue
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if int(data["N_Delta"]) <= MAX_N_DELTA:
            ids.add(contest_id)
    return ids


def load_early_sigma() -> pd.DataFrame:
    if not EARLY_PARAMETER_PATH.exists():
        raise FileNotFoundError(
            f"Missing {EARLY_PARAMETER_PATH}. Run structural_early75_parameters.py first."
        )
    frame = pd.read_csv(EARLY_PARAMETER_PATH)
    required = {"contest_id", "sigma"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {EARLY_PARAMETER_PATH}: {sorted(missing)}")
    return frame[["contest_id", "sigma"]].rename(columns={"sigma": "sigma_hat_early"})


def private_late_volatility(json_path: Path) -> dict | None:
    contest_id = contest_id_from_path(json_path)
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    n_delta = int(data["N_Delta"])
    if n_delta > MAX_N_DELTA:
        return None
    cutoff = int(np.floor(SPLIT * n_delta))
    private = np.asarray(data["real_y"], dtype=float)
    late_private = private[cutoff:]
    if len(late_private) <= MIN_LATE_INCREMENTS:
        return None
    increments = np.diff(late_private)
    abs_increments = np.abs(increments)
    return {
        "contest_id": contest_id,
        "full_N_Delta": n_delta,
        "cutoff": cutoff,
        "late_N_Delta": int(len(late_private)),
        "V_late_sd": float(np.std(increments, ddof=1)),
        "V_late_rms": float(np.sqrt(np.mean(increments**2))),
        "V_late_abs_mean": float(np.mean(abs_increments)),
        "V_late_abs_q95": float(np.quantile(abs_increments, 0.95)),
        "V_late_abs_max": float(np.max(abs_increments)),
    }


def load_frame() -> tuple[pd.DataFrame, dict]:
    eligible_ids = eligible_contest_ids()
    sigma = load_early_sigma()
    rows = []
    for json_path in sorted(DATA_DIR.glob("contest_*.json")):
        contest_id = contest_id_from_path(json_path)
        if contest_id not in eligible_ids:
            continue
        row = private_late_volatility(json_path)
        if row is not None:
            rows.append(row)
    volatility = pd.DataFrame(rows)
    frame = volatility.merge(sigma, on="contest_id", how="inner")
    frame = frame.sort_values("contest_id").reset_index(drop=True)
    frame["log1p_sigma_hat_early"] = np.log1p(frame["sigma_hat_early"])
    for col in ["V_late_sd", "V_late_rms", "V_late_abs_mean", "V_late_abs_q95", "V_late_abs_max"]:
        frame[f"log1p_{col}"] = np.log1p(frame[col])
    if len(frame) >= 3:
        frame["sigma_early_tercile"] = pd.qcut(
            frame["sigma_hat_early"],
            min(3, frame["sigma_hat_early"].nunique()),
            labels=False,
            duplicates="drop",
        )
    diagnostics = {
        "eligible_contests": len(eligible_ids),
        "early_parameter_contests": int(sigma["contest_id"].nunique()),
        "merged_contests": len(frame),
    }
    return frame, diagnostics


def fit_ols(df: pd.DataFrame, outcome: str):
    x = sm.add_constant(df["log1p_sigma_hat_early"])
    return sm.OLS(df[outcome], x).fit(cov_type="HC1")


def summarize_regressions(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    outcomes = {
        "late_path_sd": "log1p_V_late_sd",
        "late_path_rms": "log1p_V_late_rms",
        "late_abs_q95": "log1p_V_late_abs_q95",
        "late_abs_max": "log1p_V_late_abs_max",
    }
    if len(df) < 4:
        return pd.DataFrame(
            [
                {
                    "outcome": label,
                    "dependent_variable": outcome,
                    "coef_log1p_sigma_early": np.nan,
                    "std_error": np.nan,
                    "p_value": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "r2": np.nan,
                    "n": len(df),
                }
                for label, outcome in outcomes.items()
            ]
        )
    for label, outcome in outcomes.items():
        fit = fit_ols(df, outcome)
        ci_low, ci_high = fit.conf_int().loc["log1p_sigma_hat_early"].tolist()
        rows.append(
            {
                "outcome": label,
                "dependent_variable": outcome,
                "coef_log1p_sigma_early": float(fit.params["log1p_sigma_hat_early"]),
                "std_error": float(fit.bse["log1p_sigma_hat_early"]),
                "p_value": float(fit.pvalues["log1p_sigma_hat_early"]),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "r2": float(fit.rsquared),
                "n": int(fit.nobs),
            }
        )
    return pd.DataFrame(rows)


def summarize_rank(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in ["V_late_sd", "V_late_rms", "V_late_abs_q95", "V_late_abs_max"]:
        if len(df) < 4:
            spearman_stat = spearman_p = kendall_stat = kendall_p = np.nan
        else:
            spearman = stats.spearmanr(df["sigma_hat_early"], df[target], nan_policy="omit")
            kendall = stats.kendalltau(df["sigma_hat_early"], df[target], nan_policy="omit")
            spearman_stat = float(spearman.statistic)
            spearman_p = float(spearman.pvalue)
            kendall_stat = float(kendall.statistic)
            kendall_p = float(kendall.pvalue)
        rows.append(
            {
                "target": target,
                "spearman_rho": spearman_stat,
                "spearman_p": spearman_p,
                "kendall_tau": kendall_stat,
                "kendall_p": kendall_p,
            }
        )
    return pd.DataFrame(rows)


def summarize_groups(df: pd.DataFrame) -> pd.DataFrame:
    if "sigma_early_tercile" not in df.columns:
        return pd.DataFrame()
    rows = []
    labels = {0: "Low early sigma", 1: "Middle early sigma", 2: "High early sigma"}
    for target in ["V_late_sd", "V_late_rms", "V_late_abs_q95", "V_late_abs_max"]:
        for group_id, group in df.groupby("sigma_early_tercile", observed=False):
            rows.append(
                {
                    "target": target,
                    "sigma_early_tercile": labels.get(int(group_id), f"Group {group_id}"),
                    "n": int(len(group)),
                    "mean": float(group[target].mean()),
                    "median": float(group[target].median()),
                    "q25": float(group[target].quantile(0.25)),
                    "q75": float(group[target].quantile(0.75)),
                }
            )
    return pd.DataFrame(rows)


def save_plot(df: pd.DataFrame, path: Path) -> None:
    if len(df) < 4:
        return
    fit = fit_ols(df, "log1p_V_late_sd")
    x = df["log1p_sigma_hat_early"].to_numpy(dtype=float)
    y = df["log1p_V_late_sd"].to_numpy(dtype=float)
    grid = np.linspace(x.min(), x.max(), 200)
    pred = fit.predict(sm.add_constant(pd.Series(grid, name="log1p_sigma_hat_early")))

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.scatter(x, y, s=30, alpha=0.78, color="#2b6cb0", edgecolor="white", linewidth=0.45)
    ax.plot(grid, pred, color="#b83232", linewidth=1.7)
    ax.set_xlabel(r"$\log(1+\hat\sigma^{early})$")
    ax.set_ylabel(r"$\log(1+V^{late})$")
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df, diagnostics = load_frame()
    regression = summarize_regressions(df)
    rank = summarize_rank(df)
    groups = summarize_groups(df)

    df.to_csv(RESULTS_DIR / "private_volatility_timesplit_contest_level.csv", index=False)
    regression.to_csv(
        RESULTS_DIR / "private_volatility_timesplit_regression_summary.csv", index=False
    )
    rank.to_csv(RESULTS_DIR / "private_volatility_timesplit_rank_summary.csv", index=False)
    groups.to_csv(RESULTS_DIR / "private_volatility_timesplit_group_summary.csv", index=False)
    save_plot(df, PAPER_DIR / "validate_private_volatility_timesplit.pdf")

    note = "\n".join(
        [
            f"split: {SPLIT}",
            f"eligible_contests: {diagnostics['eligible_contests']}",
            f"early_parameter_contests: {diagnostics['early_parameter_contests']}",
            f"merged_contests: {diagnostics['merged_contests']}",
            "target: late private-leaderboard volatility computed only after the split",
            "predictor: sigma estimated from the early structural likelihood",
        ]
    )
    (RESULTS_DIR / "private_volatility_timesplit_diagnostics.txt").write_text(
        note + "\n", encoding="utf-8"
    )

    print(note)
    print(regression.to_string(index=False))
    print(rank.to_string(index=False))


if __name__ == "__main__":
    main()
