from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "metakaggle" / "__jsondata__"
RESULTS_DIR = ROOT / "validation" / "results"
EARLY_PARAMETER_PATH = RESULTS_DIR / "structural_time_split_holdout_parameters.csv"

SPLIT = 0.75
EXCLUDE_CONTESTS = {37190, 7456, 4031, 8540}
MAX_N_DELTA = 2300
MIN_LATE_POINTS = 10


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


def load_early_lambda() -> pd.DataFrame:
    if not EARLY_PARAMETER_PATH.exists():
        raise FileNotFoundError(
            f"Missing {EARLY_PARAMETER_PATH}. Run structural_early75_parameters.py first."
        )
    frame = pd.read_csv(EARLY_PARAMETER_PATH)
    required = {"contest_id", "lambda"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {EARLY_PARAMETER_PATH}: {sorted(missing)}")
    return frame[["contest_id", "lambda"]].rename(columns={"lambda": "lambda_hat_early"})


def late_disagreement(json_path: Path) -> dict | None:
    contest_id = contest_id_from_path(json_path)
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    n_delta = int(data["N_Delta"])
    if n_delta > MAX_N_DELTA:
        return None
    cutoff = int(np.floor(SPLIT * n_delta))
    private = np.asarray(data["real_y"], dtype=float)
    public = np.asarray(data["hat_y"], dtype=float)
    late_private = private[cutoff:]
    late_public = public[cutoff:]
    if len(late_private) < MIN_LATE_POINTS:
        return None
    diff = late_private - late_public
    return {
        "contest_id": contest_id,
        "full_N_Delta": n_delta,
        "cutoff": cutoff,
        "late_N_Delta": int(len(late_private)),
        "D_late": float(np.sqrt(np.mean(diff**2))),
        "D_late_abs_mean": float(np.mean(np.abs(diff))),
        "D_late_abs_median": float(np.median(np.abs(diff))),
        "D_late_abs_q95": float(np.quantile(np.abs(diff), 0.95)),
    }


def load_frame() -> tuple[pd.DataFrame, dict]:
    eligible_ids = eligible_contest_ids()
    lambdas = load_early_lambda()
    rows = []
    for json_path in sorted(DATA_DIR.glob("contest_*.json")):
        contest_id = contest_id_from_path(json_path)
        if contest_id not in eligible_ids:
            continue
        row = late_disagreement(json_path)
        if row is not None:
            rows.append(row)
    disagreement = pd.DataFrame(rows)
    frame = disagreement.merge(lambdas, on="contest_id", how="inner")
    frame = frame.sort_values("contest_id").reset_index(drop=True)
    frame["log1p_lambda_hat_early"] = np.log1p(frame["lambda_hat_early"])
    for col in ["D_late", "D_late_abs_mean", "D_late_abs_median", "D_late_abs_q95"]:
        frame[f"log1p_{col}"] = np.log1p(frame[col])
    diagnostics = {
        "eligible_contests": len(eligible_ids),
        "early_parameter_contests": int(lambdas["contest_id"].nunique()),
        "merged_contests": len(frame),
    }
    return frame, diagnostics


def fit_ols(df: pd.DataFrame, outcome: str):
    x = sm.add_constant(df["log1p_lambda_hat_early"])
    return sm.OLS(df[outcome], x).fit(cov_type="HC1")


def summarize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    outcomes = {
        "late_path_disagreement": "log1p_D_late",
        "late_path_disagreement_l1": "log1p_D_late_abs_mean",
        "late_abs_q95": "log1p_D_late_abs_q95",
    }
    for label, outcome in outcomes.items():
        fit = fit_ols(df, outcome)
        ci_low, ci_high = fit.conf_int().loc["log1p_lambda_hat_early"].tolist()
        rows.append(
            {
                "outcome": label,
                "dependent_variable": outcome,
                "coef_log1p_lambda_early": float(fit.params["log1p_lambda_hat_early"]),
                "std_error": float(fit.bse["log1p_lambda_hat_early"]),
                "p_value": float(fit.pvalues["log1p_lambda_hat_early"]),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "r2": float(fit.rsquared),
                "n": int(fit.nobs),
            }
        )

    rank_rows = []
    for target in ["D_late", "D_late_abs_mean", "D_late_abs_q95"]:
        spearman = stats.spearmanr(df["lambda_hat_early"], df[target], nan_policy="omit")
        kendall = stats.kendalltau(df["lambda_hat_early"], df[target], nan_policy="omit")
        rank_rows.append(
            {
                "target": target,
                "spearman_rho": float(spearman.statistic),
                "spearman_p": float(spearman.pvalue),
                "kendall_tau": float(kendall.statistic),
                "kendall_p": float(kendall.pvalue),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(rank_rows)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df, diagnostics = load_frame()
    summary, rank = summarize(df)

    df.to_csv(RESULTS_DIR / "signal_disagreement_timesplit_contest_level.csv", index=False)
    summary.to_csv(RESULTS_DIR / "signal_disagreement_timesplit_summary.csv", index=False)
    rank.to_csv(RESULTS_DIR / "signal_disagreement_timesplit_rank_summary.csv", index=False)

    note = "\n".join(
        [
            f"split: {SPLIT}",
            f"eligible_contests: {diagnostics['eligible_contests']}",
            f"early_parameter_contests: {diagnostics['early_parameter_contests']}",
            f"merged_contests: {diagnostics['merged_contests']}",
            "target: late public-private path disagreement computed only after the split",
            "predictor: lambda estimated from the early structural likelihood",
        ]
    )
    (RESULTS_DIR / "signal_disagreement_timesplit_diagnostics.txt").write_text(
        note + "\n", encoding="utf-8"
    )

    print(note)
    print(summary.to_string(index=False))
    print(rank.to_string(index=False))


if __name__ == "__main__":
    main()
