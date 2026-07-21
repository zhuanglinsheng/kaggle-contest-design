from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIRS = [
    ROOT / "metakaggle" / "__jsondata_robust23__",
    Path("/Volumes/RAWDATA/__jsondata_robust23__"),
]
RESULTS_DIR = ROOT / "validation" / "results"
PAPER_DIR = ROOT / "paper"
MAX_N_DELTA = 2300

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "validation" / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "validation" / ".cache"))

from matplotlib import pyplot as plt


def contest_id_from_path(path: Path) -> int:
    return int(path.stem.split("_")[1])


def default_data_dir() -> Path:
    for data_dir in DEFAULT_DATA_DIRS:
        if data_dir.exists():
            return data_dir
    return DEFAULT_DATA_DIRS[0]


def submission_event_indices(data: dict) -> list[int]:
    n_delta = int(data["N_Delta"])
    times = [float(t) for t in data["hat_t_i"]] + [float(t) for t in data["hat_t_j"]]
    return sorted(max(0, min(n_delta - 1, math.ceil(t))) for t in times)


def calculate_lambda_mle_submission_events(data: dict) -> tuple[float, int, float]:
    hat_y = np.asarray(data["hat_y"], dtype=float)
    real_y = np.asarray(data["real_y"], dtype=float)
    event_idx = submission_event_indices(data)
    if not event_idx:
        return float("nan"), 0, float("nan")
    event_gap = hat_y[event_idx] - real_y[event_idx]
    sse = float(np.sum(event_gap**2))
    if sse == 0.0:
        return float("nan"), len(event_idx), sse
    lambda_mle = len(event_idx) / (float(data["Delta2f"]) * sse)
    return float(lambda_mle), len(event_idx), sse


def load_frame(data_dir: Path, max_n_delta: int = MAX_N_DELTA) -> pd.DataFrame:
    rows = []
    for json_path in sorted(data_dir.glob("contest_*.json")):
        posterior_path = json_path.with_suffix(".pkl")
        if not posterior_path.exists():
            continue
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if int(data["N_Delta"]) > max_n_delta:
            continue
        with posterior_path.open("rb") as f:
            posterior = pickle.load(f)

        lambda_mle, event_n, event_sse = calculate_lambda_mle_submission_events(data)
        rows.append(
            {
                "contest_id": contest_id_from_path(json_path),
                "lambda_post_mean": float(np.asarray(posterior["lambda"], dtype=float).mean()),
                "lambda_mle_submission": lambda_mle,
                "event_n": event_n,
                "event_sse": event_sse,
                "Delta2f": float(data["Delta2f"]),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "contest_id",
                "lambda_post_mean",
                "lambda_mle_submission",
                "event_n",
                "event_sse",
                "Delta2f",
            ]
        )
    return pd.DataFrame(rows).sort_values("contest_id").reset_index(drop=True)


def fit_ols(frame: pd.DataFrame):
    df = frame.dropna(subset=["lambda_post_mean", "lambda_mle_submission"]).copy()
    df["bayes"] = np.log1p(df["lambda_post_mean"])
    df["mle"] = np.log1p(df["lambda_mle_submission"])
    x = sm.add_constant(df["bayes"])
    fit = sm.OLS(df["mle"], x).fit()
    return df, fit


def write_summary(fit, output_path: Path) -> None:
    ci_low, ci_high = fit.conf_int().loc["bayes"].tolist()
    lines = [
        f"n_contests: {int(fit.nobs)}",
        f"const: {float(fit.params['const']):.9g}",
        f"const_se: {float(fit.bse['const']):.9g}",
        f"slope: {float(fit.params['bayes']):.9g}",
        f"slope_se: {float(fit.bse['bayes']):.9g}",
        f"slope_p: {float(fit.pvalues['bayes']):.9g}",
        f"slope_ci_low: {float(ci_low):.9g}",
        f"slope_ci_high: {float(ci_high):.9g}",
        f"r2: {float(fit.rsquared):.9g}",
        f"adj_r2: {float(fit.rsquared_adj):.9g}",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_plot(df: pd.DataFrame, fit, output_path: Path) -> None:
    x = df["bayes"].to_numpy(dtype=float)
    y = df["mle"].to_numpy(dtype=float)
    order = np.argsort(x)
    x_grid = x[order]
    y_fit = fit.predict(sm.add_constant(pd.Series(x_grid, name="bayes")))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 15})
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.scatter(x, y, s=20, label="Contest", color="black")
    ax.plot(x_grid, y_fit, label="OLS Fit", color="black", linestyle="--")
    ax.set_xlabel(r"$\log(1+\hat\lambda)$")
    ax.set_ylabel(r"$\log(1+\lambda^{MLE})$")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--max-n-delta", type=int, default=MAX_N_DELTA)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_frame(args.data_dir, args.max_n_delta)
    if frame.empty:
        print(f"No robust23 json/pkl pairs found in {args.data_dir}")
        return 1

    df, fit = fit_ols(frame)
    frame.to_csv(RESULTS_DIR / "lambda_mle_submission_events_robust23.csv", index=False)
    write_summary(fit, RESULTS_DIR / "lambda_mle_submission_events_robust23_ols.txt")
    save_plot(df, fit, PAPER_DIR / "validate_lamb_robust23.pdf")

    print(f"Wrote {len(frame)} contests from {args.data_dir}")
    print(f"Slope: {fit.params['bayes']:.6g}")
    print(f"p-value: {fit.pvalues['bayes']:.6g}")
    print(f"R-squared: {fit.rsquared:.6g}")
    print(f"Adj. R-squared: {fit.rsquared_adj:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
