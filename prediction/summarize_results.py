from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import RESULTS_DIR, load_prediction_file, parse_common_args, save_predictions


MODEL_COLUMNS = {
    "FP": {"G": "G_hat_FP", "W": "W_hat_FP", "V": "V_hat_FP"},
    "RF": {"G": "G_hat_RF", "W": "W_hat_RF", "V": "V_hat_RF"},
    "S": {"G": "G_hat_S", "W": "W_hat_S", "V": "V_hat_S"},
}


def rmse(actual: pd.Series, predicted: pd.Series) -> float:
    err = actual.to_numpy(dtype=float) - predicted.to_numpy(dtype=float)
    return float(np.sqrt(np.mean(err**2)))


def mae(actual: pd.Series, predicted: pd.Series) -> float:
    err = np.abs(actual.to_numpy(dtype=float) - predicted.to_numpy(dtype=float))
    return float(np.mean(err))


def winner_accuracy(actual: pd.Series, predicted: pd.Series) -> float:
    return float(np.mean(actual.to_numpy(dtype=int) == predicted.to_numpy(dtype=int)))


def merge_predictions(results_dir: Path) -> pd.DataFrame:
    fp = load_prediction_file(results_dir, "predictions_fp.csv")
    rf = load_prediction_file(results_dir, "predictions_rf.csv")
    st = load_prediction_file(results_dir, "predictions_structural.csv")

    merged = fp.merge(
        rf.drop(columns=["G", "W", "V"], errors="ignore"),
        on="contest_id",
        how="inner",
    ).merge(
        st.drop(columns=["G", "W", "V"], errors="ignore"),
        on="contest_id",
        how="inner",
    )
    return merged.sort_values("contest_id").reset_index(drop=True)


def summarize(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    labels = {
        "FP": "Final public leaderboard",
        "RF": "Reduced-form activity model",
        "S": "Structural Bayesian model",
    }
    for code, cols in MODEL_COLUMNS.items():
        rows.append(
            {
                "model": labels[code],
                "model_code": code,
                "n_contests": int(len(merged)),
                "private_gap_rmse": rmse(merged["G"], merged[cols["G"]]),
                "private_gap_mae": mae(merged["G"], merged[cols["G"]]),
                "winner_accuracy": winner_accuracy(merged["W"], merged[cols["W"]]),
                "private_volatility_rmse": rmse(merged["V"], merged[cols["V"]]),
                "private_volatility_mae": mae(merged["V"], merged[cols["V"]]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = parse_common_args("Summarize prediction benchmark results.")
    args = parser.parse_args()
    results_dir = args.results_dir

    merged = merge_predictions(results_dir)
    summary = summarize(merged)

    save_predictions(merged, results_dir / "predictions_all.csv")
    save_predictions(summary, results_dir / "summary_metrics.csv")

    metadata = {
        "n_contests": int(len(merged)),
        "contest_ids": [int(x) for x in merged["contest_id"].tolist()],
        "results_dir": str(results_dir),
    }
    (results_dir / "summary_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"Wrote merged predictions and summary metrics to {results_dir}")


if __name__ == "__main__":
    main()

