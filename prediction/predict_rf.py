from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import warnings

from common import build_base_frame, load_records, parse_common_args, save_predictions


FEATURES = [
    "final_public_gap",
    "public_volatility",
    "Ni",
    "Nj",
    "submission_imbalance",
    "theta",
    "duration",
    "late_public_improvement",
]


def loo_linear_predictions(df: pd.DataFrame, target: str) -> np.ndarray:
    y_hat = np.empty(len(df), dtype=float)
    x = df[FEATURES].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)

    for idx in range(len(df)):
        train = np.ones(len(df), dtype=bool)
        train[idx] = False
        model = make_pipeline(StandardScaler(), LinearRegression())
        model.fit(x[train], y[train])
        y_hat[idx] = model.predict(x[[idx]])[0]
    return y_hat


def loo_logistic_predictions(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    p_hat = np.empty(len(df), dtype=float)
    w_hat = np.empty(len(df), dtype=int)
    x = df[FEATURES].to_numpy(dtype=float)
    y = df["W"].to_numpy(dtype=int)

    for idx in range(len(df)):
        train = np.ones(len(df), dtype=bool)
        train[idx] = False
        y_train = y[train]
        if len(np.unique(y_train)) < 2:
            prob = float(np.mean(y_train))
        else:
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=5000, solver="lbfgs"),
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model.fit(x[train], y_train)
            prob = float(model.predict_proba(x[[idx]])[0, 1])
        p_hat[idx] = prob
        w_hat[idx] = int(prob >= 0.5)
    return p_hat, w_hat


def main() -> None:
    parser = parse_common_args("Generate reduced-form activity benchmark predictions.")
    args = parser.parse_args()

    records = load_records(
        args.data_dir,
        include_all=args.include_all,
        max_n_delta=args.max_n_delta,
    )
    base = build_base_frame(records)
    pred = base[["contest_id", "G", "W", "V"] + FEATURES].copy()
    pred["G_hat_RF"] = loo_linear_predictions(base, "G")
    pred["V_hat_RF"] = loo_linear_predictions(base, "V")
    pred["W_prob_RF"], pred["W_hat_RF"] = loo_logistic_predictions(base)

    save_predictions(pred, args.results_dir / "predictions_rf.csv")
    print(f"Wrote {len(pred)} RF predictions to {args.results_dir / 'predictions_rf.csv'}")


if __name__ == "__main__":
    main()

