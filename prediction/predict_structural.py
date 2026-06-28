from __future__ import annotations

import numpy as np
import pandas as pd
import pickle

from common import build_base_frame, load_records, parse_common_args, save_predictions


def structural_predictions(records) -> pd.DataFrame:
    rows = []
    for rec in records:
        data = rec.data
        with rec.posterior_path.open("rb") as f:
            posterior = pickle.load(f)
        delta = float(data["Delta2f"])

        tilde_y = np.asarray(posterior["tilde_y"], dtype=float)
        terminal_gap_draws = tilde_y[:, -1]
        vol_draws = np.std(np.diff(tilde_y, axis=1) / np.sqrt(delta), axis=1, ddof=0)
        g_hat = float(np.mean(terminal_gap_draws))
        v_hat = float(np.mean(vol_draws))

        rows.append(
            {
                "contest_id": rec.contest_id,
                "G_hat_S": g_hat,
                "W_hat_S": int(g_hat > 0),
                "V_hat_S": v_hat,
                "G_hat_S_sd": float(np.std(terminal_gap_draws, ddof=0)),
                "V_hat_S_sd": float(np.std(vol_draws, ddof=0)),
            }
        )
    return pd.DataFrame(rows).sort_values("contest_id").reset_index(drop=True)


def main() -> None:
    parser = parse_common_args("Generate structural Bayesian predictions.")
    args = parser.parse_args()

    records = load_records(
        args.data_dir,
        include_all=args.include_all,
        max_n_delta=args.max_n_delta,
        load_posterior=False,
    )
    base = build_base_frame(records)[["contest_id", "G", "W", "V"]]
    structural = structural_predictions(records)
    pred = base.merge(structural, on="contest_id", how="inner")

    save_predictions(pred, args.results_dir / "predictions_structural.csv")
    print(f"Wrote {len(pred)} structural predictions to {args.results_dir / 'predictions_structural.csv'}")


if __name__ == "__main__":
    main()
