from __future__ import annotations

from common import build_base_frame, load_records, parse_common_args, save_predictions


def main() -> None:
    parser = parse_common_args("Generate final-public benchmark predictions.")
    args = parser.parse_args()

    records = load_records(
        args.data_dir,
        include_all=args.include_all,
        max_n_delta=args.max_n_delta,
    )
    base = build_base_frame(records)
    pred = base[
        [
            "contest_id",
            "G",
            "W",
            "V",
            "final_public_gap",
            "public_winner",
            "public_volatility",
        ]
    ].copy()
    pred.rename(
        columns={
            "final_public_gap": "G_hat_FP",
            "public_winner": "W_hat_FP",
            "public_volatility": "V_hat_FP",
        },
        inplace=True,
    )
    save_predictions(pred, args.results_dir / "predictions_fp.csv")
    print(f"Wrote {len(pred)} FP predictions to {args.results_dir / 'predictions_fp.csv'}")


if __name__ == "__main__":
    main()

