#!/usr/bin/env python3
"""Validate and finalize the accepted robustness posterior summaries."""

from __future__ import annotations

import csv
import pandas as pd

from pipeline.config import ROBUSTNESS_ESTIMATION_DIR

FIT_DIR = ROBUSTNESS_ESTIMATION_DIR
SOURCE = FIT_DIR / "posterior_summary.csv"
FINAL = FIT_DIR / "posterior_summary_final.csv"
MANIFEST = FIT_DIR / "accepted_chain_sources.csv"


def main() -> None:
    frame = pd.read_csv(SOURCE)
    rhat_columns = [column for column in frame if column.endswith("_rhat")]
    max_rhat = frame[rhat_columns].max(axis=1)
    problems = frame.loc[
        (max_rhat > 1.05)
        | (frame["divergences"] > 0)
        | (frame["max_treedepth"] >= 12)
    ]
    if not problems.empty:
        raise RuntimeError(
            "Unresolved robustness fits:\n"
            + problems[["contest_id", "divergences", "max_treedepth"] + rhat_columns]
            .to_string(index=False)
        )
    frame.to_csv(FINAL, index=False)
    pd.DataFrame(
        {
            "contest_id": frame["contest_id"].astype(int),
            "mode": frame["mode"],
            "accepted_source": "initial",
        }
    ).to_csv(MANIFEST, index=False)
    print(
        f"Accepted {len(frame)} fits; max Rhat={max_rhat.max():.4f}, "
        f"divergences={int(frame['divergences'].sum())}, "
        f"max treedepth={int(frame['max_treedepth'].max())}"
    )


if __name__ == "__main__":
    main()
