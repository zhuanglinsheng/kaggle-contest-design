"""Collect reproducible numbers for the Chapter 5 text and appendix."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.config import (
    MAIN_DATA_DIR,
    MAIN_ESTIMATION_DIR,
    MAIN_VALIDATION_DIR,
    resolve_manifest_path,
)
from pipeline.empirical.estimate import read_columns


def main() -> None:
    estimates = pd.read_csv(MAIN_ESTIMATION_DIR / "posterior_summary_final.csv")
    diagnostics = pd.read_csv(MAIN_VALIDATION_DIR / "chapter5_contest_level.csv")
    regressions = pd.read_csv(MAIN_VALIDATION_DIR / "chapter5_regressions.csv")
    for row in estimates.itertuples():
        raw = MAIN_DATA_DIR / f"contest_{row.contest_id}.json"
        if row.input_sha256 != hashlib.sha256(raw.read_bytes()).hexdigest():
            raise RuntimeError(f"Cannot report obsolete estimates: {raw.name}")
    manifest_path = MAIN_ESTIMATION_DIR / "accepted_chain_sources.csv"
    with manifest_path.open() as handle:
        manifest = {int(row["contest_id"]): row for row in csv.DictReader(handle)
                    if row["mode"] == "full"}
    player_medians, contest_medians = [], []
    appendix_rows = []
    for _, row in diagnostics.iterrows():
        contest_id = int(row["contest_id"])
        run_dir = resolve_manifest_path(manifest[contest_id]["run_dir"], manifest_path)
        chains = [read_columns(run_dir / f"chain_{chain}.csv")
                  for chain in range(1, 5)]
        sigma = np.concatenate([chain["sigma"] for chain in chains])
        w = [float(row["theta"]) / (sigma**2 *
             np.concatenate([chain[cost] for chain in chains]))
             for cost in ("c_i", "c_j")]
        player_medians.extend(float(np.median(values)) for values in w)
        contest_medians.append(float(np.median(np.maximum(*w))))
        cells = [str(contest_id), str(int(row["Ni"])), str(int(row["Nj"])),
                 f"{row['theta']:.4f}", f"{row['public_data_pct']:.0f}"]
        cells.extend(f"{row[name]:.3f}" for name in
                     ("lambda", "sigma", "mu_0", "c_i", "c_j", "r",
                      "lambda_bench", "sigma_bench"))
        appendix_rows.append(" & ".join(r"\textcolor{blue}{" + cell + "}"
                                        for cell in cells) + r" \\")
    report = {
        "fits": len(estimates),
        "divergences": int(estimates.divergences.sum()),
        "maximum_rhat": float(estimates.filter(regex="_rhat$").max().max()),
        "estimation_sha256": hashlib.sha256((MAIN_ESTIMATION_DIR / "posterior_summary_final.csv").read_bytes()).hexdigest(),
        "median_player_posterior_median_w": float(np.median(player_medians)),
        "players_with_posterior_median_w_below_one": int(np.sum(np.array(player_medians) < 1)),
        "contests_with_posterior_median_max_w_below_one": int(np.sum(np.array(contest_medians) < 1)),
        "regressions": regressions.to_dict(orient="records"),
        "appendix_rows": "\n".join(appendix_rows) + "\n",
    }
    output = MAIN_VALIDATION_DIR / "chapter5_paper_numbers.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key != "appendix_rows"}, indent=2))


if __name__ == "__main__":
    main()
