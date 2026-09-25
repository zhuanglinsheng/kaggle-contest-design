#!/usr/bin/env python3
"""Rerun the three nonconverged USD-100,000 empirical fits."""

from __future__ import annotations

import csv
import json
from pipeline.config import MAIN_ESTIMATION_DIR
from pipeline.empirical import estimate as estimation


BASE_DIR = MAIN_ESTIMATION_DIR
RERUN_DIR = BASE_DIR / "reruns"
PROBLEM_GROUPS = ((3288, "full"), (21669, "full"), (7115, "full"))


def main() -> None:
    estimation.OUTPUT_DIR = RERUN_DIR
    rows = []
    for contest_id, mode in PROBLEM_GROUPS:
        raw = json.loads(
            (estimation.DATA_DIR / f"contest_{contest_id}.json").read_text()
        )
        paths = estimation.run_contest(
            contest_id,
            mode,
            estimation.stan_data(raw, mode),
            warmup=2000,
            samples=2000,
            seed=3100000 + contest_id,
            adapt_delta=0.995,
            max_depth=15,
        )
        rows.append(estimation.summarize(contest_id, mode, paths))
        estimation.write_summary(rows)
        print(f"rerun completed contest {contest_id} ({mode})", flush=True)

    base_path = BASE_DIR / "posterior_summary.csv"
    with base_path.open() as handle:
        base = {
            (int(row["contest_id"]), row["mode"]): row
            for row in csv.DictReader(handle)
        }
    for row in rows:
        base[(int(row["contest_id"]), str(row["mode"]))] = row
    final = [base[key] for key in sorted(base)]
    final_path = BASE_DIR / "posterior_summary_final.csv"
    with final_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(final[0]))
        writer.writeheader()
        writer.writerows(final)

    manifest_path = BASE_DIR / "accepted_chain_sources.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("contest_id", "mode", "accepted_source")
        )
        writer.writeheader()
        for key in sorted(base):
            writer.writerow({
                "contest_id": key[0],
                "mode": key[1],
                "accepted_source": "rerun" if key in PROBLEM_GROUPS else "initial",
            })


if __name__ == "__main__":
    main()
