#!/usr/bin/env python3
"""Rerun empirical groups with divergences or split R-hat above 1.01."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import metakaggle.run_revised_estimation as estimation


BASE_SUMMARY = ROOT / "metakaggle" / "revised_estimation" / "posterior_summary.csv"
RERUN_DIR = ROOT / "metakaggle" / "revised_estimation" / "reruns"


def problem_groups() -> list[tuple[int, str]]:
    with BASE_SUMMARY.open() as handle:
        rows = list(csv.DictReader(handle))
    groups = []
    for row in rows:
        max_rhat = max(float(value) for key, value in row.items() if key.endswith("_rhat"))
        divergences = int(float(row["divergences"]))
        if divergences > 0 or max_rhat > 1.01:
            groups.append((int(row["contest_id"]), row["mode"]))
    return groups


def main() -> None:
    RERUN_DIR.mkdir(parents=True, exist_ok=True)
    estimation.OUTPUT_DIR = RERUN_DIR
    rows = []
    for contest_id, mode in problem_groups():
        raw = json.loads(
            (estimation.DATA_DIR / f"contest_{contest_id}.json").read_text()
        )
        paths = estimation.run_contest(
            contest_id,
            mode,
            estimation.stan_data(raw, mode),
            warmup=1000,
            samples=1000,
            seed=1900000 + contest_id + (100000 if mode == "early" else 0),
            adapt_delta=0.99,
            max_depth=14,
        )
        rows.append(estimation.summarize(contest_id, mode, paths))
        estimation.write_summary(rows)
        print(f"rerun completed contest {contest_id} ({mode})", flush=True)


if __name__ == "__main__":
    main()
