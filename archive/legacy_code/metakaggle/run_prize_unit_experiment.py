#!/usr/bin/env python3
"""Prize-unit experiment for the empirical contest with the largest fitted w."""

from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import metakaggle.run_revised_estimation as estimation


CONTEST_ID = 5144
OUTPUT_DIR = ROOT / "metakaggle" / "prize_unit_experiment"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--divisor", type=float, default=10.0)
    parser.add_argument("--label", default="ten_thousand_usd")
    args = parser.parse_args()
    raw_path = estimation.DATA_DIR / f"contest_{CONTEST_ID}.json"
    raw = json.loads(raw_path.read_text())
    data = estimation.stan_data(raw, "full", prize_divisor=args.divisor)

    estimation.OUTPUT_DIR = OUTPUT_DIR
    paths = estimation.run_contest(
        CONTEST_ID,
        args.label,
        data,
        warmup=1000,
        samples=1000,
        seed=2514400,
        adapt_delta=0.99,
        max_depth=14,
    )
    summary = estimation.summarize(CONTEST_ID, args.label, paths)
    estimation.write_summary([summary])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
