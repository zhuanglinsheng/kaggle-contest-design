#!/usr/bin/env python3
"""Merge accepted reruns into the canonical empirical posterior summary."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIT_DIR = ROOT / "metakaggle" / "revised_estimation"
BASE = FIT_DIR / "posterior_summary.csv"
RERUN = FIT_DIR / "reruns" / "posterior_summary.csv"
FINAL = FIT_DIR / "posterior_summary_final.csv"
MANIFEST = FIT_DIR / "accepted_chain_sources.csv"


def read(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def diagnostics(row: dict[str, str]) -> tuple[int, float]:
    divergences = int(float(row["divergences"]))
    max_rhat = max(float(value) for key, value in row.items() if key.endswith("_rhat"))
    return divergences, max_rhat


def main() -> None:
    base = {(row["contest_id"], row["mode"]): row for row in read(BASE)}
    rerun = {(row["contest_id"], row["mode"]): row for row in read(RERUN)}
    manifest = []
    for key, revised in rerun.items():
        original = base[key]
        old_div, old_rhat = diagnostics(original)
        new_div, new_rhat = diagnostics(revised)
        accept = new_div < old_div or (new_div == old_div and new_rhat < old_rhat)
        if accept:
            base[key] = revised
        manifest.append({
            "contest_id": key[0],
            "mode": key[1],
            "accepted_source": "rerun" if accept else "initial",
            "initial_divergences": old_div,
            "initial_max_rhat": old_rhat,
            "accepted_divergences": diagnostics(base[key])[0],
            "accepted_max_rhat": diagnostics(base[key])[1],
        })
    final = [base[key] for key in sorted(base, key=lambda value: (int(value[0]), value[1]))]
    write(FINAL, final)
    write(MANIFEST, manifest)


if __name__ == "__main__":
    main()
