#!/usr/bin/env python3
"""Assemble the accepted USD-100,000 contest fits and record provenance."""

from __future__ import annotations

import csv
from pathlib import Path

from pipeline.config import MAIN_ESTIMATION_DIR


FIT_DIR = MAIN_ESTIMATION_DIR
ACCEPTED_RERUNS = {
    (21669, "full"): "reruns",
    (7115, "full"): "reruns",
    (3288, "full"): "reruns_long",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    base = {
        (int(row["contest_id"]), row["mode"]): row
        for row in read_rows(FIT_DIR / "posterior_summary.csv")
    }
    sources = {key: "initial" for key in base}
    for key, source in ACCEPTED_RERUNS.items():
        candidates = {
            (int(row["contest_id"]), row["mode"]): row
            for row in read_rows(FIT_DIR / source / "posterior_summary.csv")
        }
        base[key] = candidates[key]
        sources[key] = source

    final = [base[key] for key in sorted(base)]
    write_rows(FIT_DIR / "posterior_summary_final.csv", final)
    manifest = [
        {
            "contest_id": key[0],
            "mode": key[1],
            "accepted_source": sources[key],
        }
        for key in sorted(base)
    ]
    write_rows(FIT_DIR / "accepted_chain_sources.csv", manifest)


if __name__ == "__main__":
    main()
