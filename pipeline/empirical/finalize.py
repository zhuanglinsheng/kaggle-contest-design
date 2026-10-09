#!/usr/bin/env python3
"""Assemble the accepted USD-100,000 contest fits and record provenance."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from shutil import copy2

from pipeline.config import MAIN_DATA_DIR, MAIN_ESTIMATION_DIR, manifest_path
from pipeline.empirical.estimate import stan_data


FIT_DIR = MAIN_ESTIMATION_DIR
def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    old_final = FIT_DIR / "posterior_summary_final.csv"
    required_ids = {int(row["contest_id"]) for row in read_rows(old_final)
                    if row["mode"] == "full"}
    base, sources, run_dirs, snapshots = {}, {}, {}, {}
    for mode, source_names in (
        ("full", ("full_current_data", "full_current_refinements", "chapter5_corrected", "chapter5_corrected_refinements", "chapter5_all_submissions", "chapter5_all_submissions_upper", "chapter5_all_submissions_tail", "chapter5_all_submissions_tail_upper", "chapter5_all_submissions_refinements", "chapter5_all_submissions_dense_refinements", "chapter5_all_submissions_dense_refinements_long")),
        ("early", ("early_deadline_corrected", "early_deadline_refinements", "chapter5_corrected", "chapter5_corrected_refinements", "chapter5_all_submissions", "chapter5_all_submissions_upper", "chapter5_all_submissions_tail", "chapter5_all_submissions_tail_upper", "chapter5_all_submissions_refinements", "chapter5_all_submissions_dense_refinements")),
    ):
        for source in source_names:
            path = FIT_DIR.parent / source / "posterior_summary.csv"
            if not path.exists():
                continue
            for row in read_rows(path):
                if row["mode"] != mode:
                    if source.startswith(("chapter5_corrected", "chapter5_all_submissions")):
                        continue
                    raise RuntimeError(f"Unexpected estimation mode in {path}")
                key = (int(row["contest_id"]), mode)
                base[key], sources[key] = row, source
                run_dirs[key] = path.parent / mode / f"contest_{key[0]}"
    required = {(contest_id, mode) for contest_id in required_ids
                for mode in ("full", "early")}
    if set(base) != required:
        raise RuntimeError(f"Matched re-estimation is incomplete: {len(base)}/{len(required)}")
    for key, row in base.items():
        rhats = [float(row[f"{p}_rhat"]) for p in
                 ("c_i", "c_j", "sigma", "lambda", "mu_0", "r")]
        if int(row["divergences"]) or not all(map(math.isfinite, rhats)) or max(rhats) > 1.01:
            raise RuntimeError(f"Matched fit requires refinement: {key}")
        raw_path = MAIN_DATA_DIR / f"contest_{key[0]}.json"
        raw_bytes = raw_path.read_bytes()
        raw = json.loads(raw_bytes)
        saved_input = json.loads((run_dirs[key] / "data.json").read_text())
        if saved_input != stan_data(raw, key[1]):
            raise RuntimeError(f"Current data do not match the fitted inputs: {key}")
        row["deadline_preserved"] = 1
        row["input_sha256"] = hashlib.sha256(raw_bytes).hexdigest()
        snapshot = (FIT_DIR / "input_snapshots" /
                    f"contest_{key[0]}_{row['input_sha256']}.json")
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if snapshot.exists() and snapshot.read_bytes() != raw_bytes:
            raise RuntimeError(f"Input snapshot checksum conflict: {snapshot}")
        if not snapshot.exists():
            copy2(raw_path, snapshot)
        snapshots[key] = snapshot

    # Preserve the previously published estimates before replacing both modes.
    backup = FIT_DIR.parent / "early_deadline_corrected" / "previous_posterior_summary_final.csv"
    if old_final.exists() and not backup.exists():
        copy2(old_final, backup)

    final = [base[key] for key in sorted(base)]
    write_rows(FIT_DIR / "posterior_summary_final.csv", final)
    manifest_file = FIT_DIR / "accepted_chain_sources.csv"
    manifest = [
        {
            "contest_id": key[0],
            "mode": key[1],
            "accepted_source": sources[key],
            "run_dir": manifest_path(run_dirs[key], manifest_file),
            "input_sha256": base[key]["input_sha256"],
            "raw_input_path": manifest_path(snapshots[key], manifest_file),
        }
        for key in sorted(base)
    ]
    write_rows(manifest_file, manifest)


if __name__ == "__main__":
    main()
