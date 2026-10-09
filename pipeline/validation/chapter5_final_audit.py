"""Audit every accepted posterior draw and the fixed-grid precision diagnostic."""
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
from pipeline.validation.chapter5 import event_indices, fit_row, parameter_frame
from pipeline.validation.count_floor_audit import audit


def main():
    rows = []
    manifest_path = MAIN_ESTIMATION_DIR / "accepted_chain_sources.csv"
    manifest = pd.read_csv(manifest_path)
    for fit in manifest.itertuples():
        run = resolve_manifest_path(fit.run_dir, manifest_path)
        count = len(read_columns(run / "chain_1.csv")["lp__"])
        rows.append(dict(contest_id=fit.contest_id, mode=fit.mode,
                         **audit(run, count)))
    floor = pd.DataFrame(rows)
    floor.to_csv(MAIN_VALIDATION_DIR / "chapter5_accepted_count_floor_audit.csv", index=False)
    frame = parameter_frame("full")
    benchmarks = []
    for fit in frame.itertuples():
        raw = json.loads((MAIN_DATA_DIR / f"contest_{fit.contest_id}.json").read_text())
        indices = event_indices(raw)
        difference = np.asarray(raw["real_y"])[indices] - np.asarray(raw["hat_y"])[indices]
        denominator = raw["Delta2f"] * np.sum(difference**2)
        benchmarks.append(len(indices) / denominator if denominator > 0 else np.nan)
    frame["fixed_grid_benchmark"] = np.log1p(benchmarks)
    frame["log_precision"] = np.log1p(frame["lambda"])
    regression, _ = fit_row(frame, "log_precision", "fixed_grid_benchmark", "fixed_grid_precision")
    report = dict(accepted_fits=len(floor), audited_draws=int(floor.draws.sum()),
                  floor_affected_draws=int(floor.affected_draws.sum()),
                  maximum_floor_loglikelihood_advantage=float(floor.maximum_loglikelihood_advantage.max()),
                  fixed_grid_precision=regression)
    (MAIN_VALIDATION_DIR / "chapter5_final_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
