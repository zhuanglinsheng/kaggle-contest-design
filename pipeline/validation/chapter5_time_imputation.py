"""Compare two preselected time-imputation seeds with the primary fits.

This is a sensitivity check, not repeated-sample inference. Preserve the same
focal pairs, horizons, model, and priors; validate each saved fitted input.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

from pipeline.config import MAIN_DATA_DIR, MAIN_ESTIMATION_DIR, MAIN_VALIDATION_DIR
from pipeline.empirical.estimate import stan_data

IDS = [7878, 4453, 4986, 3641, 23652, 37077, 21669, 3960]
SEEDS = [20261008, 20261009]
PARAMETERS = ["c_i", "c_j", "sigma", "lambda", "mu_0", "r"]


def main():
    root = MAIN_ESTIMATION_DIR.parent
    baseline = pd.read_csv(MAIN_ESTIMATION_DIR / "posterior_summary_final.csv").set_index(["contest_id", "mode"])
    rows = []
    for seed in SEEDS:
        sources = [root / f"chapter5_time_seed_{seed}", root / f"chapter5_time_seed_{seed}_endpoint",
                   root / f"chapter5_time_seed_{seed}_refinements"]
        accepted = {}
        for source in sources:
            if not (source / "posterior_summary.csv").exists():
                continue
            for row in pd.read_csv(source / "posterior_summary.csv").to_dict("records"):
                accepted[(int(row["contest_id"]), row["mode"])] = (row, source)
        required = {(cid, mode) for cid in IDS for mode in ("full", "early")}
        if set(accepted) != required:
            raise RuntimeError(f"Incomplete time-imputation fits: seed {seed}")
        for (cid, mode), (fit, source) in accepted.items():
            rhats = np.asarray([fit[f"{p}_rhat"] for p in PARAMETERS])
            if fit["divergences"] or not np.isfinite(rhats).all() or rhats.max() > 1.01:
                raise RuntimeError(f"Unconverged sensitivity fit: {seed}, {cid}, {mode}")
            path = root / "chapter5_score_audit" / f"seed_{seed}" / f"contest_{cid}.json"
            raw = json.loads(path.read_text())
            saved = json.loads((source / mode / f"contest_{cid}" / "data.json").read_text())
            if saved != stan_data(raw, mode):
                raise RuntimeError(f"Changed time-imputation input: {seed}, {cid}, {mode}")
            primary = baseline.loc[(cid, mode)]
            if primary.input_sha256 != hashlib.sha256((MAIN_DATA_DIR / f"contest_{cid}.json").read_bytes()).hexdigest():
                raise RuntimeError("Obsolete baseline estimates")
            for p in PARAMETERS:
                difference = fit[f"{p}_mean"] - primary[f"{p}_mean"]
                rows.append(dict(seed=seed, contest_id=cid, mode=mode, parameter=p,
                    baseline_mean=primary[f"{p}_mean"], alternate_mean=fit[f"{p}_mean"],
                    difference=difference, absolute_shift_in_baseline_sd=abs(difference) / primary[f"{p}_sd"],
                    relative_mean_change=difference / primary[f"{p}_mean"] if p != "mu_0" else np.nan,
                    intervals_overlap=max(primary[f"{p}_q025"], fit[f"{p}_q025"]) <= min(primary[f"{p}_q975"], fit[f"{p}_q975"])))
    frame = pd.DataFrame(rows)
    frame.to_csv(MAIN_VALIDATION_DIR / "chapter5_time_imputation.csv", index=False)
    summaries = []
    for (mode, parameter), work in frame.groupby(["mode", "parameter"]):
        summaries.append(dict(mode=mode, parameter=parameter, comparisons=len(work),
            median_absolute_sd_shift=float(work.absolute_shift_in_baseline_sd.median()),
            max_absolute_sd_shift=float(work.absolute_shift_in_baseline_sd.max()),
            intervals_overlap=int(work.intervals_overlap.sum()),
            median_absolute_relative_change=float(work.relative_mean_change.abs().median()) if parameter != "mu_0" else None))
    report = dict(contest_ids=IDS, seeds=SEEDS, summaries=summaries,
                  interpretation="Sensitivity conditional on reconstructed intraday times, not robustness to all possible time imputations")
    (MAIN_VALIDATION_DIR / "chapter5_time_imputation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
