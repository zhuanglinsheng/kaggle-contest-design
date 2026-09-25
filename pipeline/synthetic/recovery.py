"""Regenerate the PaperJK7 synthetic recovery tables with CmdStan.

The script uses the same parameter supports and likelihood for every experiment.
Table 1 reports the 90-day baseline with the effort-to-submission scale fixed
or estimated, Table 2 pools independent 90-day contests, and Table 3 compares
longer contests at a common integrated submission intensity per day.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

from pipeline.config import (
    SYNTHETIC_DIR,
    SYNTHETIC_POOLED_MODEL,
    SYNTHETIC_RESULTS_DIR,
    SYNTHETIC_SINGLE_MODEL,
)
from synthetic_data.synthetic_data import synthetic_data_simulation


OUT_DIR = SYNTHETIC_RESULTS_DIR / "recovery"
SINGLE_MODEL = SYNTHETIC_SINGLE_MODEL
POOLED_MODEL = SYNTHETIC_POOLED_MODEL
PARAMS = ("c_i", "c_j", "sigma", "lambda", "mu_0")
BASELINE_R = 15.0
SIGMA_TRUE = 2.0
LAMBDA_TRUE = 1.0
# Calibrated in pilot simulations so that the integrated submission intensity
# per day matches the 90-day baseline value (0.6185).  The calibration uses
# the same DGP seeds as the reported longer-horizon realizations.
LONG_HORIZON_R = {180: 18.6484375, 360: 47.4375}
CALIBRATED_RATE_PER_DAY = 0.6184661122311594


def simulate_contest(
    *,
    days: int,
    c_i: float,
    c_j: float,
    seed: int,
    ratio: float = BASELINE_R,
    estimate_r: int = 0,
) -> tuple[dict, dict]:
    start = datetime(2026, 1, 1)
    end = start + timedelta(days=days)
    result = synthetic_data_simulation(
        theta=1.0,
        c_i=c_i,
        c_j=c_j,
        sigma=SIGMA_TRUE,
        lamb=LAMBDA_TRUE,
        intensity_effort_ratio=ratio,
        hour_arrival_ub=1.0,
        start_time=start,
        end_time=end,
        time_unit=timedelta(hours=1),
        time_unit_2f=1 / 24,
        mu_0=0.0,
        seed_brownian=seed,
        seed_poisson=seed + 1,
        seed_uniform=seed + 2,
        seed_initial=seed + 3,
    )
    time_grid, effort_i, effort_j, _real, _perceived, observed, sub_i, sub_j = result
    data = {
        "theta": 1.0,
        "ratio": ratio,
        "estimate_r": estimate_r,
        "N_Delta": len(time_grid),
        "Delta2f": 1 / 24,
        "Ni": len(sub_i),
        "Nj": len(sub_j),
        "hat_t_i": [(t - start).total_seconds() / 3600 for t in sub_i],
        "hat_t_j": [(t - start).total_seconds() / 3600 for t in sub_j],
        "hat_y": observed.tolist(),
    }
    diagnostics = {
        "Ni": len(sub_i),
        "Nj": len(sub_j),
        "expected_Ni": float(ratio * effort_i.sum() / 24),
        "expected_Nj": float(ratio * effort_j.sum() / 24),
        "ratio": ratio,
    }
    return data, diagnostics


def pooled_data(contests: list[dict]) -> dict:
    ni_max = max(d["Ni"] for d in contests)
    nj_max = max(d["Nj"] for d in contests)

    def padded(values: list[float], length: int) -> list[float]:
        return values + [0.0] * (length - len(values))

    return {
        "N_tests": len(contests),
        "theta": 1.0,
        "ratio": 15.0,
        "Delta2f": 1 / 24,
        "N_Delta": contests[0]["N_Delta"],
        "Ni_arr": [d["Ni"] for d in contests],
        "Nj_arr": [d["Nj"] for d in contests],
        "Ni_max": ni_max,
        "Nj_max": nj_max,
        "hat_t_i_arr": [padded(d["hat_t_i"], ni_max) for d in contests],
        "hat_t_j_arr": [padded(d["hat_t_j"], nj_max) for d in contests],
        "hat_y_arr": [d["hat_y"] for d in contests],
    }


def run_chains(
    model: Path,
    data_path: Path,
    name: str,
    *,
    warmup: int,
    samples: int,
    seed: int,
) -> list[Path]:
    csv_paths = [OUT_DIR / f"{name}_chain_{chain}.csv" for chain in range(1, 5)]
    processes = []
    log_handles = []
    for chain, csv_path in enumerate(csv_paths, start=1):
        command = [
            str(model),
            "sample",
            f"num_warmup={warmup}",
            f"num_samples={samples}",
            "adapt",
            "delta=0.95",
            "algorithm=hmc",
            "engine=nuts",
            "max_depth=12",
            "random",
            f"seed={seed}",
            f"id={chain}",
            "data",
            f"file={data_path}",
            "output",
            f"file={csv_path}",
            "refresh=200",
        ]
        log_handle = (OUT_DIR / f"{name}_chain_{chain}.log").open("w")
        log_handles.append(log_handle)
        processes.append(
            subprocess.Popen(
                command,
                cwd=OUT_DIR,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
        )
    failures = [process.wait() for process in processes]
    for log_handle in log_handles:
        log_handle.close()
    if any(failures):
        raise RuntimeError(f"CmdStan failed for {name}: exit codes {failures}")
    return csv_paths


def read_chain(path: Path) -> dict[str, np.ndarray]:
    with path.open() as handle:
        lines = [line for line in handle if not line.startswith("#")]
    reader = csv.DictReader(lines)
    columns: dict[str, list[float]] = {}
    for row in reader:
        for key, value in row.items():
            columns.setdefault(key, []).append(float(value))
    return {key: np.asarray(values) for key, values in columns.items()}


def split_rhat(chains: list[np.ndarray]) -> float:
    n = min(len(chain) for chain in chains)
    half = n // 2
    split = np.asarray(
        [piece for chain in chains for piece in (chain[:half], chain[n - half :])]
    )
    within = float(np.mean(np.var(split, axis=1, ddof=1)))
    between = half * float(np.var(np.mean(split, axis=1), ddof=1))
    variance = (half - 1) * within / half + between / half
    return math.sqrt(variance / within)


def summarize(
    csv_paths: list[Path],
    true_values: dict[str, float],
    *,
    include_r: bool = False,
) -> dict:
    raw = [read_chain(path) for path in csv_paths]
    names = list(PARAMS) + (["r"] if include_r else [])
    result: dict[str, dict[str, float]] = {}
    for name in names:
        chain_values = [np.exp(chain["log_r"]) if name == "r" else chain[name] for chain in raw]
        values = np.concatenate(chain_values)
        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        lo, hi = np.quantile(values, [0.025, 0.975])
        truth = true_values[name]
        result[name] = {
            "true": truth,
            "mean": mean,
            "sd": sd,
            "rmsd": math.sqrt((mean - truth) ** 2 + sd**2),
            "q2_5": float(lo),
            "q97_5": float(hi),
            "rhat": split_rhat(chain_values),
        }
    result["diagnostics"] = {
        "divergences": int(sum(chain["divergent__"].sum() for chain in raw)),
        "max_treedepth": int(max(chain["treedepth__"].max() for chain in raw)),
    }
    return result


def write_data(name: str, data: dict) -> Path:
    path = OUT_DIR / f"{name}.json"
    path.write_text(json.dumps(data))
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument(
        "--only",
        choices=("all", "baseline", "pooled", "longer"),
        default="all",
        help="Run the full recovery suite or update one experiment group.",
    )
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    output = OUT_DIR / "recovery_results.json"
    if args.only == "all" or not output.exists():
        results: dict[str, dict] = {}
    else:
        results = json.loads(output.read_text())
    true_baseline = {"c_i": 1.2, "c_j": 1.5, "sigma": 2.0, "lambda": 1.0, "mu_0": 0.0, "r": 15.0}

    if args.only in ("all", "baseline"):
        results.pop("table1", None)
        results.pop("table2", None)
        baseline_data = json.loads(
            (SYNTHETIC_DIR / "baseline_data.json").read_text()
        )
        baseline_data["estimate_r"] = 0
        for name, estimate_r in (("baseline_fixed_r", 0), ("baseline_estimated_r", 1)):
            data = dict(baseline_data)
            data["estimate_r"] = estimate_r
            paths = run_chains(
                SINGLE_MODEL,
                write_data(name, data),
                name,
                warmup=args.warmup,
                samples=args.samples,
                seed=7100 + estimate_r * 100,
            )
            results[name] = summarize(paths, true_baseline, include_r=bool(estimate_r))

    if args.only in ("all", "pooled"):
        pool_contests = [simulate_contest(days=90, c_i=1.2, c_j=1.5, seed=3000 + 10 * i)[0] for i in range(20)]
        for size in (10, 20):
            name = f"pool{size}"
            paths = run_chains(
                POOLED_MODEL,
                write_data(name, pooled_data(pool_contests[:size])),
                name,
                warmup=args.warmup,
                samples=args.samples,
                seed=7300 + size,
            )
            results[name] = summarize(paths, true_baseline)
            results[name]["submission_counts"] = {
                "Ni": sum(d["Ni"] for d in pool_contests[:size]),
                "Nj": sum(d["Nj"] for d in pool_contests[:size]),
            }

    if args.only in ("all", "longer"):
        for days in (180, 360):
            name = f"days{days}"
            ratio = LONG_HORIZON_R[days]
            data, counts = simulate_contest(
                days=days,
                c_i=1.2,
                c_j=1.5,
                seed=5000 + days,
                ratio=ratio,
            )
            integrated_rate = (counts["expected_Ni"] + counts["expected_Nj"]) / days
            if not math.isclose(integrated_rate, CALIBRATED_RATE_PER_DAY, abs_tol=5e-5):
                raise RuntimeError(
                    f"submission-rate calibration failed for {days} days: {integrated_rate}"
                )
            truth = {"c_i": 1.2, "c_j": 1.5, "sigma": 2.0, "lambda": 1.0, "mu_0": 0.0}
            paths = run_chains(
                SINGLE_MODEL,
                write_data(name, data),
                name,
                warmup=args.warmup,
                samples=args.samples,
                seed=7500 + days,
            )
            results[name] = summarize(paths, truth)
            results[name]["submission_counts"] = counts

    output.write_text(json.dumps(results, indent=2) + "\n")
    print(output)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
