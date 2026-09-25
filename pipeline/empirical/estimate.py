#!/usr/bin/env python3
"""Run resumable PaperJK7 estimation for the 73 empirical contests.

The runner calls the CmdStan executable directly, launches four chains for one
contest in parallel, and moves to the next contest only after all chains finish.
Completed chain files are reused, so an interrupted batch can be resumed safely.
All prizes are expressed in common units of USD 100,000.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
import numpy as np

from pipeline.config import (
    EMPIRICAL_MODEL,
    MAIN_DATA_DIR,
    MAIN_ESTIMATION_DIR,
    PRIZE_DIVISOR_FROM_THOUSANDS,
    PRIZE_UNIT_USD,
)

DATA_DIR = MAIN_DATA_DIR
MODEL = EMPIRICAL_MODEL
OUTPUT_DIR = MAIN_ESTIMATION_DIR
EXCLUDED = {4031, 8540}
PARAMETERS = ("c_i", "c_j", "sigma", "lambda", "mu_0", "log_r")


def empirical_files() -> list[Path]:
    paths = []
    for path in DATA_DIR.glob("contest_*.json"):
        contest_id = int(path.stem.split("_")[-1])
        if contest_id not in EXCLUDED:
            paths.append(path)
    return sorted(paths, key=lambda p: int(p.stem.split("_")[-1]))


def stan_data(
    raw: dict,
    mode: str,
    *,
    prize_divisor: float = PRIZE_DIVISOR_FROM_THOUSANDS,
) -> dict:
    n_delta = int(raw["N_Delta"])
    if mode == "early":
        n_delta = max(2, int(math.floor(0.75 * n_delta)))
    times_i = [float(t) for t in raw["hat_t_i"] if 0 < float(t) <= n_delta]
    times_j = [float(t) for t in raw["hat_t_j"] if 0 < float(t) <= n_delta]
    return {
        # The JSON files store prizes in thousands of U.S. dollars. Dividing
        # by 100 expresses every contest prize in units of $100,000.
        "theta": float(raw["theta"]) / prize_divisor,
        "Delta2f": float(raw["Delta2f"]),
        "N_Delta": n_delta,
        "Ni": len(times_i),
        "Nj": len(times_j),
        "hat_t_i": times_i,
        "hat_t_j": times_j,
        "hat_y": [float(x) for x in raw["hat_y"][:n_delta]],
    }


def chain_paths(run_dir: Path) -> list[Path]:
    return [run_dir / f"chain_{chain}.csv" for chain in range(1, 5)]


def csv_complete(path: Path, samples: int) -> bool:
    if not path.exists():
        return False
    with path.open(errors="replace") as handle:
        return sum(1 for line in handle if line and not line.startswith("#")) >= samples + 1


def run_contest(
    contest_id: int,
    mode: str,
    data: dict,
    *,
    warmup: int,
    samples: int,
    seed: int,
    adapt_delta: float = 0.95,
    max_depth: int = 12,
) -> list[Path]:
    run_dir = OUTPUT_DIR / mode / f"contest_{contest_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    data_path = run_dir / "data.json"
    data_path.write_text(json.dumps(data))
    paths = chain_paths(run_dir)
    processes: list[tuple[subprocess.Popen, object]] = []
    for chain, output_path in enumerate(paths, start=1):
        if csv_complete(output_path, samples):
            continue
        log_handle = (run_dir / f"chain_{chain}.log").open("w")
        command = [
            str(MODEL),
            "sample",
            f"num_warmup={warmup}",
            f"num_samples={samples}",
            "adapt",
            f"delta={adapt_delta}",
            "algorithm=hmc",
            "engine=nuts",
            f"max_depth={max_depth}",
            "random",
            f"seed={seed}",
            f"id={chain}",
            "data",
            f"file={data_path}",
            "output",
            f"file={output_path}",
            "refresh=100",
        ]
        process = subprocess.Popen(
            command,
            cwd=run_dir,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((process, log_handle))
    failures = []
    for process, handle in processes:
        failures.append(process.wait())
        handle.close()
    if any(failures):
        raise RuntimeError(
            f"CmdStan failed for contest {contest_id} ({mode}): {failures}"
        )
    return paths


def read_columns(path: Path) -> dict[str, np.ndarray]:
    with path.open() as handle:
        reader = csv.DictReader(line for line in handle if not line.startswith("#"))
        columns: dict[str, list[float]] = {}
        for row in reader:
            for key, value in row.items():
                columns.setdefault(key, []).append(float(value))
    return {key: np.asarray(values) for key, values in columns.items()}


def split_rhat(chains: list[np.ndarray]) -> float:
    n = min(len(chain) for chain in chains)
    half = n // 2
    split = np.asarray(
        [part for chain in chains for part in (chain[:half], chain[n - half :])]
    )
    within = float(np.mean(np.var(split, axis=1, ddof=1)))
    between = half * float(np.var(np.mean(split, axis=1), ddof=1))
    variance = (half - 1) * within / half + between / half
    return math.sqrt(variance / within) if within > 0 else 1.0


def summarize(contest_id: int, mode: str, paths: list[Path]) -> dict:
    chains = [read_columns(path) for path in paths]
    row: dict[str, float | int | str] = {"contest_id": contest_id, "mode": mode}
    for parameter in PARAMETERS:
        values_by_chain = [chain[parameter] for chain in chains]
        values = np.concatenate(values_by_chain)
        reported = np.exp(values) if parameter == "log_r" else values
        name = "r" if parameter == "log_r" else parameter
        row[f"{name}_mean"] = float(np.mean(reported))
        row[f"{name}_sd"] = float(np.std(reported, ddof=1))
        row[f"{name}_q025"] = float(np.quantile(reported, 0.025))
        row[f"{name}_q975"] = float(np.quantile(reported, 0.975))
        row[f"{name}_rhat"] = split_rhat(values_by_chain)
    row["divergences"] = int(sum(chain["divergent__"].sum() for chain in chains))
    row["max_treedepth"] = int(max(chain["treedepth__"].max() for chain in chains))
    return row


def write_summary(rows: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "posterior_summary.csv"
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def existing_summary() -> list[dict]:
    path = OUTPUT_DIR / "posterior_summary.csv"
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--adapt-delta", type=float, default=0.95)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--modes", nargs="+", choices=("full", "early"), default=("full", "early"))
    parser.add_argument("--contest-ids", nargs="*", type=int)
    args = parser.parse_args()
    if not MODEL.exists():
        raise FileNotFoundError(f"Compile the Stan model first: {MODEL}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_config = {
        "prize_unit_usd": PRIZE_UNIT_USD,
        "source_prize_unit_usd": 1_000.0,
        "prize_divisor": PRIZE_DIVISOR_FROM_THOUSANDS,
        "positive_parameter_lower_bound": 0.01,
        "positive_parameter_upper_bound": None,
        "warmup": args.warmup,
        "samples": args.samples,
        "adapt_delta": args.adapt_delta,
        "max_depth": args.max_depth,
        "modes": args.modes,
    }
    (OUTPUT_DIR / "run_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n"
    )

    selected = set(args.contest_ids or [])
    summary_by_key = {
        (int(row["contest_id"]), row["mode"]): row for row in existing_summary()
    }
    for path in empirical_files():
        contest_id = int(path.stem.split("_")[-1])
        if selected and contest_id not in selected:
            continue
        raw = json.loads(path.read_text())
        for mode in args.modes:
            data = stan_data(raw, mode)
            paths = run_contest(
                contest_id,
                mode,
                data,
                warmup=args.warmup,
                samples=args.samples,
                seed=900000 + contest_id + (100000 if mode == "early" else 0),
                adapt_delta=args.adapt_delta,
                max_depth=args.max_depth,
            )
            summary_by_key[(contest_id, mode)] = summarize(contest_id, mode, paths)
            rows = [summary_by_key[key] for key in sorted(summary_by_key)]
            write_summary(rows)
            print(f"completed contest {contest_id} ({mode})", flush=True)


if __name__ == "__main__":
    main()
