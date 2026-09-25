from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "metakaggle" / "__jsondata__"
STAN_FILE = ROOT / "validation" / "real_data_early75_parameters.stan"
RESULTS_DIR = ROOT / "validation" / "results"
OUTPUT_DIR = RESULTS_DIR / "stan_outputs_early75"
SUMMARY_PATH = RESULTS_DIR / "structural_time_split_holdout_parameters.csv"
FAILURE_PATH = RESULTS_DIR / "structural_time_split_holdout_parameter_failures.csv"

SPLIT = 0.75
CHAINS = 4
PARALLEL_CHAINS = 1
ITER_WARMUP = 1000
ITER_SAMPLING = 2000
SEED_BASE = 20260625
MAX_N_DELTA = 2300
EXCLUDE_CONTESTS = {37190, 7456, 4031, 8540}
PARAMETERS = ["mu_0", "c_i", "c_j", "sigma", "lambda", "r"]


def contest_id_from_path(path: Path) -> int:
    return int(path.stem.split("_")[1])


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def early_times(times: list[float], cutoff: int) -> list[float]:
    out = []
    for t in times:
        value = float(t)
        if math.ceil(value) < cutoff:
            out.append(value)
    return out


def build_stan_data(data: dict) -> tuple[dict, dict]:
    full_n_delta = int(data["N_Delta"])
    cutoff = int(math.floor(SPLIT * full_n_delta))
    if cutoff < 2:
        raise ValueError(f"cutoff too small: {cutoff}")

    hat_t_i = early_times(data["hat_t_i"], cutoff)
    hat_t_j = early_times(data["hat_t_j"], cutoff)
    if len(hat_t_i) + len(hat_t_j) < 2:
        raise ValueError("fewer than two early submissions")

    stan_data = {
        "theta": float(data["theta"]),
        "Delta2f": float(data["Delta2f"]),
        "N_Delta": cutoff,
        "Ni": len(hat_t_i),
        "Nj": len(hat_t_j),
        "hat_t_i": hat_t_i,
        "hat_t_j": hat_t_j,
        "hat_y": [float(x) for x in data["hat_y"][:cutoff]],
    }
    meta = {
        "full_N_Delta": full_n_delta,
        "early_N_Delta": cutoff,
        "early_Ni": len(hat_t_i),
        "early_Nj": len(hat_t_j),
    }
    return stan_data, meta


def compile_model() -> CmdStanModel:
    # Mirrors the local metakaggle compilation style while keeping optimization modest.
    try:
        subprocess.run(
            ["make", "clean-all"],
            cwd=ROOT,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass
    return CmdStanModel(
        stan_file=str(STAN_FILE),
        cpp_options={"CXXFLAGS": "-O2", "LDFLAGS": "-headerpad_max_install_names"},
    )


def completed_contests() -> set[int]:
    if not SUMMARY_PATH.exists():
        return set()
    frame = pd.read_csv(SUMMARY_PATH)
    if "contest_id" not in frame.columns:
        return set()
    return set(frame["contest_id"].astype(int))


def append_row(path: Path, row: dict) -> None:
    frame = pd.DataFrame([row])
    header = not path.exists()
    frame.to_csv(path, mode="a", index=False, header=header)


def summarize_fit(contest_id: int, meta: dict, fit) -> dict:
    draws = fit.draws_pd(vars=PARAMETERS)
    row = {
        "contest_id": contest_id,
        "split": SPLIT,
        "chains": CHAINS,
        "iter_warmup": ITER_WARMUP,
        "iter_sampling": ITER_SAMPLING,
        **meta,
    }
    for parameter in PARAMETERS:
        values = draws[parameter].to_numpy(dtype=float)
        row[parameter] = float(np.mean(values))
        row[f"{parameter}_sd"] = float(np.std(values, ddof=1))
        row[f"{parameter}_q025"] = float(np.quantile(values, 0.025))
        row[f"{parameter}_q975"] = float(np.quantile(values, 0.975))
    return row


def contest_paths() -> list[Path]:
    paths = []
    for path in sorted(DATA_DIR.glob("contest_*.json")):
        contest_id = contest_id_from_path(path)
        if contest_id in EXCLUDE_CONTESTS:
            continue
        if not path.with_suffix(".pkl").exists():
            continue
        data = load_json(path)
        if int(data["N_Delta"]) > MAX_N_DELTA:
            continue
        paths.append(path)
    return paths


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = compile_model()
    done = completed_contests()
    paths = contest_paths()
    print(f"Found {len(paths)} contests; {len(done)} already completed.")

    for index, json_path in enumerate(paths, start=1):
        contest_id = contest_id_from_path(json_path)
        if contest_id in done:
            print(f"[{index}/{len(paths)}] contest {contest_id}: skip existing", flush=True)
            continue
        data = load_json(json_path)
        try:
            stan_data, meta = build_stan_data(data)
            contest_output_dir = OUTPUT_DIR / f"contest_{contest_id}"
            if contest_output_dir.exists():
                shutil.rmtree(contest_output_dir)
            contest_output_dir.mkdir(parents=True, exist_ok=True)

            print(
                f"[{index}/{len(paths)}] contest {contest_id}: "
                f"N={meta['early_N_Delta']}, events={meta['early_Ni'] + meta['early_Nj']}",
                flush=True,
            )
            fit = model.sample(
                data=stan_data,
                chains=CHAINS,
                parallel_chains=PARALLEL_CHAINS,
                iter_warmup=ITER_WARMUP,
                iter_sampling=ITER_SAMPLING,
                seed=SEED_BASE + contest_id,
                output_dir=str(contest_output_dir),
                show_progress=False,
                refresh=200,
            )
            row = summarize_fit(contest_id, meta, fit)
            append_row(SUMMARY_PATH, row)
            done.add(contest_id)
            print(f"[{index}/{len(paths)}] contest {contest_id}: saved parameters", flush=True)
        except Exception as exc:
            append_row(
                FAILURE_PATH,
                {
                    "contest_id": contest_id,
                    "split": SPLIT,
                    "error": repr(exc),
                },
            )
            print(f"[{index}/{len(paths)}] contest {contest_id}: FAILED {exc!r}", flush=True)

    print(f"Wrote parameter summary to {SUMMARY_PATH}")
    if FAILURE_PATH.exists():
        print(f"Failures, if any, are in {FAILURE_PATH}")


if __name__ == "__main__":
    main()
