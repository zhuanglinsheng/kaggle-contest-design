from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from structural_effort_intensity_cv import (
    EPS,
    RESULTS_DIR,
    structural_intensity_from_parameters,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "metakaggle" / "__jsondata__"
FULL_PARAMETER_PATH = RESULTS_DIR / "structural_full_parameters.csv"
SPLIT = 0.75
EXCLUDE_CONTESTS = {37190, 7456, 4031, 8540}
MAX_N_DELTA = 2300


def contest_id_from_path(path: Path) -> int:
    return int(path.stem.split("_")[1])


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_parameters() -> dict[int, dict[str, float]]:
    frame = pd.read_csv(FULL_PARAMETER_PATH)
    return {
        int(row.contest_id): {
            "mu_0": float(row.mu_0),
            "c_i": float(row.c_i),
            "c_j": float(row.c_j),
            "sigma": float(row.sigma),
            "lambda": float(row["lambda"]),
            "r": float(row.r),
        }
        for _, row in frame.iterrows()
    }


def event_hours(times: list[float], cutoff: int, n_delta: int) -> list[int]:
    out = []
    for value in times:
        hour = int(math.ceil(float(value)))
        if cutoff <= hour < n_delta:
            out.append(hour)
    return sorted(out)


def count_events(times: list[float], start: int, end: int) -> int:
    total = 0
    for value in times:
        hour = int(math.ceil(float(value)))
        if start <= hour < end:
            total += 1
    return total


def time_rescaled_increments(event_idx: list[int], intensity: np.ndarray, cutoff: int) -> list[float]:
    increments = []
    previous = cutoff
    for hour in event_idx:
        if hour < previous:
            continue
        delta = float(np.sum(intensity[previous : hour + 1]))
        if delta > EPS and np.isfinite(delta):
            increments.append(delta)
        previous = hour + 1
    return increments


def scaled_intensities(
    data: dict,
    intensity_i: np.ndarray,
    intensity_j: np.ndarray,
    cutoff: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    early_i = count_events(data["hat_t_i"], 0, cutoff)
    early_j = count_events(data["hat_t_j"], 0, cutoff)
    expected_early_i = float(np.sum(intensity_i[:cutoff]))
    expected_early_j = float(np.sum(intensity_j[:cutoff]))

    contest_kappa = (early_i + early_j) / max(expected_early_i + expected_early_j, EPS)

    return {
        "raw_structural": (intensity_i, intensity_j),
        "early_contest_calibrated": (contest_kappa * intensity_i, contest_kappa * intensity_j),
    }


def add_oracle_global_calibration(
    counts: pd.DataFrame,
    increments: pd.DataFrame,
    blocks: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_counts = counts.loc[counts["model"] == "raw_structural"].copy()
    raw_increments = increments.loc[increments["model"] == "raw_structural"].copy()
    raw_blocks = blocks.loc[blocks["model"] == "raw_structural"].copy()

    kappa = raw_counts["observed_late"].sum() / max(raw_counts["expected_late"].sum(), EPS)

    raw_counts["model"] = "oracle_global_calibrated"
    raw_counts["expected_late"] = raw_counts["expected_late"] * kappa
    raw_counts["calibration_ratio"] = raw_counts["observed_late"] / np.maximum(
        raw_counts["expected_late"], EPS
    )
    raw_counts["poisson_z"] = (
        raw_counts["observed_late"] - raw_counts["expected_late"]
    ) / np.sqrt(np.maximum(raw_counts["expected_late"], EPS))

    raw_increments["model"] = "oracle_global_calibrated"
    raw_increments["delta_lambda"] = raw_increments["delta_lambda"] * kappa
    raw_increments["u_uniform"] = 1.0 - np.exp(-raw_increments["delta_lambda"])

    raw_blocks["model"] = "oracle_global_calibrated"
    raw_blocks["expected"] = raw_blocks["expected"] * kappa
    raw_blocks["pearson_component"] = (
        raw_blocks["observed"] - raw_blocks["expected"]
    ) ** 2 / np.maximum(raw_blocks["expected"], EPS)

    return (
        pd.concat([counts, raw_counts], ignore_index=True),
        pd.concat([increments, raw_increments], ignore_index=True),
        pd.concat([blocks, raw_blocks], ignore_index=True),
    )


def build_diagnostics_frame() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    params = load_parameters()
    count_rows = []
    increment_rows = []
    block_rows = []

    for json_path in sorted(DATA_DIR.glob("contest_*.json")):
        contest_id = contest_id_from_path(json_path)
        if contest_id in EXCLUDE_CONTESTS or contest_id not in params:
            continue
        data = load_json(json_path)
        n_delta = int(data["N_Delta"])
        if n_delta > MAX_N_DELTA:
            continue
        cutoff = int(math.floor(SPLIT * n_delta))
        intensity_i, intensity_j = structural_intensity_from_parameters(data, params[contest_id])

        intensity_variants = scaled_intensities(data, intensity_i, intensity_j, cutoff)
        for model, (model_intensity_i, model_intensity_j) in intensity_variants.items():
            for player, times, intensity in [
                ("i", data["hat_t_i"], model_intensity_i),
                ("j", data["hat_t_j"], model_intensity_j),
            ]:
                late_events = event_hours(times, cutoff, n_delta)
                observed = len(late_events)
                expected = float(np.sum(intensity[cutoff:n_delta]))
                z_score = (observed - expected) / math.sqrt(max(expected, EPS))
                count_rows.append(
                    {
                        "model": model,
                        "contest_id": contest_id,
                        "player": player,
                        "observed_late": observed,
                        "expected_late": expected,
                        "calibration_ratio": observed / max(expected, EPS),
                        "poisson_z": z_score,
                    }
                )

                for delta in time_rescaled_increments(late_events, intensity, cutoff):
                    increment_rows.append(
                        {
                            "model": model,
                            "contest_id": contest_id,
                            "player": player,
                            "delta_lambda": delta,
                            "u_uniform": 1.0 - math.exp(-delta),
                        }
                    )

                for block_size in [24, 48, 72]:
                    for start in range(cutoff, n_delta, block_size):
                        end = min(start + block_size, n_delta)
                        observed_block = count_events(times, start, end)
                        expected_block = float(np.sum(intensity[start:end]))
                        block_rows.append(
                            {
                                "model": model,
                                "contest_id": contest_id,
                                "player": player,
                                "block_size_hours": block_size,
                                "start_hour": start,
                                "end_hour": end,
                                "observed": observed_block,
                                "expected": expected_block,
                                "pearson_component": (observed_block - expected_block) ** 2
                                / max(expected_block, EPS),
                            }
                        )

    return pd.DataFrame(count_rows), pd.DataFrame(increment_rows), pd.DataFrame(block_rows)


def summarize_counts(counts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in counts.groupby("model", observed=False):
        observed = group["observed_late"].to_numpy(dtype=float)
        expected = group["expected_late"].to_numpy(dtype=float)
        z = group["poisson_z"].to_numpy(dtype=float)
        rows.append(
            {
                "model": model,
                "n_player_contests": int(len(group)),
                "observed_total": float(observed.sum()),
                "expected_total": float(expected.sum()),
                "total_ratio": float(observed.sum() / max(expected.sum(), EPS)),
                "mean_z": float(np.mean(z)),
                "sd_z": float(np.std(z, ddof=1)),
                "share_abs_z_le_1_96": float(np.mean(np.abs(z) <= 1.96)),
                "corr_observed_expected": float(np.corrcoef(observed, expected)[0, 1]),
            }
        )
    return pd.DataFrame(rows)


def summarize_time_rescaling(increments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in increments.groupby("model", observed=False):
        delta = group["delta_lambda"].to_numpy(dtype=float)
        uniform = group["u_uniform"].to_numpy(dtype=float)
        ks_exp = stats.kstest(delta, "expon", args=(0, 1))
        ks_uniform = stats.kstest(uniform, "uniform", args=(0, 1))
        rows.append(
            {
                "model": model,
                "n_intervals": int(len(group)),
                "mean_delta_lambda": float(np.mean(delta)),
                "median_delta_lambda": float(np.median(delta)),
                "ks_exp_stat": float(ks_exp.statistic),
                "ks_exp_p": float(ks_exp.pvalue),
                "ks_uniform_stat": float(ks_uniform.statistic),
                "ks_uniform_p": float(ks_uniform.pvalue),
            }
        )
    return pd.DataFrame(rows)


def summarize_overdispersion(blocks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, block_size), group in blocks.groupby(["model", "block_size_hours"], observed=False):
        observed = group["observed"].to_numpy(dtype=float)
        expected = group["expected"].to_numpy(dtype=float)
        pearson = group["pearson_component"].to_numpy(dtype=float)
        rows.append(
            {
                "model": model,
                "block_size_hours": int(block_size),
                "n_blocks": int(len(group)),
                "observed_total": float(observed.sum()),
                "expected_total": float(expected.sum()),
                "total_ratio": float(observed.sum() / max(expected.sum(), EPS)),
                "pearson_dispersion": float(pearson.sum() / max(len(group), 1)),
                "observed_mean": float(np.mean(observed)),
                "observed_variance": float(np.var(observed, ddof=1)),
                "variance_mean_ratio": float(np.var(observed, ddof=1) / max(np.mean(observed), EPS)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    if not FULL_PARAMETER_PATH.exists():
        raise FileNotFoundError(
            f"Missing {FULL_PARAMETER_PATH}. Run validation/export_full_parameters.py first."
        )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    counts, increments, blocks = build_diagnostics_frame()
    counts, increments, blocks = add_oracle_global_calibration(counts, increments, blocks)
    count_summary = summarize_counts(counts)
    time_summary = summarize_time_rescaling(increments)
    overdispersion_summary = summarize_overdispersion(blocks)

    counts.to_csv(RESULTS_DIR / "point_process_count_calibration.csv", index=False)
    increments.to_csv(RESULTS_DIR / "point_process_time_rescaling.csv", index=False)
    blocks.to_csv(RESULTS_DIR / "point_process_block_counts.csv", index=False)
    count_summary.to_csv(RESULTS_DIR / "point_process_count_summary.csv", index=False)
    time_summary.to_csv(RESULTS_DIR / "point_process_time_rescaling_summary.csv", index=False)
    overdispersion_summary.to_csv(RESULTS_DIR / "point_process_overdispersion_summary.csv", index=False)

    diagnostics = "\n".join(
        [
            "parameter_source: full_parameters_internal",
            "state_source: realized_public_leaderboard_path",
            "estimation_note: this is an internal diagnostic until early-75 parameters are substituted.",
        ]
    )
    (RESULTS_DIR / "point_process_diagnostics_note.txt").write_text(
        diagnostics + "\n", encoding="utf-8"
    )

    print(count_summary.to_string(index=False))
    print(time_summary.to_string(index=False))
    print(overdispersion_summary.to_string(index=False))


if __name__ == "__main__":
    main()
