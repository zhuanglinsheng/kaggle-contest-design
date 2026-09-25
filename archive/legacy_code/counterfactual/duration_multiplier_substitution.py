#!/usr/bin/env python3
"""Best-output-preserving prize-duration counterfactuals.

For each contest and duration multiplier a, set T_a = a * T_obs and compute
the minimum prize theta_a that keeps predicted expected best output at least
as large as its observed-design baseline. Predicted best output combines the
ODE-based expected total effort with the leading-order sampling component from
Lemma 2 of the paper.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
sys.path.insert(0, str(SCRIPT_DIR))

from all_contests_joint_optimize import (  # noqa: E402
    CONTEST_PARAMETERS,
    ContestParams,
    contest_params_from_row,
    total_expected_effort,
)


DEFAULT_OUTPUT_DIR = REPO_ROOT / "counterfactual_duration_multiplier_results"
DEFAULT_MULTIPLIERS = (1.0, 1.01, 1.02, 1.05, 1.10)
DEFAULT_EXCLUDE_CONTEST_IDS = (4031, 8540)


def sampling_component(T: float, params: ContestParams) -> float:
    """Leading-order selection gain sigma * sqrt(T) / sqrt(2*pi)."""
    if T <= 0:
        raise ValueError("contest duration must be positive")
    return params.sigma * math.sqrt(T) / math.sqrt(2.0 * math.pi)


def expected_best_output(
    theta: float,
    T: float,
    params: ContestParams,
    *,
    z_boundary: float,
    dz_target: float,
) -> float:
    """Predicted best output: one-half total effort plus sampling value."""
    effort = total_expected_effort(
        theta,
        T,
        params,
        z_boundary=z_boundary,
        dz_target=dz_target,
    )
    return 0.5 * effort + sampling_component(T, params)


def find_min_theta_fast(
    T: float,
    B0: float,
    params: ContestParams,
    *,
    theta_min: float,
    theta0: float,
    z_boundary: float,
    dz_target: float,
    iterations: int = 40,
) -> float | None:
    """Smallest prize in the search interval that preserves best output."""
    model_kwargs = {"z_boundary": z_boundary, "dz_target": dz_target}
    B_theta0 = expected_best_output(theta0, T, params, **model_kwargs)
    if B_theta0 < B0 - 1e-7:
        return None

    B_theta_min = expected_best_output(theta_min, T, params, **model_kwargs)
    if B_theta_min >= B0:
        return theta_min

    theta_check = np.linspace(theta_min, theta0, 9)
    output_check = np.array([
        expected_best_output(float(value), T, params, **model_kwargs)
        for value in theta_check
    ])
    tolerance = 1e-7 * max(1.0, float(np.max(np.abs(output_check))))
    if np.any(np.diff(output_check) < -tolerance):
        raise RuntimeError(
            f"expected best output is not monotone in prize for contest {params.contest_id}"
        )

    lo, hi = theta_min, theta0
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        B_mid = expected_best_output(mid, T, params, **model_kwargs)
        if B_mid >= B0:
            hi = mid
        else:
            lo = mid
    return hi


def evaluate_one(
    params: ContestParams,
    multiplier: float,
    *,
    b0: float,
    m0: float,
    theta_min_abs: float,
    theta_min_frac: float,
    z_boundary: float,
    dz_target: float,
) -> dict[str, Any]:
    theta_min = max(theta_min_abs, theta_min_frac * params.theta0)
    theta_min = min(theta_min, params.theta0)
    t_new = multiplier * params.T0_days

    if math.isclose(multiplier, 1.0, rel_tol=0.0, abs_tol=1e-12):
        return {
            "contest_id": params.contest_id,
            "multiplier": multiplier,
            "status": "ok",
            "theta0": params.theta0,
            "T0_days": params.T0_days,
            "T_new_days": t_new,
            "B0": b0,
            "B_new": b0,
            "M0": m0,
            "M_new": m0,
            "sampling0": sampling_component(params.T0_days, params),
            "sampling_new": sampling_component(t_new, params),
            "theta_new": params.theta0,
            "required_prize_change_pct": 0.0,
            "required_prize_change_abs": 0.0,
            "lower_prize": False,
        }

    theta_new = find_min_theta_fast(
        t_new,
        b0,
        params,
        theta_min=theta_min,
        theta0=params.theta0,
        z_boundary=z_boundary,
        dz_target=dz_target,
    )

    if theta_new is None:
        return {
            "contest_id": params.contest_id,
            "multiplier": multiplier,
            "status": "infeasible",
            "theta0": params.theta0,
            "T0_days": params.T0_days,
            "T_new_days": t_new,
            "B0": b0,
            "B_new": math.nan,
            "M0": m0,
            "M_new": math.nan,
            "sampling0": sampling_component(params.T0_days, params),
            "sampling_new": sampling_component(t_new, params),
            "theta_new": math.nan,
            "required_prize_change_pct": math.nan,
            "required_prize_change_abs": math.nan,
            "lower_prize": False,
        }

    m_new = total_expected_effort(
        theta_new,
        t_new,
        params,
        z_boundary=z_boundary,
        dz_target=dz_target,
    )
    b_new = 0.5 * m_new + sampling_component(t_new, params)
    prize_change_abs = theta_new - params.theta0
    prize_change_pct = 100.0 * prize_change_abs / params.theta0
    return {
        "contest_id": params.contest_id,
        "multiplier": multiplier,
        "status": "ok",
        "theta0": params.theta0,
        "T0_days": params.T0_days,
        "T_new_days": t_new,
        "B0": b0,
        "B_new": b_new,
        "M0": m0,
        "theta_new": theta_new,
        "M_new": m_new,
        "sampling0": sampling_component(params.T0_days, params),
        "sampling_new": sampling_component(t_new, params),
        "required_prize_change_pct": prize_change_pct,
        "required_prize_change_abs": prize_change_abs,
        "lower_prize": theta_new < params.theta0 - 1e-6,
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    multipliers = sorted({float(r["multiplier"]) for r in rows})
    for multiplier in multipliers:
        ok_rows = [
            r for r in rows
            if r["status"] == "ok" and float(r["multiplier"]) == multiplier
        ]
        if not ok_rows:
            summary.append({
                "multiplier": multiplier,
                "n_contests": 0,
                "contests_with_lower_prize": 0,
                "mean_required_prize_change_pct": math.nan,
                "median_required_prize_change_pct": math.nan,
                "n_at_least_1pct_lower": 0,
                "n_at_least_5pct_lower": 0,
                "n_at_least_10pct_lower": 0,
                "mean_required_prize_change_abs": math.nan,
                "median_required_prize_change_abs": math.nan,
            })
            continue

        pct = np.array([float(r["required_prize_change_pct"]) for r in ok_rows])
        absolute = np.array([float(r["required_prize_change_abs"]) for r in ok_rows])
        summary.append({
            "multiplier": multiplier,
            "n_contests": len(ok_rows),
            "contests_with_lower_prize": int(sum(bool(r["lower_prize"]) for r in ok_rows)),
            "mean_required_prize_change_pct": float(np.mean(pct)),
            "median_required_prize_change_pct": float(np.median(pct)),
            "n_at_least_1pct_lower": int(np.sum(pct <= -1.0)),
            "n_at_least_5pct_lower": int(np.sum(pct <= -5.0)),
            "n_at_least_10pct_lower": int(np.sum(pct <= -10.0)),
            "mean_required_prize_change_abs": float(np.mean(absolute)),
            "median_required_prize_change_abs": float(np.median(absolute)),
        })
    return summary


def write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_csv = output_dir / "duration_multiplier_detail.csv"
    summary_csv = output_dir / "duration_multiplier_summary.csv"
    summary_md = output_dir / "duration_multiplier_summary.md"

    if rows:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with detail_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    summary = summarize(rows)
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "multiplier",
            "n_contests",
            "contests_with_lower_prize",
            "n_at_least_1pct_lower",
            "n_at_least_5pct_lower",
            "n_at_least_10pct_lower",
            "mean_required_prize_change_pct",
            "median_required_prize_change_pct",
            "mean_required_prize_change_abs",
            "median_required_prize_change_abs",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)

    with summary_md.open("w", encoding="utf-8") as f:
        f.write("# Best-output-preserving prize-duration substitution\n\n")
        f.write(
            "For each duration multiplier a, the counterfactual duration is "
            "T_a = a T_obs. The required prize is the minimum prize that keeps "
            "predicted expected best output at least at the "
            "observed-design baseline.\n\n"
        )
        f.write(
            "| Duration multiplier | Contests with lower prize | "
            "At least 1% lower | At least 5% lower | At least 10% lower | "
            "Mean required prize change | Median required prize change | "
            "Mean absolute prize change |\n"
        )
        f.write("|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in summary:
            f.write(
                f"| {row['multiplier']:.2f} | "
                f"{row['contests_with_lower_prize']}/{row['n_contests']} | "
                f"{row['n_at_least_1pct_lower']}/{row['n_contests']} | "
                f"{row['n_at_least_5pct_lower']}/{row['n_contests']} | "
                f"{row['n_at_least_10pct_lower']}/{row['n_contests']} | "
                f"{row['mean_required_prize_change_pct']:.2f}% | "
                f"{row['median_required_prize_change_pct']:.2f}% | "
                f"{row['mean_required_prize_change_abs']:.2f} |\n"
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--multipliers",
        type=float,
        nargs="+",
        default=list(DEFAULT_MULTIPLIERS),
    )
    parser.add_argument("--theta-min-abs", type=float, default=0.01)
    parser.add_argument("--theta-min-frac", type=float, default=0.0)
    parser.add_argument("--z-boundary", type=float, default=10.0)
    parser.add_argument("--dz-target", type=float, default=0.01)
    parser.add_argument(
        "--exclude-contests",
        type=int,
        nargs="*",
        default=list(DEFAULT_EXCLUDE_CONTEST_IDS),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    rows: list[dict[str, Any]] = []
    excluded = set(args.exclude_contests)
    for raw_row in CONTEST_PARAMETERS:
        contest_id = int(raw_row[0])
        if contest_id in excluded:
            continue

        params = contest_params_from_row(raw_row)
        if params is None:
            for multiplier in args.multipliers:
                rows.append({
                    "contest_id": contest_id,
                    "multiplier": multiplier,
                    "status": "missing_json",
            })
            continue

        m0 = total_expected_effort(
            params.theta0,
            params.T0_days,
            params,
            z_boundary=args.z_boundary,
            dz_target=args.dz_target,
        )
        b0 = 0.5 * m0 + sampling_component(params.T0_days, params)
        ordered_multipliers = sorted(set(float(value) for value in args.multipliers))
        duration_output = np.array([
            expected_best_output(
                params.theta0,
                params.T0_days * multiplier,
                params,
                z_boundary=args.z_boundary,
                dz_target=args.dz_target,
            )
            for multiplier in ordered_multipliers
        ])
        tolerance = 1e-7 * max(1.0, float(np.max(np.abs(duration_output))))
        if np.any(np.diff(duration_output) < -tolerance):
            raise RuntimeError(
                f"expected best output is not monotone over the requested "
                f"duration multipliers for contest {contest_id}"
            )
        for multiplier in args.multipliers:
            rows.append(
                evaluate_one(
                    params,
                    float(multiplier),
                    b0=b0,
                    m0=m0,
                    theta_min_abs=args.theta_min_abs,
                    theta_min_frac=args.theta_min_frac,
                    z_boundary=args.z_boundary,
                    dz_target=args.dz_target,
                )
            )

    write_outputs(rows, args.output_dir)
    print(f"Saved results to {args.output_dir}")

    for row in summarize(rows):
        print(
            f"a={row['multiplier']:.2f}: "
            f"lower prize {row['contests_with_lower_prize']}/{row['n_contests']}, "
            f"mean required prize change {row['mean_required_prize_change_pct']:.2f}%"
        )


if __name__ == "__main__":
    main()
