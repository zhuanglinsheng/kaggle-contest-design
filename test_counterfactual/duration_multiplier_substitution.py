#!/usr/bin/env python3
"""Prize-duration substitution counterfactuals at fixed duration multipliers.

For each contest and duration multiplier a, set T_a = a * T_obs and compute
the minimum prize theta_a that keeps the theorem-defined expected total effort
at least as large as the observed-design baseline M(theta_obs, T_obs).
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
DEFAULT_MULTIPLIERS = (1.0, 1.1, 1.2, 1.5, 2.0, 3.0)
DEFAULT_EXCLUDE_CONTEST_IDS = (4031, 8540)


def find_min_theta_fast(
    T: float,
    M0: float,
    params: ContestParams,
    *,
    theta_min: float,
    theta0: float,
    z_boundary: float,
    dz_target: float,
    iterations: int = 40,
) -> float | None:
    effort_kwargs = {"z_boundary": z_boundary, "dz_target": dz_target}
    M_theta0 = total_expected_effort(theta0, T, params, **effort_kwargs)
    if M_theta0 < M0 - 1e-7:
        return None

    M_theta_min = total_expected_effort(theta_min, T, params, **effort_kwargs)
    if M_theta_min >= M0:
        return theta_min

    theta_check = np.linspace(theta_min, theta0, 9)
    effort_check = np.array([
        total_expected_effort(float(value), T, params, **effort_kwargs)
        for value in theta_check
    ])
    tolerance = 1e-7 * max(1.0, float(np.max(np.abs(effort_check))))
    if np.any(np.diff(effort_check) < -tolerance):
        raise RuntimeError(
            f"expected total effort is not monotone in prize for contest {params.contest_id}"
        )

    lo, hi = theta_min, theta0
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        M_mid = total_expected_effort(mid, T, params, **effort_kwargs)
        if M_mid >= M0:
            hi = mid
        else:
            lo = mid
    return hi


def evaluate_one(
    params: ContestParams,
    multiplier: float,
    *,
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
            "M0": m0,
            "theta_new": params.theta0,
            "M_new": m0,
            "required_prize_change_pct": 0.0,
            "required_prize_change_abs": 0.0,
            "lower_prize": False,
        }

    theta_new = find_min_theta_fast(
        t_new,
        m0,
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
            "M0": m0,
            "theta_new": math.nan,
            "M_new": math.nan,
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
    prize_change_abs = theta_new - params.theta0
    prize_change_pct = 100.0 * prize_change_abs / params.theta0
    return {
        "contest_id": params.contest_id,
        "multiplier": multiplier,
        "status": "ok",
        "theta0": params.theta0,
        "T0_days": params.T0_days,
        "T_new_days": t_new,
        "M0": m0,
        "theta_new": theta_new,
        "M_new": m_new,
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
            "mean_required_prize_change_pct",
            "median_required_prize_change_pct",
            "mean_required_prize_change_abs",
            "median_required_prize_change_abs",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)

    with summary_md.open("w", encoding="utf-8") as f:
        f.write("# Duration-multiplier prize substitution\n\n")
        f.write(
            "For each duration multiplier a, the counterfactual duration is "
            "T_a = a T_obs. The required prize is the minimum prize that keeps "
            "the theorem-defined expected total effort at least at the "
            "observed-design baseline.\n\n"
        )
        f.write(
            "| Duration multiplier | Contests with lower prize | "
            "Mean required prize change | Median required prize change | "
            "Mean absolute prize change |\n"
        )
        f.write("|---:|---:|---:|---:|---:|\n")
        for row in summary:
            f.write(
                f"| {row['multiplier']:.2f} | "
                f"{row['contests_with_lower_prize']}/{row['n_contests']} | "
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
        ordered_multipliers = sorted(set(float(value) for value in args.multipliers))
        duration_effort = np.array([
            total_expected_effort(
                params.theta0,
                params.T0_days * multiplier,
                params,
                z_boundary=args.z_boundary,
                dz_target=args.dz_target,
            )
            for multiplier in ordered_multipliers
        ])
        tolerance = 1e-7 * max(1.0, float(np.max(np.abs(duration_effort))))
        if np.any(np.diff(duration_effort) < -tolerance):
            raise RuntimeError(
                f"expected total effort is not monotone over the requested "
                f"duration multipliers for contest {contest_id}"
            )
        for multiplier in args.multipliers:
            rows.append(
                evaluate_one(
                    params,
                    float(multiplier),
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
