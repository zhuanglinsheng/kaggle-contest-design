"""Coarse exact-HJB scan of precision regimes at fixed prize and horizon."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import csv
from pathlib import Path

import numpy as np

from pipeline.theory.joint_design_exact import full_disclosure_output, solve_output


INITIAL_VARIANCES = (0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0)
PRIZES = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
PRECISIONS = tuple(float(value) for value in np.logspace(-1.5, 1.5, 13)) + (float("inf"),)
OUTPUT = Path(__file__).resolve().parents[2] / "results" / "theory" / "precision_regimes.csv"


def _solve(arguments: tuple[float, float, float]) -> tuple[float, float, float, float]:
    initial, prize, precision = arguments
    if np.isinf(precision):
        output = full_disclosure_output(
            theta=prize,
            horizon=1.0,
            initial_variance=initial,
            dt=1 / 1000,
        )
        return initial, prize, precision, output
    scale = np.sqrt(initial + 1.0)
    spacing = min(0.025 * scale, 0.08 * precision ** (-0.25))
    base_steps = max(600, int(np.ceil(2 * precision * initial)))
    for refinement in (1, 2):
        try:
            result = solve_output(
                theta=prize,
                horizon=1.0,
                precision=precision,
                initial_variance=initial,
                dy=spacing,
                dt=1 / (base_steps * refinement),
                extent=6.0,
            )
            return initial, prize, precision, result.output
        except RuntimeError:
            pass
    return initial, prize, precision, float("nan")


def main() -> None:
    jobs = [
        (initial, prize, precision)
        for initial in INITIAL_VARIANCES
        for prize in PRIZES
        for precision in PRECISIONS
    ]
    with ProcessPoolExecutor(max_workers=4) as executor:
        rows = list(executor.map(_solve, jobs, chunksize=3))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("S0", "theta", "lambda", "output"))
        writer.writerows(rows)

    for initial in INITIAL_VARIANCES:
        for prize in PRIZES:
            profile = sorted(
                (row for row in rows if row[0] == initial and row[1] == prize),
                key=lambda row: row[2],
            )
            values = np.asarray([row[3] for row in profile])
            if np.isnan(values).all():
                print(f"S0={initial:4g} theta={prize:4g} all solves failed")
                continue
            index = int(np.nanargmax(values))
            location = (
                "interior" if 0 < index < len(profile) - 1 else
                "low" if index == 0 else "high"
            )
            print(
                f"S0={initial:4g} theta={prize:4g} "
                f"best_lambda={profile[index][2]:8.3g} location={location:8s} "
                f"gain_vs_low={values[index] - values[0]:.6g} "
                f"gain_vs_high={values[index] - values[-1]:.6g}"
                f" failed={int(np.isnan(values).sum())}"
            )


if __name__ == "__main__":
    main()
