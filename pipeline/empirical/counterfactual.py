#!/usr/bin/env python3
"""Posterior counterfactuals for the PaperJK7 precision mechanism."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np

from pipeline.config import (
    COUNTERFACTUAL_DIR,
    MAIN_DATA_DIR,
    MAIN_ESTIMATION_DIR,
    PRIZE_DIVISOR_FROM_THOUSANDS,
)

FIT_ROOT = MAIN_ESTIMATION_DIR
FIT_DIR = FIT_ROOT / "full"
DATA_DIR = MAIN_DATA_DIR
OUT_DIR = COUNTERFACTUAL_DIR
MULTIPLIERS = (1.00, 1.01, 1.05, 1.10)
PRIZE_DIVISOR = PRIZE_DIVISOR_FROM_THOUSANDS


def normal_cdf(x: np.ndarray | float) -> np.ndarray | float:
    values = np.asarray(x, dtype=float)
    result = 0.5 * (1.0 + np.vectorize(math.erf)(values / math.sqrt(2.0)))
    return float(result) if result.ndim == 0 else result


def precision_best_response(mu0: float, sigma: float, duration: float) -> float:
    threshold = sigma * math.sqrt(duration)
    if abs(mu0) <= threshold:
        return math.inf
    return (sigma / (mu0 * mu0 - sigma * sigma * duration)) ** 2


def no_prize_best_output(mu0: float, sigma: float, duration: float) -> float:
    scale = sigma * math.sqrt(duration)
    z = abs(mu0) / scale
    return (
        scale * math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        + abs(mu0) * (float(normal_cdf(z)) - 0.5)
    )


def marginal_prize_output(
    mu0: float,
    sigma: float,
    duration: float,
    precision: float,
    cost: float,
) -> float:
    bar_s = 0.0 if math.isinf(precision) else sigma / math.sqrt(precision)
    q0 = sigma * sigma * duration + bar_s
    return duration * math.exp(-mu0 * mu0 / (2.0 * q0)) / (
        cost * math.sqrt(2.0 * math.pi * q0)
    )


def expected_best_output(
    mu0: float,
    theta: float,
    duration: float,
    precision: float,
    sigma: float,
    cost: float,
) -> float:
    return no_prize_best_output(mu0, sigma, duration) + theta * marginal_prize_output(
        mu0, sigma, duration, precision, cost
    )


def required_prize(
    baseline: float,
    *,
    mu0: float,
    duration: float,
    precision: float,
    sigma: float,
    cost: float,
    theta_min: float,
    theta_max: float,
) -> float:
    slope = marginal_prize_output(mu0, sigma, duration, precision, cost)
    required = max(theta_min, (baseline - no_prize_best_output(mu0, sigma, duration)) / slope)
    return required if required <= theta_max + 1e-10 else math.nan


def read_draws(contest_id: int) -> dict[str, np.ndarray]:
    chains: list[dict[str, list[float]]] = []
    with (FIT_ROOT / "accepted_chain_sources.csv").open() as handle:
        manifest = {
            (int(row["contest_id"]), row["mode"]): row["accepted_source"]
            for row in csv.DictReader(handle)
        }
    source = manifest[(contest_id, "full")]
    contest_dir = (
        FIT_DIR / f"contest_{contest_id}"
        if source == "initial"
        else FIT_ROOT / source / "full" / f"contest_{contest_id}"
    )
    for chain in range(1, 5):
        path = contest_dir / f"chain_{chain}.csv"
        with path.open() as handle:
            reader = csv.DictReader(line for line in handle if not line.startswith("#"))
            columns: dict[str, list[float]] = {}
            for row in reader:
                for name in ("c_i", "c_j", "sigma", "lambda", "mu_0"):
                    columns.setdefault(name, []).append(float(row[name]))
            chains.append(columns)
    return {
        name: np.concatenate([np.asarray(chain[name]) for chain in chains])
        for name in ("c_i", "c_j", "sigma", "lambda", "mu_0")
    }


def analyze_contest(contest_id: int, theta_min: float) -> tuple[dict, list[dict]]:
    raw = json.loads((DATA_DIR / f"contest_{contest_id}.json").read_text())
    draws = read_draws(contest_id)
    theta0 = float(raw["theta"]) / PRIZE_DIVISOR
    duration0 = float(raw["N_Delta"]) * float(raw["Delta2f"])
    finite = np.abs(draws["mu_0"]) > draws["sigma"] * math.sqrt(duration0)
    precision_row = {
        "contest_id": contest_id,
        "finite_precision_probability": float(np.mean(finite)),
        "posterior_draws": int(finite.size),
    }

    detail: list[dict] = []
    for draw in range(finite.size):
        mu0 = float(draws["mu_0"][draw])
        sigma = float(draws["sigma"][draw])
        observed_precision = float(draws["lambda"][draw])
        c_i = float(draws["c_i"][draw])
        c_j = float(draws["c_j"][draw])
        cost = 2.0 / (1.0 / c_i + 1.0 / c_j)
        baseline = expected_best_output(
            mu0, theta0, duration0, observed_precision, sigma, cost
        )
        for multiplier in MULTIPLIERS:
            duration = multiplier * duration0
            for rule in ("fixed", "optimized"):
                precision = (
                    observed_precision
                    if rule == "fixed"
                    else precision_best_response(mu0, sigma, duration)
                )
                theta = required_prize(
                    baseline,
                    mu0=mu0,
                    duration=duration,
                    precision=precision,
                    sigma=sigma,
                    cost=cost,
                    theta_min=theta_min,
                    theta_max=theta0,
                )
                detail.append({
                    "contest_id": contest_id,
                    "draw": draw,
                    "duration_multiplier": multiplier,
                    "precision_rule": rule,
                    "theta_required": theta,
                    "prize_change_pct": 100.0 * (theta - theta0) / theta0,
                })
    return precision_row, detail


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_frontier(detail: list[dict]) -> list[dict]:
    contest_draw_means: dict[tuple[int, float, str], list[float]] = {}
    for row in detail:
        value = float(row["prize_change_pct"])
        if math.isfinite(value):
            key = (int(row["contest_id"]), float(row["duration_multiplier"]), str(row["precision_rule"]))
            contest_draw_means.setdefault(key, []).append(value)
    rows: list[dict] = []
    for multiplier in MULTIPLIERS:
        for rule in ("fixed", "optimized"):
            values = np.asarray([
                np.mean(draw_values)
                for (contest_id, a, p), draw_values in contest_draw_means.items()
                if a == multiplier and p == rule
            ])
            rows.append({
                "duration_multiplier": multiplier,
                "precision_rule": rule,
                "n_contests": int(values.size),
                "lower_prize": int(np.sum(values < -1e-8)),
                "at_least_5pct_lower": int(np.sum(values <= -5.0)),
                "mean_change_pct": float(np.mean(values)),
                "median_change_pct": float(np.median(values)),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--theta-min", type=float, default=0.0001)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    precision_rows: list[dict] = []
    detail: list[dict] = []
    for contest_dir in sorted(FIT_DIR.glob("contest_*")):
        contest_id = int(contest_dir.name.split("_")[-1])
        row, contest_detail = analyze_contest(contest_id, args.theta_min)
        precision_rows.append(row)
        detail.extend(contest_detail)
    write_csv(OUT_DIR / "precision_recommendations.csv", precision_rows)
    write_csv(OUT_DIR / "counterfactual_draws.csv", detail)
    write_csv(OUT_DIR / "counterfactual_summary.csv", summarize_frontier(detail))


if __name__ == "__main__":
    main()
