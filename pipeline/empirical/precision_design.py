"""Posterior-average precision policies for Table 7; no new model fits."""
from __future__ import annotations

import csv
import hashlib
import json
import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

from pipeline.empirical.counterfactual import (
    DATA_DIR, FIT_ROOT, OUT_DIR, PRIZE_DIVISOR, read_draws, write_csv,
)


def optimal_noise(q, mu2, sigma, weight, grid_size=513):
    """Maximize the posterior incentive term in x=1/sqrt(lambda).

    All draw-specific derivatives are negative beyond max((mu²-q)/sigma).
    Check the full-disclosure endpoint and all positive-to-negative derivative
    crossings on a log grid, with a doubled-grid numerical stability check.
    """
    critical = (mu2 - q) / sigma
    positive = critical[critical > 0]

    def value(x):
        variance = q + sigma * x
        return float(np.mean(weight * np.exp(-mu2 / (2 * variance)) / np.sqrt(variance)))

    def derivative(x):
        variance = q + sigma * x
        return float(np.mean(weight * np.exp(-mu2 / (2 * variance))
                             * sigma * (mu2 - variance) / (2 * variance**2.5)))

    if positive.size == 0:
        return 0.0, value(0.0), 1
    grid = np.r_[0.0, np.geomspace(float(positive.min()) * 1e-4,
                                 float(positive.max()) * 1.001, grid_size)]
    slopes = [derivative(x) for x in grid]
    candidates = [0.0]
    for a, b, da, db in zip(grid[:-1], grid[1:], slopes[:-1], slopes[1:]):
        if da > 0 and db < 0:
            candidates.append(brentq(derivative, a, b, xtol=1e-12, rtol=1e-12))
    values = [value(x) for x in candidates]
    index = int(np.argmax(values))
    return candidates[index], values[index], len(candidates)


def analyze(contest_id):
    path = DATA_DIR / f"contest_{contest_id}.json"
    raw = json.loads(path.read_text())
    draws = read_draws(contest_id)  # Checks accepted-chain/input provenance.
    theta = float(raw["theta"]) / PRIZE_DIVISOR
    duration = float(raw["N_Delta"]) * float(raw["Delta2f"])
    sigma = draws["sigma"]
    mu2 = draws["mu_0"] ** 2
    q = sigma**2 * duration
    cost = 2 / (1 / draws["c_i"] + 1 / draws["c_j"])
    weight = theta * duration / (cost * math.sqrt(2 * math.pi))
    x, incentive, candidates = optimal_noise(q, mu2, sigma, weight)
    x_check, incentive_check, _ = optimal_noise(q, mu2, sigma, weight, 1025)
    if not np.isclose(incentive, incentive_check, rtol=1e-9, atol=1e-12):
        raise RuntimeError(f"Unstable precision optimization: {contest_id}")
    if not np.isclose(x, x_check, rtol=1e-7, atol=1e-10):
        raise RuntimeError(f"Unstable precision policy: {contest_id}")
    scale = np.sqrt(q)
    abs_mu = np.sqrt(mu2)
    no_effort = scale * np.exp(-mu2 / (2*q)) / math.sqrt(2*math.pi) + abs_mu * (ndtr(abs_mu / scale) - .5)
    current_x = 1 / np.sqrt(draws["lambda"])
    current_variance = q + sigma * current_x
    baseline = float(np.mean(no_effort + weight * np.exp(-mu2 / (2*current_variance)) / np.sqrt(current_variance)))
    policy_output = float(np.mean(no_effort)) + incentive
    # Directional support integrates uncertainty in the optimal target as well
    # as current precision; gains use the single implementable policy above.
    draw_optimal_x = np.maximum(0, (mu2 - q) / sigma)
    increase = float(np.mean(draw_optimal_x < current_x))
    decrease = float(np.mean(draw_optimal_x > current_x))
    category = "Increase precision" if increase > .75 else (
        "Decrease precision" if decrease > .75 else "Direction uncertain")
    return dict(contest_id=contest_id, input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                posterior_draws=len(q), theta=theta, duration=duration,
                optimal_noise=x, optimal_precision=math.inf if x == 0 else x**-2,
                increase_probability=increase, decrease_probability=decrease,
                recommendation=category, current_output=baseline, policy_output=policy_output,
                output_gain_pct=100*(policy_output / baseline - 1),
                drawwise_finite_probability=float(np.mean(mu2 > q)),
                optimizer_candidates=candidates,
                doubled_grid_objective_difference=incentive_check-incentive)


def main():
    with (FIT_ROOT / "accepted_chain_sources.csv").open() as handle:
        ids = sorted(int(row["contest_id"]) for row in csv.DictReader(handle) if row["mode"] == "full")
    rows = []
    for contest_id in ids:
        rows.append(analyze(contest_id))
        print(f"{contest_id}: {rows[-1]['recommendation']}, gain={rows[-1]['output_gain_pct']:.3f}%", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "precision_policy_contests.csv", rows)
    groups = []
    for category in ("All contests", "Increase precision", "Decrease precision", "Direction uncertain"):
        selected = rows if category == "All contests" else [r for r in rows if r["recommendation"] == category]
        gains = [r["output_gain_pct"] for r in selected]
        quartiles = np.quantile(gains, [.25,.5,.75]).tolist() if gains else [None]*3
        groups.append(dict(group=category, n_contests=len(selected), share_pct=100*len(selected)/len(rows),
                           gain_q25_pct=quartiles[0], gain_median_pct=quartiles[1], gain_q75_pct=quartiles[2]))
    report = dict(n_contests=len(rows), full_disclosure_policies=sum(r["optimal_noise"] == 0 for r in rows),
                  finite_policies=sum(r["optimal_noise"] > 0 for r in rows),
                  negative_expected_gains=sum(r["output_gain_pct"] < 0 for r in rows),
                  groups=groups, interpretation="Single policy maximizing posterior mean calibrated output; current output integrates joint posterior including current precision.")
    (OUT_DIR / "precision_policy_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
