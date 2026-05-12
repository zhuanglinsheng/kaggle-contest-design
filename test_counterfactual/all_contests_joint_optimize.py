#!/usr/bin/env python3
"""Run theta/T counterfactual screening for all contests in CONTEST_PARAMETERS.

For each contest, the script checks whether there exists a counterfactual pair
(theta, T) such that total effort is weakly higher than the baseline and prize
theta is weakly lower than the baseline.  Results are written to the
counterfactual_results folder by default.

The effort metric is the deterministic mean‑path proxy described in the paper.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from metakaggle.contest_parameters import CONTEST_PARAMETERS  # noqa: E402


# ---------------------------------------------------------------------------
# Helper / model functions – deterministic mean‑path proxy
# ---------------------------------------------------------------------------


def _fn_gamma(u: float) -> float:
    """Gamma function from Ryvkin's model."""
    if not (-1 <= u <= 1):
        raise ValueError(f"_fn_gamma: -1 < u < 1, got {u}")
    if u == -1:
        return -math.inf
    if u == 1:
        return math.inf
    return u / (1.0 - u * u) + math.atanh(u)


def _fn_invgamma_approx(z: float) -> float:
    """Approximate inverse-gamma."""
    return np.arctan(0.856 * z) * 2.0 / np.pi


def _fn_rho(z: float, rho_i: float, rho_j: float) -> float:
    """Deterministic rho(z) for the mean‑path proxy."""
    gamma_i = _fn_gamma(rho_i)
    gamma_j = _fn_gamma(rho_j)

    # Handle infinite extremes
    if math.isinf(gamma_i) or math.isinf(gamma_j):
        if gamma_i > 0 and gamma_j > 0:
            if z == 0:
                return 0.0
            return 1.0 if z > 0 else -1.0
        if gamma_i > 0:
            return 1.0
        if gamma_j > 0:
            return 1.0
        return -1.0

    cdf = float(stats.norm.cdf(z))
    loc = cdf * (gamma_i + gamma_j) - gamma_j
    return _fn_invgamma_approx(loc)


def _effort_one_step(
    y: float,
    t_days: float,
    T_days: float,
    theta: float,
    sigma: float,
    c_i: float,
    c_j: float,
) -> tuple[float, float]:
    """Eq. effort intensities m_i, m_j at the deterministic mean."""
    remaining = T_days - t_days
    if remaining <= 0:
        return 0.0, 0.0

    s2 = sigma * sigma
    w_i = theta / (s2 * c_i)
    w_j = theta / (s2 * c_j)

    rho_i = (math.exp(w_i) + math.exp(-w_j) - 2.0) / (math.exp(w_i) - math.exp(-w_j))
    rho_j = (math.exp(w_j) + math.exp(-w_i) - 2.0) / (math.exp(w_j) - math.exp(-w_i))

    y_stderr = sigma * math.sqrt(remaining)
    z = y / y_stderr if y_stderr > 0 else 0.0
    rho_z = _fn_rho(z, rho_i, rho_j)

    density = float(stats.norm.pdf(y, loc=0.0, scale=y_stderr))

    K_factor = s2 / 2.0 * (_fn_gamma(rho_i) + _fn_gamma(rho_j)) * (1.0 - rho_z * rho_z)
    m_i = density * K_factor * (1.0 + rho_z)
    m_j = density * K_factor * (1.0 - rho_z)
    return float(m_i), float(m_j)


@dataclass
class ContestParams:
    """Minimal parameters for one contest (used by the counterfactual scan)."""
    contest_id: int
    theta0: float        # baseline prize  (k USD)
    T0_days: float       # baseline duration
    lamb: float
    sigma: float
    mu0: float
    c_i: float
    c_j: float
    r: float


# ---------------------------------------------------------------------------
# Core counterfactual functions
# ---------------------------------------------------------------------------


def total_effort_mean_path(
    theta: float,
    T_days: float,
    params: ContestParams,
    dt_days: float = 0.5,
) -> float:
    """Deterministic mean‑path total effort.

    The state y starts at mu0 and evolves as
        dy/dt = m_i(y, t) – m_j(y, t)
    (no Kalman‑filter correction because we only use the mean path).
    Total effort = Σ_t dt · (m_i + m_j).
    """
    y = float(params.mu0)
    total = 0.0
    t = 0.0
    while t < T_days:
        step = min(dt_days, T_days - t)
        m_i, m_j = _effort_one_step(
            y, t, T_days, theta,
            params.sigma, params.c_i, params.c_j,
        )
        total += (m_i + m_j) * step
        # Deterministic update (no Kalman correction)
        y += (m_i - m_j) * step
        t += step
    return total


def find_min_theta_for_no_loss(
    T: float,
    M0: float,
    params: ContestParams,
    theta_min: float,
    theta0: float,
    dt_days: float = 0.5,
) -> float | None:
    """Smallest theta ∈ [theta_min, theta0] s.t. M(T, theta) >= M0 – eps.

    Returns None when no feasible theta exists.
    """
    # quick guard: is theta0 itself feasible?
    M_theta0 = total_effort_mean_path(theta0, T, params, dt_days=dt_days)
    if M_theta0 < M0 - 1e-7:
        return None

    lo, hi = theta_min, theta0
    for _ in range(60):  # binary‑search refinement
        mid = 0.5 * (lo + hi)
        M_mid = total_effort_mean_path(mid, T, params, dt_days=dt_days)
        if M_mid >= M0 - 1e-7:
            hi = mid
        else:
            lo = mid
    return hi  # guaranteed ≤ theta0, and M(hi) >= M0 – eps


def max_M_with_theta_cap(
    params: ContestParams,
    T_min: float,
    T_max: float,
    dt_days: float = 0.5,
) -> dict[str, float]:
    """Max total effort when theta ≤ theta0 and T ∈ [T_min, T_max].

    Returns {'theta', 'T_days', 'M'} for the maximiser (grid search).
    """
    best = {"theta": params.theta0, "T_days": T_max, "M": -math.inf}
    # coarse‑fine grid to keep runtime low
    for T in np.linspace(T_min, T_max, 30):
        M = total_effort_mean_path(params.theta0, float(T), params, dt_days=dt_days)
        if M > best["M"]:
            best = {"theta": params.theta0, "T_days": float(T), "M": M}
    return best


# ---------------------------------------------------------------------------
# Contest‑parameter loading
# ---------------------------------------------------------------------------

JSON_DIR = REPO_ROOT / "metakaggle" / "__jsondata__"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "counterfactual_results"


def contest_params_from_row(row: tuple[Any, ...]) -> ContestParams | None:
    """Build ContestParams from one row of CONTEST_PARAMETERS and its json file."""
    (
        contest_id,
        _N_i,
        _N_j,
        prize,
        _public_data_pct,
        lamb,
        sigma,
        mu0,
        c_i,
        c_j,
        r,
        _lambda_mle,
        _sigma_mle,
    ) = row

    json_path = JSON_DIR / f"contest_{contest_id}.json"
    if not json_path.exists():
        return None

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Prefer the json value for theta because it is the exact estimation input;
    # fall back to the table prize if absent.
    theta0 = float(data.get("theta", prize))
    T0_days = float(data["N_Delta"]) / 24.0

    return ContestParams(
        contest_id=int(contest_id),
        theta0=theta0,
        T0_days=T0_days,
        lamb=float(lamb),
        sigma=float(sigma),
        mu0=float(mu0),
        c_i=float(c_i),
        c_j=float(c_j),
        r=float(r),
    )


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------


def scan_one_contest(
    params: ContestParams,
    *,
    theta_min_abs: float,
    theta_min_frac: float,
    min_t_multiplier: float,
    max_t_multiplier: float,
    grid: int,
    dt_days: float,
) -> dict[str, Any]:
    """Return one summary row for a contest."""
    theta_min = max(theta_min_abs, theta_min_frac * params.theta0)
    theta_min = min(theta_min, params.theta0)
    T_min = min_t_multiplier * params.T0_days
    T_max = max_t_multiplier * params.T0_days

    M0 = total_effort_mean_path(params.theta0, params.T0_days, params, dt_days=dt_days)

    feasible_rows: list[dict[str, float]] = []
    for T in np.linspace(T_min, T_max, grid):
        theta_star = find_min_theta_for_no_loss(
            float(T),
            M0,
            params,
            theta_min=theta_min,
            theta0=params.theta0,
            dt_days=dt_days,
        )
        if theta_star is None:
            continue
        M_star = total_effort_mean_path(theta_star, float(T), params, dt_days=dt_days)
        feasible_rows.append(
            {
                "theta": float(theta_star),
                "T_days": float(T),
                "M": float(M_star),
                "theta_reduction_pct": 100.0 * (params.theta0 - theta_star) / params.theta0,
                "T_change_pct": 100.0 * (float(T) - params.T0_days) / params.T0_days,
                "M_change_pct": 100.0 * (M_star - M0) / M0 if M0 != 0 else math.nan,
            }
        )

    feasible_rows.sort(key=lambda x: (x["theta"], abs(x["T_days"] - params.T0_days)))
    best = feasible_rows[0] if feasible_rows else None
    max_m = max_M_with_theta_cap(params, T_min=T_min, T_max=T_max, dt_days=dt_days)

    row: dict[str, Any] = {
        "contest_id": params.contest_id,
        "status": "ok",
        "has_pareto_opportunity": bool(
            best is not None
            and best["theta"] <= params.theta0
            and best["M"] >= M0 - 1e-7
            and (best["theta"] < params.theta0 - 1e-6 or best["M"] > M0 + 1e-7)
        ),
        "strict_theta_reduction": bool(best is not None and best["theta"] < params.theta0 - 1e-6),
        "theta0": params.theta0,
        "T0_days": params.T0_days,
        "M0": M0,
        "theta_best": best["theta"] if best else math.nan,
        "T_best_days": best["T_days"] if best else math.nan,
        "M_best": best["M"] if best else math.nan,
        "theta_reduction_pct_best": best["theta_reduction_pct"] if best else math.nan,
        "T_change_pct_best": best["T_change_pct"] if best else math.nan,
        "M_change_pct_best": best["M_change_pct"] if best else math.nan,
        "theta_for_max_M": max_m["theta"],
        "T_for_max_M_days": max_m["T_days"],
        "M_max_with_theta_cap": max_m["M"],
        "M_max_change_pct": 100.0 * (max_m["M"] - M0) / M0 if M0 != 0 else math.nan,
        "sigma": params.sigma,
        "lambda": params.lamb,
        "mu0": params.mu0,
        "c_i": params.c_i,
        "c_j": params.c_j,
        "r": params.r,
        "theta_min_used": theta_min,
        "T_min_days": T_min,
        "T_max_days": T_max,
        "n_feasible_grid_points": len(feasible_rows),
    }
    return row


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_results(rows: list[dict[str, Any]], output_dir: Path, config: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "counterfactual_summary.csv"
    json_path = output_dir / "counterfactual_summary.json"
    txt_path = output_dir / "counterfactual_summary.md"

    if rows:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"config": config, "results": rows}, f, indent=2)

    ok_rows = [r for r in rows if r["status"] == "ok"]
    pareto_rows = [r for r in ok_rows if r["has_pareto_opportunity"]]
    strict_rows = [r for r in ok_rows if r["strict_theta_reduction"]]
    top_rows = sorted(strict_rows, key=lambda r: r["theta_reduction_pct_best"], reverse=True)[:20]

    with txt_path.open("w", encoding="utf-8") as f:
        f.write("# Counterfactual theta/T screening results\n\n")
        f.write(f"- Total rows: {len(rows)}\n")
        f.write(f"- Successfully evaluated: {len(ok_rows)}\n")
        f.write(f"- Pareto opportunities (M>=M0, theta<=theta0): {len(pareto_rows)}\n")
        f.write(f"- Strict prize-reduction opportunities: {len(strict_rows)}\n\n")
        f.write("## Top strict prize reductions\n\n")
        f.write("| contest_id | theta0 | T0_days | theta_best | T_best_days | theta_reduction_pct | M0 | M_best |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in top_rows:
            f.write(
                f"| {r['contest_id']} | {r['theta0']:.4f} | {r['T0_days']:.2f} | "
                f"{r['theta_best']:.4f} | {r['T_best_days']:.2f} | "
                f"{r['theta_reduction_pct_best']:.2f} | {r['M0']:.4f} | {r['M_best']:.4f} |\n"
            )


def write_one_contest_result(row: dict[str, Any], output_dir: Path, config: dict[str, Any]) -> None:
    """Persist one contest immediately, so long runs leave local per‑contest output."""
    per_contest_dir = output_dir / "per_contest"
    per_contest_dir.mkdir(parents=True, exist_ok=True)
    contest_id = row["contest_id"]
    json_path = per_contest_dir / f"contest_{contest_id}.json"
    md_path = per_contest_dir / f"contest_{contest_id}.md"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"config": config, "result": row}, f, indent=2)

    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# Contest {contest_id} counterfactual result\n\n")
        f.write(f"- status: {row.get('status')}\n")
        if row.get("status") == "ok":
            f.write(f"- has Pareto opportunity: {row['has_pareto_opportunity']}\n")
            f.write(f"- strict theta reduction: {row['strict_theta_reduction']}\n")
            f.write(f"- baseline theta0: {row['theta0']:.6f}\n")
            f.write(f"- baseline T0 days: {row['T0_days']:.6f}\n")
            f.write(f"- baseline M0: {row['M0']:.6f}\n")
            f.write(f"- best theta: {row['theta_best']:.6f}\n")
            f.write(f"- best T days: {row['T_best_days']:.6f}\n")
            f.write(f"- best M: {row['M_best']:.6f}\n")
            f.write(f"- theta reduction pct: {row['theta_reduction_pct_best']:.6f}\n")
            f.write(f"- T change pct: {row['T_change_pct_best']:.6f}\n")
            f.write(f"- feasible grid points: {row['n_feasible_grid_points']}\n")
        else:
            f.write(f"- error: {row.get('error', '')}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--theta-min-abs", type=float, default=0.01)
    parser.add_argument("--theta-min-frac", type=float, default=0.0)
    parser.add_argument("--min-t-multiplier", type=float, default=0.25)
    parser.add_argument("--max-t-multiplier", type=float, default=3.0)
    parser.add_argument("--grid", type=int, default=80)
    parser.add_argument("--dt-hours", type=float, default=4.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    dt_days = args.dt_hours / 24.0
    config = {
        "theta_min_abs": args.theta_min_abs,
        "theta_min_frac": args.theta_min_frac,
        "min_t_multiplier": args.min_t_multiplier,
        "max_t_multiplier": args.max_t_multiplier,
        "grid": args.grid,
        "dt_hours": args.dt_hours,
        "effort_metric": "deterministic_mean_path_proxy",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for raw_row in CONTEST_PARAMETERS:
        contest_id = int(raw_row[0])
        params = contest_params_from_row(raw_row)
        if params is None:
            row = {"contest_id": contest_id, "status": "missing_json"}
            rows.append(row)
            write_one_contest_result(row, args.output_dir, config)
            print(f"[{contest_id}] skipped: missing json", flush=True)
            continue

        try:
            row = scan_one_contest(
                params,
                theta_min_abs=args.theta_min_abs,
                theta_min_frac=args.theta_min_frac,
                min_t_multiplier=args.min_t_multiplier,
                max_t_multiplier=args.max_t_multiplier,
                grid=args.grid,
                dt_days=dt_days,
            )
            rows.append(row)
            write_one_contest_result(row, args.output_dir, config)
            flag = "YES" if row["has_pareto_opportunity"] else "NO"
            print(
                f"[{contest_id}] {flag}: theta0={row['theta0']:.4f}, T0={row['T0_days']:.2f}, "
                f"theta_best={row['theta_best']:.4f}, T_best={row['T_best_days']:.2f}, "
                f"theta_reduction={row['theta_reduction_pct_best']:.2f}%",
                flush=True,
            )
        except Exception as exc:  # keep going so every contest is represented
            row = {"contest_id": contest_id, "status": "error", "error": repr(exc)}
            rows.append(row)
            write_one_contest_result(row, args.output_dir, config)
            print(f"[{contest_id}] error: {exc!r}", flush=True)

    write_results(rows, args.output_dir, config)

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    pareto_rows = [r for r in ok_rows if r.get("has_pareto_opportunity")]
    strict_rows = [r for r in ok_rows if r.get("strict_theta_reduction")]
    print("\nSaved results to:", args.output_dir)
    print(f"Successfully evaluated: {len(ok_rows)} / {len(rows)}")
    print(f"Pareto opportunities: {len(pareto_rows)}")
    print(f"Strict theta reductions: {len(strict_rows)}")


if __name__ == "__main__":
    main()