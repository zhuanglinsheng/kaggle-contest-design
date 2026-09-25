#!/usr/bin/env python3
"""Run theta/T counterfactual screening for all contests in CONTEST_PARAMETERS.

For each contest, the script checks whether there exists a counterfactual pair
(theta, T) such that total effort is weakly higher than the baseline and prize
theta is weakly lower than the baseline.  Results are written to the
counterfactual_results folder by default.

The effort metric is the theorem-defined expected total effort

    M(tilde_y_0, 0) = sigma * sqrt(T) * q(z_0),

where q solves the boundary-value ODE in the paper's equilibrium theorem. The
code deliberately does not replace this expectation with effort integrated
along a deterministic mean state path.
"""

from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import linalg, stats

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from metakaggle.contest_parameters import CONTEST_PARAMETERS  # noqa: E402


# ---------------------------------------------------------------------------
# Helper / model functions
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


def _rho_parameters(theta: float, sigma: float, c_i: float, c_j: float) -> tuple[float, float]:
    """Return rho_i and rho_j using a cancellation-safe parameterization."""
    if theta <= 0 or sigma <= 0 or c_i <= 0 or c_j <= 0:
        raise ValueError("theta, sigma, c_i, and c_j must all be positive")

    w_i = theta / (sigma * sigma * c_i)
    w_j = theta / (sigma * sigma * c_j)
    denominator = math.expm1(w_i) - math.expm1(-w_j)
    if denominator <= 0:
        raise ValueError("invalid equilibrium parameter denominator")

    rho_i = (math.expm1(w_i) + math.expm1(-w_j)) / denominator
    rho_j = (math.expm1(w_j) + math.expm1(-w_i)) / (
        math.expm1(w_j) - math.expm1(-w_i)
    )
    eps = 1e-12
    return (
        float(np.clip(rho_i, -1.0 + eps, 1.0 - eps)),
        float(np.clip(rho_j, -1.0 + eps, 1.0 - eps)),
    )


def _gamma_array(u: np.ndarray | float) -> np.ndarray | float:
    """Vectorized gamma(u) from equation (eq-gamma)."""
    return u / (1.0 - np.asarray(u) ** 2) + np.arctanh(u)


@lru_cache(maxsize=20_000)
def _solve_total_effort_q(
    theta: float,
    sigma: float,
    c_i: float,
    c_j: float,
    z_max: float,
    dz_target: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the theorem's linear boundary-value ODE for q on [-z_max,z_max].

    The infinite-boundary conditions q(+/- infinity)=0 are imposed at a
    finite, adaptively chosen truncation boundary.  The equilibrium paper uses
    the analytical inverse-gamma approximation in its empirical implementation,
    so the same rho(z) approximation is used here.
    """
    rho_i, rho_j = _rho_parameters(theta, sigma, c_i, c_j)
    gamma_sum = float(_gamma_array(rho_i) + _gamma_array(rho_j))

    intervals = max(400, int(math.ceil(2.0 * z_max / dz_target)))
    if intervals % 2:
        intervals += 1
    z_grid = np.linspace(-z_max, z_max, intervals + 1)
    dz = float(z_grid[1] - z_grid[0])
    z = z_grid[1:-1]

    loc = stats.norm.cdf(z) * gamma_sum - float(_gamma_array(rho_j))
    rho_z = (2.0 / np.pi) * np.arctan(0.856 * loc)
    exp_term = np.exp(-0.5 * z * z)
    a_minus = gamma_sum * (1.0 - rho_z * rho_z) / (2.0 * math.sqrt(2.0 * math.pi))
    a_plus = a_minus * rho_z

    drift = 4.0 * a_plus * exp_term + z
    source = 4.0 * a_minus * exp_term
    inv_dz2 = 1.0 / (dz * dz)
    lower = inv_dz2 - drift / (2.0 * dz)
    diagonal = np.full_like(z, -2.0 * inv_dz2 - 1.0)
    upper = inv_dz2 + drift / (2.0 * dz)

    # scipy.linalg.solve_banded stores the upper, main, and lower diagonals by
    # column. Boundary values are zero, so they add nothing to the right side.
    banded = np.zeros((3, z.size))
    banded[0, 1:] = upper[:-1]
    banded[1, :] = diagonal
    banded[2, :-1] = lower[1:]
    q_interior = linalg.solve_banded((1, 1), banded, -source, check_finite=False)

    min_q = float(np.min(q_interior))
    if min_q < -1e-7:
        raise RuntimeError(f"total-effort ODE produced materially negative q: {min_q}")
    q_interior = np.maximum(q_interior, 0.0)
    q_grid = np.concatenate(([0.0], q_interior, [0.0]))
    z_grid.setflags(write=False)
    q_grid.setflags(write=False)
    return z_grid, q_grid


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


def total_expected_effort(
    theta: float,
    T_days: float,
    params: ContestParams,
    *,
    z_boundary: float = 10.0,
    dz_target: float = 0.01,
) -> float:
    """The theorem-defined expected total effort M(tilde_y_0, 0)."""
    if T_days <= 0:
        raise ValueError("contest duration must be positive")
    if dz_target <= 0 or z_boundary <= 0:
        raise ValueError("ODE boundary and grid spacing must be positive")

    z0 = params.mu0 / (params.sigma * math.sqrt(T_days))
    adaptive_boundary = max(float(z_boundary), math.ceil(abs(z0) + 2.0))
    z_grid, q_grid = _solve_total_effort_q(
        float(theta),
        float(params.sigma),
        float(params.c_i),
        float(params.c_j),
        float(adaptive_boundary),
        float(dz_target),
    )
    q0 = float(np.interp(z0, z_grid, q_grid))
    return params.sigma * math.sqrt(T_days) * q0


def find_min_theta_for_no_loss(
    T: float,
    M0: float,
    params: ContestParams,
    theta_min: float,
    theta0: float,
    z_boundary: float = 10.0,
    dz_target: float = 0.01,
) -> float | None:
    """Smallest theta ∈ [theta_min, theta0] s.t. M(T, theta) >= M0 – eps.

    Returns None when no feasible theta exists.
    """
    # quick guard: is theta0 itself feasible?
    effort_kwargs = {"z_boundary": z_boundary, "dz_target": dz_target}
    M_theta0 = total_expected_effort(theta0, T, params, **effort_kwargs)
    if M_theta0 < M0 - 1e-7:
        return None

    M_theta_min = total_expected_effort(theta_min, T, params, **effort_kwargs)
    if M_theta_min >= M0:
        return theta_min

    # The paper proves prize monotonicity analytically only for the symmetric,
    # small-w approximation. Verify it numerically before using bisection for
    # the contest-specific asymmetric model.
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
    for _ in range(60):  # binary‑search refinement
        mid = 0.5 * (lo + hi)
        M_mid = total_expected_effort(mid, T, params, **effort_kwargs)
        if M_mid >= M0:
            hi = mid
        else:
            lo = mid
    return hi  # guaranteed ≤ theta0, and M(hi) >= M0 – eps


def max_M_with_theta_cap(
    params: ContestParams,
    T_min: float,
    T_max: float,
    z_boundary: float = 10.0,
    dz_target: float = 0.01,
) -> dict[str, float]:
    """Max total effort when theta ≤ theta0 and T ∈ [T_min, T_max].

    Returns {'theta', 'T_days', 'M'} for the maximiser (grid search).
    """
    best = {"theta": params.theta0, "T_days": T_max, "M": -math.inf}
    # coarse‑fine grid to keep runtime low
    for T in np.linspace(T_min, T_max, 30):
        M = total_expected_effort(
            params.theta0,
            float(T),
            params,
            z_boundary=z_boundary,
            dz_target=dz_target,
        )
        if M > best["M"]:
            best = {"theta": params.theta0, "T_days": float(T), "M": M}
    return best


# ---------------------------------------------------------------------------
# Contest‑parameter loading
# ---------------------------------------------------------------------------

JSON_DIR = REPO_ROOT / "metakaggle" / "__jsondata__"
STRUCTURAL_PARAMETERS_CSV = (
    REPO_ROOT / "validation" / "results" / "structural_full_parameters.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "counterfactual_results"


@lru_cache(maxsize=1)
def _duration_hours_by_contest() -> dict[int, float]:
    """Fallback contest durations from the exported structural input summary."""
    durations: dict[int, float] = {}
    if not STRUCTURAL_PARAMETERS_CSV.exists():
        return durations
    with STRUCTURAL_PARAMETERS_CSV.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            durations[int(record["contest_id"])] = float(record["full_N_Delta"])
    return durations


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
    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # Prefer the json value for theta because it is the exact estimation
        # input; fall back to the appendix table prize if absent.
        theta0 = float(data.get("theta", prize))
        T0_days = float(data["N_Delta"]) / 24.0
    else:
        duration_hours = _duration_hours_by_contest().get(int(contest_id))
        if duration_hours is None:
            return None
        theta0 = float(prize)
        T0_days = duration_hours / 24.0

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
    z_boundary: float,
    dz_target: float,
) -> dict[str, Any]:
    """Return one summary row for a contest."""
    theta_min = max(theta_min_abs, theta_min_frac * params.theta0)
    theta_min = min(theta_min, params.theta0)
    T_min = min_t_multiplier * params.T0_days
    T_max = max_t_multiplier * params.T0_days

    M0 = total_expected_effort(
        params.theta0,
        params.T0_days,
        params,
        z_boundary=z_boundary,
        dz_target=dz_target,
    )

    feasible_rows: list[dict[str, float]] = []
    for T in np.linspace(T_min, T_max, grid):
        theta_star = find_min_theta_for_no_loss(
            float(T),
            M0,
            params,
            theta_min=theta_min,
            theta0=params.theta0,
            z_boundary=z_boundary,
            dz_target=dz_target,
        )
        if theta_star is None:
            continue
        M_star = total_expected_effort(
            theta_star,
            float(T),
            params,
            z_boundary=z_boundary,
            dz_target=dz_target,
        )
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
    max_m = max_M_with_theta_cap(
        params,
        T_min=T_min,
        T_max=T_max,
        z_boundary=z_boundary,
        dz_target=dz_target,
    )

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
            writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
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
    parser.add_argument("--z-boundary", type=float, default=10.0)
    parser.add_argument("--dz-target", type=float, default=0.01)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = {
        "theta_min_abs": args.theta_min_abs,
        "theta_min_frac": args.theta_min_frac,
        "min_t_multiplier": args.min_t_multiplier,
        "max_t_multiplier": args.max_t_multiplier,
        "grid": args.grid,
        "z_boundary": args.z_boundary,
        "dz_target": args.dz_target,
        "effort_metric": "theorem_expected_total_effort_ode",
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
                z_boundary=args.z_boundary,
                dz_target=args.dz_target,
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
