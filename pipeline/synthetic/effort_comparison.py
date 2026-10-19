"""Direct player-i effort comparison for the current PaperJK7 revision.

Run ``python -m pipeline.synthetic.effort_comparison``. No estimation is run.
Six configurations vary theta=1,2,4, lambda=.1,10, or the opponent cost c_j=.6
around the synthetic baseline. Only player i is displayed. This controlled
sensitivity exercise does not claim to represent
the empirical posterior or validate optimized counterfactuals.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/synthetic/effort_comparison/controlled_incentives"
SOURCE = ROOT / "pipeline/theory/joint_design_exact.py"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))

from pipeline.synthetic.nonlinear import Primitives, variance
from pipeline.theory.joint_design_exact import solve_equilibrium_adaptive


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def select():
    cases = [dict(key=f"theta_{theta}", label=rf"$\theta={theta}$",
                  primitives=asdict(replace(Primitives(), theta=float(theta))))
             for theta in (1, 2, 4)]
    for key, label, changes in (
        ("lambda_low", r"Low precision: $\lambda=0.1$", dict(lamb=.1)),
        ("lambda_high", r"High precision: $\lambda=10$", dict(lamb=10.)),
        ("opponent_cost", r"Lower opponent cost: $c_j=0.6$", dict(c_j=.6)),
    ):
        cases.append(dict(key=key, label=label,
                          primitives=asdict(replace(Primitives(), **changes))))
    selection = dict(rule="Top row: vary only theta over 1,2,4. Bottom row: independently change only lambda to .1, lambda to 10, or c_j to .6, keeping theta=1 and all other baseline primitives, including S0=4, fixed. User-authorized controlled comparison; no empirical representativeness claim.",
                     cases=cases)
    save(OUT / "selection.json", selection)
    return cases


def solve(case, level):
    p = Primitives(**case["primitives"])
    wi, wj = (p.theta / (p.sigma**2 * c) for c in (p.c_i, p.c_j))
    # Dimensionless x=y/(sigma*sqrt(T)), tau=t/T; effort unit sigma/sqrt(T).
    h = p.S0 / (p.sigma**2 * p.T)
    terminal_h = float(variance(p.T, p)) / (p.sigma**2 * p.T)
    dx = min(.0125, .2 * math.sqrt(terminal_h)) / 2**level
    nt = 1024 * 2**level
    extent = 7.0
    start = time.time()
    scaled = Primitives(T=1, theta=1, c_i=1 / wi, c_j=1 / wj, sigma=1,
                        lamb=p.lamb * p.sigma**2 * p.T**2, S0=h)
    eq = solve_equilibrium_adaptive(scaled, dy=dx, base_steps=nt, extent=extent)
    result = dict(time=eq.t, state=eq.y, effort=eq.effort[:, :, 0].astype(float),
                  diagnostics=dict(eq.diagnostics, level=level))
    if not np.all(np.isfinite(result["effort"])) or np.min(result["effort"]) < -1e-9:
        raise RuntimeError("Invalid nonlinear effort")
    print(f"{case['key']} level={level} solved in {time.time()-start:.1f}s", flush=True)
    return result


def evaluate(solution, case, times, z):
    p = Primitives(**case["primitives"])
    scale = p.sigma / math.sqrt(p.T)
    q = variance(times * p.T, p) / (p.sigma**2 * p.T) + 1 - times
    states = np.sqrt(q[:, None]) * z[None, :]
    effort = np.empty_like(states)
    for j, tau in enumerate(times):
        index = int(np.clip(np.searchsorted(solution["time"], tau, side="right") - 1,
                            0, len(solution["time"]) - 2))
        weight = (tau - solution["time"][index]) / (solution["time"][index+1] - solution["time"][index])
        profile = ((1-weight) * solution["effort"][index]
                   + weight * solution["effort"][index+1])
        effort[j] = np.interp(states[j], solution["state"], profile) * scale
    wi = p.theta / (p.sigma**2 * p.c_i)
    leading = scale * wi / np.sqrt(2 * np.pi * q[:, None]) * np.exp(-z[None, :]**2 / 2)
    return effort, leading


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = select()
    print("Selected", [c["key"] for c in cases], flush=True)
    z = np.linspace(-3, 3, 601)
    # Whole-horizon audit with extra observations near the terminal layer.
    times = np.unique(np.r_[np.linspace(0, 1, 257), 1-np.geomspace(1e-5, .1, 121)])
    metrics, curves = [], []
    for case in cases:
        coarse, fine = solve(case, 1), solve(case, 2)
        nonlinear, leading = evaluate(fine, case, times, z)
        previous, _ = evaluate(coarse, case, times, z)
        peak = float(nonlinear.max())
        discrepancy = float(np.max(np.abs(nonlinear-leading)) / peak)
        grid_error = float(np.max(np.abs(nonlinear-previous)) / peak)
        time_specific_grid_error = float(np.max(np.max(abs(nonlinear-previous), axis=1)/nonlinear.max(axis=1)))
        # Refine again if the numerical uncertainty is material.
        if max(grid_error, time_specific_grid_error) > .005:
            extra = solve(case, 3)
            nonlinear_extra, _ = evaluate(extra, case, times, z)
            grid_error = float(np.max(np.abs(nonlinear_extra-nonlinear)) / nonlinear_extra.max())
            time_specific_grid_error = float(np.max(np.max(abs(nonlinear_extra-nonlinear), axis=1)/nonlinear_extra.max(axis=1)))
            fine, nonlinear = extra, nonlinear_extra
            peak = float(nonlinear.max())
            discrepancy = float(np.max(np.abs(nonlinear-leading)) / peak)
        if max(grid_error, time_specific_grid_error) > .005:
            save(OUT / f"{case['key']}_unresolved.json", dict(
                case=case, maximum_grid_difference=grid_error,
                maximum_time_specific_grid_difference=time_specific_grid_error,
                status="Not accepted for manuscript: refinement difference exceeds 0.5% of peak effort."))
            raise RuntimeError(f"Unresolved nonlinear policy for {case['key']}: grid difference {grid_error:.1%}")
        display, display_leading = evaluate(fine, case, np.array([.5]), z)
        p = Primitives(**case["primitives"])
        row = dict(key=case["key"], w_i=p.theta/(p.sigma**2*p.c_i),
                   w_j=p.theta/(p.sigma**2*p.c_j), peak_effort=peak,
                   maximum_normalized_error=discrepancy,
                   maximum_grid_difference=grid_error,
                   maximum_time_specific_grid_difference=time_specific_grid_error,
                   maximum_time_specific_error=float(np.max(np.max(abs(nonlinear-leading), axis=1)/nonlinear.max(axis=1))),
                   solver_diagnostics=fine["diagnostics"],
                   displayed_slice_error=float(np.max(abs(display-display_leading))/display.max()))
        np.savez_compressed(OUT / f"{case['key']}_audit.npz", time_fraction=times,
                            z=z, nonlinear=nonlinear, leading=leading)
        metrics.append(row)
        curves.append((display[0], display_leading[0]))
        print(json.dumps(row), flush=True)
    save(OUT / "summary.json", dict(cases=cases, metrics=metrics,
        evaluation=dict(player="i", standardized_gap=[-3, 3], state_points=len(z),
                        time_points=len(times), time_fraction_range=[0, 1], displayed_time_fraction=.5,
                        normalization="Primary manuscript metric: maximum over time of maximum absolute effort difference divided by the contemporaneous peak nonlinear effort, on z in [-3,3]. Also retain the whole-grid peak-normalized metric and the plotted midpoint slice metric."),
        producer_sha256=digest(__file__), solver_sha256=digest(SOURCE),
        primitives_source_sha256=digest(ROOT / "pipeline/synthetic/nonlinear.py")))
    np.savez_compressed(OUT / "curves.npz", z=z, nonlinear=np.array([c[0] for c in curves]),
                        leading=np.array([c[1] for c in curves]))
    plot(cases, metrics, curves, z)


def plot(cases, metrics, curves, z):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "serif", "font.size": 12,
                         "mathtext.fontset": "stix", "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 3, figsize=(10.6, 6.3), sharex=True)
    for index, (ax, case, row, (nonlinear, leading)) in enumerate(zip(axes.flat, cases, metrics, curves)):
        ax.plot(z, nonlinear, color="black", linewidth=1.6)
        ax.plot(z, leading, color="black", linewidth=1.3, linestyle=(0, (5, 3)))
        title = case["label"] + (" (baseline)" if case["key"] == "theta_1" else "")
        parameter = rf"$w_i={row['w_i']:.2f}$"
        if case["key"] == "opponent_cost":
            title = "Lower opponent cost"
            parameter = rf"$c_j=0.6,\quad w_i={row['w_i']:.2f}$"
        ax.set_title(f"({chr(97+index)}) " + title + "\n" + parameter, fontsize=12)
        ax.text(.97, .94, rf"$E_i={100*row['maximum_time_specific_error']:.2f}\%$",
                transform=ax.transAxes, ha="right", va="top", fontsize=11)
        ax.set_xlabel(r"Perceived gap $\tilde y/\sqrt{Q_t}$")
        ax.set_ylabel(r"Player $i$ effort")
        ax.set_xticks([-3, -1.5, 0, 1.5, 3])
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color=".9", linewidth=.6)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3), useMathText=True)
    fig.legend(handles=[Line2D([], [], color="black", lw=1.6, label="Nonlinear equilibrium"),
                        Line2D([], [], color="black", lw=1.3, linestyle=(0, (5, 3)), label="First-order approximation")],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(.5, .0))
    fig.tight_layout(rect=(0, .07, 1, 1), w_pad=1.4, h_pad=1.8)
    fig.savefig(OUT / "effort_policy_comparison.pdf", bbox_inches="tight")
    fig.savefig(OUT / "effort_policy_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only", action="store_true", help="Redraw saved curves without resolving equilibria.")
    args = parser.parse_args()
    if args.plot_only:
        summary = json.loads((OUT / "summary.json").read_text())
        with np.load(OUT / "curves.npz") as curves:
            plot(summary["cases"], summary["metrics"],
                 list(zip(curves["nonlinear"], curves["leading"])), curves["z"])
        summary["render_source_sha256"] = digest(__file__)
        summary["render_only"] = "Saved curves redrawn; numerical metrics and calculation-producer hash retained unchanged."
        save(OUT / "summary.json", summary)
    else:
        main()
