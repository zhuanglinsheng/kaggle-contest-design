"""Reproducible nonlinear-HJB joint-design grid, including both precision endpoints.

This is numerical evidence, not a proof or an exhaustive continuous optimization.
Failures are retained and screened with analytical output upper bounds; a failed
potentially competitive grid design makes a cell unresolved, never inferior.
The utility and delay shape floors are 64. The unquantified universal threshold
in Proposition 1 is NOT asserted to be at most 64.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import tempfile
import time

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="joint-plane-mpl-"))
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import solve_bvp, simpson
from scipy.special import ndtr

from archive.experiments.exact_joint_design import filtering_variance, solve_exact
from pipeline.config import ROOT
from pipeline.theory.zero_signal_series import coefficients

S0, GAMMA, AMPLITUDE = 1.0, 64.0, 3.0
B0 = math.sqrt(S0/(2*math.pi))
RATE = GAMMA/B0
POWER = GAMMA
# Normalize the no-prize deadline at sigma=1 to one.
LOG_D = (math.log(AMPLITUDE*RATE)-RATE*(math.sqrt(2/(2*math.pi))-B0)
         -math.log(2*math.sqrt(4*math.pi))-math.log(POWER))
SIGMAS = np.array([.04, .08, .16, .32, .55, .8, 1., 1.2, 1.5, 1.8, 2.2, 2.6])
OUT = ROOT/"results/theory/numerical_joint_plane_strong64"
FIELDS = ["sigma", "alpha", "T", "lambda", "level", "B", "error", "status"]


def tail(b):
    return AMPLITUDE*math.exp(-RATE*(b-B0))


def delay(t):
    return 0.0 if t == 0 else math.exp(LOG_D+POWER*math.log(t))


def zero_design(sigma):
    def equation(log_t):
        t = math.exp(log_t)
        q = S0+sigma*sigma*t
        b = math.sqrt(q/(2*math.pi))
        return (LOG_D+math.log(POWER)+(POWER-1)*log_t
                -math.log(AMPLITUDE*RATE)-2*math.log(sigma)
                +math.log(2*math.sqrt(2*math.pi*q))+RATE*(b-B0))
    t = math.exp(brentq(equation, -50, (math.log(AMPLITUDE)-LOG_D)/POWER))
    b = math.sqrt((S0+sigma*sigma*t)/(2*math.pi))
    ce = 2*POWER*delay(t)/(sigma*sigma)
    return t, tail(b)+delay(t), ce


@lru_cache(maxsize=1)
def central_series():
    a = 1/math.sqrt(2*math.pi)
    return np.array([float(p[0])*a**(n+1)/(n+1)
                     for n, p in enumerate(coefficients(28))])


def full_profile_output(sigma, alpha, horizon, level):
    """Infinite-precision output from its self-similar linear BVP.

    beta''+(x+2*w*f'*delta')*beta'-beta=-w*delta'. Subtract the
    analytic no-effort beta before solving, avoiding the terminal-time spike
    and the discretization bias of diffusing an absolute-value terminal payoff.
    """
    w = alpha/(sigma*sigma)
    if w > 40:
        raise RuntimeError("full-disclosure BVP beyond validated w range")
    delta_grid = np.linspace(0., 1., 128001 if level>=2 else 32001 if level else 8001)
    k_one = (1+math.sinh(w)/w)/2
    ratio = (delta_grid+np.sinh(w*delta_grid)/w)/(2*k_one)

    def ode(x, y):
        delta = np.interp(2*ndtr(x)-1, ratio, delta_grid)
        derivative = (2*k_one*np.exp(-x*x/2)/math.sqrt(2*math.pi)
                      /((1+np.cosh(w*delta))/2))
        drift = w*np.tanh(w*delta/2)*derivative
        source = w*derivative+2*drift*(ndtr(x)-.5)
        return np.vstack((y[1], y[0]-(x+2*drift)*y[1]-source))

    tiny = math.exp(-min(w+3, 44))
    n = 450 if level>=2 else 300 if level else 140
    x = np.unique(np.r_[0., np.geomspace(tiny, .05, n), np.linspace(.05, 9., 2*n)])
    guess = np.vstack((min(w, 3)*np.exp(-x*x/2)/math.sqrt(2*math.pi),
                       -x*min(w, 3)*np.exp(-x*x/2)/math.sqrt(2*math.pi)))
    solution = solve_bvp(ode, lambda left,right: np.array([left[1],right[0]]),
                         x, guess, tol=2e-9 if level>=2 else 1e-8 if level else 2e-6, max_nodes=25000 if level>=2 else 15000)
    if not solution.success:
        raise RuntimeError("full-disclosure BVP: "+solution.message)
    s = S0/(sigma*sigma*horizon)
    weighted = solution.y[0]*np.exp(-solution.x**2/(2*s))/math.sqrt(2*math.pi*s)
    gain = 2*simpson(weighted, x=solution.x)
    return math.sqrt((S0+sigma*sigma*horizon)/(2*math.pi))+sigma*math.sqrt(horizon)*gain


@lru_cache(maxsize=1024)
def finite_baseline(sigma, horizon, precision, level, dx):
    s0 = S0/(sigma*sigma*horizon)
    result = solve_exact(w=0., ell=precision*sigma*sigma*horizon*horizon,
                         s0=s0, x0=0., dx=dx, steps=840 if level>=2 else 420 if level else 150, extent=7)
    return sigma*math.sqrt(horizon)*result.best_output


def sample_key(row):
    return tuple(float(row[k]) for k in FIELDS[:5])


def calculate(job):
    np.seterr(over="raise", invalid="raise", divide="raise")
    sigma, alpha, horizon, precision, level = job
    row = dict(zip(FIELDS[:5], job))
    row.update(B=float("nan"), error=float("nan"), status="")
    try:
        q = S0+sigma*sigma*horizon
        baseline = math.sqrt(q/(2*math.pi))
        exposure = alpha*horizon/q
        if precision == 0 and exposure <= .25:
            terms = central_series()*exposure**np.arange(29)
            b = baseline+alpha*horizon/math.sqrt(q)*float(terms.sum())
            row["error"] = alpha*horizon/math.sqrt(q)*abs(float(terms[21:].sum()))
            row["status"] = "analytic_zero_series"
        elif math.isinf(precision):
            b = full_profile_output(sigma, alpha, horizon, level)
            row["status"] = "full_disclosure_bvp"
        else:
            s0 = S0/(sigma*sigma*horizon)
            ell = precision*sigma*sigma*horizon*horizon
            dx = min((.02 if level>=2 else .035 if level else .065)*math.sqrt(1+s0),
                     (.08 if level>=2 else .14 if level else .24)*math.sqrt(float(filtering_variance(1, s0, ell))))
            solution = solve_exact(w=alpha/(sigma*sigma), ell=ell, s0=s0,
                x0=0, dx=dx, steps=840 if level>=2 else 420 if level else 150, extent=7)
            raw = sigma*math.sqrt(horizon)*solution.best_output
            b = raw-finite_baseline(sigma, horizon, precision, level, dx)+baseline
            row["status"] = "nonlinear_hjb"
        upper = upper_output(sigma, alpha, horizon, precision)
        if not math.isfinite(b) or b < baseline-1e-5 or b > upper+1e-5:
            raise RuntimeError("output violates exact energy bounds")
        row["B"] = b
    except (RuntimeError, ValueError, OverflowError, FloatingPointError, np.linalg.LinAlgError) as exc:
        row["status"] = "failed: "+str(exc)
    return row


def run_jobs(jobs, cache, workers, destination):
    missing = [j for j in jobs if j not in cache]
    if not missing:
        return
    fresh = not destination.exists()
    start = time.monotonic()
    with destination.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if fresh:
            writer.writeheader()
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(calculate, job) for job in missing]
            for n, future in enumerate(as_completed(futures), 1):
                row = future.result()
                writer.writerow(row)
                cache[sample_key(row)] = row
                if n % 200 == 0 or n == len(missing):
                    handle.flush()
                    print(f"{n}/{len(missing)} HJB/output solves; {time.monotonic()-start:.1f}s", flush=True)


def upper_output(sigma, alpha, t, precision):
    base = math.sqrt((S0+sigma*sigma*t)/(2*math.pi))
    if math.isinf(precision):
        return base+5*sigma*sigma*t/math.sqrt(2*math.pi*S0)
    return base+math.sqrt(2*alpha*t)


def profile(sigma, cost, designs, cache, alpha_max, precision_cap=500., relative_floor=.005, required_level=0):
    t0, loss0, ce = zero_design(sigma)
    best = dict(loss=loss0, theta=0., alpha=0., T=t0, precision=float("nan"), B=math.sqrt((1+sigma*sigma*t0)/(2*math.pi)))
    finite, full = None, None
    missing_lower = math.inf
    fine_keys = []
    for a, t, l in designs:
        key = (float(sigma), float(a), float(t), float(l), 0.)
        coarse = cache[key]
        versions = [(level,cache[key[:-1]+(float(level),)]) for level in range(3)
                    if key[:-1]+(float(level),) in cache]
        level, row = versions[-1]
        if not math.isfinite(row["B"]):
            lower = tail(upper_output(sigma, a, t, l))+cost*a+delay(t)
            missing_lower = min(missing_lower, lower)
            continue
        loss = tail(row["B"])+cost*a+delay(t)
        item = dict(loss=loss, theta=cost*a, alpha=a, T=t, precision=l, B=row["B"], key=key)
        # Refinement differences are diagnostics, not rigorous error bounds.
        earlier = [v for _,v in versions[:-1] if math.isfinite(v["B"])]
        error_b = (3*abs(row["B"]-earlier[-1]["B"]) if earlier else
                   math.inf if level>0 and row["status"]!="analytic_zero_series" else 0.)
        error_b = max(error_b, float(row["error"]) if math.isfinite(row["error"]) else 0.)
        item["uncertainty"] = (math.inf if not math.isfinite(error_b) else
                               tail(max(B0, row["B"]-error_b))-tail(row["B"]))
        item["level"] = level
        target = full if math.isinf(l) else finite
        if target is None or loss < target["loss"]:
            if math.isinf(l):
                full = item
            else:
                finite = item
        if loss < best["loss"]:
            best = item
    for item in (finite, full):
        if item is not None and item["key"][:-1]+(float(max(1,required_level)),) not in cache:
            fine_keys.append(item["key"][:-1]+(float(max(1,required_level)),))
    tol = max(1e-300, relative_floor*best["loss"],
              *(item["uncertainty"] for item in (finite, full) if item is not None))
    if (not math.isfinite(tol) or (required_level and fine_keys)
            or missing_lower < best["loss"]-tol or cost*alpha_max < best["loss"]-tol):
        regime = "unresolved"
    elif best["theta"] == 0:
        runner = min((v["loss"] for v in (finite, full) if v is not None), default=math.inf)
        regime = ("unresolved" if cost < ce else
                  "zero_on_grid" if runner-loss0 > tol else "near_tie")
    else:
        competitor = min(loss0, (full if not math.isinf(best["precision"]) else finite)["loss"]
                         if (full if not math.isinf(best["precision"]) else finite) is not None else math.inf)
        if competitor-best["loss"] <= tol:
            regime = "near_tie"
        elif math.isinf(best["precision"]):
            regime = "full_on_grid"
        elif best["precision"] == precision_cap:
            regime = "finite_tail_unresolved"
        else:
            regime = "finite_on_grid"
    return dict(sigma=sigma, c=cost, regime=regime, **{k:v for k,v in best.items() if k not in ("key", "uncertainty", "level")},
                zero_loss=loss0, ce=ce, finite_loss=finite["loss"] if finite else math.inf,
                full_loss=full["loss"] if full else math.inf, missing_lower=missing_lower,
                tolerance=tol), fine_keys


def draw(rows, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    labels = ["zero_on_grid", "finite_on_grid", "full_on_grid", "near_tie", "unresolved", "finite_tail_unresolved"]
    colors = ["#80b398", "#5187bc", "#dfa14c", "#d6d6d6", "#ece1ef", "#b992c6"]
    sigmas = sorted({r["sigma"] for r in rows})
    costs = sorted({r["c"] for r in rows})
    values = np.array([[labels.index(next(r["regime"] for r in rows if r["sigma"]==s and r["c"]==c))
                        for s in sigmas] for c in costs])
    fig, ax = plt.subplots(figsize=(10, 6.4), constrained_layout=True)
    sigma_mid = (np.array(sigmas[:-1])+np.array(sigmas[1:]))/2
    sigma_edges = np.r_[max(0., 2*sigmas[0]-sigma_mid[0]), sigma_mid, 2*sigmas[-1]-sigma_mid[-1]]
    log_cost = np.log(costs)
    cost_mid = (log_cost[:-1]+log_cost[1:])/2
    cost_edges = np.exp(np.r_[2*log_cost[0]-cost_mid[0], cost_mid, 2*log_cost[-1]-cost_mid[-1]])
    ax.pcolormesh(sigma_edges, cost_edges, values, shading="flat", cmap=ListedColormap(colors),
                  norm=BoundaryNorm(np.arange(-.5, 6.5), 6), rasterized=True)
    ax.set_yscale("log")
    ax.set_xlabel(r"Innovation risk $\sigma$")
    ax.set_ylabel(r"Effort cost $c$ (log scale)")
    ax.set_title("Nonlinear-equilibrium joint design: evaluated numerical grid")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=color, label=label.replace("_", " ")) for color,label in zip(colors,labels)],
              loc="upper right", fontsize=9)
    ax.text(.01, .01, "Shape floors: 64; S0=1. Unresolved cells retain failures/caps.\n"
            "Full on grid uses the separate infinite-precision model, not max finite lambda.",
            transform=ax.transAxes, fontsize=8, va="bottom", bbox=dict(facecolor="white", alpha=.85))
    for ext in ("png", "pdf"):
        fig.savefig(destination/f"numerical_regions.{ext}", dpi=190)
    ax.set_xlim(0., .7)
    ax.set_ylim(1e-6, 1e3)
    ax.set_title("Low-risk detail: the same evaluated numerical grid")
    for ext in ("png", "pdf"):
        fig.savefig(destination/f"numerical_regions_zoom.{ext}", dpi=190)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-refine", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sigmas = SIGMAS[::2] if args.pilot else SIGMAS
    tmax = math.exp((math.log(AMPLITUDE)-LOG_D)/POWER)
    tu = math.exp((math.log(AMPLITUDE/(POWER+1))-LOG_D)/POWER)
    cf = 4*tu*POWER*AMPLITUDE/(POWER+1)
    ces = np.array([zero_design(s)[2] for s in SIGMAS])
    costs = np.unique(np.r_[np.geomspace(max(1e-60, ces.min()*.2), cf*40, 39),
        np.geomspace(cf/200, cf*40, 25), ces*.65, ces*.85, ces*1.05, cf*np.array([.5,1,3,6,10,20])])
    alphas = np.unique(np.r_[np.geomspace(.00015, 6, 14), .01, .1, 1., 2.])
    precisions = [0., .005, .08, 1., 20., 500., math.inf]
    cache_path = OUT/"validated_samples.csv"
    cache = {}
    if cache_path.exists():
        with cache_path.open() as handle:
            for raw in csv.DictReader(handle):
                row = {k:(raw[k] if k=="status" else float(raw[k])) for k in FIELDS}
                cache[sample_key(row)] = row
    designs = {}
    jobs = []
    for sigma in sigmas:
        t0 = zero_design(sigma)[0]
        horizons = np.unique(np.r_[np.geomspace(.035, tmax, 10), t0*np.array([.94,1.,1.04])])
        horizons = horizons[horizons<=tmax]
        designs[float(sigma)] = [(float(a),float(t),l) for a in alphas for t in horizons for l in precisions]
        jobs.extend((float(sigma),a,t,l,0.) for a,t,l in designs[float(sigma)])
    print(json.dumps(dict(shape_floor=GAMMA, delay_coefficient=math.exp(LOG_D),
        T_max=tmax, cF=cf, sigmas=sigmas.tolist(), cost_points=len(costs), jobs=len(jobs))), flush=True)
    run_jobs(jobs, cache, args.workers, cache_path)
    refinement_passes = 0
    for iteration in range(3 if not args.no_refine else 0):
        refine = set()
        for sigma in sigmas:
            for c in costs:
                _, keys = profile(float(sigma),float(c),designs[float(sigma)],cache,float(alphas[-1]))
                refine.update(keys)
        if not refine:
            break
        print(f"Refinement pass {iteration+1}: {len(refine)} candidate checks", flush=True)
        refinement_passes += 1
        run_jobs(sorted(refine), cache, args.workers, cache_path)
    rows = [profile(float(s),float(c),designs[float(s)],cache,float(alphas[-1]))[0]
            for c in costs for s in sigmas]
    destination = OUT/("pilot" if args.pilot else "full")
    destination.mkdir(exist_ok=True)
    with (destination/"regions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    counts = {label:sum(r["regime"]==label for r in rows) for label in sorted({r["regime"] for r in rows})}
    metadata = dict(qualification="Grid optimization only; failure/cap screening and refinement differences are not global numerical certificates.",
        utility=f"3*(1-exp(-{RATE}*(B-{B0})))", delay=f"{math.exp(LOG_D)}*T**64",
        gamma_floor=GAMMA, universal_threshold_not_quantified=True, S0=S0, cF=cf,
        grids=dict(sigma=sigmas.tolist(),c=costs.tolist(),alpha=alphas.tolist(),precision=["infinity" if math.isinf(v) else v for v in precisions],
                   horizon="ten geometric points plus sigma-specific T0 neighbors; see validated_samples.csv"),
        numerical_methods="finite HJB baseline correction; self-similar infinite-precision BVP; analytic zero series on certified exposure domain",
        counts=counts, failed_samples=sum(str(r["status"]).startswith("failed") for r in cache.values()),
        evaluated_samples=len(cache),
        refined_samples=sum(k[-1]==1. for k in cache),
        refinement_passes_this_run=refinement_passes)
    (destination/"metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    draw(rows, destination)
    print(json.dumps(dict(destination=str(destination), counts=counts), indent=2), flush=True)


if __name__ == "__main__":
    main()
