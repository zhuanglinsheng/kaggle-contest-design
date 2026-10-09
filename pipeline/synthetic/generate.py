"""Data-only paired experiments: original hourly generator versus S0=4.

No inference is imported or run. The effort approximation, hourly updates,
readout noise, thinning, seeds, and old long-horizon ratios are unchanged.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta

from pipeline.config import ROOT, MATPLOTLIB_CACHE_DIR

os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pipeline.synthetic.figure3 import plot_ts_events
from synthetic_data.synthetic_data import filtering_variance, synthetic_data_simulation

OUT = ROOT / "results/synthetic/variance_only"
# Preserve the previous experiment design; do not recalibrate after changing S.
LONG_R = {180: 18.6484375, 360: 47.4375}


def generate_contest(days, seed, ratio, S0, *, effort_policy=None):
    start = datetime(2026, 1, 1)
    return synthetic_data_simulation(
        theta=1.0, c_i=1.2, c_j=1.5, sigma=2.0, lamb=1.0,
        intensity_effort_ratio=ratio, hour_arrival_ub=1.0,
        start_time=start, end_time=start + timedelta(days=days),
        time_unit=timedelta(hours=1), time_unit_2f=1/24, mu_0=0.0,
        S0=S0, effort_policy=effort_policy, seed_brownian=seed, seed_poisson=seed+1,
        seed_uniform=seed+2, seed_initial=seed+3,
    )


def save_contest(folder, name, result, S0, ratio):
    grid, mi, mj, latent, perceived, public, sub_i, sub_j = result
    start = grid[0]
    days = np.arange(len(latent)) / 24
    events_i = np.array([(t-start).total_seconds()/86400 for t in sub_i])
    events_j = np.array([(t-start).total_seconds()/86400 for t in sub_j])
    variance = np.array([filtering_variance(t, 2., 1., S0) for t in days])
    np.savez_compressed(folder / f"{name}_paths.npz", time_days=days,
                        effort_i=mi, effort_j=mj, latent=latent, perceived=perceived,
                        public=public, S=variance, events_i_days=events_i,
                        events_j_days=events_j)
    # Readout times remain hourly boundaries, not exact candidate times.
    obs_steps = np.unique(np.ceil(np.r_[events_i, events_j]*24).astype(int))
    observations = dict(time_days=(obs_steps/24).tolist(),
                        interval_days=(np.diff(np.r_[0, obs_steps])/24).tolist(),
                        public_gap=public[obs_steps].tolist(),
                        events_i_days=events_i.tolist(), events_j_days=events_j.tolist())
    (folder / f"{name}_observations.json").write_text(json.dumps(observations, indent=2)+"\n")
    return dict(submissions_i=len(sub_i), submissions_j=len(sub_j),
                initial_latent_gap=float(latent[0]),
                expected_count_per_day=float(ratio*(mi.sum()+mj.sum())/len(mi)),
                finite=bool(all(np.isfinite(x).all() for x in (mi,mj,latent,perceived,public,variance))))


def plot_baseline(result, destination):
    grid, _, _, latent, perceived, public, sub_i, sub_j = result
    fig = plot_ts_events(
        grid,
        series={r"real output gap ($y_t$)": latent[:-1],
                r"observed output gap ($\hat y_t$)": public[:-1],
                r"perceived output gap ($\tilde y_t$)": perceived[:-1]},
        events={r"player $i$": sub_i, r"player $j$": sub_j},
        colors=["blue", "black", "black"], linestyles=["solid", "solid", "dashed"],
    )
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    experiments = [("baseline", 90, 1000, 15.)]
    experiments += [(f"pool_contest{i+1:02}", 90, 3000+10*i, 15.) for i in range(20)]
    experiments += [(f"days{d}", d, 5000+d, r) for d,r in LONG_R.items()]
    manifest = dict(
        status="data generation only; no Bayesian estimation",
        primitives=dict(theta=1., c_i=1.2, c_j=1.5, sigma=2., lamb=1., mu_0=0.),
        only_changes=["initial latent variance", "S_t in the unchanged first-order effort kernel",
                      "lambda*S_t in the unchanged held-readout recursion"],
        unchanged=["hourly grid and update order", "held public readouts", "readout-noise scaling",
                   "first-order effort formula", "Poisson thinning", "four RNG seeds",
                   "all non-S parameters including previously calibrated long-horizon r"],
        simulator_sha256=hashlib.sha256((ROOT/"synthetic_data/synthetic_data.py").read_bytes()).hexdigest(),
        pools={str(n):[f"pool_contest{i+1:02}" for i in range(n)] for n in (10,20)},
        experiments={},
    )
    lines=["# Variance-only data generation", "",
           "Legacy hourly generator restored; new data change only S (S0=4). No Bayesian estimation run.",
           "First-order efforts are retained; these data are not generated with a numerical full-HJB policy.",
           "Long-horizon r stays at the original calibrated values; intensity matching is not recalibrated.", "",
           "| Experiment | Legacy i / j | S0=4 i / j | Same submission times? |",
           "|---|---:|---:|---|"]
    for name, days, seed, ratio in experiments:
        results, summaries = {}, {}
        for variant, S0 in (("legacy", None), ("nonsteady", 4.)):
            folder = OUT/variant
            folder.mkdir(exist_ok=True)
            results[variant] = generate_contest(days, seed, ratio, S0)
            summaries[variant] = save_contest(folder, name, results[variant], 2. if S0 is None else S0, ratio)
            if name == "baseline":
                plot_baseline(results[variant], folder/"figure3.png")
        old, new = results["legacy"], results["nonsteady"]
        same_events = old[6:] == new[6:]
        differences = {key:float(np.max(np.abs(a-b))) for key,a,b in
                       zip(("effort_i","effort_j","latent","perceived","public"), old[1:6], new[1:6])}
        manifest["experiments"][name] = dict(days=days, ratio=ratio,
            seeds=dict(brownian=seed, poisson=seed+1, uniform=seed+2, initial=seed+3),
            results=summaries, same_submission_times=same_events, max_absolute_difference=differences)
        a,b = summaries["legacy"], summaries["nonsteady"]
        lines.append(f'| {name} | {a["submissions_i"]} / {a["submissions_j"]} | {b["submissions_i"]} / {b["submissions_j"]} | {same_events} |')
        print(name, summaries, flush=True)
    (OUT/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    (OUT/"README.md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    main()
