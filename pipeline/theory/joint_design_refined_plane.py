"""Adaptive design search and denser risk slices; no interpolation of labels.

Each displayed column is calculated. Two local design-search rounds and a
solver-resolution check precede plotting. This remains a numerical search,
not a global certificate on the continuous design space.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math

import numpy as np

from pipeline.theory.joint_design_plane import (
    OUT, FIELDS, SIGMAS, AMPLITUDE, LOG_D, POWER, RATE, B0,
    delay, tail, zero_design, sample_key, run_jobs, profile,
)

PRECISIONS = (0., .005, .08, 1., 20., 500., 10000., math.inf)
CAP = 10000.


def load_cache(path):
    cache = {}
    if path.exists():
        with path.open() as handle:
            for raw in csv.DictReader(handle):
                row = {k:raw[k] if k == "status" else float(raw[k]) for k in FIELDS}
                cache[sample_key(row)] = row
    return cache


def winning_designs(sigma, costs, designs, cache):
    """Both endpoint families are reoptimized, not just the current winner."""
    found = set()
    for infinite in (False, True):
        available = []
        losses, alphas = [], []
        for a, t, l in sorted(designs):
            if math.isinf(l) != infinite:
                continue
            key = (sigma, a, t, l, 0.)
            row = cache.get(key[:-1]+(2.,),cache.get(key[:-1]+(1.,), cache.get(key)))
            if row is None or not math.isfinite(row["B"]):
                continue
            available.append((a, t, l))
            losses.append(tail(row["B"])+delay(t))
            alphas.append(a)
        if available:
            values = np.array(losses)[None, :]+np.array(costs)[:, None]*np.array(alphas)[None, :]
            found.update(available[k] for k in np.unique(np.argmin(values, axis=1)))
    return found


def plot(rows, destination, focus_only=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch
    from matplotlib.ticker import LogLocator

    kinds = {"zero_on_grid":0, "finite_on_grid":1, "full_on_grid":2}
    colors = ["#97C6AD", "#3C82B6", "#E2A04B", "#E6E7E9"]
    sigmas = sorted({r["sigma"] for r in rows})
    costs = sorted({r["c"] for r in rows})
    lookup = {(r["sigma"],r["c"]):kinds.get(r["regime"],3) for r in rows}
    values = np.array([[lookup[s,c] for s in sigmas] for c in costs])
    mids = (np.array(sigmas[:-1])+sigmas[1:])/2
    edges_s = np.r_[max(0.,2*sigmas[0]-mids[0]),mids,2*sigmas[-1]-mids[-1]]
    logs = np.log(costs)
    mids_c = (logs[:-1]+logs[1:])/2
    edges_c = np.exp(np.r_[2*logs[0]-mids_c[0],mids_c,2*logs[-1]-mids_c[-1]])
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-.5,4.5),4)
    legends = [Patch(facecolor=v, label=k) for v,k in zip(colors,
               ["Zero prize", "Positive prize · finite precision", "Positive prize · full disclosure", "Inconclusive"])]
    views = [("numerical_regions_detail",True)] if focus_only else [("numerical_regions_refined",False),("numerical_regions_detail",True)]
    for name, zoom in views:
        fig, ax = plt.subplots(figsize=(10.5,6.5), layout="constrained")
        ax.pcolormesh(edges_s,edges_c,values,cmap=cmap,norm=norm,shading="flat",rasterized=True)
        ax.set_yscale("log")
        ax.set_xlim(.03,.7 if zoom else 2.65)
        ax.set_ylim(1e-4 if zoom else min(costs),800.)
        ax.set_xlabel(r"Innovation risk $\sigma$", fontsize=12)
        ax.set_ylabel(r"Effort cost $c$ (log scale)", fontsize=12)
        ax.set_title("Joint design: refined nonlinear-equilibrium search"+
                     (" — low-risk detail" if zoom else ""),fontsize=14,pad=14)
        ax.legend(handles=legends,loc="upper right",framealpha=.96,fontsize=10)
        ax.yaxis.set_major_locator(LogLocator(base=10,numticks=9))
        ax.grid(which="major",axis="y",color="white",alpha=.28,linewidth=.5)
        ax.tick_params(labelsize=10)
        if zoom:
            for x,y,text,color,kind in [(.42,50,"Zero prize","#244C37",0),
                                       (.16,.0015,"Finite precision","white",1),
                                       (.35,.045,"Full disclosure","#55370F",2)]:
                nearest_s = min(sigmas,key=lambda s:abs(s-x))
                nearest_c = min(costs,key=lambda c:abs(math.log(c/y)))
                if lookup[nearest_s,nearest_c]==kind:
                    ax.text(x,y,text,fontsize=15,color=color,ha="center")
        fig.get_layout_engine().set(rect=(0,.085,1,.915))
        fig.text(.08,.027,"Each risk slice is computed; gray cells are not assigned a regime.\n"
                 "Colors describe the searched design grid, not certified continuous-domain boundaries.",
                 fontsize=9,color="#555555")
        for ext in ("png","pdf"):
            fig.savefig(destination/f"{name}.{ext}",dpi=260)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--plot-only",action="store_true")
    args = parser.parse_args()
    destination = OUT/"refined"
    destination.mkdir(exist_ok=True)
    if args.plot_only:
        with (destination/"regions.csv").open() as handle:
            rows = [{k:v if k=="regime" else float(v) for k,v in raw.items()}
                    for raw in csv.DictReader(handle)]
        plot(rows,destination)
        return
    cache_path = OUT/"validated_samples.csv"
    cache = load_cache(cache_path)
    original = sorted({k[0] for k in cache if k[0] in SIGMAS})
    sigmas = np.unique(np.r_[SIGMAS,np.linspace(.04,.68,27)])
    with (OUT/"full/regions.csv").open() as handle:
        old_rows = list(csv.DictReader(handle))
    old_costs = sorted({float(r["c"]) for r in old_rows})
    ce = np.array([zero_design(float(s))[2] for s in sigmas])
    costs = np.unique(np.r_[old_costs,np.geomspace(1e-4,800,90),
                           (ce[:,None]*np.array([.5,.65,.8,.95,1.05,1.2])).ravel()]).astype(float)
    tmax = math.exp((math.log(AMPLITUDE)-LOG_D)/POWER)
    designs = {}
    for value in sigmas:
        s = float(value)
        t0 = zero_design(s)[0]
        proposed = set()
        neighbors = sorted(original,key=lambda v:abs(v-s))[:2]
        for neighbor in neighbors:
            candidates = {(k[1],k[2],k[3]) for k in cache if k[0]==neighbor and k[-1]==0.}
            if neighbor==s:
                proposed.update(candidates)
            winners = winning_designs(neighbor,costs,candidates,cache)
            for a,t,l in winners:
                scaled_t = min(tmax,t*t0/zero_design(neighbor)[0])
                proposed.update((a,scaled_t,p) for p in PRECISIONS)
        # Broad endpoint probes retain alternatives away from transported winners.
        for a in [.00015,.001,.01,.1,1.,3.,6.]:
            horizons = np.unique(np.r_[np.geomspace(.035,tmax,8),t0*np.array([.5,.8,.94,1.,1.04])])
            for t in horizons[horizons<=tmax]:
                proposed.update((a,float(t),l) for l in (0.,math.inf))
        designs[s] = proposed
    jobs = [(s,a,t,l,0.) for s,items in designs.items() for a,t,l in sorted(items)]
    print(json.dumps(dict(risk_slices=len(sigmas),cost_points=len(costs),initial_designs=len(jobs))),flush=True)
    run_jobs(jobs,cache,args.workers,cache_path)
    history = []

    def summarize(stage):
        rows = [profile(float(s),float(c),sorted(designs[float(s)]),cache,6.,CAP)[0]
                for c in costs for s in sigmas]
        history.append(dict(stage=stage,counts=dict(Counter(r["regime"] for r in rows))))
        return rows

    previous = summarize("transported_seed")
    for round_no in range(2):
        jobs = []
        for s,items in designs.items():
            winners = winning_designs(s,costs,items,cache)
            extra = set()
            for a,t,l in winners:
                # Coordinate refinement, including mixed prize/deadline moves.
                factors_a = [.7,1.,1.4] if round_no==0 else [.85,1.,1.18]
                factors_t = [.97,1.,1.03] if round_no==0 else [.985,1.,1.015]
                for af in factors_a:
                    for tf in factors_t:
                        na,nt = a*af,t*tf
                        if .00001<=na<=6. and .02<=nt<=tmax:
                            extra.add((na,nt,l))
                            # Same prize/deadline compared across both endpoints.
                            extra.add((na,nt,math.inf if not math.isinf(l) else CAP))
                if not math.isinf(l):
                    ls = [.001] if l==0 else [l*.5,l*2.] if round_no==0 else [l*.7,l*1.4]
                    extra.update((a,t,min(CAP,p)) for p in ls)
            items.update(extra)
            jobs.extend((s,a,t,l,0.) for a,t,l in sorted(extra))
        print(f"Local design round {round_no+1}: {len(jobs)} samples",flush=True)
        run_jobs(jobs,cache,args.workers,cache_path)
        current = summarize(f"local_round_{round_no+1}")
        history[-1]["class_changes"] = sum(a["regime"]!=b["regime"] for a,b in zip(previous,current))
        previous = current
    # Reprofile after each solver-resolution check; new winners get checked too.
    for check in range(3):
        jobs = set()
        for s,items in designs.items():
            for a,t,l in winning_designs(s,costs,items,cache):
                key = (s,a,t,l,1.)
                if key not in cache:
                    jobs.add(key)
        if not jobs:
            break
        print(f"Resolution check {check+1}: {len(jobs)} samples",flush=True)
        run_jobs(sorted(jobs),cache,args.workers,cache_path)
    rows = summarize("resolution_checked")
    history[-1]["class_changes"] = sum(a["regime"]!=b["regime"] for a,b in zip(previous,rows))
    for name,records in [("regions.csv",rows),
                         ("designs.csv",[dict(sigma=s,alpha=a,T=t,precision=l)
                                         for s,items in designs.items() for a,t,l in sorted(items)])]:
        with (destination/name).open("w",newline="") as handle:
            writer = csv.DictWriter(handle,fieldnames=list(records[0]))
            writer.writeheader();writer.writerows(records)
    metadata = dict(qualification="Adaptive numerical search only; no continuous-domain global certificate.",
                    S0=1,utility=f"3*(1-exp(-{RATE}*(B-{B0})))",delay=f"{math.exp(LOG_D)}*T**64",
                    universal_curvature_threshold_not_quantified=True,
                    sigma=sigmas.tolist(),c=costs.tolist(),precision_cap=CAP,
                    history=history,computed_designs=sum(len(v) for v in designs.values()),
                    refined_candidates=sum(k[-1]==1 and k[0] in sigmas for k in cache))
    (destination/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    plot(rows,destination)
    print(json.dumps(history,indent=2),flush=True)


if __name__=="__main__":
    main()
