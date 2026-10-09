"""Targeted higher-resolution search; white remains one unclassified category."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import json
import math
import numpy as np
from pipeline.theory.joint_design_plane import OUT, B0, tail, delay, zero_design, upper_output, run_jobs
from pipeline.theory.joint_design_refined_plane import load_cache, winning_designs, plot, CAP, PRECISIONS


def profiles(sigma, costs, designs, cache, relative_floor=0., required_level=2, branch_screen=True):
    """Classify precision families; keep optimizer-location uncertainty separate."""
    costs = np.asarray(costs)
    t0, loss0, ce = zero_design(sigma)
    valid, errors, levels, failed = [], [], [], []
    for a,t,l in sorted(designs):
        versions = [(k,cache[(sigma,a,t,l,float(k))]) for k in range(3)
                    if (sigma,a,t,l,float(k)) in cache]
        if not versions:
            raise RuntimeError("missing design output")
        level,row = versions[-1]
        if not math.isfinite(row["B"]):
            failed.append((a,tail(upper_output(sigma,a,t,l))+delay(t),l))
            continue
        earlier = [r for _,r in versions[:-1] if math.isfinite(r["B"])]
        err = 3*abs(row["B"]-earlier[-1]["B"]) if earlier else (
            math.inf if level>0 and row["status"]!="analytic_zero_series" else 0.)
        err = max(err,row["error"] if math.isfinite(row["error"]) else 0.)
        errors.append(math.inf if not math.isfinite(err) else tail(max(B0,row["B"]-err))-tail(row["B"]))
        valid.append((a,t,l,row["B"]))
        levels.append(level)
    if not valid:
        raise RuntimeError("no valid positive-prize candidate")
    data = np.asarray(valid)
    losses = costs[:,None]*data[:,0]+np.array([tail(b)+delay(t) for a,t,l,b in valid])[None,:]
    indices, family_losses = [], []
    for mask in [np.isfinite(data[:,2]),np.isinf(data[:,2])]:
        sub = np.flatnonzero(mask)
        if not len(sub):
            raise RuntimeError("missing precision family")
        idx = sub[np.argmin(losses[:,sub],axis=1)]
        indices.append(idx)
        family_losses.append(losses[np.arange(len(costs)),idx])
    fi,ui = indices
    fl,ul = family_losses
    pi = np.where(fl<=ul,fi,ui)
    positive_loss = np.minimum(fl,ul)
    best_loss = np.minimum(loss0,positive_loss)
    errors = np.asarray(errors)
    floor = np.maximum(1e-300,max(relative_floor,64*np.finfo(float).eps)*best_loss)
    tol = np.maximum.reduce([floor,errors[fi],errors[ui]])
    lower_by_family = []
    for infinite in [False,True]:
        group = [(a,v) for a,v,l in failed if math.isinf(l)==infinite]
        if group:
            fails = np.asarray(group)
            lower_by_family.append(np.min(costs[:,None]*fails[:,0]+fails[:,1],axis=1))
        else:
            lower_by_family.append(np.full(len(costs),math.inf))
    if failed:
        lower = np.minimum(*lower_by_family)
    else:
        lower = np.full(len(costs),math.inf)
    levels = np.asarray(levels)
    rows = []
    for k,c in enumerate(costs):
        a,t,l,b = valid[pi[k]]
        positive = positive_loss[k]<loss0
        theta = c*a if positive else 0.
        if not positive:
            a,t,l,b = 0.,t0,float("nan"),math.sqrt((1+sigma*sigma*t0)/(2*math.pi))
        if (not math.isfinite(tol[k]) or levels[fi[k]]<required_level or levels[ui[k]]<required_level
                or lower[k]<best_loss[k]-tol[k] or c*6<best_loss[k]-tol[k]):
            regime = "unresolved"
        elif not positive:
            regime = "unresolved" if c<ce else "zero_on_grid" if positive_loss[k]-loss0>tol[k] else "near_tie"
        elif min(loss0,ul[k] if pi[k]==fi[k] else fl[k])-best_loss[k]<=tol[k]:
            regime = "near_tie"
        elif math.isinf(l):
            regime = "full_on_grid"
        else:
            regime = "finite_tail_unresolved" if l==CAP else "finite_on_grid"
        design_unresolved = int(lower[k]<best_loss[k]+max(errors[fi[k]],errors[ui[k]]) or l==CAP)
        if branch_screen:
            # Distinguish precision-family ranking from locating its optimizer.
            # A missing value in the winning family can only improve that family.
            ef,eu = errors[fi[k]],errors[ui[k]]
            failed_finite,failed_full = lower_by_family[0][k],lower_by_family[1][k]
            invalid = (not math.isfinite(tol[k]) or levels[fi[k]]<required_level
                       or levels[ui[k]]<required_level or c*6<best_loss[k]-floor[k])
            if invalid:
                regime = "unresolved"
            elif (fl[k]+ef+floor[k]<min(loss0,ul[k]-eu,failed_full)):
                regime = "finite_on_grid"
            elif (ul[k]+eu+floor[k]<min(loss0,fl[k]-ef,failed_finite)):
                regime = "full_on_grid"
            elif (loss0+floor[k]<min(fl[k]-ef,ul[k]-eu,failed_finite,failed_full) and c>=ce):
                regime = "zero_on_grid"
            else:
                regime = "unresolved" if lower[k]<best_loss[k]+max(ef,eu) or (not positive and c<ce) else "near_tie"
        rows.append(dict(sigma=sigma,c=float(c),regime=regime,loss=best_loss[k],theta=theta,alpha=a,T=t,
                         precision=l,B=b,zero_loss=loss0,ce=ce,finite_loss=fl[k],full_loss=ul[k],
                         missing_lower=lower[k],tolerance=tol[k],design_unresolved=design_unresolved))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--plot-only",action="store_true")
    parser.add_argument("--reclassify",action="store_true")
    args = parser.parse_args()
    dest = OUT/"boundary_polish"
    dest.mkdir(exist_ok=True)
    if args.plot_only:
        with (dest/"regions.csv").open() as f:
            rows = [{k:v if k=="regime" else float(v) for k,v in r.items()} for r in csv.DictReader(f)]
        plot(rows,dest,focus_only=True)
        return
    cache_path = OUT/"validated_samples.csv"
    cache = load_cache(cache_path)
    if args.reclassify:
        with (dest/"regions.csv").open() as f:
            saved = [{k:v if k=="regime" else float(v) for k,v in r.items()} for r in csv.DictReader(f)]
        costs = sorted({r["c"] for r in saved})
        sigmas = sorted({r["sigma"] for r in saved})
        designs = defaultdict(set)
        with (dest/"designs.csv").open() as f:
            for r in csv.DictReader(f):
                designs[float(r["sigma"])].add((float(r["alpha"]),float(r["T"]),float(r["precision"])))
        rows = [r for s in sigmas for r in profiles(s,costs,designs[s],cache)]
        fixed = [r for s in sigmas for r in profiles(s,costs,designs[s],cache,relative_floor=.005)]
        pooled = [r for s in sigmas for r in profiles(s,costs,designs[s],cache,branch_screen=False)]
        white = lambda rs:sum(r["regime"] not in ["zero_on_grid","finite_on_grid","full_on_grid"] for r in rs)
        with (OUT/"refined/regions.csv").open() as f:
            original_keys = {(float(r["sigma"]),float(r["c"])) for r in csv.DictReader(f)}
        metadata = json.loads((dest/"metadata.json").read_text())
        metadata.update(relative_floor=0.,roundoff_relative_floor=64*np.finfo(float).eps,
                        classification_is_not_complete_design_solution=True,
                        infinite_precision_statement_is_grid_only=True,
                        counts=dict(Counter(r["regime"] for r in rows)),
                        white_error_floor_same_upgraded_grid=white(rows),
                        white_fixed_floor_same_upgraded_grid=white(fixed),
                        white_pooled_failure_screen_same_upgraded_grid=white(pooled),
                        branch_screen="winning-family failures do not invalidate a strict family comparison",
                        white_same_old_cells_after=white([r for r in rows if (r["sigma"],r["c"]) in original_keys]))
        with (dest/"regions.csv").open("w",newline="") as f:
            writer = csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        (dest/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
        plot(rows,dest,focus_only=True)
        print(json.dumps({k:v for k,v in metadata.items() if k not in ["sigma","c"]},indent=2),flush=True)
        return
    with (OUT/"refined/regions.csv").open() as f:
        old = [{k:v if k=="regime" else float(v) for k,v in r.items()} for r in csv.DictReader(f)]
    old = [r for r in old if r["sigma"]<=.7 and 1e-4<=r["c"]<=800]
    sigmas = sorted({r["sigma"] for r in old})
    costs = sorted({r["c"] for r in old})
    lookup = {(r["sigma"],r["c"]):r["regime"] for r in old}
    transitions = []
    extra_costs = set()
    for s in sigmas:
        for a,b in zip(costs,costs[1:]):
            if lookup[s,a]!=lookup[s,b]:
                extra_costs.add(math.sqrt(a*b))
    for a,b in zip(sigmas,sigmas[1:]):
        transitions.append((sum(lookup[a,c]!=lookup[b,c] for c in costs),(a+b)/2))
    extra_sigmas = [v for _,v in sorted(transitions,reverse=True)[:6]]
    costs = sorted(set(costs)|extra_costs)
    designs = defaultdict(set)
    with (OUT/"refined/designs.csv").open() as f:
        for r in csv.DictReader(f):
            s = float(r["sigma"])
            if s in sigmas:
                designs[s].add((float(r["alpha"]),float(r["T"]),float(r["precision"])))
    tmax = max(t for items in designs.values() for a,t,l in items)
    jobs = []
    for s in extra_sigmas:
        for neighbor in sorted(sigmas,key=lambda v:abs(v-s))[:2]:
            for a,t,l in winning_designs(neighbor,costs,designs[neighbor],cache):
                nt = min(tmax,t*zero_design(s)[0]/zero_design(neighbor)[0])
                designs[s].update((a,nt,p) for p in PRECISIONS)
        jobs.extend((s,a,t,l,0.) for a,t,l in sorted(designs[s]))
    sigmas = sorted(sigmas+extra_sigmas)
    print(json.dumps(dict(sigma_slices=len(sigmas),cost_points=len(costs),new_slice_designs=len(jobs))),flush=True)
    run_jobs(jobs,cache,args.workers,cache_path)
    jobs = []
    for s,items in designs.items():
        extra = set()
        for a,t,l in winning_designs(s,costs,items,cache):
            for af,tf in [(0.95,1),(1.05,1),(1,.995),(1,1.005)]:
                na,nt = a*af,t*tf
                if 1e-6<=na<=6 and .02<=nt<=tmax:
                    extra.add((na,nt,l))
                    extra.add((na,nt,CAP if math.isinf(l) else math.inf))
            if 0<l<CAP:
                extra.update((a,t,min(CAP,l*f)) for f in [.85,1.18])
        items.update(extra)
        jobs.extend((s,a,t,l,0.) for a,t,l in extra)
    print(f"Tighter joint-design probes: {len(jobs)}",flush=True)
    run_jobs(jobs,cache,args.workers,cache_path)
    for level in [1.,2.]:
        for check in range(4):
            jobs = {(s,a,t,l,level) for s,items in designs.items()
                    for a,t,l in winning_designs(s,costs,items,cache) if (s,a,t,l,level) not in cache}
            if not jobs:
                break
            print(f"Resolution level {level:g}, pass {check+1}: {len(jobs)}",flush=True)
            run_jobs(sorted(jobs),cache,args.workers,cache_path)
    rows = [r for s in sigmas for r in profiles(s,costs,designs[s],cache)]
    # Controlled comparison: the same upgraded designs/resolution, old fixed floor.
    fixed = [r for s in sigmas for r in profiles(s,costs,designs[s],cache,relative_floor=.005)]
    pooled = [r for s in sigmas for r in profiles(s,costs,designs[s],cache,branch_screen=False)]
    white = lambda rs:sum(r["regime"] not in ["zero_on_grid","finite_on_grid","full_on_grid"] for r in rs)
    paired = [r for r in rows if (r["sigma"],r["c"]) in lookup]
    metadata = dict(qualification="Numerical grid search and resolution-difference diagnostics, not certified global boundaries.",
                    sigma=sigmas,c=costs,relative_floor=0.,roundoff_relative_floor=64*np.finfo(float).eps,old_relative_floor=.005,
                    required_resolution=2,counts=dict(Counter(r["regime"] for r in rows)),
                    white_same_old_cells_before=white(old),white_same_old_cells_after=white(paired),
                    white_fixed_floor_same_upgraded_grid=white(fixed),white_error_floor_same_upgraded_grid=white(rows),
                    white_pooled_failure_screen_same_upgraded_grid=white(pooled),
                    branch_screen="winning-family failures do not invalidate a strict family comparison",
                    output_level2_spacing=".02/.08 normalized spatial controls; 840 base time steps",
                    shapes_and_primitives_unchanged=True,
                    classification_is_not_complete_design_solution=True,
                    infinite_precision_statement_is_grid_only=True)
    for name,records in [("regions.csv",rows),("designs.csv",[dict(sigma=s,alpha=a,T=t,precision=l)
                                                            for s,items in designs.items() for a,t,l in sorted(items)])]:
        with (dest/name).open("w",newline="") as f:
            writer = csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    (dest/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    plot(rows,dest,focus_only=True)
    print(json.dumps(metadata,indent=2),flush=True)


if __name__=="__main__":
    main()
