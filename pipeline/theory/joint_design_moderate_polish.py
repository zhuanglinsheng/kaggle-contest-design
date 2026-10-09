"""Recover competitive failed designs without relaxing plane classification."""
from __future__ import annotations
import csv
import json
import math
import time
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed

import numpy as np

from pipeline.config import ROOT
from pipeline.theory.joint_design_plane import FIELDS,upper_output,run_jobs
from pipeline.theory.joint_design_refined_plane import load_cache
from pipeline.theory.joint_design_moderate_plane import OUT,latest,loss_output,classify,draw,candidate_output_upper,winners
from pipeline.theory.joint_design_stable_audit import calculate_stable


def recover_jobs(jobs,cache,workers=4,destination=None):
    path=(OUT if destination is None else destination)/"stable_audit_samples.csv"
    missing=[j for j in jobs if j not in cache]
    if not missing:return
    fresh=not path.exists()
    start=time.monotonic()
    with path.open("a",newline="") as f:
        w=csv.DictWriter(f,fieldnames=FIELDS)
        if fresh:w.writeheader()
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures=[pool.submit(calculate_stable,j) for j in missing]
            for n,future in enumerate(as_completed(futures),1):
                row=future.result();w.writerow(row)
                key=tuple(float(row[k]) for k in FIELDS[:5]);cache[key]=row
                if n%10==0 or n==len(missing):
                    f.flush();print(f"{n}/{len(missing)} stable checks; {time.monotonic()-start:.1f}s",flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--finest",action="store_true",help="Audit remaining no-signal designs at dx=0.0025")
    parser.add_argument("--weak-signal",action="store_true",help="Probe positive precisions adjacent to competitive no-signal designs")
    args=parser.parse_args()
    old=list(csv.DictReader((OUT/"regions.csv").open()))
    sigmas=sorted({float(r["sigma"]) for r in old})
    cache=load_cache(ROOT/"results/theory/numerical_joint_plane_strong64/validated_samples.csv")
    cache.update(load_cache(OUT/"output_samples.csv"))
    cache.update(load_cache(OUT/"stable_audit_samples.csv"))
    designs={s:{(k[1],k[2],k[3]) for k in cache if k[0]==s and .00005<=k[1]<=12
                and .001<=k[2]<=math.sqrt(6.)} for s in sigmas}
    # Interval-envelope minimizers can differ from point-estimate minimizers.
    # Check their resolution too; otherwise a coarse losing candidate can
    # unnecessarily prevent a classification even after the winner is refined.
    for level in (1.,2.):
        jobs={(s,a,t,l,level) for s,items in designs.items()
              for a,t,l in winners(s,items,cache)
              if latest(cache,(s,a,t,l))["level"]<level}
        print(f"Envelope resolution checks at level {level:g}: {len(jobs)}",flush=True)
        run_jobs(sorted(jobs),cache,4,OUT/"output_samples.csv")
    if args.weak_signal:
        probes=set()
        for s,items in designs.items():
            unresolved=[r for r in classify(s,items,cache)
                        if r["regime"]=="unresolved" and r["c"]>=.03-1e-12]
            endpoints=[(a,t) for a,t,l in items if l==0]
            # At each unresolved cost retain the most optimistic failed
            # endpoint; the endpoint itself remains in every comparison.
            for r in unresolved:
                candidates=[(float(loss_output(candidate_output_upper(s,a,t,0.)))
                             +t*t/2+r["c"]*a,a,t) for a,t in endpoints
                            if not math.isfinite(latest(cache,(s,a,t,0.))["B"])]
                if not candidates:continue
                bound,a,t=min(candidates)
                if bound>=r["comparison_upper"]:continue
                for l in (.0001,.001,.01):probes.add((s,a,t,l))
        for s,a,t,l in probes:designs[s].add((a,t,l))
        print(f"Adjacent positive-precision designs: {len(probes)}",flush=True)
        for level in (3.,4.,5.):
            recover_jobs(sorted(key+(level,) for key in probes),cache)
        if args.finest:
            recover_jobs(sorted(key+(6.,) for key in probes),cache,workers=2)
    targets=set()
    for s,items in designs.items():
        rows=[r for r in classify(s,items,cache) if r["regime"]=="unresolved" and r["c"]>=.03-1e-12]
        costs=np.asarray([r["c"] for r in rows]);best=np.asarray([r["comparison_upper"] for r in rows])
        for a,t,l in items:
            r=latest(cache,(s,a,t,l))
            if not len(costs) or math.isfinite(r["B"]) or math.isinf(l):continue
            lower=loss_output(candidate_output_upper(s,a,t,l))+t*t/2+costs*a
            if np.any(lower<best):targets.add((s,a,t,l))
    # Include previously audited targets on resume: all must have three meshes.
    targets.update(k[:4] for k in cache if k[-1]>=3)
    print(f"{len(targets)} competitive failed designs; three vanishing-viscosity meshes",flush=True)
    for level in (3.,4.,5.):
        recover_jobs(sorted(key+(level,) for key in targets),cache)
    if args.finest:
        # Only unresolved competitive endpoint designs, not all prior targets.
        fine=set()
        for s,items in designs.items():
            unresolved=[r for r in classify(s,items,cache) if r["regime"]=="unresolved" and r["c"]>=.03-1e-12]
            costs=np.asarray([r["c"] for r in unresolved])
            best=np.asarray([r["comparison_upper"] for r in unresolved])
            for a,t,l in items:
                if l!=0 or not len(costs):continue
                row=latest(cache,(s,a,t,l))
                if row["level"]<3:continue
                lower=loss_output(candidate_output_upper(s,a,t,l))+t*t/2+costs*a
                if np.any(lower<best):fine.add((s,a,t,l,6.))
        print(f"Endpoint finest-mesh checks: {len(fine)}",flush=True)
        recover_jobs(sorted(fine),cache,workers=2)
    rows=[r for s,items in designs.items() for r in classify(s,items,cache)]
    before_path=OUT/"regions_before_stable_audit.csv"
    baseline=list(csv.DictReader(before_path.open())) if before_path.exists() else old
    if not before_path.exists():
        with before_path.open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(old[0]));w.writeheader();w.writerows(old)
    with (OUT/"regions.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    metadata=json.loads((OUT/"metadata.json").read_text())
    viewed=lambda rr:[r for r in rr if float(r["c"])>=.03-1e-12]
    before=Counter(r["regime"] for r in viewed(baseline));after=Counter(r["regime"] for r in viewed(rows))
    def unresolved_area(rr):
        rr=viewed(rr)
        def widths(xs):
            z=np.log(xs);m=(z[:-1]+z[1:])/2
            return dict(zip(xs,np.diff(np.r_[z[0],m,z[-1]])))
        ss=sorted({float(r["sigma"]) for r in rr})
        cs=sorted({float(r["c"]) for r in rr})
        ws,wc=widths(ss),widths(cs)
        area=sum(ws[float(r["sigma"])]*wc[float(r["c"])]
                 for r in rr if r["regime"]=="unresolved")
        return area/(math.log(ss[-1]/ss[0])*math.log(cs[-1]/cs[0]))
    metadata.update(counts=dict(Counter(r["regime"] for r in rows)),
        stable_audit=dict(targets=len(targets),levels=[3,4,5,6] if args.finest else [3,4,5],
            dx=[.02,.01,.005,.0025] if args.finest else [.02,.01,.005],
            paper_counts_before=dict(before),paper_counts_after=dict(after),
            unresolved_log_panel_area_before=unresolved_area(baseline),
            unresolved_log_panel_area_after=unresolved_area(rows),
            family_comparison="Optimized optimistic/pessimistic envelopes; not error at a single point winner",
            no_signal_value_verification_tolerance=2e-3,
            finest_endpoint_audit=args.finest,
            adjacent_positive_precision_audit=args.weak_signal,
            method="Backward Euler + CFL LF; physical upwind organizer PDE; up to four mesh checks; unchanged payoff tolerance"))
    (OUT/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    draw(rows)
    print(json.dumps(metadata["stable_audit"],indent=2),flush=True)


if __name__=="__main__":main()
