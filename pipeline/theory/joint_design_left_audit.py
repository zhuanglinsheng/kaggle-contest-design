"""Targeted direct comparisons of the low-risk unresolved cells.

Keeps the original domain, failed-design guards and payoff tolerance.
Physical-policy value verification independently checks fine weak-signal solves.
"""
from __future__ import annotations
import csv
import json
import math
import shutil
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.optimize import minimize
from pipeline.config import ROOT
from pipeline.theory.joint_design_plane import FIELDS,calculate,run_jobs
from pipeline.theory.joint_design_refined_plane import load_cache
from pipeline.theory.joint_design_moderate_plane import (
    OUT,latest,classify,winners,families,zero_design,loss_output,candidate_output_upper,draw,
)
from pipeline.theory.joint_design_moderate_dense import DENSE,COSTS,unresolved_linear_area
from pipeline.theory.joint_design_moderate_polish import recover_jobs
from pipeline.theory.joint_design_stable_audit import calculate_verified_stable


def verified_jobs(jobs,cache):
    path=DENSE/"verified_policy_samples.csv"
    missing=[k for k in jobs if k not in load_cache(path)]
    if not missing:return
    fresh=not path.exists()
    with path.open("a",newline="") as f:
        w=csv.DictWriter(f,fieldnames=FIELDS)
        if fresh:w.writeheader()
        with ProcessPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(calculate_verified_stable,k) for k in missing]
            for n,future in enumerate(as_completed(futures),1):
                row=future.result();w.writerow(row)
                cache[tuple(float(row[k]) for k in FIELDS[:5])]=row
                if n%10==0 or n==len(missing):
                    f.flush();print(f"{n}/{len(missing)} fine physical-policy checks",flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--reclassify",action="store_true",help="Reapply current comparison rules without new solves")
    parser.add_argument("--coverage",action="store_true",help="Resolve unrefined optimistic envelopes after independent-value failures")
    args=parser.parse_args()
    before=list(csv.DictReader((DENSE/"regions.csv").open()))
    backup=DENSE/"regions_before_left_audit.csv"
    if not backup.exists():shutil.copyfile(DENSE/"regions.csv",backup)
    cache=load_cache(ROOT/"results/theory/numerical_joint_plane_strong64/validated_samples.csv")
    for p in (OUT/"output_samples.csv",OUT/"stable_audit_samples.csv",
              DENSE/"output_samples.csv",DENSE/"stable_audit_samples.csv",
              DENSE/"verified_policy_samples.csv"):
        cache.update(load_cache(p))
    sigmas=sorted({float(r["sigma"]) for r in before})
    targets={s:[float(r["c"]) for r in before if float(r["sigma"])==s and
                r["regime"]=="unresolved" and float(r["c"])>=.3]
             for s in sigmas if s<=.6}
    targets={s:cc for s,cc in targets.items() if cc}
    designs={s:{k[1:4] for k in cache if k[0]==s and .00005<=k[1]<=12
                and .001<=k[2]<=math.sqrt(6)} for s in sigmas}
    if args.reclassify:
        publish_rows(cache,designs,targets,(),(),reclassify=True)
        return
    if args.coverage:
        cc=COSTS[COSTS>=.3]
        for iteration in range(2):
            changed=False
            for level in (4.,5.):
                jobs={(s,a,t,l,level) for s in targets
                      for a,t,l in winners(s,designs[s],cache,cc)
                      if math.isfinite(l) and latest(cache,(s,a,t,l))["level"]<level}
                print(f"Coverage round {iteration+1}, level {level:g}: {len(jobs)}",flush=True)
                if jobs:changed=True
                recover_jobs(sorted(jobs),cache,destination=DENSE)
            # At a new family optimum verify the physical payoff, not only
            # differences between artificial-viscosity meshes.
            saved=load_cache(DENSE/"verified_policy_samples.csv")
            jobs={(s,a,t,l,6.) for s in targets
                  for a,t,l in winners(s,designs[s],cache,cc)
                  if math.isfinite(l) and (s,a,t,l,6.) not in saved}
            print(f"Coverage physical-value checks: {len(jobs)}",flush=True)
            verified_jobs(sorted(jobs),cache)
            if not changed and not jobs:break
        publish_rows(cache,designs,targets,(),())
        return
    # Directly optimize full disclosure at low/middle/high unresolved costs.
    with (DENSE/"output_samples.csv").open("a",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS)
        for s,cc in targets.items():
            for c in sorted({cc[0],cc[len(cc)//2],cc[-1]}):
                data,_,(_,ui),_,_=families(s,designs[s],cache,[c])
                a,t=data[int(ui[0]),:2]
                t0,_,_=zero_design(s)
                def objective(log_design):
                    aa,tt=np.exp(log_design);key=(s,float(aa),float(tt),math.inf)
                    row=latest(cache,key)
                    if row is None:
                        row=calculate(key+(0.,));cache[key+(0.,)]=row;writer.writerow(row)
                    designs[s].add(key[1:])
                    return float(loss_output(row["B"]))+c*aa+tt*tt/2 if math.isfinite(row["B"]) else 1e6
                for initial in ((a,t),(.01,max(.001,t0))):
                    minimize(objective,np.log(initial),method="L-BFGS-B",
                             bounds=[(math.log(.00005),math.log(12.)),
                                     (math.log(.001),math.log(math.sqrt(6)))],
                             options=dict(maxiter=35,ftol=1e-12,eps=1e-4,maxls=12))
            f.flush()
    print(f"Direct full-disclosure search: {len(targets)} risk slices",flush=True)
    probes=set()
    for s in targets:
        for t in (.5,.7,.9):
            for exposure in (.9,1.2,1.5,1.8):
                a=exposure*(1+s*s*t)/t
                for l in (1e-6,1e-5,1e-4,.001,.01,.1,1.):
                    probes.add((s,a,t,l))
        # Include the exact failed endpoint's neighboring positive signals.
        failed=[(a,t,l) for a,t,l in designs[s] if l==0 and
                not math.isfinite(latest(cache,(s,a,t,l))["B"])]
        for c in (targets[s][0],targets[s][-1]):
            if not failed:continue
            a,t,_=min(failed,key=lambda v:float(loss_output(candidate_output_upper(s,*v)))+v[1]**2/2+c*v[0])
            for l in (1e-6,1e-5,1e-4,.001,.01,.1,1.):probes.add((s,a,t,l))
    for s,a,t,l in probes:designs[s].add((a,t,l))
    print(f"Targeted positive-precision designs: {len(probes)}",flush=True)
    recover_jobs(sorted(k+(3.,) for k in probes),cache,destination=DENSE)
    for level in (4.,5.):
        jobs={(s,a,t,l,level) for s in targets
              for a,t,l in winners(s,designs[s],cache,targets[s])
              if math.isfinite(l) and latest(cache,(s,a,t,l))["level"]<level}
        print(f"Targeted envelope refinements at level {level:g}: {len(jobs)}",flush=True)
        recover_jobs(sorted(jobs),cache,destination=DENSE)
    fine=set()
    for s,cc in targets.items():
        fine.update((s,a,t,l,6.) for a,t,l in winners(s,designs[s],cache,cc)
                    if math.isfinite(l))
        # Audit every failed endpoint whose output bound could affect a target.
        rr=classify(s,designs[s],cache,cc)
        best=np.array([r["comparison_upper"] for r in rr])
        for a,t,l in designs[s]:
            if l!=0:continue
            row=latest(cache,(s,a,t,l))
            if math.isfinite(row["B"]):continue
            bound=loss_output(candidate_output_upper(s,a,t,l))+t*t/2+np.array(cc)*a
            if np.any(bound<best):fine.add((s,a,t,l,6.))
    print(f"Finest independent value checks: {len(fine)}",flush=True)
    verified_jobs(sorted(fine),cache)
    # Full-disclosure optimizers are checked independently at both resolutions.
    for level in (1.,2.):
        jobs={(s,a,t,l,level) for s in targets for a,t,l in winners(s,designs[s],cache,targets[s])
              if math.isinf(l) and latest(cache,(s,a,t,l))["level"]<level}
        run_jobs(sorted(jobs),cache,4,DENSE/"output_samples.csv")
    publish_rows(cache,designs,targets,probes,fine)


def publish_rows(cache,designs,targets,probes,fine,reclassify=False):
    rows=[r for s in sorted(designs) for r in classify(s,designs[s],cache,COSTS)]
    with (DENSE/"regions.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    original=list(csv.DictReader((DENSE/"regions_before_left_audit.csv").open()))
    meta=json.loads((DENSE/"metadata.json").read_text())
    meta["counts"]=dict(Counter(r["regime"] for r in rows))
    meta["linear_area_comparison"]["dense_unresolved_fraction"]=unresolved_linear_area(rows)
    previous=meta.get("targeted_left_audit",{})
    meta["targeted_left_audit"]=dict(risk_slices=len(targets),positive_designs=len(probes) or previous.get("positive_designs",0),
        finest_physical_checks=len(fine) or previous.get("finest_physical_checks",0),before_counts=dict(Counter(r["regime"] for r in original)),
        after_counts=meta["counts"],unresolved_area_before=unresolved_linear_area(original),
        unresolved_area_after=unresolved_linear_area(rows),
        physical_value_residual_tolerance=.002,
        single_mesh_is_not_zero_error=True,
        reclassified_without_new_solves=reclassify,
        qualification="Direct local optimizations; unchanged payoff threshold; resolution-protected comparisons and guarded failures")
    checks=load_cache(DENSE/"verified_policy_samples.csv")
    meta["targeted_left_audit"]["finest_physical_checks"]=len(checks)
    meta["targeted_left_audit"]["verified_successes"]=sum(math.isfinite(r["B"]) for r in checks.values())
    meta["targeted_left_audit"]["verified_failures"]=sum(not math.isfinite(r["B"]) for r in checks.values())
    fields=["sigma","c","regime","finite_loss","full_loss","zero_loss",
            "finite_lower","finite_upper","full_lower","full_upper",
            "failed_finite_lower","failed_full_lower","best_point_estimate_family","unresolved_reason"]
    with (DENSE/"left_direct_comparisons.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:
            if r["sigma"] not in targets or r["c"]<.3:continue
            rr={k:r[k] for k in fields if k in r}
            rr["best_point_estimate_family"]=min(("finite","full","zero"),key=lambda k:r[k+"_loss"])
            rr["unresolved_reason"]=("competitive failed finite-feedback design" if r["failed_finite_lower"]<r["comparison_upper"]
                                      else "payoff/resolution overlap") if r["regime"]=="unresolved" else ""
            w.writerow(rr)
    (DENSE/"metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
    draw(rows,destination=DENSE,xlim=(0.,3.))
    print(json.dumps(meta["targeted_left_audit"],indent=2),flush=True)


if __name__=="__main__":main()
