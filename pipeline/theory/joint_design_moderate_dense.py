"""Denser measured risk slices; no interpolation of regime classifications.

Neighboring designs initialize new independent solves. Old failed competitive
endpoint designs are transferred too; they cannot be erased by continuation.
This is a local/adaptive design search, not a continuous-domain certificate.
"""
from __future__ import annotations
import csv
import json
import math
import argparse
from collections import Counter
import numpy as np

from pipeline.config import ROOT
from pipeline.theory.joint_design_plane import run_jobs,calculate,FIELDS
from pipeline.theory.joint_design_refined_plane import load_cache
from pipeline.theory.joint_design_moderate_plane import (
    OUT, latest, winners, classify, draw, candidate_output_upper, loss_output,families,zero_design,
)
from pipeline.theory.joint_design_moderate_polish import recover_jobs

DENSE=ROOT/"results/theory/numerical_joint_plane_moderate_dense"
COSTS=np.round(np.arange(.03,.500001,.0025),8)
SEED_COSTS=np.array([.03,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5])


def unresolved_linear_area(rows):
    """Midpoint-cell area on the common measured rectangle, not cell counts."""
    rows=[r for r in rows if .06-1e-12<=float(r["sigma"])<=3+1e-12
          and .03-1e-12<=float(r["c"])<=.5+1e-12]
    def widths(values,lo,hi):
        values=sorted(values)
        mid=(np.array(values[:-1])+values[1:])/2
        return dict(zip(values,np.maximum(0,np.diff(np.r_[lo,mid,hi]))))
    sw=widths({float(r["sigma"]) for r in rows},.06,3.)
    cw=widths({float(r["c"]) for r in rows},.03,.5)
    return sum(sw[float(r["sigma"])]*cw[float(r["c"])]
               for r in rows if r["regime"]=="unresolved")/(2.94*.47)


def polish_full_at_islands(designs,cache):
    """Independent continuous full-disclosure search at isolated finite cells.

    Connected components select diagnostic locations only. No labels are
    changed here; new measured outputs enter the same interval comparison.
    """
    from scipy.optimize import minimize
    ss=sorted(designs)
    rows=[classify(s,designs[s],cache,COSTS) for s in ss]
    active={(i,j) for i,rr in enumerate(rows) for j,r in enumerate(rr)
            if r["regime"]=="finite"}
    seen=set();targets=set()
    for cell in sorted(active):
        if cell in seen:continue
        component=set();pending=[cell]
        while pending:
            i,j=pending.pop()
            if (i,j) in component:continue
            component.add((i,j));seen.add((i,j))
            pending.extend(v for v in ((i-1,j),(i+1,j),(i,j-1),(i,j+1))
                           if v in active and v not in component)
        if not any(i==0 or j==0 for i,j in component):targets.update(component)
    path=DENSE/"output_samples.csv"
    print(f"Continuous full-disclosure checks at isolated cells: {len(targets)}",flush=True)
    with path.open("a",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=FIELDS)
        for n,(i,j) in enumerate(sorted(targets),1):
            s,c=ss[i],float(COSTS[j])
            data,_,(_,ui),_,_=families(s,designs[s],cache,[c])
            a,t=data[int(ui[0]),:2]
            t0,_,_=zero_design(s)
            def objective(log_design):
                aa,tt=np.exp(log_design)
                key=(s,float(aa),float(tt),math.inf)
                row=latest(cache,key)
                if row is None:
                    row=calculate(key+(0.,));cache[key+(0.,)]=row
                    writer.writerow(row)
                designs[s].add(key[1:])
                if not math.isfinite(row["B"]):return 1e6
                return float(loss_output(row["B"]))+c*aa+tt*tt/2
            for initial in ((a,t),(.1,t0)):
                minimize(objective,np.log(initial),method="L-BFGS-B",
                         bounds=[(math.log(.00005),math.log(12.)),
                                 (math.log(.001),math.log(math.sqrt(6)))],
                         options=dict(maxiter=35,ftol=1e-12,eps=1e-4,maxls=12))
            handle.flush()
            if n%10==0 or n==len(targets):print(f"{n}/{len(targets)} continuous checks",flush=True)
    return len(targets)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--neighbor-polish",action="store_true",
                        help="Transfer both-family envelope candidates between adjacent measured slices")
    parser.add_argument("--full-island-check",action="store_true",
                        help="Reoptimize full disclosure continuously at isolated finite cells")
    args=parser.parse_args()
    DENSE.mkdir(parents=True,exist_ok=True)
    cache=load_cache(ROOT/"results/theory/numerical_joint_plane_strong64/validated_samples.csv")
    for p in (OUT/"output_samples.csv",OUT/"stable_audit_samples.csv",
              DENSE/"output_samples.csv",DENSE/"stable_audit_samples.csv",
              DENSE/"verified_policy_samples.csv"):
        cache.update(load_cache(p))
    old=list(csv.DictReader((OUT/"regions.csv").open()))
    old_sigmas=sorted({float(r["sigma"]) for r in old})
    sigmas=np.unique(np.r_[.03,np.arange(.05,3.0001,.05),old_sigmas])
    old_designs={s:{k[1:4] for k in cache if k[0]==s and
                    .00005<=k[1]<=12 and .001<=k[2]<=math.sqrt(6)} for s in old_sigmas}
    seeds={s:winners(s,items,cache,SEED_COSTS) for s,items in old_designs.items()}
    for s,items in old_designs.items():
        # Failed no-signal candidates nearest the envelope are not lost.
        failed=[(a,t,l) for a,t,l in items if l==0 and
                not math.isfinite(latest(cache,(s,a,t,l))["B"])]
        for c in SEED_COSTS:
            if failed:
                a,t,l=min(failed,key=lambda v:float(loss_output(candidate_output_upper(s,*v)))
                          +v[1]**2/2+c*v[0])
                seeds[s].add((a,t,l))
    designs={}
    for s in sigmas:
        if s in old_designs:
            items=set(old_designs[s])
        else:
            neighbors=sorted(old_sigmas,key=lambda v:abs(v-s))[:2]
            items=set().union(*(seeds[n] for n in neighbors))
            # Weak-prize controls and alternative disclosure levels.
            items.update((a,t,l) for a in (.0002,.05,.2,1.,4.)
                         for t in (.05,.4,1.) for l in (0.,1.,math.inf))
        items.update(k[1:4] for k in cache if k[0]==s and k[-1]>=0 and
                     .00005<=k[1]<=12 and .001<=k[2]<=math.sqrt(6))
        designs[float(s)]=items
    print(f"{len(sigmas)} risk slices x {len(COSTS)} cost points; independent output solves",flush=True)
    run_jobs(sorted((s,a,t,l,0.) for s,items in designs.items() for a,t,l in items
                    if latest(cache,(s,a,t,l)) is None),cache,4,DENSE/"output_samples.csv")
    if args.neighbor_polish:
        proposed={s:winners(s,items,cache,SEED_COSTS) for s,items in designs.items()}
        jobs=set()
        ordered=sorted(designs)
        for i,s in enumerate(ordered):
            for neighbor in ordered[max(0,i-1):min(len(ordered),i+2)]:
                if neighbor==s:continue
                for a,t,l in proposed[neighbor]:
                    if (a,t,l) not in designs[s]:
                        designs[s].add((a,t,l));jobs.add((s,a,t,l,0.))
        print(f"Neighboring-family cross-checks: {len(jobs)}",flush=True)
        run_jobs(sorted(jobs),cache,4,DENSE/"output_samples.csv")
    # New measured slices reoptimize locally; existing audited slices remain intact.
    jobs=set()
    for s,items in designs.items():
        if s in old_sigmas:continue
        for a,t,l in winners(s,items,cache,SEED_COSTS):
            for af,tf in ((.8,1.),(1.25,1.),(1.,.8),(1.,1.25)):
                aa,tt=a*af,t*tf
                if .00005<=aa<=12 and .001<=tt<=math.sqrt(6):
                    items.add((aa,tt,l));jobs.add((s,aa,tt,l,0.))
    print(f"Local design probes: {len(jobs)}",flush=True)
    run_jobs(sorted(jobs),cache,4,DENSE/"output_samples.csv")
    for level in (1.,2.):
        jobs={(s,a,t,l,level) for s,items in designs.items()
              for a,t,l in winners(s,items,cache,COSTS)
              if latest(cache,(s,a,t,l))["level"]<level}
        print(f"Envelope refinement level {level:g}: {len(jobs)}",flush=True)
        run_jobs(sorted(jobs),cache,4,DENSE/"output_samples.csv")
    # Independent stable audit for failed candidates that could change a cell.
    targets=set()
    for s,items in designs.items():
        unresolved=[r for r in classify(s,items,cache,COSTS) if r["regime"]=="unresolved"]
        cc=np.array([r["c"] for r in unresolved])
        upper=np.array([r["comparison_upper"] for r in unresolved])
        for a,t,l in items:
            row=latest(cache,(s,a,t,l))
            if not len(cc) or math.isfinite(row["B"]) or math.isinf(l):continue
            lower=loss_output(candidate_output_upper(s,a,t,l))+t*t/2+cc*a
            if np.any(lower<upper):targets.add((s,a,t,l))
    print(f"Stable competitive-design checks: {len(targets)}",flush=True)
    for level in (3.,4.,5.):
        recover_jobs(sorted(k+(level,) for k in targets if
                            latest(cache,k)["level"]<=level),cache,destination=DENSE)
    island_checks=polish_full_at_islands(designs,cache) if args.full_island_check else 0
    # Recovered designs can move either family optimum.
    for level in (1.,2.):
        jobs={(s,a,t,l,level) for s,items in designs.items()
              for a,t,l in winners(s,items,cache,COSTS)
              if latest(cache,(s,a,t,l))["level"]<level}
        run_jobs(sorted(jobs),cache,4,DENSE/"output_samples.csv")
    rows=[r for s,items in designs.items() for r in classify(s,items,cache,COSTS)]
    with (DENSE/"regions.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    metadata=dict(S0=1.,utility="3*(1-exp(-2*B))",delay="T**2/2",
        sigma=sigmas.tolist(),c=COSTS.tolist(),counts=dict(Counter(r["regime"] for r in rows)),
        risk_step=.05,cost_step=.0025,display_sigma=[0.,3.],display_c=[0.,.5],
        qualification="Independent solves and adaptive local design search; not a global certificate. No label interpolation.",
        excluded_zero_strip="c<0.03 and sigma<0.03 are hatched, not extrapolated.",
        competitive_failed_audits=len(targets))
    metadata["neighbor_design_cross_check"]=args.neighbor_polish
    metadata["continuous_full_disclosure_island_checks"]=island_checks
    metadata["linear_area_comparison"]=dict(
        rectangle_sigma=[.06,3.],rectangle_c=[.03,.5],
        old_unresolved_fraction=unresolved_linear_area(old),
        dense_unresolved_fraction=unresolved_linear_area(rows),
        interpretation="Midpoint-cell visualization weights, not certified boundary areas")
    (DENSE/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    draw(rows,destination=DENSE,xlim=(0.,3.))
    print(json.dumps(metadata["counts"],indent=2),flush=True)


if __name__=="__main__":main()
