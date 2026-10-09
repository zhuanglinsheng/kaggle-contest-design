"""Three-way point-estimate decisions for previously unresolved cells.

Does not discard failed competitors or convert numerical diagnostics into
certificates. Keeps the guarded classification intact and publishes a separate
exploratory panel and decision table, with uncertainty recorded independently.
No new equilibrium solves are performed by this reporting pass.
"""
from __future__ import annotations

import csv
import json
from collections import Counter

from pipeline.theory.joint_design_moderate_dense import DENSE
from pipeline.theory.joint_design_moderate_plane import draw


def decision(row):
    losses = {name: float(row[name + "_loss"])
              for name in ("zero", "full", "finite")}
    ordered = sorted(losses, key=losses.get)
    winner, runner = ordered[:2]
    best_upper = min(float(row["comparison_upper"]), losses["zero"])
    failed_competitor = min(float(row["failed_finite_lower"]),
                            float(row["failed_full_lower"])) < best_upper
    return dict(
        sigma=float(row["sigma"]), c=float(row["c"]),
        classification=winner,
        classification_basis="minimum loss among successfully evaluated designs",
        payoff_margin=losses[runner] - losses[winner],
        runner_up=runner,
        zero_loss=losses["zero"], full_loss=losses["full"],
        finite_loss=losses["finite"],
        interval_winner=min(("zero", "full", "finite"), key=lambda name:
            losses["zero"] if name == "zero" else float(row[name + "_upper"])),
        competitive_failed_design=failed_competitor,
        qualification=("competitive design still lacks a reliable payoff"
                       if failed_competitor else "payoff-resolution intervals overlap"),
        global_classification_verified=False,
    )


def main():
    rows=list(csv.DictReader((DENSE/"regions.csv").open()))
    selected=[decision(r) for r in rows if r["regime"]=="unresolved"]
    out=DENSE/"three_class_comparison"
    out.mkdir(exist_ok=True)
    with (out/"unresolved_point_decisions.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(selected[0]))
        writer.writeheader();writer.writerows(selected)
    lookup={(r["sigma"],r["c"]):r["classification"] for r in selected}
    panel=[]
    for row in rows:
        converted={k:(v if k=="regime" else float(v))
                   for k,v in row.items() if k in ("sigma","c","regime")}
        key=(converted["sigma"],converted["c"])
        converted["regime"]=lookup.get(key,converted["regime"])
        panel.append(converted)
    with (out/"three_class_regions.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["sigma","c","regime"])
        writer.writeheader();writer.writerows(panel)
    draw(panel,destination=out,xlim=(.03,3.),ylim=(.03,.5),
         three_class_estimates=True,smooth_boundaries=True)
    meta=dict(previously_unresolved=len(selected),
              point_estimate_counts=dict(Counter(r["classification"] for r in selected)),
              competitive_failed_designs=sum(r["competitive_failed_design"] for r in selected),
              conservative_envelope_disagreements=sum(
                  r["classification"]!=r["interval_winner"] for r in selected),
              strict_source_preserved=True,new_equilibrium_solves=0,
              display_smoothing=dict(method="smoothing splines of region envelopes",
                                     cost_rms_fit_residual=.0035,
                                     changes_classification_data=False),
              qualification="Exploratory three-class decisions, not verified global classifications")
    (out/"metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(json.dumps(meta,indent=2))


if __name__=="__main__":main()
