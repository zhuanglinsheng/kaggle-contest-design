"""Grid-refinement and large-precision checks at the candidate regime switch."""
import json
import math
from pathlib import Path

from archive.experiments.exact_joint_design import solve_exact, filtering_variance
from pipeline.theory.joint_design_exact import full_disclosure_output


def main():
    folder = Path("results/theory/precision_conditions")
    source = json.loads((folder / "disclosure_region_transition_fine.json").read_text())
    case = next(row for row in source["results"] if row["sigma"] == 0.73)
    sigma = case["sigma"]
    b0 = 1 / math.sqrt(2*math.pi)
    rows = []
    for kind in ("full", "finite"):
        theta, horizon = case[kind]["theta"], case[kind]["T"]
        precisions = ([math.inf,3000,1e4,1e5,1e6] if kind == "full"
                      else [case[kind]["precision"]])
        for precision in precisions:
            for refinement in (1,2) if math.isinf(precision) else (1,):
                if math.isinf(precision):
                    b = full_disclosure_output(
                        theta=theta,horizon=horizon,initial_variance=1,sigma=sigma,
                        dy=.008*math.sqrt(1+sigma*sigma*horizon)/refinement,
                        dt=horizon/(3600*refinement))
                    diag = {}
                else:
                    s0 = 1/(sigma*sigma*horizon)
                    ell = precision*sigma*sigma*horizon*horizon
                    dx = min(.008*math.sqrt(1+s0),
                             .08*math.sqrt(float(filtering_variance(1,s0,ell))))
                    result = solve_exact(w=theta/(sigma*sigma),ell=ell,s0=s0,x0=0,
                                         dx=dx,steps=2400)
                    b = sigma*math.sqrt(horizon)*result.best_output
                    diag = result.diagnostics
                score = -3*math.expm1(-5*(b-b0))-theta-horizon*horizon/2
                row = dict(kind=kind,theta=theta,T=horizon,precision=precision,
                           refinement=refinement,output=b,score=score,diagnostics=diag)
                rows.append(row)
                print(json.dumps(row),flush=True)
    (folder/"disclosure_tail_audit.json").write_text(json.dumps(dict(
        sigma=sigma,rows=rows,qualification="Fixed-design refinement and tail samples; not a uniform all-precision bound."
    ),indent=2)+"\n")


if __name__ == "__main__":
    main()
