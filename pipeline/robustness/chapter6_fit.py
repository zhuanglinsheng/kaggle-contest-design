"""Resumable matched Chapter 6 fits; preserve existing published runs."""
import concurrent.futures
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.config import MAIN_ESTIMATION_DIR
from pipeline.empirical import estimate
from pipeline.robustness.chapter6_data import ROOT


def fit_one(scenario, cid, baseline):
    raw_path = ROOT/"inputs"/scenario/f"contest_{cid}.json"
    raw = json.loads(raw_path.read_text())
    data = estimate.stan_data(raw,"full")
    checksum = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    cached = ROOT/"accepted"/scenario/f"contest_{cid}.json"
    if cached.exists():
        row = json.loads(cached.read_text())
        if row["input_sha256"] != checksum:
            raise ValueError(f"Cached input differs: {scenario}, {cid}")
        return row
    a = float(raw.get("score_scale_a",1))
    initial = {"c_i":max(.011,float(baseline.c_i_mean)/a**2),
               "c_j":max(.011,float(baseline.c_j_mean)/a**2),
               "sigma":max(.011,float(baseline.sigma_mean)*a),
               "lambda":max(.011,float(baseline.lambda_mean)/a**2),
               "log_r":math.log(max(.011,float(baseline.r_mean)/a)),
               "mu_0":float(baseline.mu_0_mean)*a if scenario != "pair" else raw["hat_y"][0]}
    for attempt,(warmup,samples,metric) in enumerate(((1000,1000,"diag_e"),
            (2000,2000,"dense_e"),(4000,4000,"dense_e")),start=1):
        mode = f"{scenario}_attempt{attempt}"
        init_dir = ROOT/"initializations"
        init_dir.mkdir(parents=True,exist_ok=True)
        for chain in range(1,5):
            rng = np.random.default_rng(800000+cid+chain+attempt*10000)
            init = {k:max(.0101,v*math.exp(rng.normal(0,.15))) if k not in ("log_r","mu_0")
                    else v+float(rng.normal(0,.15)) for k,v in initial.items()}
            (init_dir/f"contest_{cid}_{mode}_chain_{chain}.json").write_text(json.dumps(init))
        paths = estimate.run_contest(cid,mode,data,warmup=warmup,samples=samples,
            seed=20261008+cid,adapt_delta=.99,max_depth=13,init_dir=init_dir,metric=metric)
        row = estimate.summarize(cid,"full",paths)
        rhats = [row[f"{p}_rhat"] for p in ("c_i","c_j","sigma","lambda","mu_0","r")]
        # Check saturation separately from the maximum reached depth.
        saturations = sum(int(np.sum(estimate.read_columns(p)["treedepth__"]>=13)) for p in paths)
        row.update(scenario=scenario,scale_a=a,input_sha256=checksum,
                   run_dir=str(paths[0].parent),saturated_transitions=saturations,
                   draws=4*samples,attempt=attempt)
        (paths[0].parent/"diagnostics.json").write_text(json.dumps(row,indent=2)+"\n")
        if row["divergences"]==0 and max(rhats)<1.01 and saturations==0:
            cached.parent.mkdir(parents=True,exist_ok=True)
            cached.write_text(json.dumps(row,indent=2)+"\n")
            print(f"ACCEPTED {scenario} {cid}: max Rhat={max(rhats):.5f}",flush=True)
            return row
        for name in ("c_i","c_j","sigma","lambda","mu_0"):
            initial[name] = max(.0101,row[f"{name}_mean"]) if name!="mu_0" else row[f"{name}_mean"]
        initial["log_r"] = math.log(row["r_mean"])
        print(f"REFINE {scenario} {cid}: attempt {attempt}, Rhat={max(rhats):.4f}, divergences={row['divergences']}",flush=True)
    raise RuntimeError(f"Convergence not established: {scenario} {cid}")


def main():
    estimate.OUTPUT_DIR = ROOT/"fits"
    baseline = pd.read_csv(MAIN_ESTIMATION_DIR/"posterior_summary_final.csv")
    baseline = baseline.loc[baseline["mode"]=="full"].set_index("contest_id")
    scenarios = ("pair","L1","L2","Linf")
    provenance = dict(model_sha256=hashlib.sha256(estimate.MODEL.read_bytes()).hexdigest(),
                     baseline_sha256=hashlib.sha256((MAIN_ESTIMATION_DIR/"posterior_summary_final.csv").read_bytes()).hexdigest(),
                     scenarios=scenarios,priors="unchanged from Chapter 5, in transformed score units",
                     chains=4,parallel_contests=2,adapt_delta=.99,max_treedepth=13)
    (ROOT/"fit_config.json").write_text(json.dumps(provenance,indent=2)+"\n")
    accepted,failed = [],[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs = {pool.submit(fit_one,scenario,int(cid),row):(scenario,int(cid))
                for scenario in scenarios for cid,row in baseline.iterrows()}
        for future in concurrent.futures.as_completed(jobs):
            try:
                accepted.append(future.result())
            except Exception as error:
                failed.append(dict(scenario=jobs[future][0],contest_id=jobs[future][1],error=str(error)))
                print(f"FAILED {jobs[future]}: {error}",flush=True)
            pd.DataFrame(accepted).to_csv(ROOT/"posterior_summary.csv",index=False)
            (ROOT/"failures.json").write_text(json.dumps(failed,indent=2)+"\n")
    if failed or len(accepted)!=4*len(baseline):
        raise RuntimeError("Matched Chapter 6 estimation incomplete; see failures.json")
    print(f"Complete: {len(accepted)} accepted fits",flush=True)
    from pipeline.robustness.chapter6_report import main as report
    report()


if __name__ == "__main__":
    main()
