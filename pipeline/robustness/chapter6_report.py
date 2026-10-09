"""Verify and summarize completed Chapter 6 fits without editing the paper."""
import argparse
import hashlib
import json
import math
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from pipeline.config import MAIN_ESTIMATION_DIR, MAIN_VALIDATION_DIR, manifest_path
from pipeline.empirical import counterfactual
from pipeline.empirical import precision_design
from pipeline.empirical.estimate import stan_data
from pipeline.robustness.chapter6_data import ROOT
from pipeline.validation.chapter5 import benchmark_row, fit_row, save_plot


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--wait",action="store_true")
    parser.add_argument("--scenarios", nargs="+", choices=["pair", "L1", "L2", "Linf"])
    args=parser.parse_args()
    scenarios = args.scenarios or ["pair", "L1", "L2", "Linf"]
    output = ROOT if args.scenarios is None else ROOT/"reports"/"_".join(scenarios)
    output.mkdir(parents=True, exist_ok=True)
    if args.wait:
        while True:
            files=list((ROOT/"accepted").glob("*/contest_*.json"))
            if len(files)==292:
                # The fit driver writes its combined summary after each job.
                if (ROOT/"posterior_summary.csv").exists() and len(pd.read_csv(ROOT/"posterior_summary.csv"))==292:
                    break
            failures_path=ROOT/"failures.json"
            if failures_path.exists():
                try:
                    failures=json.loads(failures_path.read_text())
                    if failures and len(files)+len(failures)>=292:
                        raise RuntimeError("Some fits failed; resolve failures.json before reporting")
                except json.JSONDecodeError:
                    pass
            time.sleep(10)
    fit_config = json.loads((ROOT/"fit_config.json").read_text())
    baseline_path = MAIN_ESTIMATION_DIR/"posterior_summary_final.csv"
    if hashlib.sha256(baseline_path.read_bytes()).hexdigest()!=fit_config["baseline_sha256"]:
        raise RuntimeError("Baseline changed during the matched Chapter 6 run")
    fits = pd.read_csv(ROOT/"posterior_summary.csv")
    fits = fits.loc[fits.scenario.isin(scenarios)].copy()
    if len(fits)!=73*len(scenarios) or fits.duplicated(["scenario","contest_id"]).any():
        raise RuntimeError("All matched fits in the requested scenarios are required before reporting")
    baseline = pd.read_csv(baseline_path)
    baseline = baseline.loc[baseline["mode"]=="full"].set_index("contest_id")
    for scenario, group in fits.groupby("scenario"):
        if set(group.contest_id)!=set(baseline.index):
            raise RuntimeError(f"Contest set differs: {scenario}")
    columns = [f"{p}_rhat" for p in ("c_i","c_j","sigma","lambda","mu_0","r")]
    if fits.divergences.sum() or fits[columns].max().max()>=1.01 or fits.saturated_transitions.sum():
        raise RuntimeError("Some fits still require convergence refinement")
    details, regressions, scale_rows, policy_rows = [],[],[],[]
    for scenario, group in fits.groupby("scenario"):
        directory=ROOT/"accepted"/scenario
        manifest_file=directory/"accepted_chain_sources.csv"
        manifest=[]
        for _,row in group.iterrows():
            cid=int(row.contest_id)
            path=ROOT/"inputs"/scenario/f"contest_{cid}.json"
            raw=json.loads(path.read_text())
            if hashlib.sha256(path.read_bytes()).hexdigest()!=row.input_sha256:
                raise RuntimeError(f"Input checksum differs: {scenario} {cid}")
            run_path=ROOT/"fits"/f"{scenario}_attempt{int(row.attempt)}"/f"contest_{cid}"
            if json.loads((run_path/"data.json").read_text())!=stan_data(raw,"full"):
                raise RuntimeError(f"Fitted data differ: {scenario} {cid}")
            params=pd.Series({p:float(row[f"{p}_mean"]) for p in ("c_i","c_j","sigma","lambda","mu_0","r")})
            benchmarks=benchmark_row(raw,params)
            a=float(row.scale_a)
            details.append(dict(scenario=scenario,contest_id=cid,scale_a=a,
                **{p:params[p] for p in params.index},**benchmarks,
                common_lambda=a*a*params["lambda"],common_sigma=params["sigma"]/a))
            manifest.append(dict(contest_id=cid,mode="full",accepted_source="chapter6",
                run_dir=manifest_path(run_path,manifest_file),input_sha256=row.input_sha256))
        pd.DataFrame(manifest).to_csv(manifest_file,index=False)
        # Reuse the exact Chapter 5 policy evaluator for the scale exercises.
        if scenario!="pair":
            counterfactual.FIT_ROOT=directory
            counterfactual.DATA_DIR=ROOT/"inputs"/scenario
            precision_design.DATA_DIR=counterfactual.DATA_DIR
            policies=[precision_design.analyze(int(cid)) for cid in group.contest_id]
            for row in policies:
                row["scenario"]=scenario
                policy_rows.append(row)
            frame=pd.DataFrame([d for d in details if d["scenario"]==scenario]).set_index("contest_id")
            full=sum(math.isinf(p["optimal_precision"]) for p in policies)
            scale_rows.append(dict(scenario=scenario,scale_a=float(group.scale_a.iloc[0]),
                lambda_rank_correlation=float(spearmanr(frame.common_lambda,baseline.loc[frame.index,"lambda_mean"]).statistic),
                sigma_rank_correlation=float(spearmanr(frame.common_sigma,baseline.loc[frame.index,"sigma_mean"]).statistic),
                full_disclosure_policies=full,
                median_relative_output_gain_pct=float(np.median([p["output_gain_pct"] for p in policies]))))
    frame=pd.DataFrame(details)
    frame.to_csv(output/"contest_diagnostics.csv",index=False)
    pd.DataFrame(policy_rows).to_csv(output/"scale_precision_policies.csv",index=False)
    for scenario, work in frame.groupby("scenario"):
        work=work.copy()
        for p in ("lambda","sigma","lambda_bench","sigma_bench","D_path","V_path"):
            work[f"log1p_{p}"]=np.log1p(work[p])
        for p in ("lambda","sigma"):
            row,fit=fit_row(work,f"log1p_{p}",f"log1p_{p}_bench",f"{scenario}_{p}_benchmark")
            regressions.append(row)
            if scenario=="pair":
                save_plot(work,f"log1p_{p}",f"log1p_{p}_bench",fit,
                    output/f"validate_{p}_pair.pdf",rf"$\log(1+\hat{{\{p}}})$",rf"$\log(1+\{p}^{{bench}})$",show_statistics=True)
    pd.DataFrame(regressions).to_csv(output/"regressions.csv",index=False)
    pd.DataFrame(scale_rows).to_csv(output/"scale_summary.csv",index=False)
    report=dict(accepted_fits=len(fits),divergences=int(fits.divergences.sum()),
        maximum_split_rhat=float(fits[columns].max().max()),scale_results=scale_rows,
        pair_results=[r for r in regressions if r["specification"].startswith("pair_")])
    (output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    main()
