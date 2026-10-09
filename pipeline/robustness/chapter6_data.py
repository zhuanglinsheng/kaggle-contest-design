"""Matched Chapter 6 data: alternative rivals and public-only scale choices."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.special import ndtri

from pipeline.config import MAIN_DATA_DIR, MAIN_ESTIMATION_DIR, RESULTS_DIR
from pipeline.empirical.rebuild_main import AUDIT, FIELDS, assign_times, logits, rebuild

ROOT = RESULTS_DIR / "robustness" / "chapter6"


def weighted_quantile(values, weights, probabilities):
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    positions = (np.cumsum(weights) - weights / 2) / weights.sum()
    return np.interp(probabilities, positions, values)


def norm_scales(values, weights):
    probabilities = (np.arange(999) + .5) / 999
    quantiles = weighted_quantile(values, weights, probabilities)
    normal = ndtri(probabilities)
    nonzero = np.abs(quantiles) > 1e-12
    ratios = normal[nonzero] / quantiles[nonzero]
    ratio_weights = np.abs(quantiles[nonzero])
    order = np.argsort(ratios)
    pos = np.searchsorted(np.cumsum(ratio_weights[order]), ratio_weights.sum()/2)
    a1 = float(ratios[order[pos]])
    a2 = float(quantiles @ normal / (quantiles @ quantiles))
    constraints = np.vstack([np.column_stack([quantiles, -np.ones(999)]),
                             np.column_stack([-quantiles, -np.ones(999)])])
    solution = linprog([0,1], A_ub=constraints, b_ub=np.r_[normal,-normal],
                       bounds=[(0,None),(0,None)], method="highs")
    if not solution.success:
        raise RuntimeError(solution.message)
    scales = {"L1": a1, "L2": a2, "Linf": float(solution.x[0])}
    if min(scales.values()) <= 0:
        raise ValueError("Degenerate normal-reference scale")
    losses = {}
    for name, a in {"baseline":1., **scales}.items():
        residual = a*quantiles-normal
        losses[name] = dict(L1=float(np.mean(abs(residual))),
                            L2=float(np.sqrt(np.mean(residual**2))),
                            Linf=float(np.max(abs(residual))))
    return scales, losses


def build():
    # The audited cache retains every distinct, pre-deadline submission.
    source = pd.read_pickle(AUDIT / "submissions.pkl").drop_duplicates("Id").dropna(subset=FIELDS)
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest, public_values, public_weights = [], [], []
    fits = pd.read_csv(MAIN_ESTIMATION_DIR/"posterior_summary_final.csv")
    ids = sorted(fits.loc[fits["mode"]=="full","contest_id"].astype(int))
    paths = [MAIN_DATA_DIR/f"contest_{cid}.json" for cid in ids]
    for path in paths:
        raw = json.loads(path.read_text())
        cid = int(path.stem.split("_")[-1])
        frame = source.loc[source.CompetitionId == cid]
        last = frame.sort_values(["SubmissionDate","Id"]).groupby("TeamId").tail(1)
        ranking = last.sort_values(["PrivateScoreFullPrecision","TeamId"], ascending=[False,True])
        ranking_ids = ranking.TeamId.astype(int).tolist()
        leader = int(raw["team_i"])
        candidates = [tid for tid in ranking_ids[:20] if tid != leader]
        selected = None
        errors = []
        for left, team_i in enumerate(candidates):
            for team_j in candidates[left+1:]:
                try:
                    alt, _ = rebuild(frame, raw, (team_i,team_j), 1234, raw["score_metric"])
                    selected = alt
                    break
                except ValueError as error:
                    errors.append(str(error))
            if selected is not None:
                break
        if selected is None:
            raise ValueError(f"No eligible alternative pair: {cid}; {errors[-3:]}")
        # Use exactly the baseline observation window, prize and transformation.
        destination = ROOT / "inputs" / "pair" / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(selected)+"\n")
        manifest.append(dict(contest_id=cid, baseline_i=leader, baseline_j=raw["team_j"],
                             alternative_i=selected["team_i"], alternative_j=selected["team_j"],
                             reconstructed_rank_i=ranking_ids.index(selected["team_i"])+1,
                             reconstructed_rank_j=ranking_ids.index(selected["team_j"])+1,
                             Ni=selected["Ni"], Nj=selected["Nj"],
                             input_sha256=hashlib.sha256(destination.read_bytes()).hexdigest()))

        # Calibrate only on submitted public scores used in the baseline window.
        timed = assign_times(frame,1234)
        start = raw["start_time"] + raw["count_start_hour"]*3600
        end = raw["start_time"] + raw["N_Delta"]*3600
        kept = timed.loc[timed.TeamId.isin([raw["team_i"],raw["team_j"]])
                         & (timed.imputed_time > start) & (timed.imputed_time <= end)]
        public = logits(kept, raw["signed_score_normalization"])[:,1]
        centered = public - np.median(public)
        public_values.extend(centered)
        public_weights.extend(np.full(len(public),1/len(public)))
        print(f"Prepared {cid}: alternative pair {selected['team_i']}, {selected['team_j']}",flush=True)
    scales, losses = norm_scales(np.array(public_values),np.array(public_weights))
    for norm,a in scales.items():
        directory = ROOT / "inputs" / norm
        directory.mkdir(parents=True,exist_ok=True)
        for path in paths:
            raw = json.loads(path.read_text())
            for key in ("hat_y","real_y"):
                raw[key] = (a*np.array(raw[key])).tolist()
            raw["score_scale_a"] = a
            (directory/path.name).write_text(json.dumps(raw)+"\n")
    pd.DataFrame(manifest).to_csv(ROOT/"pair_manifest.csv",index=False)
    (ROOT/"scale_calibration.json").write_text(json.dumps(dict(scales=scales,losses=losses,
        n_contests=len(manifest),n_public_scores=len(public_values),quantile_grid=999,
        centering="contest-specific public-score median, calibration only; gaps unchanged except scaling",
        weighting="equal contest weight; equal submission weight within contest",
        input_data="baseline focal-pair submitted public scores; no private scores used for calibration"),indent=2)+"\n")
    print(json.dumps(scales),flush=True)


if __name__ == "__main__":
    build()
