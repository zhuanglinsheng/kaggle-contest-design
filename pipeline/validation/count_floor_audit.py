"""Check whether the numerical event-intensity floor affects retained draws."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pipeline.empirical.estimate import read_columns


def audit(run, draws_per_chain=100):
    data = json.loads((run / "data.json").read_text())
    chains = [read_columns(run / f"chain_{k}.csv") for k in range(1, 5)]
    values = {}
    for key in ("c_i", "c_j", "sigma", "lambda", "mu_0", "log_r"):
        values[key] = np.concatenate([c[key][np.linspace(0, len(c[key])-1, draws_per_chain, dtype=int)] for c in chains])
    gap = values["mu_0"].copy()
    sigma, precision = values["sigma"], values["lambda"]
    dt, horizon = data["Delta2f"], data["T_contest"]
    variance = sigma / np.sqrt(precision)
    gain = -np.expm1(-np.sqrt(precision) * sigma * dt)
    advantage = np.zeros(len(gap))
    counts = [np.bincount(np.ceil(data[key]).astype(int), minlength=data["N_Delta"]+1)[1:]
              for key in ("hat_t_i", "hat_t_j")]
    for index in range(data["N_Delta"]):
        q = sigma**2 * (horizon - index * dt) + variance
        log_kernel = -.5 * np.log(2 * np.pi * q) - gap**2 / (2*q)
        for cost, count in zip(("c_i", "c_j"), counts):
            log_intensity = values["log_r"] + np.log(dt * data["theta"]) - np.log(values[cost]) + log_kernel
            if count[index]:
                advantage += count[index] * np.maximum(0, np.log(1e-12) - log_intensity)
        gap += (data["theta"] * np.exp(log_kernel) * (1/values["c_i"]-1/values["c_j"]) * dt
                + gain * (data["hat_y"][index]-gap))
    return dict(draws=len(gap), affected_draws=int((advantage>0).sum()),
                median_loglikelihood_advantage=float(np.median(advantage)),
                maximum_loglikelihood_advantage=float(np.max(advantage)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--draws-per-chain", type=int, default=100)
    args = parser.parse_args()
    rows = []
    for row in pd.read_csv(args.directory / "posterior_summary.csv").itertuples():
        run = args.directory / row.mode / f"contest_{row.contest_id}"
        rows.append(dict(contest_id=row.contest_id, mode=row.mode, **audit(run,args.draws_per_chain)))
    frame = pd.DataFrame(rows)
    frame.to_csv(args.directory / "count_floor_audit.csv", index=False)
    print(frame.loc[frame.affected_draws>0].to_string(index=False))
    print(f"Audited {len(frame)} completed fits")


if __name__ == "__main__":
    main()
