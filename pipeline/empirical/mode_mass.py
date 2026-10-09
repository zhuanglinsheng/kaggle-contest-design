"""Numerically audit the relative posterior mass of separated volatility modes.

Use disjoint posterior halves for Gaussian proposal fitting and bridge sampling.
Report region-specific normalizing constants; this is a numerical diagnostic,
not a certificate that no additional modes exist. The Stan target and all
unconstraining Jacobians are retained.
"""
import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from pipeline.config import EMPIRICAL_MODEL
from pipeline.empirical.estimate import PARAMETERS, read_columns


def unconstrain(chain):
    values = np.column_stack([chain[p] for p in PARAMETERS])
    values[:, :4] = np.log(values[:, :4] - .01)
    values[:, 5] = np.log(values[:, 5] - np.log(.01))
    return values


def bridge(logfp, logqp, logfg, logqg):
    n, m = len(logfp), len(logfg)
    log_s1, log_s2 = np.log([n / (n + m), m / (n + m)])
    value = logsumexp(logfg - logqg) - np.log(m)
    for iteration in range(1000):
        denominator_g = np.logaddexp(log_s1 + logfg, log_s2 + value + logqg)
        denominator_p = np.logaddexp(log_s1 + logfp, log_s2 + value + logqp)
        update = (logsumexp(logfg - denominator_g) - np.log(m)
                  - logsumexp(logqp - denominator_p) + np.log(n))
        if abs(update - value) < 1e-9:
            return float(update), iteration + 1
        value = update
    raise RuntimeError("Bridge iteration failed to converge")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--split-sigma", type=float, default=.2)
    parser.add_argument("--draws", type=int, default=3000)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    out = run / "mode_mass"
    out.mkdir(exist_ok=True)
    chains = [read_columns(run / f"chain_{i}.csv") for i in range(1, 5)]
    groups = {"low": [], "high": []}
    for chain in chains:
        low = chain["sigma"] < args.split_sigma
        if not (low.all() or (~low).all()):
            raise RuntimeError("Chain crosses the proposed split; use mixed-chain estimation instead")
        groups["low" if low.all() else "high"].append(chain)
    reports = []
    for region, group in groups.items():
        if not group:
            raise RuntimeError("The mass comparison needs samples from both regions")
        fit, posterior, log_p = [], [], []
        for chain in group:
            transformed = unconstrain(chain)
            half = len(transformed) // 2
            fit.append(transformed[:half])
            posterior.append(transformed[half:])
            log_p.append(chain["lp__"][half:])
        fit, posterior, log_p = np.concatenate(fit), np.concatenate(posterior), np.concatenate(log_p)
        proposal = multivariate_normal(mean=fit.mean(axis=0), cov=np.cov(fit.T) + np.eye(6) * 1e-8)
        for repeat in range(3):
            draws = proposal.rvs(size=args.draws, random_state=np.random.default_rng(2748 + repeat))
            path = out / f"{region}_{repeat}_parameters.csv"
            # Append posterior points to verify Stan's target matches lp__.
            constrained = np.vstack([draws, posterior[:3]]).copy()
            constrained[:, :4] = .01 + np.exp(constrained[:, :4])
            constrained[:, 5] = np.log(.01) + np.exp(constrained[:, 5])
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(PARAMETERS)
                writer.writerows(constrained)
            output = out / f"{region}_{repeat}.csv"
            with (out / f"{region}_{repeat}.log").open("w") as handle:
                subprocess.run([str(EMPIRICAL_MODEL), "log_prob", f"constrained_params={path}",
                    "jacobian=true", "data", f"file={run / 'data.json'}", "output",
                    f"file={output}", "sig_figs=16"], stdout=handle, stderr=subprocess.STDOUT, check=True)
            evaluated = read_columns(output)
            density = evaluated["lp__"]
            if np.max(np.abs(density[-3:] - log_p[:3])) > .01:
                raise RuntimeError("Stan target does not match the unconstrained posterior density")
            log_g = density[:-3].copy()
            inside = (.01 + np.exp(draws[:, 2])) < args.split_sigma
            if region == "high":
                inside = ~inside
            log_g[~inside] = -np.inf
            log_z, iterations = bridge(log_p, proposal.logpdf(posterior), log_g, proposal.logpdf(draws))
            reports.append(dict(region=region, repeat=repeat, log_normalizer=log_z,
                                bridge_iterations=iterations, proposal_draws=args.draws,
                                evaluation_posterior_draws=len(log_p)))
    low = np.mean([r["log_normalizer"] for r in reports if r["region"] == "low"])
    high = np.mean([r["log_normalizer"] for r in reports if r["region"] == "high"])
    report = dict(split_sigma=args.split_sigma, low_to_high_mass_ratio=float(np.exp(low - high)),
                  log_mass_ratio=float(low - high), estimates=reports,
                  qualification="Region-specific numerical bridge estimates, not a global mode-exhaustiveness certificate")
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
