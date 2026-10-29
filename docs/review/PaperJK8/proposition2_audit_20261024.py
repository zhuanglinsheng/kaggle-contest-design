"""Read-only audit of Proposition 2; writes only its own review artifacts.

Run with a Python environment containing NumPy and SciPy. This does not run
Stan, modify the manuscript, or recreate posterior-chain diagnostics.
"""
from pathlib import Path
import csv
import hashlib
import json
import math

import numpy as np
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
EMP = ROOT / "results/reestimation/lambda100_sd225"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixed_input_moments(p, dt=.25, steps=12, initial_public=.3):
    """No-previous-readout history, with exact Riccati path on the grid."""
    ci, cj, sigma, precision, mu, r, s0 = p
    horizon = dt * steps
    t = np.arange(steps) * dt
    steady = sigma / math.sqrt(precision)
    h = np.tanh(sigma * math.sqrt(precision) * t)
    variance = steady * (s0 + steady * h) / (steady + s0 * h)
    perceived = mu
    rates, drifts = [], []
    for q in range(steps):
        Q = variance[q] + sigma**2 * (horizon - t[q])
        kernel = math.exp(-perceived**2 / (2 * Q)) / math.sqrt(2 * math.pi * Q)
        efforts = np.array([kernel / ci, kernel / cj])  # theta = 1
        rates.append(r * efforts)
        drift = dt * (efforts[0] - efforts[1])
        drifts.append(drift)
        gain = -math.expm1(-precision * variance[q] * dt)
        perceived += drift + gain * (initial_public - perceived)
    times = np.arange(1, steps + 1) * dt
    return (np.array(rates), mu + np.cumsum(drifts),
            s0 + sigma**2 * times + 1 / (precision * times), times)


def algebra_checks():
    p = np.array([1.1, 1.8, .8, 1.4, .2, 12., 1.6])
    rates, means, variances, t = fixed_input_moments(p)
    # Identified zero-count probabilities determine rates without using readouts.
    recovered_rates = -np.log(np.exp(-.25 * rates)) / .25
    s0, sig2, inverse_precision = np.linalg.solve(
        np.column_stack((np.ones(3), t[:3], 1 / t[:3])), variances[:3])
    cumulative_difference = .25 * np.cumsum(rates[:, 0] - rates[:, 1])
    r = ((cumulative_difference[1] - cumulative_difference[0]) /
         (means[1] - means[0]))
    mu = means[0] - cumulative_difference[0] / r
    Q0 = s0 + sig2 * t[-1]
    kernel0 = math.exp(-mu**2 / (2 * Q0)) / math.sqrt(2 * math.pi * Q0)
    recovered = np.array([r * kernel0 / rates[0, 0], r * kernel0 / rates[0, 1],
                          math.sqrt(sig2), 1 / inverse_precision, mu, r, s0])
    assert np.allclose(recovered, p, rtol=1e-11, atol=1e-11)
    assert np.allclose(recovered_rates, rates, rtol=1e-12, atol=1e-12)
    # Exact invariance on the symmetric submodel.
    symmetric = p.copy()
    symmetric[1] = symmetric[0]
    scaled = symmetric.copy()
    scaled[[0, 1, 5]] *= 2.3
    sym_error = max(float(np.max(np.abs(a - b))) for a, b in zip(
        fixed_input_moments(symmetric)[:3], fixed_input_moments(scaled)[:3]))
    assert sym_error < 1e-12
    # Common-scale sensitivity vanishes as cost asymmetry vanishes.
    sensitivities = []
    for gap in (1., .1, .01, .001, 0.):
        a = p.copy()
        a[1] = a[0] * (1 + gap)
        b = a.copy()
        b[[0, 1, 5]] *= 1.2
        sensitivities.append(dict(relative_cost_gap=gap,
            maximum_first_readout_mean_change=float(np.max(np.abs(
                fixed_input_moments(a)[1] - fixed_input_moments(b)[1])))))
    # Probabilities below refer to separate possible first-readout histories.
    survival = np.exp(-.25 * np.r_[0., np.cumsum(rates.sum(1))[:-1]])
    first_prob = survival * -np.expm1(-.25 * rates.sum(1))
    return dict(recovered_parameters=recovered.tolist(),
                max_recovery_error=float(np.max(np.abs(recovered - p))),
                symmetric_scale_invariance_error=sym_error,
                first_three_history_probabilities=first_prob[:3].tolist(),
                near_symmetry_scale_sensitivity=sensitivities)


def gaussian_checks():
    """Scalar innovations vs dense Brownian residual covariance."""
    times = np.array([.25, .75, 1., 2.5])
    elapsed = np.diff(np.r_[0., times])
    residual = np.array([.1, -.2, .4, .3])
    sigma, precision, s0, jitter = .8, 1.4, 1.6, 1e-9
    covariance = (s0 + sigma**2 * np.minimum.outer(times, times)
                  + np.diag(1 / (precision * elapsed) + jitter))
    sign, logdet = np.linalg.slogdet(covariance)
    assert sign == 1
    dense = -.5 * (len(times) * math.log(2 * math.pi) + logdet
                    + residual @ np.linalg.solve(covariance, residual))
    mean, variance, scalar = 0., s0, 0.
    for z, h in zip(residual, elapsed):
        prediction = variance + sigma**2 * h
        measurement = 1 / (precision * h) + jitter
        total = prediction + measurement
        innovation = z - mean
        scalar += -.5 * (math.log(2 * math.pi * total) + innovation**2 / total)
        mean += prediction / total * innovation
        variance = prediction * measurement / total
    assert abs(dense - scalar) < 1e-12
    # A counterexample to automatically normalizing a plug-in future-data input.
    # This linear example is NOT an empirical effect estimate or a replacement model.
    rho = .3
    integral = quad(lambda y: math.exp(-((1 - rho) * y)**2 / 2) /
                    math.sqrt(2 * math.pi), -np.inf, np.inf)[0]
    assert abs(integral - 1 / (1 - rho)) < 1e-10
    return dict(dense_log_density=float(dense), scalar_log_density=float(scalar),
                absolute_difference=abs(float(dense - scalar)),
                endogenous_input_counterexample=dict(rho=rho, integral=integral,
                    normalized=False, interpretation="Generic counterexample only; not fitted to Kaggle."))


def fixed_window_checks():
    """Check precisely which steps survive a fixed, unobserved count prefix."""
    p = np.array([1.1, 1.8, .8, 1.4, .2, 12., 1.6])
    rates, means, variances, times = fixed_input_moments(p)
    prefix = 4
    dt = .25
    pre_drift = float(dt * np.sum(rates[:prefix, 0] - rates[:prefix, 1]) / p[5])
    observed_I = dt * np.cumsum(rates[prefix:, 0] - rates[prefix:, 1])
    recovered_r = (observed_I[1] - observed_I[0]) / (means[prefix+1] - means[prefix])
    recovered_intercept = means[prefix] - observed_I[0] / recovered_r
    coefficients = np.linalg.solve(np.column_stack((np.ones(3), times[prefix:prefix+3],
                                   1 / times[prefix:prefix+3])), variances[prefix:prefix+3])
    assert np.allclose(coefficients, [p[6], p[2]**2, 1/p[3]], atol=1e-12)
    assert abs(recovered_r - p[5]) < 1e-12
    assert abs(recovered_intercept - (p[4] + pre_drift)) < 1e-12
    assert abs(recovered_intercept - p[4]) > 1e-4
    return dict(prefix_intervals=prefix, recovered_r=float(recovered_r),
        true_mu=float(p[4]), recovered_intercept=float(recovered_intercept),
        prefix_drift=pre_drift, recovered_variance_coefficients=coefficients.tolist(),
        interpretation="Original subtraction still recovers r, but the intercept is mu plus unobserved-prefix drift.")


def empirical_recursion(raw, p, initial_input=None):
    """Replay effort and drift, optionally varying only the backfilled input."""
    ci, cj, sigma, precision, mu, r = p
    dt, n = raw["Delta2f"], raw["N_Delta"]
    horizon = n * dt
    steady = sigma / math.sqrt(precision)
    gain = -math.expm1(-sigma * math.sqrt(precision) * dt)
    public = np.array(raw["hat_y"], dtype=float)
    if initial_input is not None:
        # Change the prefix only, not subsequent measurements or the EB prior.
        public[:raw["count_start_hour"]] = initial_input
    x = mu
    rates = np.empty((n, 2))
    drifts = np.empty(n)
    for q in range(n):
        Q = steady + sigma**2 * (horizon - q * dt)
        kernel = raw["theta"] / 100 * math.exp(-x*x/(2*Q)) / math.sqrt(2*math.pi*Q)
        efforts = np.array([kernel / ci, kernel / cj])
        rates[q] = r * efforts  # per day, before multiplication by dt
        drifts[q] = dt * (efforts[0] - efforts[1])
        x += drifts[q] + gain * (public[q] - x)
    return rates, np.cumsum(drifts)


def empirical_checks():
    summaries = {int(r["contest_id"]): r for r in csv.DictReader(
        (EMP / "validation/chapter5_contest_level.csv").open())}
    fits = json.loads((ROOT / "results/releases/paperjk7-review-20261016/accepted_fits.json").read_text())["fits"]
    main_full = [f for f in fits if f["scenario"] == "main" and f["mode"] == "full"]
    assert len(main_full) == 73
    rows = []
    max_match_error = 0.
    missing_chains = 0
    for fit in main_full:
        path = ROOT / fit["input_snapshot"]
        assert digest(path) == fit["input_sha256"]
        raw = json.loads(path.read_text())
        cid = fit["contest_id"]
        c = raw["count_start_hour"]
        events = np.array(raw["hat_t_i"] + raw["hat_t_j"])
        indices = np.unique(np.ceil(events).astype(int))
        first = int(indices[0])  # Stan's one-based interval index
        assert all(v == raw["hat_y"][0] for v in raw["hat_y"][:c+1])
        assert events.min() > c and first >= c + 1
        p = np.array([float(summaries[cid][k]) for k in ("c_i", "c_j", "sigma", "lambda", "mu_0", "r")])
        rates, drifts = empirical_recursion(raw, p)
        intensity_integrals = raw["Delta2f"] * np.cumsum(rates[:, 0] - rates[:, 1])
        error = float(np.max(np.abs(drifts - intensity_integrals / p[5])))
        max_match_error = max(max_match_error, error)
        # Isolate future-value dependence, holding every parameter fixed.
        u, eps = raw["hat_y"][0], 1e-5
        _, plus = empirical_recursion(raw, p, u + eps)
        _, minus = empirical_recursion(raw, p, u - eps)
        derivative = float((plus[first-1] - minus[first-1]) / (2*eps))
        rows.append(dict(contest_id=cid, count_start_hour=c,
            first_readout_interval=first, first_readout_equals_backfill=raw["hat_y"][first-1] == u,
            first_readout_in_first_count_interval=first == c+1,
            initial_public=u, first_readout=raw["hat_y"][first-1],
            backfill_to_first_drift_derivative_at_marginal_medians=derivative,
            matched_drift_max_error=error, snapshot_sha256=fit["input_sha256"]))
        missing_chains += sum(not (ROOT / f).exists() for f in fit["chain_paths"])
    assert max_match_error < 1e-10
    assert sum(r["first_readout_equals_backfill"] for r in rows) == 51
    path = OUT / "proposition2_input_audit_20261024.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    floor_path = EMP / "validation/chapter5_accepted_count_floor_audit.csv"
    floor_rows = list(csv.DictReader(floor_path.open()))
    derivatives = np.array([r["backfill_to_first_drift_derivative_at_marginal_medians"]
                            for r in rows if r["first_readout_equals_backfill"]])
    return dict(contests=len(rows), frozen_input_hashes_verified=len(rows),
        same_value_backfill_and_first_readout=51,
        other_first_readout_values=22, data_dependent_prefix_above_23_hours=sum(r["count_start_hour"]>23 for r in rows),
        prefix_hours_range=[min(r["count_start_hour"] for r in rows),max(r["count_start_hour"] for r in rows)],
        maximum_matched_drift_error=max_match_error,
        prefix_input_drift_derivative_absolute_max_at_marginal_medians=float(np.max(abs(derivatives))),
        prefix_input_drift_derivative_absolute_median_at_marginal_medians=float(np.median(abs(derivatives))),
        saved_floor_audit=dict(fits=len(floor_rows),affected_draws=sum(int(r["affected_draws"]) for r in floor_rows),
                               source="Previously saved report; not rerun from chains."),
        missing_main_full_chain_files=missing_chains,
        note="Numerical replay at marginal posterior medians is a dependency check, not a posterior effect or bias estimate.")


def main():
    result = dict(date="2026-10-24", algebra=algebra_checks(), gaussian=gaussian_checks(),
                  fixed_window=fixed_window_checks(), empirical=empirical_checks())
    source_paths = ["paper/PaperJK8.tex", "metakaggle/real_data.stan",
                    "results/validation/estimation_details/detail_model.stan",
                    "pipeline/empirical/rebuild_main.py", "pipeline/empirical/estimate.py",
                    "pipeline/empirical/unified_prior_reestimate.py"]
    result["source_sha256"] = {p: digest(ROOT / p) for p in source_paths}
    result["limitations"] = ["No posterior resampling or manuscript edits.",
        "No fresh CmdStan executable-to-source parity test; compiled executables and chain bundle are absent.",
        "Numerical identities supplement the analytical proof; they do not prove empirical identification."]
    (OUT / "proposition2_audit_20261024.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
