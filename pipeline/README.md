# Current PaperJK7 pipeline

All modules are invoked from the repository root with `python -m`. Shared paths
and the common USD 100,000 prize unit are defined in `pipeline/config.py`.
Run `make models` after changing any Stan source file.

The compact publication release is verified by `make verify-release`; its
hashes are recorded in `results/release_manifest.json`. The table below lists
the current entry points. Later paragraphs retain detailed provenance for the
estimation shards and numerical experiments but do not define alternative
publication versions.

| Stage | Module | Main output |
| --- | --- | --- |
| Optimal-policy synthetic DGP only | `pipeline.synthetic.optimal_generate` | `results/synthetic/optimal_hourly/` (steady and S0=4 data, two-panel figures) |
| Retained first-order comparison | `pipeline.synthetic.generate` | `results/synthetic/variance_only/` |
| Table 1 initialization comparison | `pipeline.synthetic.model_comparison` | `results/synthetic/initialization_comparison/` |
| Section 4.1 steady-state recovery | `pipeline.synthetic.richer_recovery` | `results/synthetic/richer_steady/` |
| Previous fixed/free-r comparison (retained) | `pipeline.synthetic.table1` | `results/synthetic/table1_optimal_hourly/` |
| Original synthetic recovery (deferred) | `pipeline.synthetic.recovery` | `results/synthetic/recovery/` |
| Main estimation | `pipeline.empirical.estimate` | `results/estimation/main/` |
| Failed-fit reruns | `pipeline.empirical.rerun_failed` | `results/estimation/main/reruns/` |
| Accepted-fit manifest | `pipeline.empirical.finalize` | final summary and accepted-chain manifest |
| Counterfactuals | `pipeline.empirical.counterfactual` | `results/counterfactual/main/` |
| Chapter 5 validation | `pipeline.validation.chapter5` | `results/validation/main/` and paper figures |
| Matched Chapter 6 data | `pipeline.robustness.chapter6_data` | `results/robustness/chapter6/inputs/` |
| Matched Chapter 6 estimation | `pipeline.robustness.chapter6_fit` | `results/robustness/chapter6/fits/` |
| Chapter 6 release report | `pipeline.robustness.chapter6_report --scenarios pair L1 L2` | `results/robustness/chapter6/reports/pair_L1_L2/` and paper figures |
| Release verification | `pipeline.release` | `results/release_manifest.json` |
| Moderate-curvature joint-design Figure 3 | `pipeline.theory.joint_design_moderate_plane` | `results/theory/numerical_joint_plane_moderate/` |

The main and Chapter 6 estimators use the same Stan likelihood. Raw CmdStan
chains are resumable; finalization records exactly which run is accepted for
every contest. Manifest paths are stored relative to their manifest, so the
repository can be moved without invalidating provenance.

Early fits retain `T_contest` from the full contest while `N_Delta`
controls the truncated observation window. The split excludes the
per-contest backfilled initialization prefix. Finalization requires all 146 fits to pass
convergence checks and verifies each saved Stan input against the current
contest data. The accepted-chain manifest records the actual run directory
and a checksum of the complete input file. Validation and counterfactuals
refuse mismatched data versions. The former accepted summary is preserved.

Historical notebooks are under `archive/notebooks/`. They are useful for
provenance but are not authoritative implementations of the current model.

The authoritative Chapter 5 correction batch is
`results/estimation/chapter5_all_submissions/`. It retains the 73 original
focal pairs and full strategic horizons. `pipeline.empirical.score_audit`
caches the raw records; `pipeline.empirical.rebuild_main` verifies the pair
IDs against the legacy joint score-gap signatures and reconstructs the
inputs from every submission, rather than deduplicating calendar dates.
Prizes for 2489, 3064, 3353, 3366, 3774, 4493, 7115, and 14420 are
USD 250, 150, 6,000, 200, 250, 5,000, 10,000, and 8,000, respectively,
from the archived competition descriptions. Contest 2489 uses the
prediction-task prize and 4493 the ordinary winner prize; the others are
cash first-minus-second differences. Noncash and separately conditioned
supplementary awards (including travel reimbursements and software-specific
prizes) are excluded.

Calendar dates are assigned distinct, sorted random second offsets in
submission-ID order (seed 1234). These are imputed, not observed intraday
times. Hourly scores are aligned with interval endpoints, including events
exactly at an endpoint. The per-contest `count_start_hour`,
`initialization_hours`, and `path_start_index` exclude the artificial prefix
from count exposure, split placement, and validation outcomes, respectively.
Signed scores are mapped from [-1,1] to [0,1] before clipping to
[0.0001,0.9999] and applying the logit; no 0.5-domain filter is used.
The scalar Gaussian innovations likelihood is algebraically identical to
the previous dense Brownian covariance likelihood, but avoids its cubic
factorization cost. Re-estimate both modes using
`python -m pipeline.empirical.estimate --accepted-sample --output-dir
results/estimation/chapter5_all_submissions --adapt-delta 0.99`.
Publication is blocked until matched, converged corrections exist for all
146 fits. The earlier `chapter5_corrected` exposure-only batch is superseded
and must not be published with rebuilt data.
The first 37 contests run in `chapter5_all_submissions`, the next 24 in
`chapter5_all_submissions_upper`, and the last 12 in the two six-contest
shards `chapter5_all_submissions_tail` and `chapter5_all_submissions_tail_upper`.
The tail shards use dispersed prior-centered
initializations. Accepted refinements are in
`chapter5_all_submissions_refinements`; the longer dense-metric fit for
8219/full is in `chapter5_all_submissions_dense_refinements_long`.
Only numerical sampling settings change in these refinements.
Finalization verifies the actual
input of each accepted fit, so the superseded 3064 fits with the former prize
cannot enter the paper.

The corrected publication batch is complete: 73 contests, 13,311 retained
submissions, and 146 full/early fits. All accepted fits have no divergences
and maximum parameter R-hat 1.00753. Their source manifests and input
snapshots are in `results/estimation/main`. Validation tables and Figures
6--7 were regenerated from these estimates. The full retained-draw intensity
floor audit (616,000 draws) and the fixed-grid benchmark check are in
`results/validation/main/chapter5_final_audit.json`. The time-imputation
sensitivity report is `chapter5_time_imputation.json` in that directory.
This publication update covers Sections 5.1--5.2 and their related appendix;
the subsequent counterfactual and alternative-pair results are not regenerated
by the Chapter 5 validation command.

Time-imputation checks use seeds 20261008 and 20261009 for eight contests
preselected at evenly spaced ranks in total retained submission counts:
7878, 4453, 4986, 3641, 23652, 37077, 21669, and 3960. Each uses the same
model, prior, full horizon, and 75% early-window rule as the primary batch.
The endpoint correction for seed 20261009/contest 3960 is stored separately
in `chapter5_time_seed_20261009_endpoint`. Refinements for contest 23652 use
dispersed starts informed by the initial multi-start fits. Region-specific
bridge checks (`pipeline.empirical.mode_mass`) retained all Jacobians and
used disjoint posterior halves; three independently drawn Gaussian proposals
gave negligible mass to the low-volatility local mode for both time seeds.
These checks are numerical diagnostics, not global mode-exhaustiveness proofs.
`pipeline.validation.chapter5_time_imputation` verifies inputs and convergence
before comparing the two time seeds with the accepted primary estimates.
`pipeline.validation.rank_agreement` independently reproduces Table 4 from
the official jointly available ranks, excluding benchmark and unranked teams;
its 10,000 contest-level bootstrap resamples use seed 20261008.

Run `make chapter4` or `make figure3` for data generation only. Both use the
original hourly generator with the nonlinear HJB equilibrium policy m in
place of the first-order policy. Paired S0=2 and S0=4 runs each use their
own HJB solution. Seeds, hourly timing, thinning, held-readout recursion,
noise scaling, and all other parameters are unchanged. Spatial/time grid
refinement checks accompany all six policy solutions. Bayesian estimation
still uses the first-order approximation and is not run at this stage.
The long-horizon r values remain 18.6484375 and 47.4375; no recalibration is
performed. The first 10 of the same 20 independent contests form the smaller pool.

The previous continuous-signal/full-HJB experiment (`chapter4.py`,
`nonlinear.py`, `chapter4_checks.py`, `chapter4_report.py`, and
`results/synthetic/nonsteady_hjb/`) is retained for audit only. It is not a
variance-only comparison and is no longer run by the default targets.
The manuscript describes the optimal-policy hourly DGP and its regenerated
figure. Table 1 uses `python -m pipeline.synthetic.model_comparison`:
both models estimate r on the same complete nonsteady baseline, with S0
either free or fixed to sigma/sqrt(lambda). No data are regenerated or discarded.
Section 4.1 is estimated separately with `python -m pipeline.synthetic.richer_recovery`.
This estimator preserves original synthetic supports and priors, including
mu0 N(0,1) and log(r) N(log(15),0.35^2), with the documented free-S0 prior.
Hourly Gaussian innovations are checked against the original dense covariance
density; readouts shared by multiple submissions are included only once.

Initialization comparison uses normalized bridge sampling with disjoint
posterior halves for proposal fitting and evidence evaluation. All proper
priors include their truncation constants and all unconstraining Jacobians
are retained. S0 prior medians 1 and 4 (log-SD 0.75) supplement the predeclared
median-2 baseline. Multi-start likelihood-only optimization supplies auxiliary
BIC, with n=45 readouts and alternative counting conventions reported separately.
The hourly integration grid is not treated as a sample size. Use `--reuse` to
reuse hash-verified chains and recompute the comparison; the older fixed/free-r
results remain untouched.

Section 4.1 uses the saved nonlinear-policy, S0=4 data without regenerating or
trimming observations. The pools contain the first 10 and all 20 independent
90-day contests; filters and latent initial draws reset at each contest, and
the common-parameter prior enters once per pool. The two other fits use the
180- and 360-day samples. All fits impose S0=sigma/sqrt(lambda) and estimate
(c_i,c_j,sigma,lambda,mu0,r) jointly. The log-r prior has SD 0.35 and is centered
at log of the generating scale: 15, 18.6484375, or 47.4375. Other priors/supports
match Table 1. These are four recovery fits, not 20 separate paired model
comparisons or repeated-sampling bias/coverage experiments. `--reuse` reuses
chains only after verifying model, data and source-file hashes. The original
synthetic recovery runner is retained for historical reproduction.

### Signal-disagreement outlier sensitivity

Run `.venv/bin/python -m pipeline.validation.chapter5_robust` to compare the
full-path and time-split OLS diagnostics with all-contest Huber regressions.
The procedure fixes Huber's tuning constant at 1.345, estimates residual scale
using Huber's proposal 2 (d=2.5), and reports H3 asymptotic inference alongside
2,000 contest-pairs bootstrap percentile intervals (seeds 20261008 and 20261009).
No contests are deleted or winsorized, and the original OLS files and structural
estimates are unchanged. Outputs in `results/validation/main/` are
`chapter5_signal_robust_regressions.csv`, `chapter5_signal_robust_weights.csv`,
and `chapter5_signal_robust_bootstrap.csv`.

Huber downweights large residuals; it does not remove all predictor-leverage
effects. These are post-hoc sensitivity checks, not replacement confirmatory
tests. Both inference methods condition on structural point estimates and do
not propagate their posterior uncertainty. Report both intervals when they
disagree about exclusion of zero rather than selecting a favorable method.

Run `.venv/bin/python -m pipeline.validation.chapter5_influence` for a separate
one-pass Cook-distance sensitivity check. Each full-sample OLS regression flags
Cook D > 4/n, then refits once without those observations using HC1 covariance.
The threshold uses the original valid sample size; it is not recomputed through
repeated deletions. The two specifications can flag different contests.
`chapter5_signal_cook_regressions.csv` reports the baseline, screened estimates,
sample sizes, exclusions, and input hash; `chapter5_signal_cook_flags.csv` records
all contest-level diagnostics. Source data, primary estimates, paper, and main
regression files remain unchanged. This is a post-hoc influence sensitivity
check: flagged observations are not established data errors, and reported
p-values/intervals do not adjust for outcome-dependent screening or structural
parameter uncertainty.
