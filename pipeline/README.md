# Pipeline roles and current publication

`../current_release.toml` selects the current empirical revision under
`results/reestimation/lambda100_sd225/`. It has 397 accepted fits; the main
publication and auxiliary results use posterior medians and the unified lambda
prior. Current publication checks use `pipeline.publication`, via
`make verify-release` and `make verify-chains`.

See `../docs/REPRODUCIBILITY.md` for the current stage map and limitations, and
`../docs/PAPER_EVIDENCE.csv` for figure/table provenance. Numerical samplers and
source-hashed protocols remain in their original locations.

| Role | Modules | Handling |
| --- | --- | --- |
| Current estimation and reporting | `unified_prior_medians`, `unified_prior_outputs`, `unified_prior_aux_summary`, `unified_prior_verify` | Read frozen protocols; expensive or report-writing runners |
| Current design comparisons | `fixed_initial_design`, `fixed_initial_manuscript` | Fixed initial distributions; asymmetric first-order output; both design tables and the full-disclosure comparator; no-feedback limits labeled separately |
| Historical joint-design frontier | `design_duration_sensitivity` | Superseded steady-state output map; preserve original artifacts |
| Historical manuscript integration | `unified_prior_manuscript`, `update_joint_design_manuscript` | Preserve as dated revision tools; do not replay over current prose |
| Prior/nonlinear/recovery exploration | `nonlinear_*`, `sigma_lambda_*`, `paired_prior_*`, `activity_*` and related trials | Retain: some outputs are active recovery or accepted-fit dependencies |
| Publication maintenance | `publication` | Standard-library checks; no sampler imports |
| Previous empirical baseline | `empirical.estimate`, `empirical.finalize`, `validation.chapter5*`, `robustness.chapter6*`, `release` | Keep original paths; historical Make targets require `LEGACY=1` |

## Historical workflow reference

The following stage table and experiment notes are retained for provenance.
References to the original main directories and old release manifest describe
the previous empirical baseline. They do not override `current_release.toml`.

All modules are invoked from the repository root with `python -m`. Shared paths
and the common USD 100,000 prize unit are defined in `pipeline/config.py`.
Run `make models` after changing any Stan source file.

The historical release was checked by `pipeline.release` and recorded in
`results/release_manifest.json`. The table below lists its entry points. Later paragraphs retain detailed provenance for the
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

### Nonlinear-policy reestimation of the ten largest incentive indices

`pipeline.empirical.nonlinear_reestimate` provides a paired estimator outside
the publication release. `prepare` selects ten contests by the accepted full
posterior median of `max(w_i,w_j)`, preserving joint draws and breaking ties by
contest ID. Data, observation likelihood, priors, prize units, and steady-state
initialization match the main estimator. Only effort is replaced by a numerical
solution of the coupled nonlinear HJB equations.

The active implicit solver is `--backend fast`; its outputs are isolated in
`results/estimation/nonlinear_top10/fast_v1/`. Earlier numerical audits and
proposal pilots remain in the parent directory and `implicit_v1/`.

```sh
.venv/bin/python -m pipeline.empirical.nonlinear_reestimate prepare --backend fast
.venv/bin/python -m pipeline.empirical.nonlinear_reestimate explore --backend fast --contest ID --dy .05 --nt 256
.venv/bin/python -m pipeline.empirical.nonlinear_reestimate calibrate --backend fast --contest ID --dy .0125 --nt 512
.venv/bin/python -m pipeline.empirical.nonlinear_reestimate sample --backend fast --contest ID --chain N --warmup 400 --samples 2400 --dy .0125 --nt 256 --pilot
.venv/bin/python -m pipeline.empirical.nonlinear_reestimate posterior-grid --backend fast --contest ID
.venv/bin/python -m pipeline.empirical.nonlinear_reestimate baseline-target --backend fast
.venv/bin/python -m pipeline.empirical.nonlinear_reestimate summarize --backend fast
.venv/bin/python -m pipeline.empirical.nonlinear_report
```

For a batch with completed pilots, run `python -m pipeline.empirical.nonlinear_validation_batch refine` to resume sampling improvement, or `python -m pipeline.empirical.nonlinear_validation_batch validate --workers 2` to resume posterior grid checks. Run one driver per stage; validation caches are protected by
contest-level file locks. `posterior-grid --grid-level 1 --grid-workers 4`
refines both grids again, reuses already computed reference likelihoods,
archives the previous level and parallelizes independent solves. A default
resume retains the highest established grid level. Exploration/calibration outputs are proposal aids,
not posterior estimates or evidence of a likelihood optimum.

Sampling uses four independently seeded chains, a fixed Student-t independence
proposal and random-walk moves. Delayed acceptance uses a Gaussian screen,
or `sample-prior` uses the original prior as its screening density when a
Gaussian screen explores posterior tails too slowly. Both second acceptance
steps retain the same full finite-grid target. For a curved posterior tail, `learn-mixture --proposal-generation 3` archives a completed pilot and constructs two fixed Student-t proposal components. `sample-mixture` uses their normalized mixture density in both acceptance stages, with the unchanged prior as its screen. If the prior screen impedes transitions between those regions, `sample-mixture-gaussian` uses a frozen Gaussian mixture fitted to the same completed pilot as its screen; its second stage still uses the full original posterior. Component weights are computational proposal choices, not posterior probabilities or revised priors. Proposal adaptation ends at warmup. `learn-proposal` archives
a completed pilot and learns a new fixed proposal; `--proposal-generation 2`
retains a second completed pilot in a separate archive; `extend` preserves the final
state and random-number generator. Numerical solver failures abort rather than
silently changing posterior support. Checkpoints preserve data, chain, solver,
proposal and sampler provenance.

Independent checks compare the observation likelihood with its dense Gaussian
law and CmdStan, equilibrium policies with a separate value solver, and the
actual sampler and importance correction with known distributions.

`--pilot` permits explicitly provisional finite-grid sampling. A single-point
audit is insufficient to approve estimates. `posterior-grid` requires
rank-normalized folded R-hat below 1.01 and bulk/tail ESS at least 400 for six
parameters and three incentive indices. It evaluates a finer likelihood on
1,024 balanced posterior draws, corrects the posterior by importance weights,
and checks another refined grid on representative and extreme draws. Importance
weight ESS and autocorrelation-adjusted mean ESS must both reach 400; changes
between the refined posterior summaries must remain below 0.05 posterior SD.
Centered posterior log-density errors also have explicit bounds. Domain expansion
from seven to nine standard-deviation units is checked at representative and
parameter-extreme posterior points.

`parameter_comparison.csv` contains only validated contests; provisional
summaries have a separate filename. The final report refuses to run until all
ten contests pass. Its parameter comparison reports medians, 95% intervals,
and shifts in baseline posterior SD. True parameters are unknown: small model
differences, if observed, would not establish small errors relative to truth.

`precision_parameter_sensitivity.csv` uses the original Table 7 steady-state,
leading-order design formula with the validated nonlinear parameter posterior.
This isolates parameter sensitivity; it does not evaluate the nonlinear design
objective. The publication estimates, full 73-contest analysis and Table 7 are
not overwritten by this workflow.

`effort_approximation_summary.csv` compares leading-order and nonlinear policies
at the same joint representative draw from each validated nonlinear posterior.
The common states are normalized gaps -2, -1, 0, 1, 2, at time fractions
0, 0.25, 0.5, 0.75, 0.9. A separate grid check keeps numerical differences below
0.5% of the largest nonlinear effort on those checked states. The reported
approximation difference is also scaled by that checked peak; it is not a
relative error in realized effort or expected output.

The completed ten-contest comparison is recorded in
`results/estimation/nonlinear_top10/fast_v1/completion.json`. Approved outputs
include `comparison_report.txt`, `parameter_comparison.csv`,
`w_comparison.csv` / `.png`, `effort_approximation_summary.csv`,
`precision_parameter_sensitivity.csv`, and `posterior_audit_summary.csv`.
All ten posterior audits passed. The selected incentive-index medians declined
from 11.01–53.82 to 1.43–3.05, but 40 of 60 primitive parameter means shifted
by more than one baseline posterior SD. Representative-policy differences
range from 6.4% to 51.4% of the checked nonlinear peak. Thus these results
do not establish negligible approximation error. Under the original design
formula with replaced parameters, the two previously finite recommendations
become full disclosure; this remains parameter sensitivity, not a nonlinear
design counterfactual.

## Bayesian detail audit (three contests)

`results/validation/estimation_details/` contains a separate audit of contests
4699, 4657, and 5390. The effort formula is unchanged. Data-driven prior
hyperparameters are introduced only in a copied model; the canonical empirical
model, publication posteriors, and Table 7 are not replaced.

The audit isolates prior strength, initialization, monetary-unit conversion,
feedback timing, and numerical intensity floors. A paired hourly recovery
control uses empirical-scale parameters and both leading and nonlinear data
generation. In the leading-policy control, the true maximum incentive index is
2.101; its estimated median is 8.237 under the original empirical priors and
4.462 when only the log-SD of r is increased from 0.5 to 1.0. This is one
realization, not a repeated-sample coverage study.

The wider cost/r prior for 4699 is a sensitivity configuration. Its initial fit
had 13 divergences and is excluded from final conclusions. The same target was
rerun with 2,000 warmup iterations, 3,000 retained draws per chain, and
adapt_delta=0.99; the final median maximum w is 5.386 with no divergences.
These changed-prior results cannot establish agreement with the old-prior
nonlinear estimates. The estimated-effort reference comparison likewise uses
different fitted parameters and priors, unlike the same-parameter policy check
in the ten-contest nonlinear report.

Regenerate the consolidated report and final manifest from retained outputs:

```sh
.venv/bin/python -m pipeline.empirical.estimation_detail_report
```

Replay the validated wider-scale-prior fit (saved configuration and starts are
checked before reusing existing chains):

```sh
.venv/bin/python -m pipeline.empirical.estimation_detail_audit --stage run --contests 4699 --variants scale_weak --warmup 2000 --samples 3000 --adapt-delta .99 --run-label delta99
```

`detail_audit_report.txt` explains the findings and limits in Chinese.
`completion.json` identifies the final fits, excluded pilot, source hashes,
and retained artifacts. Final fits require rank R-hat <= 1.01, ESS >= 400,
zero divergences and tree-depth hits, and BFMI >= 0.3.

## Common-prior improvement of both estimators

`results/validation/paired_prior_improvement/` keeps the same widened priors
on both sides of the comparison. The `r_weak` candidate changes only the log-SD
of r to 1.0; `scale_weak` also widens both cost log-SDs to 1.25. Prior means,
innovation priors, monetary units and observed data remain fixed. Existing
nonlinear posteriors have insufficient overlap under these candidates and
are used only to initialize fresh sampling, not as final reweighted estimates.

```sh
.venv/bin/python -m pipeline.empirical.paired_prior_improvement --stage screen
.venv/bin/python -m pipeline.empirical.paired_prior_improvement --stage leading --profile scale_weak --contests 4657 5390
.venv/bin/python -m pipeline.empirical.paired_prior_improvement --stage nonlinear --profile scale_weak
.venv/bin/python -m pipeline.empirical.paired_prior_report
```

The retained, validated 4699 `scale_weak_delta99` fit supplies the identical
leading target. A stopped duplicate fit is preserved but excluded. Numerical
grid workers explicitly receive the new prior when macOS launches fresh
processes. Fresh nonlinear chains are usable only after the existing posterior
grid audit passes; optimizer modes and pilot chains are not final estimates.

`activity_prior_design.json` declares an additional common scale sensitivity:
log r has center log 80 and SD 1.5, with the wider cost priors. Its broad center
uses public progress in the other 63 contests as a noisy scale reference,
not as an estimate of r or a fitted hierarchical prior. Importance estimates
must pass both weight ESS and autocorrelation-adjusted mean ESS thresholds;
low-overlap results require fresh sampling.

```sh
.venv/bin/python -m pipeline.empirical.paired_activity_sensitivity
.venv/bin/python -m pipeline.empirical.activity_prior_control
.venv/bin/python -m pipeline.empirical.paired_prior_recovery --profile scale_weak
.venv/bin/python -m unittest tests.test_paired_prior_targets -v
```

The recovery kernel adds only a separate interval-start feedback input,
retaining endpoint score observations and the audited nonlinear policy.
Target parity is checked against the copied Stan model. Recovery on a single
realization, conditional submission diagnostics and common-prior parameter
agreement answer different questions; none alone validates the nonlinear
design counterfactual or replaces Table 7.

The initial single-proposal nonlinear pilots did not pass chain diagnostics.
The final retained fits use frozen pilot-informed mixtures under the same
target (`paired_prior_mixture`); pilot folders remain separately archived.
The leading 5390 fit likewise uses `leading_prior_mixture`, with density
parity against Stan established by `fast_leading_prior`. The new leading
sampler was independently checked against the accepted 4657 NUTS fit.

The common activity-scale prior has two fresh, matched recovery fits on the
retained nonlinear-generated data (`leading_on_nonlinear_control` and
`activity_recovery_mixture`), as well as a leading-generated control
(`fast_leading_prior --kind recovery`). These do not imply that recovery on
nonlinear data is satisfactory for the leading approximation: its constant
effort ratio remains a structural limitation. Activity-scale importance
updates of the three empirical nonlinear fits failed effective-sample gates
and are excluded, rather than treated as valid estimates.

Regenerate the final Chinese report and source/artifact manifest:

```sh
.venv/bin/python -m pipeline.empirical.paired_prior_final_report
```

`final_report.txt` distinguishes successful scale recovery, numerical
validation, residual approximation differences, and the remaining design
validation. `validated_empirical_fits.json` and
`validated_recovery_comparison.json` check all reported combinations and
joint cost ratios. The original publication estimates and Table 7 remain
unchanged.

## Separate common and relative cost priors

`cost_prior_trial` tests prespecified Gaussian priors on common log cost
and the difference between log costs, shared by both estimators. It retains
the original prior uncertainty about relative costs while separately checking
wider common scale and wider innovation/precision priors. Source posteriors
must be validated; importance results require sufficient dependence-adjusted
effective samples and a fresh posterior-weighted numerical grid check.
The 5390 leading fit for the first candidate uses fresh direct sampling because
reweighting did not meet the effective-sample threshold.

```sh
.venv/bin/python -m pipeline.empirical.cost_prior_trial
.venv/bin/python -m pipeline.empirical.cost_prior_fresh_leading
.venv/bin/python -m pipeline.empirical.cost_prior_trial_report
.venv/bin/python -m unittest tests.test_cost_prior_trial -v
```

`cost_structure_trial/final_report.txt` documents the modest, mixed recovery
improvement and remaining empirical disagreement. These trials do not select
a publication prior from one retained simulation realization.

## Sigma and lambda focused trials

`results/validation/sigma_lambda_trials/` isolates the ten original high-w
contests. `sigma_lambda_trials` tests prior and count-likelihood sensitivities
without changing the leading effort function. Synthetic truth is a joint
representative of the validated nonlinear reference; two exploratory and two
confirmation realizations per configuration retain fixed seeds. Empirical
references are explicitly distinguished from known synthetic truth.

The practical lambda candidate changes its log prior from mean0/SD0.75 to
mean log100/SD2.25. A separate 12-point sigma-prior grid is selected on the
exploratory realizations and checked on a new realization. Count weighting
is a generalized-Bayes sensitivity, not a prior-only modification. A later
posterior-based sigma calibration is reported as retrospective and is not
presented as a fresh blinded validation.

```sh
.venv/bin/python -m pipeline.empirical.sigma_lambda_trials --stage scan
.venv/bin/python -m pipeline.empirical.sigma_lambda_posterior --dataset recovery
.venv/bin/python -m pipeline.empirical.sigma_lambda_posterior --dataset empirical --variants lambda_external activity_lambda_external
.venv/bin/python -m pipeline.empirical.sigma_prior_grid --stage train
.venv/bin/python -m pipeline.empirical.sigma_prior_grid --stage posterior
.venv/bin/python -m pipeline.empirical.sigma_posterior_calibration
.venv/bin/python -m pipeline.empirical.sigma_lambda_trial_report
.venv/bin/python -m unittest tests.test_sigma_lambda_trials -v
```

The saved input JSON files replay the exact retained realizations; generation
helpers never reroll existing inputs. Failed chains are separately archived
before fixed proposal mixtures are improved. All final tables require valid
four-chain diagnostics, including nuisance parameters and behavioral
combinations. `final_report.txt` documents both the lambda recovery improvement
and the lack of a stable sigma improvement; no publication prior or Table 7
replacement is made automatically.

## Sigma uncertainty calibration

`sigma_interval_calibration` preserves the lambda_external posterior and
parameter point estimates. It fits one multiplier to log-scale posterior
interval halfwidths using only the20 fits from exploratory realizations0/1.
The rule is frozen before generating the prospective realization4. Existing
realizations2/3 are retrospective checks. This produces an additional
simulation-calibrated uncertainty interval, explicitly distinct from the
original95% Bayesian credible interval; it does not reduce point errors or
establish population coverage from a targeted ten-configuration sample.

```sh
.venv/bin/python -m pipeline.empirical.sigma_interval_calibration --stage design
.venv/bin/python -m pipeline.empirical.sigma_interval_calibration --stage fit_train
.venv/bin/python -m pipeline.empirical.sigma_interval_calibration --stage train
.venv/bin/python -m pipeline.empirical.sigma_lambda_replication --replicate 4
.venv/bin/python -m pipeline.empirical.sigma_interval_calibration --stage fit_test
.venv/bin/python -m pipeline.empirical.sigma_interval_calibration --stage check
.venv/bin/python -m pipeline.empirical.sigma_interval_empirical
.venv/bin/python -m unittest tests.test_sigma_interval_calibration -v
```

The separate `sigma_interval_calibration/report.txt` reports both coverage and
interval widths. The empirical application reports bounds only, because real
parameter truths are unknown. No posterior files or paper tables are replaced.

## Additional lambda checks

`lambda_prior_refinement` screens12 lambda-only prior combinations using the20
training fits from realizations0/1, then freezes one choice before generating
realization5. The broader selected prior fails the new test and is explicitly
rejected in `lambda_refinement/decision.json`. Three failed importance updates
on older realizations have fresh posterior replacements. Mean-versus-median
results and Monte Carlo precision of long-tailed posterior means are reported
separately; the paper's current mean convention is not silently changed.

`lambda_interval_check` independently freezes a common interval coefficient
from0/1, retaining the previous lambda_external posterior. The new boundaries
are simulation-calibrated intervals and are explicitly distinct from original
95% posterior credible intervals.

```sh
.venv/bin/python -m pipeline.empirical.lambda_prior_refinement --stage select
.venv/bin/python -m pipeline.empirical.lambda_prior_refinement --stage retrospective
.venv/bin/python -m pipeline.empirical.sigma_lambda_replication --replicate 5
.venv/bin/python -m pipeline.empirical.lambda_interval_check --stage freeze
.venv/bin/python -m pipeline.empirical.lambda_prior_refinement --stage fit
.venv/bin/python -m pipeline.empirical.lambda_retrospective_refits
.venv/bin/python -m pipeline.empirical.lambda_interval_check --stage check
.venv/bin/python -m pipeline.empirical.lambda_refinement_report
```

The Chinese `lambda_refinement/report.txt` retains all comparisons, including
the prospective failure, the distinction between summary error and estimation
error, interval widths, and the absence of empirical truth. Publication
estimates and Table7 remain unchanged.
