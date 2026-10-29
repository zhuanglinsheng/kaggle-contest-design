# Task 1: Manuscript–implementation consistency

**Manuscript:** PaperJK9  
**Checked:** 24 October 2026  
**Scope:** Matched left-point effort and drift, empirical initialization, and feedback/readout timing across the reported exercises.

## Response to Jussi

I have checked the manuscript against the implementation for these three points. The current wording is consistent with the implementation, including the differences between the empirical and simulation exercises.

1. **Matched left-point discretization.** In the empirical implementation, each interval's submission intensities and latent-state drift use the same interval-start effort values. The cumulative drift is the cumulative sum of the effort difference multiplied by the grid spacing. Thus, the implementation uses the matched left-point convention stated in Proposition 2 and the sequential-likelihood appendix.

2. **Data-dependent initialization.** The empirical procedure backfills the earliest jointly available public gap into the initialization window, centers the prior for the initial mean on the first public gap, and imposes the steady-state variance restriction. The backfilled hours initialize the recursion but contribute neither submission-count exposure nor validation outcomes. These operations are described in the manuscript. Proposition 2 holds the initial public input fixed, and the manuscript explicitly states that the empirical initialization procedure is outside its scope.

3. **Feedback and readout timing.** The empirical belief recursion uses the current interval-end public value after computing that interval's effort. The synthetic exercises use the public value available at the interval start in the belief recursion, matching their simulator. In both cases, fresh submission-triggered observations enter the Gaussian likelihood at interval endpoints. Thus, “interval-start feedback” in the simulations refers to the feedback used in the belief update; it does not mean that new endpoint observations are scored as start-of-interval measurements. The nonlinear-recovery appendix explicitly distinguishes these conventions and limits the interpretation of its recovery results accordingly.

On this basis, no additional main-text revision is needed for Task 1. This confirms the correspondence between the manuscript and implementation; it does not extend Proposition 2 to the empirical initialization or establish equivalence between the two feedback conventions.

## Implementation correspondence

Paths below are relative to the repository root. Manuscript line numbers refer to the reviewed version and may change with subsequent edits.

| Item | Implementation checked | Corresponding manuscript statement | Conclusion |
|---|---|---|---|
| Actual empirical estimation model | `pipeline/empirical/unified_prior_reestimate.py`: `MODEL` points to `results/validation/estimation_details/detail_model`; `fit_one` records the model and input signatures | Section “Bayesian Estimation Framework” and the empirical application | The check concerns the model used by the unified-prior re-estimation runner, rather than only the older generic estimation entry point. |
| Interval effort, submission intensity, and drift | `results/validation/estimation_details/detail_model.stan`: `fn_efforts` is evaluated at `(i - 1) * Delta2f`; `intensity_i = r * m_i * Delta2f`; `integrated_drift = cumulative_sum(m_i - m_j) * Delta2f` | `paper/PaperJK9.tex`, lines 655–663, 708, 1739, and 2585–2592 | The count means and drift use the same left-point effort arrays. |
| Endpoint observation likelihood | The empirical Stan model constructs retained observation indices from submission-containing hours and evaluates `hat_y[obs_idx[ii]]` against accumulated drift and Gaussian predictive uncertainty | `paper/PaperJK9.tex`, lines 665–678 and the sequential-likelihood appendix | Observations are retained at interval endpoints; carried-forward hours are recursion inputs, not additional fresh observations. |
| Empirical backfill and observation window | `pipeline/empirical/rebuild_main.py`, `rebuild`: determines `start_hour`, backfills `gaps[:start_hour]`, retains submissions after the count-window start, and stores initialization metadata; `pipeline/empirical/estimate.py`, `stan_data`, passes `count_start_idx` | `paper/PaperJK9.tex`, line 998 and the preprocessing description around line 2680 | Initialization and count-exposure windows are distinguished consistently. |
| Initial mean prior and variance restriction | `pipeline/empirical/unified_prior_reestimate.py`, `prior`: initial-mean prior centered on `data["hat_y"][0]`; empirical Stan model: `bar_S = sigma / sqrt(lambda)` and initial residual variance `bar_S` | `paper/PaperJK9.tex`, line 946 and its prior footnote | The empirical-Bayes prior and steady-state initial variance match the description. These are separate operations from backfilling. |
| Initialization excluded from validation outcomes | `pipeline/validation/chapter5.py`, `benchmark_row`: uses `path_start_index` for path outcomes and starts private-gap increments from the first retained readout | `paper/PaperJK9.tex`, lines 1120, 1163, and 1225–1255 | Artificial initialization values are not treated as additional validation outcomes. |
| Fixed-input identification scope | Proposition 2 and its proof fix the initial public input before comparing parameter-induced data distributions | `paper/PaperJK9.tex`, lines 705–712, 1740, and 2680 | The manuscript does not claim that this proof covers data-dependent empirical initialization. |
| Synthetic initialization comparison | `pipeline/synthetic/model_comparison.py` uses `synthetic_data/hourly_comparison.stan`; the likelihood uses `public_gap[n]` in the belief update and `public_gap[n+1]` for the new observation | Synthetic initialization comparison; `paper/PaperJK9.tex`, lines 780–843 | Interval-start feedback and endpoint observation likelihood are distinguished in the implementation. Fixed and free initial-variance specifications are compared on the same sample. |
| Pooled and longer-horizon simulations | `pipeline/synthetic/richer_recovery.py` uses `synthetic_data/hourly_steady_pooled.stan`; generation is traced through `pipeline/synthetic/optimal_generate.py` to `synthetic_data/synthetic_data.py` | Pooled and longer-horizon synthetic exercises; `paper/PaperJK9.tex`, lines 851–933 | Generation and estimation retain the simulator's interval-start feedback convention. The stated generating initial variance and imposed estimation restriction are intentionally different. |
| Nonlinear parameter-recovery exercise | `pipeline/empirical/sigma_lambda_trials.py` and `sigma_lambda_replication.py` store `hat_y = public[1:]` separately from `feedback_gap = public[:-1]`; `unified_prior_interval_report.py` aggregates replicates 2–5 and records the timing distinction | `paper/appendix_effort_recovery_jk9.tex`, especially its final paragraph | The recovery likelihood uses simulator-aligned interval-start feedback. The reported 40-dataset recovery results are not presented as validation of empirical endpoint feedback and preprocessing. |

## The timing distinction in equations

For interval \((s_{\ell-1},s_\ell]\), let \(u_{\ell-1}\) denote the perceived state used to choose effort. The empirical implementation computes

\[
m_{k,\ell}=m_k^{(1)}(u_{\ell-1},S_{s_{\ell-1}},s_{\ell-1}),\qquad
\mathbb E[n^k_\ell\mid\text{past}]=r m_{k,\ell}\Delta,
\]

and uses the same effort difference in the drift increment:

\[
d_\ell=(m_{i,\ell}-m_{j,\ell})\Delta.
\]

Writing \(H_\ell\) for the held public value at endpoint \(s_\ell\), the empirical belief recursion uses

\[
u_\ell=u_{\ell-1}+d_\ell+w_\ell(H_\ell-u_{\ell-1}).
\]

The simulator-aligned recursion instead uses \(H_{\ell-1}\) in this feedback term. The simulator generates a fresh noisy observation at the endpoint if the interval contains submissions, and the likelihood evaluates that observation at the endpoint. Therefore, the feedback index differs while the use of endpoint observations in the Gaussian likelihood remains shared. Here \(w_\ell\) denotes the implemented feedback weight, not an incentive index.

## Verification record and boundaries

The checks above combine direct source inspection with the completed JK9 numerical audit:

- All 73 frozen empirical-input hashes match the recorded estimation inputs.
- Rebuilding the six empirical diagnostic measures from those inputs and the reported posterior medians reproduces the saved values up to floating-point error.
- The 80 saved-result checks pass, including table entries, regression statistics, design summaries, and simulation-recovery summaries.

Reproducible scripts and outputs are in this directory:

- `input_diagnostics_audit_20261024.py` and `.json`;
- `numerical_audit_20261024.py` and `.json`.

These numerical checks support input/result traceability; the discretization and timing conclusions above follow from the specific source operations identified in the table. This review does not rerun posterior sampling or the full nonlinear simulation/design exercises, and does not claim to validate the maintained measurement assumptions. No manuscript, estimation code, data, or reported result was changed in completing this response.
