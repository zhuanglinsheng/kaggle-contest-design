# Results map

The active version is selected by `../current_release.toml`. Its file hashes
and portable fit index are under `releases/paperjk7-review-20261016/`.

| Path | Role |
| --- | --- |
| `reestimation/lambda100_sd225/estimation/` | Current 146 full/early posterior summaries |
| `reestimation/lambda100_sd225/auxiliary/` | Current 251 robustness and time-imputation fits |
| `reestimation/lambda100_sd225/validation/` | Empirical diagnostics and nonlinear-generated recovery results |
| `reestimation/lambda100_sd225/counterfactual/` | Precision-only benchmark, with observed prize and duration |
| `reestimation/lambda100_sd225/fixed_initial_design/` | Current fixed-initial-distribution, asymmetric first-order design tables; no-feedback limits distinguished |
| `reestimation/lambda100_sd225/design_sensitivity/` | Historical steady-state joint-design scenarios, superseded for the current manuscript |
| `synthetic/optimal_hourly/`, `synthetic/initialization_comparison/`, `synthetic/richer_steady/` | Retained Chapter 4 evidence |
| `theory/numerical_joint_plane_moderate_dense/three_class_comparison/` | Retained theoretical regime illustration |
| `validation/sigma_lambda_trials/` | Recovery development and source simulations; includes active dependencies |
| `validation/main/chapter5_rank_agreement.*` | Retained data-only leaderboard agreement |
| `estimation/main/`, `counterfactual/main/`, `robustness/chapter6/` | Previous empirical baseline and retained dependencies |
| `release_manifest.json` | Historical publication lock; unchanged by current release commands |
| `manuscript_reviews/` | Dated manuscript checks and snapshots; records describe their own revision |

A directory's age or name does not establish whether it can be deleted. Check
`../docs/PAPER_EVIDENCE.csv` and the accepted-fit index first. Original accepted
run records remain immutable, including absolute paths recorded at run time;
the new fit index provides relative paths without rewriting those records.

Compact publication metadata is eligible for Git. Large chains, posterior
arrays, logs and executables remain local or in a separate artifact bundle.
See `../docs/REPRODUCIBILITY.md` for verification scope and reproduction limits.
