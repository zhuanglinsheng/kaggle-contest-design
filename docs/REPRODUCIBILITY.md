# PaperJK9 reproducibility

The active release is `results/releases/paperjk9-20261029/`, selected by
`current_release.toml`. The JK7 release is historical provenance.

All intermediate outputs now live below `tmp/`. See `STORAGE_LAYOUT.md` for the
role classification and `storage_layout.json` for old-to-new paths. Current
formal evidence, frozen inputs and method/source records stay outside `tmp/`.

## Maintenance acceptance criterion

The requested deliverable is a reproducible repository, not a new full scientific
run. Check that model/source code, frozen inputs, configurations, random seeds,
dependencies, stage order and output contracts are available; use dependency,
build and small regression checks to detect broken links. A full estimation,
search or integration run is optional and is not a maintenance acceptance gate.
"Not rerun during maintenance" records validation scope; it is not itself a
reproducibility defect.

Keep two purposes distinct: compact final evidence reproduces the published
tables, while the declared input-to-estimation-to-analysis commands support
recomputing the scientific quantities. Historical acceptance certificates help
trace the exact old run. Their absence limits historical provenance, but does
not require running a new fit now or make them inputs to a fresh fit. A missing
source, indispensable input, configuration or executable stage is a real gap
and must remain explicitly identified. Specialized recovered tail-repair source
is preserved separately from the portable ordinary-hour integration entry point;
preserving that source does not imply all historical repair controllers have
already been migrated into a portable command.

## Distinct verification levels

| Operation | Command | What it establishes |
| --- | --- | --- |
| Check compact source/input/evidence snapshot | `make verify-release` | Selected bytes and figure-source agreement; not scientific correctness |
| Reaggregate both design tables | `make reproduce-derived` | Complete 73-case main and appendix results reproduce manuscript numerical rows; no point caches/chains required |
| Check local JK9 table posterior bundle | `make verify-chains` | Hashes for 73 accepted records and 292 chains, using portable paths |
| Compile manuscript | `make paper` | LaTeX build only |
| Check ordinary tests | `make test` | Numerical/implementation regressions with deterministic fixtures |
| Compile independent Stan parity test | `JK9_STAN_INTEGRATION=1 python -m unittest tests.test_paired_prior_targets -v` | Optional compiler integration; no deleted empirical fixtures needed |

`PYTHON` can be specified for Make, e.g. `make PYTHON=.venv/bin/python test`.
Python 3.12+ is required by the data-source syntax. Start with `requirements.txt`;
`requirements-lock.txt` is the older full development environment, not a
portable operating-system lock. CmdStan 2.36.0 and TeX Live are external tools.
For Python 3.14, use `requirements-py314.txt`: these are the versions actually
used for the maintenance validation, separately from the historical environment.
The older pinned NumPy/SciPy releases may lack Python 3.14 binary wheels.

## From compact inputs to fresh estimates

All 397 historical main/auxiliary input references are in the retained portable
JK7 input index; the frozen input bytes are retained in Git. A new runner avoids
using deleted old chains merely to initialize a fresh fit:

```sh
python -m pipeline.jk9_fit --scenario main --mode full --all --dry-run
python -m pipeline.jk9 prepare-inputs
python -m pipeline.jk9_fit --scenario main --mode full --contest 2445
# Explicit full sampling, potentially expensive:
python -m pipeline.jk9_fit --scenario main --mode full --all
python -m pipeline.jk9_fit --scenario main --mode early --all
```

Use `--scenario pair`, `L1`, `L2`, `seed_20261008` or `seed_20261009` for the
corresponding retained inputs. Fresh fits go to `tmp/runs/`,
never over the publication posteriors. They use the byte-preserved model source
`pipeline/models/leading_order.stan`, the original prior and diagnostic gates,
and fixed dispersed initializations instead of depending on historical chains.
Failed attempts remain failures. This reproduces the statistical target; it
cannot reproduce old chain bytes or exactly reproduce a search driven by old
finite Monte Carlo samples. The full 397-fit rerun has not been executed by
this repository maintenance.

`make jk9-models` compiles retained leading-order, common-cost and recovered free-kernel Stan sources.
Platform-specific binaries and their hashes are not portable scientific inputs.
Historical audit records retain their original executable hashes; they must
not be rewritten to claim a new binary is the historical one.

## Nonlinear design calculations

`pipeline.empirical.nonlinear_design` is the maintained solver/search API:

```sh
python -m pipeline.empirical.nonlinear_design --help
```

The precise publication search, prize refinement, fixed-precision comparison
and sensitivity protocols are preserved in
`docs/review/PaperJK9/nonlinear_table_rebuild_20261025/` and
`docs/review/PaperJK9/task3_nonlinear_prior_sensitivity_20261026/`.
Their historical README stage logs are not current process state. Use the
portable adapter to resolve the original chain index and redirect controller
outputs, builds and SQLite caches to an isolated working tree:

```sh
python -m pipeline.jk9_design plan --contest 2445
python -m pipeline.jk9_design quadrature --contest 2445
python -m pipeline.jk9_design all --contest 2445
# Or run screen, refine, assemble, fixed individually; --all selects 73 contests.
# For fresh fits: --accepted-root tmp/runs
```

The default consumes the separately retained 73-posterior bundle; `--accepted`
accepts one fresh **full-window** fit. `--accepted-root` selects fresh main/full
fits for every case. Input and chain hashes are checked before computation.
Runtime records are ephemeral path adapters, not edits to historical accepted
records; rebuild them after moving an output tree. Controllers retain unknown
comparators and numerical failure flags. A finished command does not imply
`publication_ready=true`. The old 6322 repair certificate is never transplanted
into a new run. A full 73-case search was not run during this maintenance.

For output comparison, `python -m pipeline.jk9 tables` validates complete
coverage, reaggregates main and fixed-precision results, preserves unresolved
6322 in median bounds, and checks the actual TeX numerical rows. It never
executes manuscript integration patches or writes manuscript prose.

## Theory figure

```sh
python -m pipeline.figures.joint_design
```

This reconstructs the four-category display from retained regime and unresolved
CSV files; output is under `tmp/runs/theory_figure/`. It does not
rerun the 14,553-point HJB search or resolve failed candidates. The former
`archive.experiments.exact_joint_design` dependency is now preserved in
`pipeline.theory.dimensionless`; active theory imports use that module.

## What stays and what is disposable

Required evidence: source/models, frozen inputs, configurations/seeds, accepted
posterior index and diagnostics, final per-contest selections, failed/unresolved
candidate summaries and paper outputs. Expensive posterior draws are generated intermediate artifacts under `tmp/`,
with their provenance retained in the formal index. Raw Meta-Kaggle data must be retained by dated
snapshot and checksum; a changing live download cannot replace the historical ZIP.

Point evaluations are memoization. Their keys and payloads include the numerical
signature; failed solves remain failures in every storage mode. Default storage
is SQLite; set `JK9_CACHE_MODE=memory` for bounded in-process reuse, or `off`
for no point persistence. Legacy JSON files remain readable. Final scientific
records are never redirected into disposable cache storage.

An initial lossless compaction was stopped after the user prioritized code and
reproduction structure. Completed archives under `tmp/cache_archives/` have
independent content fingerprints and checksums; the interrupted `.partial`
archive does not authorize deletion. No further compaction is automatic.
A completed single-case archive can be restored with
`python -m pipeline.cache_store restore --contest 2445`.

## Recovered prediction comparison

Archaeology of Git tree/blob objects recovered 99 original text sources
(693,976 bytes), including the exact free-kernel Stan model, A/B quadrature,
C/D scoring, full-path C++ evaluator, independent importance integration and
specialized tail-repair methods. The sources are retained under
`docs/review/PaperJK9/recovered_prediction_sources_20261029/`; the compact
`evidence/prediction/source_recovery_manifest.json` records original Git blobs,
SHA-256 digests and source trees. Source verification needs no dangling Git
objects in a new clone.

The recovered hourly scores remain local artifacts. Git needs only 584
contest/model/fold score summaries and 146 frozen split inventories. Every
original hourly count was checked against the frozen input. The exact original
500,000-replicate paired contest bootstrap reproduces all reported means and
intervals, including the complete-date D-B endpoint near zero.

```sh
python -m pipeline.jk9_prediction summary
python -m pipeline.jk9_prediction plan --contest 2445
python -m pipeline.jk9_prediction baselines  # all 73 contests, both splits
python -m pipeline.jk9_prediction fit --model C --contest 2445 --split primary_hour
python -m pipeline.jk9_prediction fit --model D --contest 2445 --split complete_day
python -m pipeline.jk9_prediction score --model C --contest 2445 --split primary_hour \
  --accepted tmp/runs/prediction/C/primary_hour/2445/accepted.json
python -m pipeline.jk9_prediction integrate --model C --contest 2445 --split primary_hour \
  --accepted tmp/runs/prediction/C/primary_hour/2445/accepted.json
```

Fresh fitting uses only training observations and retains the full constructed
deadline. Four chains and the original parameter gates apply, including a/log(b)/b
for C. Prediction precedes that hour's readout update. Scoring averages the joint
probability over the common posterior. Rare-hour ESS/log-probability flags,
32-block MCSE and half-chain checks retain their original thresholds.
Independent integration freezes the recovered 16-cluster defensive Student
proposal using training draws and requires **two consecutive independent passing
budgets**, denominator/moment/path checks and all ordinary-hour probability gates.
It writes production blocks and uncertainty flags locally. A failed integration
exits unsuccessfully; it never becomes an accepted score. Specialized historical
tail-repair controllers are preserved as original source; their old certificates
are not automatically reused for a new posterior. For a fresh independent retry,
use a different `--output` and `--seed-offset`.

Actual validation: all 146 A/B case-splits passed fresh quadrature plus independent
adaptive checks, with maximum difference from original contest mean scores
1.35e-9. Run `python -m pipeline.prediction.parity` for the independent target check.
The recovered C/D training densities matched independent CmdStan at
three parameter points each (difference variation below 3.2e-13); native forecast
paths matched independent Python for both kernels. These checks do not certify
historical C/D Monte Carlo integration. A complete fresh 292-model/fold fit and
integration rerun has not been executed. Historical acceptance identities and integration certificates remain unverified;
source and chain recovery do not establish numerical acceptance.


Further recursive tree inspection recovered 1,400 unique text/data blobs
(1,256,958,899 bytes) into the ignored local bundle
`tmp/recovered/prediction_bundle/`. The compact
`artifact_recovery_manifest.json` records SHA-1/SHA-256 identities and paths
relative to that bundle. `recovered_chain_coverage.json` locates complete
four-chain **candidate** sets for all 292 C/D model/fold/contest targets:
five contests have identical primary/date cutoffs and reuse that training
prefix. Candidate sets include unsuccessful attempts and are not automatically
accepted. The old acceptance JSON and full importance production certificates
were not found in these trees. Preserve this separate bundle before Git object
garbage collection; the bundle has not been uploaded. `python -m
pipeline.git_recovery` can repeat recovery while the original objects exist.
A fresh clone can use the retained compact evidence and new fitting commands;
recovered historical draws require the separate bundle. No Git history rewrite
or new tracking of those 1.26 GB of blobs was performed.
