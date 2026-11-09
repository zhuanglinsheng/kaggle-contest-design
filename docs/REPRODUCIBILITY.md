# PaperJK12 reproducibility

The active manuscript is `paper/PaperJK12.tex`, selected by `current_release.toml`.
The publication lock is `results/releases/paperjk12-20261109/manifest.json`.
Long-term scientific results and their provenance remain in
`results/releases/paperjk9-20261029/`.
The JK7 release is historical provenance.

The repository retains calculation code, frozen inputs, method records and
long-term maintained results. Generated intermediates such as posterior draws,
per-hour integration arrays, point caches and temporary run logs belong under
`tmp/` and need not be retained or committed to Git. Reproduction generates
these intermediates from the corresponding code and inputs. See `PAPER_EVIDENCE.csv` and
`RESULT_CODE_AUDIT.json` for result-to-source mappings, and `STORAGE_LAYOUT.md`
for file roles.

## Reproduction contract and working directory

Reproduction here means starting from the declared source inputs and generating
all intermediate and final outputs in dependency order. Existing posterior
chains, point caches and integration arrays are not prerequisites: the generating
stages produce them. Retained final results are the long-term reference outputs.

The maintenance acceptance criterion is an explicit, connected input/output
contract: input paths, schemas and units must match the next consumer's expected
paths, schemas and units. Commands, fixed settings and stage order are documented
below. Maintenance checks the source-to-result interfaces without repeating the full
scientific computation. A concrete interface issue is documented by identifying
the producing stage, consuming stage and mismatched field or path. Generated
file retention is a storage choice, separate from the result status.

Run every command below from the repository root in a POSIX shell (macOS/Linux).
Paths in commands are relative to that root. Keep the same activated environment
through all stages. Do not use the older generic Make estimation targets for the
current empirical prior: use the explicit `pipeline.jk9_fit` commands below.

### 1. Install the environment and external compilers

```sh
git clone https://github.com/zhuanglinsheng/kaggle-contest-design.git
cd kaggle-contest-design
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c 'from cmdstanpy import install_cmdstan; install_cmdstan(version="2.36.0", cores=2)'
python -c 'from cmdstanpy import cmdstan_path; print(cmdstan_path())'
c++ --version
latexmk -v
mkdir -p tmp/cache/python tmp/cache/matplotlib tmp/runs tmp/build
export PYTHONPYCACHEPREFIX="$PWD/tmp/cache/python"
export MPLCONFIGDIR="$PWD/tmp/cache/matplotlib"
export XDG_CACHE_HOME="$PWD/tmp/cache"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
```

`python3.12` selects Python 3.12; Python 3.13 is also supported by the stated
3.12+ syntax requirement. For Python 3.14 use `requirements-py314.txt` instead
of `requirements.txt`. `version="2.36.0"` fixes the Stan compiler version;
`cores=2` controls its build concurrency. A C++17 compiler and a TeX Live
installation providing `latexmk`, `pdflatex` and the manuscript packages must be
installed on the operating system. The three thread settings cap numerical
library concurrency; they do not change model parameters. `PWD` must be the
checkout root when setting the cache paths.

### 2. Identify source inputs and prepare their working paths

```sh
python -m pipeline.jk9 verify
python -m pipeline.jk9 prepare-inputs
python -m pipeline.jk9 compile-models
python -m pipeline.jk9_fit --scenario main --mode full --all --dry-run
```

The frozen scientific inputs are
`results/reestimation/lambda100_sd225/input_snapshots/<scenario>/contest_<id>_<hash>.json`.
The input index is `results/releases/paperjk7-review-20261016/accepted_fits.json`.
It contains 397 scenario/mode references to 308 distinct frozen input files.
`prepare-inputs` checks hashes and writes 73 main working files as
`tmp/inputs/main/contest_<id>.json`; it does not fit a model or choose new pairs.
`compile-models` stages and compiles the retained leading-order, common-cost and
free-kernel Stan sources below `tmp/build/`. `--dry-run` prints the selected
input identities without sampling.

The frozen JSONs are the declared inputs to the statistical analysis, not
posterior intermediates. Core fields are `theta`, `Delta2f`, `N_Delta`, `Ni`,
`Nj`, `hat_t_i`, `hat_t_j`, `hat_y` and `real_y`, with selection and observation
window metadata such as `count_start_hour`. Counts `Ni`/`Nj` correspond to the
event arrays; the time arrays use hour coordinates and `Delta2f` converts an hour
to days. The stored prize is in thousands of USD; `stan_data` converts it to
USD 100,000 units using `PRIZE_DIVISOR_FROM_THOUSANDS=100`. Do not rescale the
input a second time. `stan_data` defines the exact Stan fields, one-based event
indices, initialization window and full/early truncation.

For the data-only official-rank exercise, supply the original dated Meta-Kaggle
ZIP at `metakaggle/__rawdata__/meta-kaggle.zip`. Its recorded identity is in
`docs/storage_layout.json`; a current online ZIP is not a substitute for that
input snapshot. Its `Teams.csv` must contain `Id`, `CompetitionId`, `IsBenchmark`,
`PublicLeaderboardRank` and `PrivateLeaderboardRank`.

### 3. Generate the nonlinear synthetic data and Figure 3

```sh
python -m pipeline.synthetic.optimal_generate
```

Inputs: `synthetic_data/synthetic_data.py`,
`pipeline/synthetic/{nonlinear,generate,optimal_generate}.py` and their fixed
configuration constants. No empirical posterior is an input. The generator
solves at `dy=0.2`, `dt=1/48`, checks against `dy=0.1`, `dt=1/96`, then simulates
hourly (`dt=1/24`). Horizons are 90, 180 and 360 days; initial variances are 2
and 4. The 90-day baseline Brownian seed is 1000; pooled contest seeds are
`3000+10*i` for `i=0..19`; longer-horizon seeds are `5000+days`. Poisson, uniform
and initial-state seeds are respectively Brownian seed +1, +2 and +3.

Outputs: `tmp/work/results/synthetic/optimal_hourly/manifest.json`,
`{steady,nonsteady}/<name>_paths.npz`, policy grids and `figure3.png`.
Names are `baseline`, `pool_contest01` through `pool_contest20`, `days180` and
`days360`. Each path NPZ supplies the `public`, `effort_i`, `effort_j` and
`events_{i,j}_days` arrays consumed by `pipeline.synthetic.table1.load_data`.
The initial-state comparison and recovery stages below consume the `nonsteady`
files directly; do not substitute the steady files.

### 4. Fit the synthetic initialization and recovery experiments

```sh
python -m pipeline.synthetic.model_comparison --warmup 1000 --samples 2000
python -m pipeline.synthetic.richer_recovery --cases pool10 pool20 days180 days360 --warmup 1000 --samples 2000
python -m pipeline.synthetic.effort_comparison
```

`--warmup` and `--samples` are iterations **per chain**, with four chains.
`model_comparison` uses `synthetic_data/hourly_comparison.stan` and the baseline
NPZ from step 3. Configurations/seeds are `steady/9100`, `nonsteady/9200`,
`nonsteady_prior1/9300`, `nonsteady_prior4/9400`; the latter two change the S0
prior median to 1 and 4. Outputs are
`tmp/work/results/synthetic/initialization_comparison/<configuration>/summary.json`,
`run.json`, `data.json`, generated chains, and the parent `comparison.json` and
`design.json`. The summary contains parameter intervals and diagnostics; the
comparison contains bridge evidence and MLE/BIC comparisons.

`richer_recovery` uses `synthetic_data/hourly_steady_pooled.stan`. `--cases`
selects the named data combinations, not new simulations. `pool10` and `pool20`
consume the first 10 and 20 pooled contests; `days180` and `days360` consume the
matching long paths. The four fitting seeds are 10100, 10200, 10300 and 10400;
the respective submission-rate prior centers are 15, 15, 18.6484375 and 47.4375.
Outputs are `tmp/work/results/synthetic/richer_steady/<case>/{summary,run,data}.json`
and generated chains. `--reuse` on either fitter reuses matching generated chains
only after their recorded input/source checks; omit it for a from-scratch run.

`effort_comparison` takes its six primitive configurations from the source,
solves the nonlinear policies, computes the leading-order comparison and writes
`tmp/work/results/synthetic/effort_comparison/controlled_incentives/summary.json`
and `effort_policy_comparison.pdf`. Its only optional argument is `--plot-only`,
which redraws already-generated curves; omit it when starting from scratch.

### 5. Estimate all 397 empirical main and auxiliary posteriors

```sh
python -m pipeline.jk9_fit --scenario main --mode full --all --output tmp/runs
python -m pipeline.jk9_fit --scenario main --mode early --all --output tmp/runs
python -m pipeline.jk9_fit --scenario pair --mode full --all --output tmp/runs
python -m pipeline.jk9_fit --scenario L1 --mode full --all --output tmp/runs
python -m pipeline.jk9_fit --scenario L2 --mode full --all --output tmp/runs
python -m pipeline.jk9_fit --scenario seed_20261008 --mode full --all --output tmp/runs
python -m pipeline.jk9_fit --scenario seed_20261008 --mode early --all --output tmp/runs
python -m pipeline.jk9_fit --scenario seed_20261009 --mode full --all --output tmp/runs
python -m pipeline.jk9_fit --scenario seed_20261009 --mode early --all --output tmp/runs
```

`--scenario` selects frozen data: `main` is the focal pair, `pair` the alternative
pair, `L1`/`L2` the alternative score scales and `seed_*` the two frozen alternative
within-date imputations. `--mode full` uses the full estimation window;
`--mode early` uses the defined early window while keeping the constructed
contest deadline. `--all` selects every indexed contest for that scenario/mode.
Use `--contest 2445` instead of `--all` to select one case. `--output` is the
parent of the generated scenario trees; keep it identical across these commands.
Expected coverage is 146 main +219 pair/scale +32 imputation fits.

Inputs: the frozen input index and `pipeline/models/leading_order.stan`.
Priors are defined by `pipeline.empirical.unified_prior_reestimate.prior`:
log centers `(0,0,log(2),log(100),hat_y[0],log(10))`, scales
`(.75,.75,.5,2.25,1,.5)`, sigma floor `.01`. Four fixed dispersed starts use
cost factors `.7,.9,1.2,1.5`, sigma 2, lambda 100 and r 10. Sampling uses four
chains, two in parallel, dense metric, max tree depth 13 and 18 significant
figures. The seed is `20261029+contest_id+100000*sum(ord(c) for c in scenario)`.
Attempts use `(warmup,samples,adapt_delta)` of `(1000,2000,.98)`,
`(2000,4000,.99)`, `(4000,8000,.995)`. These are implemented settings, not
additional undocumented CLI arguments.

Outputs per fit: `tmp/runs/<scenario>/<mode>/<id>/configuration.json`,
`attempt<N>/data.json`, chain CSVs, `attempt<N>/diagnostics.json`, and
`accepted.json` after the code's diagnostic gates pass. The accepted schema
contains `summary`, `diagnostics`, `configuration`, four `chain_paths`, four
`chain_sha256`, and `attempt`. Chain paths are relative to `accepted.json`.
Parameter medians, intervals, means and Monte Carlo diagnostics are preserved.
The next stages consume this schema, not particular historical CSV filenames.
An interrupted or existing output is preserved; select a new `--output` for a
new complete attempt rather than overwriting an accepted record.

### 6. Connect accepted fits to empirical regressions, figures and appendix rows

```sh
python -m pipeline.reproduction_reports empirical --accepted-root tmp/runs --output tmp/runs/empirical_reports
```

`--accepted-root` must be the step-5 output containing all 397 accepted records.
`--output` is a separate generated reporting tree. The adapter validates target,
input and chain identities, creates `chain_1.csv` through `chain_4.csv` symlinks
in `adapters/<scenario>/<mode>/<id>/`, and writes the chain `data.json` beside
them. This makes the portable fit outputs satisfy the original report readers'
filename contract; it does not modify chain contents or the statistical model.

The reporting implementation is
`pipeline.empirical.unified_prior_outputs` plus
`pipeline.validation.{chapter5,chapter5_report,chapter5_influence,chapter5_final_audit}`
and `pipeline.empirical.unified_prior_aux_summary`. Main summary and source CSVs
are written under `estimation/`; regressions, contest diagnostics, Cook flags,
appendix rows and count-floor checks under `validation/`; main figures under
`paper_assets/`; alternative-pair figures, regressions, parameter correlations
and time-imputation comparisons under `auxiliary/reports/`. The appendix row
string is `validation/chapter5_paper_numbers.json["appendix_rows"]`.

To reproduce only the report layer from the long-term retained posterior
summaries, the separate command is:

```sh
python -m pipeline.reproduction_reports empirical --summary-root results/reestimation/lambda100_sd225 --output tmp/runs/summary_reports
```

That command regenerates main and auxiliary manuscript regressions and figures;
it does not replace steps 2–5 in a from-scratch computation or generate the
chain-level appendix/count-floor diagnostics. This alternative is useful for
checking the report interface without executing new posterior estimation.

### 7. Recompute nonlinear main and fixed-prize/duration design tables

```sh
python -m pipeline.jk9_design plan --all --output tmp/runs/design
python -m pipeline.jk9_design all --all --accepted-root tmp/runs --output tmp/runs/design
```

`all` executes `screen`, `refine`, `assemble`, `fixed`, in that order for each
contest. `--accepted-root` supplies the full-window leading-order fits from
step 5. `--output` contains generated caches, quadrature, selections and fixed
comparisons. No C-model or training-prefix posterior may be substituted.
`--contest 2445 --accepted tmp/runs/main/full/2445/accepted.json` can replace
`--all --accepted-root tmp/runs` for one case. The separate stage commands are:

```sh
python -m pipeline.jk9_design quadrature --all --accepted-root tmp/runs --output tmp/runs/design
python -m pipeline.jk9_design screen --all --accepted-root tmp/runs --output tmp/runs/design
python -m pipeline.jk9_design refine --all --accepted-root tmp/runs --output tmp/runs/design
python -m pipeline.jk9_design assemble --all --accepted-root tmp/runs --output tmp/runs/design
python -m pipeline.jk9_design fixed --all --accepted-root tmp/runs --output tmp/runs/design
```

Choose the `all` command **or** the separate sequence; do not execute screen twice
against the same completed output. `plan` prints the input selection only.
`quadrature` prepares the draw groups; screen also prepares them if needed.
Use SQLite caches by default, `JK9_CACHE_MODE=memory` for in-process caching or
`JK9_CACHE_MODE=off` to disable persistence. Cache mode does not change the
scientific input/output contract.

Original controllers are `design_batch.py`, `refine_prizes.py`,
`assemble_refined.py`, `fixed_precision_table.py` under
`docs/review/PaperJK9/nonlinear_table_rebuild_20261025/`. The adapter supplies their
runtime paths. Inputs are frozen main JSONs and joint accepted draws; outputs
include `runtime_posteriors/<id>.json`, `contests/<id>/quadrature.json`,
`result.json`, `prize_refinement.json`, `refined_selections.json` and
`fixed_precision/<id>.json`. Main selections contain the nine deltas
`.1,.25,.5,1,2,5,10,25,100`. Failed candidates and unresolved comparisons remain
in the output schema. They are not removed by aggregation.

Aggregate the new results with the original aggregation functions, redirected to
the generated tree:

```sh
python - <<'PY'
import json
from pathlib import Path
from pipeline.jk9_design import controllers
from pipeline.publication import write_json
out = Path('tmp/runs/design').resolve()
_, _, _, fixed = controllers(out)
from pipeline.jk9_design import load_controller
builder = load_controller('build_table')
paths = sorted(out.glob('contests/*/refined_selections.json'))
rows = builder.aggregate(paths, expected=73)
write_json(out/'table_numbers.json', rows)
fixed.aggregate()
print('Wrote', out/'table_numbers.json', 'and fixed_precision/table_numbers.json')
PY
```

For comparison with the long-term published table, run `python -m pipeline.jk9 tables`.
That command reads the retained per-contest results. A newly sampled posterior
is expected to have Monte Carlo differences; do not require its chain hashes or
every discrete searched choice to equal the historical run.

### 8. Recompute prediction fits, scores and independent integrals

```sh
python -m pipeline.jk9_prediction plan
python -m pipeline.jk9_prediction baselines --output tmp/runs/prediction
python -m pipeline.jk9_prediction fit --model C --split primary_hour --output tmp/runs/prediction
python -m pipeline.jk9_prediction fit --model D --split primary_hour --output tmp/runs/prediction
python -m pipeline.jk9_prediction fit --model C --split complete_day --output tmp/runs/prediction
python -m pipeline.jk9_prediction fit --model D --split complete_day --output tmp/runs/prediction
```

A/B are constant/nonhomogeneous Poisson baselines. C is the flexible filtered
kernel (`pipeline/models/free_kernel.stan`); D is the equilibrium kernel
(`pipeline/models/leading_order.stan`). `--split` selects the exact cutoff in
`evidence/prediction/splits.json`, not an independently chosen date. Omitting
`--contest` processes all 73 contests. Baselines process both splits and write
`baselines_AB_<id>_<split>.json` plus `baseline_rerun_audit.json`.

C/D fits produce `<output>/<model>/<split>/<id>/accepted.json` with the same
chain-path convention as step 5. In addition to its ordinary schedule, the
prediction fitter can try `(2000,32000,.995)` and `(4000,64000,.9975)`.
C also diagnoses `deadline_power`, `gap_log_scale` and `gap_scale`.

Run both numerical checks for all C/D targets using their accepted records:

```sh
python -m pipeline.jk9_prediction check --accepted-root tmp/runs/prediction \
  --output tmp/runs/prediction_checks --budgets 32768 65536 131072 262144 524288 1048576 --seed-offset 0
```

`check` writes each posterior score and runs independent integration even if
the preliminary score screen flags a target. It records both outcomes in
`batch_status.json` and continues through numerical screening outcomes for
the remaining targets. If any integration needs follow-up, the command exits
with a nonzero status after recording all targets. Invalid inputs and execution
errors still stop the command. Use a new output directory for each batch.
The optional `--contest`, `--split`, and `--model` filters select targets for a
focused rerun. Execution commands require at least one matching target.

`--budgets` is the sequence of total independent proposal draws at each budget;
`--seed-offset 0` selects the declared initial integration stream. The code uses
32 blocks and requires two successive passing independent budgets. A fresh
retry uses a new `--output` and a different `--seed-offset`, preserving previous
diagnostics. Score outputs are `score_<model>_<id>_<split>.json`; integration
outputs are `importance_<model>_<id>_<split>/{proposal,status}.json` and
`production_<budget>.npz`. `status.json` links the accepted/posterior/source
identities and contains `records` and `passed`; production NPZs contain joint
hourly log scores and probability-weight diagnostics.

Use the integration records to generate fresh rare-hour results:

```sh
python -m pipeline.jk9_prediction repair --accepted-root tmp/runs/prediction \
  --integration-root tmp/runs/prediction_checks --output tmp/runs/prediction_tail \
  --budgets 32768 65536 131072 262144 524288 1048576 --seed-offset 100000000
```

An hour with probability-weight ESS below 400 or log probability below −20
triggers the tailored proposal. The repair reads the previous `status.json`,
checks its frozen split, model and accepted-record hash, and evaluates the
same predictive integral. Half the proposal weight remains on the defensive
training mixture. The rest goes to Student-t components centered at the rare
numerator modes, with covariance scales .5, 1 and 2. Each numerator is the
training density times the observed pair of hourly count probabilities.
The counts determine the numerator and its proposal centers. The training
density and the predictive probability being computed stay fixed.

Production uses new random draws and the same 32-block full-path diagnostics.
A tailored hour also requires numerator ESS at least 1,000, maximum normalized
numerator weight below .01, and log-probability MCSE at most .02. The hour
must agree across successive passing budgets within five combined MCSEs plus
.001. A low log probability alone is expected for a rare hour and can pass
these diagnostics. A repair that flags an additional rare hour can feed its
output directory into a new repair, with a new output directory and seed
offset. An unresolved record without a rare-hour trigger needs a fresh ordinary
integration, using the `integrate` command with its accepted record, contest,
split and model, and larger budgets if needed.

Register the freshly calculated scores and reproduce the comparison:

```sh
python -m pipeline.jk9_prediction summary-fresh --baseline-root tmp/runs/prediction \
  --accepted-root tmp/runs/prediction --integration-root tmp/runs/prediction_tail \
  --integration-root tmp/runs/prediction_checks --output tmp/runs/prediction_report
```

The repeated integration roots are searched in the order supplied. The command
joins passing A/B quadrature and C/D integration records against the 146 frozen
split rows. Each C/D score must match its accepted fit and have two consecutive
passing integration budgets. Outputs are `fresh_scores.json` with 584 records
and source hashes, and `fresh_summary.json` with the 73-contest equal-weight
comparison and 500,000 paired bootstrap draws. The base seed is
`20261025+1904`, with 1,000,000 added for the complete-day split. The separate
`python -m pipeline.jk9_prediction summary` command computes
`tmp/runs/prediction/historical_summary.json` from the 584 published score means.

### 9. Rebuild the theory display or regenerate its numerical search

```sh
python -m pipeline.figures.joint_design
```

Display inputs are the two long-term CSVs under
`results/releases/paperjk9-20261029/evidence/theory_source/`:
`three_class_regions.csv` (`sigma,c,regime`) and
`unresolved_point_decisions.csv` (coordinates, candidate comparison and failure
fields). The display consumes 14,553 grid points (77×189), preserving all 1,179
numerical-difficulty cells. Outputs are under `tmp/runs/theory_figure/`.

For a fresh underlying search, run the original numerical producers in order:

```sh
python -m pipeline.theory.joint_design_moderate_plane
python -m pipeline.theory.joint_design_moderate_dense --neighbor-polish --full-island-check
python -m pipeline.theory.joint_design_three_class
```

The first producer writes `tmp/work/results/theory/numerical_joint_plane_moderate/`
including `output_samples.csv` and `regions.csv`. The dense producer consumes
these and writes `numerical_joint_plane_moderate_dense/`; its two flags enable
neighbor-candidate transfers and continuous full-disclosure island checks.
The three-class report writes the two display input CSVs in its
`three_class_comparison/` subdirectory. Optional old samples are memoization;
`load_cache` returns an empty cache when an optional file is absent.
The display's maintained entry point uses the long-term CSVs by default; when
rendering a new search, point its `base` variable at the new
`three_class_comparison/` directory in an isolated copy of the display script.
Do not replace long-term evidence automatically.

### 10. Reproduce appendix recovery, cost-prior sensitivity and illustrations

#### Nonlinear-generated parameter recovery

```sh
python -m pipeline.empirical.recovery generate --output tmp/runs/nonlinear_recovery
python -m pipeline.empirical.recovery fit --output tmp/runs/nonlinear_recovery \
  --samples 30000 --workers 8
python -m pipeline.empirical.recovery report --output tmp/runs/nonlinear_recovery
```

`pipeline/empirical/recovery_design.toml` fixes the ten contest IDs, six joint
truth coordinates, activity rate, input hashes and reference calibration
multipliers. The coordinates were originally selected as joint representative
points from nonlinear empirical fits. Each truth vector was extracted from
the four retained recovery summaries for that contest, with all 40 summary
hashes checked against the published CSV. These fixed coordinates and the
full-window contest inputs supply the simulation design.

`generate` processes replicates 2–5. The seed for the Brownian path is
`271000 + 100*contest_id + 10*replicate`. Add 1, 2 and 3 for the Poisson,
uniform and initial-condition streams. The nonlinear policy uses gap spacing
.00625 and 1,024 time steps, checked against .003125 and 2,048 steps with a 1%
discrepancy tolerance. Each `recovery/<id>_<replicate>/` has `data.json`
(constructed deadline, hourly gaps, interval-start feedback and event times),
`truth.json`, `paths.npz` and `generation_audit.json` with the generated and
reference input hashes. Use `--contests` or `--replicates` for a subset of
generation or fitting.

`fit` consumes each `data.json`, builds a proposal with dispersed prior-scale
initialization, then runs four chains with 1,500 warmup and 30,000 retained
draws per chain. The leading estimator uses interval-start feedback and the
`lambda_external` prior (log median `log(100)`, log SD 2.25). It writes
`posterior/recovery/<id>_<replicate>/lambda_external/` with the proposal,
four chain NPZs, diagnostics and `summary.json`. The report requires rank
R-hat ≤1.01 and bulk and tail ESS ≥400 for all reported quantities in all 40
fits. It writes `validation/simulation_recovery_reference.csv` (80 rows with
truth, posterior mean/median, relative error and interval coverage) and
`interval_and_recovery_report.json` (aggregate error and coverage). The
appendix reports absolute relative error of the posterior median and coverage
of the original 95% credible intervals. The CSV also records coverage after
the separate frozen interval expansions. To also regenerate
`empirical_calibrated_intervals.csv`, pass
`--posterior-summary tmp/runs/empirical_reports/estimation/posterior_summary_final.csv`
to `report`, using the empirical summary generated in step 6.

#### Cost-prior sensitivity

```sh
for id in 8219 2464 5144 2551; do
  python -m pipeline.jk9_fit --scenario main --mode full --contest "$id" --output tmp/runs
done
python -m pipeline.empirical.cost_sensitivity plan --baseline-root tmp/runs
python -m pipeline.empirical.cost_sensitivity fit --baseline-root tmp/runs \
  --output tmp/runs/cost_sensitivity
python -m pipeline.empirical.cost_sensitivity evaluate --baseline-root tmp/runs \
  --output tmp/runs/cost_sensitivity
python -m pipeline.empirical.cost_sensitivity report --output tmp/runs/cost_sensitivity
```

If step 5 has already generated the four full-window accepted fits, use that
directory as `--baseline-root`. For each contest, both alternatives refit the
frozen full window with `pipeline/models/common_cost.stan`. For
`u=(log(c_i)+log(c_j))/2` and `v=log(c_i)-log(c_j)`, the mean of each is
zero, and the SD of `v` is `.75*sqrt(2)`. `common_sd_100` sets the SD of
`u` to 1 and the cost floor to .01. `cost_floor_0001` uses SD
`.75/sqrt(2)` and lowers the floor to .001. All other priors are shared.
The fitter uses dispersed starts around accepted baseline medians and the
original case seeds. The 4-chain dense NUTS schedule uses maximum tree
depth 13 and (warmup, retained draws, adapt delta)
`(1000,2000,.98), (2000,4000,.99), (4000,8000,.995),
(2000,32000,.995), (4000,64000,.9975)`. It checks the usual diagnostics plus
rank R-hat and bulk/tail ESS for `u`, `v` and the cost ratio, and writes
`fits/<scenario>/<id>/accepted.json` and hash-linked chains.

`evaluate` consumes those fits and the fixed published redesigns (default
`results/releases/paperjk9-20261029/evidence/main_design/<id>.json`).
Pass `--design-root tmp/runs/design/contests` to evaluate new selections from
step 7 instead. The four original cost weights are 2, 100, 1 and .1,
respectively. For each alternative, the paired original-game HJB calculation
uses seed `202610260000+contest_id` with 32 coarse and 4 independent fine
draws per chain. The Welch interval plus the grid error radius determines the
supported direction. The fixed published prize is evaluated first, then the
declared fractions .5, .75, .875 and .9375 toward the appropriate prize
boundary, with duration and feedback held at the selected levels. For 8219
the refinement evaluates all 128 coarse draw identities directly on the fine
grid. The pre-portability controller versions linked by the historical results are
available at commit `c47f18a28bd16dbd37d9d4c61e0b81bf222c254b`.
Baseline feedback controls for 2551 and 5144 use draws from their own
accepted full-window fits, seed `202610250000+contest_id`, and compare
feedback at the same prize and duration.

The outputs are eight `results/<id>_<scenario>.json` records, two
`small_refinement/<id>_<scenario>.json` records for 8219,
`baseline_controls/<id>.json` and `direction_summary.json`. The report
separates support at the published design from support after an allowed prize
adjustment. Use `--contests` and repeated `--scenario` flags for selected
refits or comparisons.

#### Illustrations and official ranks

```sh
python -m pipeline.figures.variance --output tmp/runs/figures/figure_S_evolve.pdf
```

The variance plot solves `S'=sigma²-lambda*S²` in closed form at 1,001
equally spaced times from 0 to 10, using sigma = lambda = 1. The two initial
values are 0.8 and approximately 1.08331, read from the retained figure.
For the two conceptual diagrams, export the complete page of each
`paper/kaggledata_diagram{,_simplified}.drawio` as a same-basename PDF in
diagrams.net.

Official-rank agreement uses the original ZIP `Teams.csv` and the
73-contest accepted-source inventory, with 10,000 bootstrap resamples and
seed 20261008:

```sh
python - <<'PYCODE'
from pathlib import Path
from pipeline.validation import rank_agreement as r
r.MAIN_ESTIMATION_DIR = Path('results/reestimation/lambda100_sd225/estimation')
r.MAIN_VALIDATION_DIR = Path('tmp/runs/rank_agreement')
r.MAIN_VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
r.main()
PYCODE
```

### 11. Compile the manuscript and inspect stage coverage

```sh
python -m pipeline.jk9 tables
python -m pipeline.jk9_prediction summary
make paper
python -m pipeline.reproducibility_audit
```

`make paper` compiles the retained manuscript and long-term figure/table assets:
`paper/PaperJK12.tex`, `paper/appendix_joint_design_jk12.tex`,
`paper/appendix_effort_recovery_jk12.tex`, `paper/appendix_100k_rows.tex` and the
figure PDFs/PNGs. It writes auxiliary files and the built PDF to `tmp/latex/`
and copies the final PDF to `paper/PaperJK12.pdf`. New scientific estimates are
written separately; building this manuscript does not silently substitute new
numbers into it. Use the figure/table index to map each generated result to its
paper location when preparing a new numerical version.

A full computation has the following interface coverage: synthetic generator
→ synthetic NPZs → initialization/pooled recovery summaries; frozen empirical
inputs →397 accepted fits → main/auxiliary reports; 73 full-window fits → design
selections → nine-row main and fixed-precision aggregates; 146 frozen prediction
splits →292 C/D fits plus A/B quadrature → predictive score/integration records
→584 means → paired bootstrap; theory solver → region CSVs → display; appendix
checks → their retained summaries; manuscript sources and assets → PDF.

For maintenance, check these links and schemas without launching the expensive
sampling, HJB search or production integration. `verify-release` checks retained
file identities; `verify-reproducibility` checks retained material completeness.
Neither is a requirement to preserve every generated intermediate in Git.


## Retained files, generated files and maintenance checks

Keep model/source code, frozen scientific inputs, configurations, seeds,
long-term results, diagnostics and source indexes. Generated chains, per-hour
probabilities, numerical grids, caches and build products live below `tmp/` and
are reproduced by the stages above. A fresh clone starts with the retained
materials and generates its own working files in dependency order.

`docs/PAPER_EVIDENCE.csv` connects manuscript items to result and producer files.
`docs/RESULT_CODE_AUDIT.json` inventories the retained result/source mappings;
`python -m pipeline.result_code_audit` regenerates it. The variance figure has an
analytic generator, and the two conceptual diagrams have editable `.drawio` sources. Historical source snapshots preserve their original content
identities. For their dated comments and review questions, consult the context
in `docs/review/README.md`.

The 6 November local inventory contains all 292 indexed main-design chains,
matching their hashes, and the recovered prediction bundle: 1,400 blobs,
1,256,958,899 bytes, matching recorded sizes, SHA-256 and Git blob SHA-1.
Local availability is recorded separately from the files distributed through Git.

The evidence source index at `results/releases/paperjk9-20261029/evidence_sources.json`
records `source_sha256` for the original artifact and `release_sha256` for the
retained copy. For 75 JSON records, the retained copies have updated `scope`
wording. The publication and reproducibility checks validate each retained copy
against its `release_sha256`.
The calculation commands above use fresh generated outputs when historical
working files are absent.

| Maintenance command | Input | Output or check |
| --- | --- | --- |
| `make verify-release` | Selected publication manifest and retained files | Source/input/result content identities and figure-source agreement |
| `make verify-reproducibility` | Frozen input indexes, source indexes, formal results | Retained-material completeness and optional local chain identities |
| `make reproduce-derived` | 73 retained main and fixed-precision result records | Numerical table rows matched to the selected manuscript |
| `make verify-chains` | Accepted-posterior index and corresponding local chain files | Four chain hashes per posterior |
| `make test` | Source and deterministic fixtures | Implementation and interface regression results |
| `make paper` | Manuscript, bibliography and figure/table assets | Compiled manuscript PDF |

Fresh fitting and integration write their own diagnostics and acceptance records.
The current numerical output schema retains concrete failed candidates and
unresolved comparisons, including the 6322 full-disclosure comparator and the
1,179 theory-grid cells shown as numerical difficulty. These are specific
calculation statuses; intermediate-file retention does not set them.

## Prediction producer and diagnostic references

The implementation includes these stages and checks:

- Posterior fitting and acceptance: `pipeline/jk9_fit.py`,
  `pipeline/empirical/unified_prior_medians.py`, and the recovered
  `task4_final_protocol_training_method_20261025.md` and
  `task4_day_acceptance_merge_method_20261025.md`.
- Joint predictive scoring: `pipeline/jk9_prediction.py`,
  `pipeline/prediction/posterior.py`, and the recovered
  `task4_calibrated_posterior_prediction_method_20261025.md` and
  `task4_complete_day_prediction_method_20261025.md`.
- Independent full-path integration: `pipeline/prediction/importance.py` and
  `pipeline/prediction/full_path.cpp`; posterior/input/chain identities,
  denominator ESS and moments, path failures, probability ESS, block MCSE,
  half-sample differences and two consecutive independent passing budgets.
- Historical tail checks and final registration: the recovered
  `task4_tail_rare_mode_verification_method_20261025.md`,
  `task4_day_verified_score_refresh_method_20261025.md` and
  `task4_cluster_bootstrap_precision_method_20261025.md` contain executable
  Python checks, including target parity, production identities, all 73 cases,
  score gates and reportability checks before final comparisons.

These historical method files are retained under
[`recovered_prediction_sources_20261029/`](review/PaperJK9/recovered_prediction_sources_20261029/).
They contain implementation code, not just narrative descriptions.
The corresponding retained result records are in
`results/releases/paperjk9-20261029/evidence/prediction/`:

| File | Role |
| --- | --- |
| `splits.json` | 146 fixed contest/split inputs and training cutoffs |
| `recovered_scores.json` | 584 contest/model/split mean joint log scores |
| `historical_summary.json` | Paired 500,000-replicate contest bootstrap |
| `baseline_rerun_audit.json` | 146 A/B quadrature checks and score comparisons |
| `native_stan_parity.json` | Native/Stan training density parity at three points per model |
| `source_recovery_manifest.json` | Content identities of the 99 retained original text sources |
| `artifact_recovery_manifest.json` | Content inventory of locally recovered generated files |
| `recovered_chain_coverage.json` | Four-chain candidate inventory for 292 C/D targets |

The recovery inventories identify files. Fitting, scoring, integration and final
registration apply the diagnostic gates shown in their implementations. For
example, candidate chain groups distinguish attempts; downstream consumers read
the accepted-posterior records written by the fitting stage.
