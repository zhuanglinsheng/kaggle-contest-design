# PaperJK7 reproducibility guide

## Active release

`current_release.toml` is the version entry point. The current empirical root
is `results/reestimation/lambda100_sd225/`; the manuscript is
`paper/PaperJK7.tex` with three new appendix/section fragments and the existing
`appendix_100k_rows.tex`. `docs/PAPER_EVIDENCE.csv` maps every figure and table
label to an evidence file and producing module. A producer is provenance, not
an instruction to rerun a one-shot manuscript integration script.

The empirical revision has 397 accepted fits: 146 main full/early, 219
alternative-pair/score-scale, and 32 time-imputation fits. Primary summaries
are posterior medians. The lambda prior is log(lambda) ~ N(log(100), 2.25²);
prizes use USD 100,000. Most fits use NUTS; two accepted robustness fits use
same-target fixed-mixture Metropolis-Hastings. Their HMC-only diagnostics are
unavailable, not zero. Original scientific verification reports document these
choices. This repository maintenance does not change their contents.

## Four different operations

| Operation | Command / entry point | Scope |
| --- | --- | --- |
| Compile current manuscript | `make paper` | LaTeX only; does not regenerate empirical results |
| Check saved snapshot | `make verify-release` | Hashes of selected code/results, frozen inputs, recursive TeX dependencies, figure-source agreement, evidence-label coverage and 397 fit references |
| Check original chain bundle | `make verify-chains` | Snapshot checks plus hashes of 397 accepted records and 1,588 chain files |
| Regenerate analyses or sample again | Scientific modules below | Separate, potentially costly operations; may write reports or manuscript snapshots |

The compact check is read-only and uses Python's standard library. It does
not recompute posterior diagnostics, reread chain draws, compare every printed
number with its formula, or certify approximation accuracy. It verifies that
the reviewed files have not changed. The compiled main PDF is deliberately
outside the byte lock because compilation can change PDF metadata; source and
figure bytes are locked. Inspect compilation output and layout separately.

`make sync-paper-assets` first checks all selected source figures against the
saved lock, then copies changed figures. It cannot silently copy the old pair
robustness figures into the new manuscript. Original paper-local diagrams have
no separate generating source registered and are locked as supplied assets.

## Updating a writing snapshot

1. Edit the manuscript and compile with `make paper`.
2. Check text, affected tables/figures and the PDF. Run relevant tests; use
   `make test` for the full suite when changing computational code.
3. Review the file differences. A failed old-snapshot hash is expected for an
   intentional edit; investigate unrelated changes before accepting them.
4. Run `make release-manifest`, then `make verify-release`.

The refresh command records the selected files. It does not rerun scientific
acceptance checks or silently update accepted-fit hashes. If the accepted fit
set changes, complete its scientific verification first, then explicitly
export a new portable index with `make export-chain-index` and review it.
The export currently requires the original checkout paths in accepted.json;
verification of an existing exported index works after moving the checkout.

## Scientific workflow and limitations

| Stage | Module(s) | Output / caution |
| --- | --- | --- |
| Main and auxiliary estimation | `pipeline.empirical.unified_prior_medians`; add `--auxiliary` for auxiliary fits | Resumable; read module CLI and frozen protocols before use |
| Same-target sampler refinements | `unified_prior_global_check`, `unified_prior_adopt_global` | Selected fits only; preserve target parity and source hashes |
| Scientific acceptance check | `pipeline.empirical.unified_prior_verify` | Checks inputs, diagnostics and chain summaries; also writes snapshots/report, so this is not the read-only release checker |
| Main empirical reports | `pipeline.empirical.unified_prior_outputs` | Validation, figures and precision-only benchmark |
| Auxiliary reports | `pipeline.empirical.unified_prior_aux_summary` | Matched pair/scale/time sensitivity |
| Recovery summary | `pipeline.empirical.unified_prior_interval_report` | Uses retained nonlinear-generated recovery experiments |
| Current joint and precision-only design | `pipeline.empirical.fixed_initial_design` | Fixed initial distributions and asymmetric first-order output; includes quadrature/search refinement and the full-disclosure comparator |
| Direct effort approximation comparison | `pipeline.synthetic.effort_comparison` | Player i only; six panels vary theta=1,2,4, feedback precision or opponent cost; midpoint curves and whole-horizon error/grid checks; copy the produced PDF into paper/ before compiling |
| Update current design passages | `pipeline.empirical.fixed_initial_manuscript` | Rebuilds Section 5.3 and its calculation/benchmark appendices from verified design outputs; saves a manuscript backup |
| Historical joint design | `pipeline.empirical.design_duration_sensitivity` | Superseded steady-state calibration; does not reproduce current design tables |
| Historical manuscript integration | `unified_prior_manuscript`, `update_joint_design_manuscript` | One-shot revision tools; do not replay over later prose edits |

There is currently **no single command rebuilding all current paper numbers
from chains**. The current design tables can be rebuilt separately with
`python -m pipeline.empirical.fixed_initial_design`, followed by
`python -m pipeline.empirical.fixed_initial_manuscript` to integrate their
text and tables. Results are in `fixed_initial_design/` under the empirical
root. Every posterior draw retains its observed-regime initial variance while
candidate precision changes the Riccati path. Cost asymmetry is retained.
The no-feedback endpoint is explicitly a limit when positive precision is
required; it is not an attained design or an exact nonlinear prediction.
The earlier `design_sensitivity/` outputs remain historical evidence only.
Score-scale robustness in the manuscript now concerns parameter rankings;
old policy classifications do not establish robustness of the revised design.

For that reason, `make reproduce-derived` now stops instead of running the
previous release's pipeline against the current manuscript. Original empirical
Make targets require `LEGACY=1` and belong in an isolated checkout. The old
`pipeline.release` verifier and `results/release_manifest.json` are retained
for historical use; current commands use `pipeline.publication`.

## Artifact storage and portability

The current lock and portable fit index live under `results/releases/`.
Compact protocols, selected reports, frozen input snapshots and recovery source
summaries are eligible for version control. Chains, arrays, executables and
bulk logs are excluded. To regenerate derived results without resampling,
retain a separate bundle containing all relative `chain_paths` and
`accepted_path` entries in `accepted_fits.json`. `make verify-chains` checks
this bundle. Original acceptance records are byte-preserved even though they
contain machine-specific absolute paths; the portable index supplies their
relocated paths. Full computational reruns may still require adapting legacy
runner path handling, rather than editing historical records.

Tested environment: Python 3.13.15, `requirements-lock.txt`, CmdStan 2.36.0,
TeX Live 2025. Run `make setup` for Python dependencies. The optional Kaggle
downloader needs separate credentials; analysis uses processed contest inputs.
Historical instructions are preserved in
`docs/history/REPRODUCIBILITY_PRE_UNIFIED_PRIOR.md` and describe that earlier
release, not the current commands.
