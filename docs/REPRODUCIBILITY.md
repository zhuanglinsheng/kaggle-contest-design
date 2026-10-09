# PaperJK7 reproducibility guide

## Release boundary

The current paper is `paper/PaperJK7.tex`. Its authoritative compact outputs
are under these paths:

- `results/estimation/main/`: 73 matched full/early posterior summaries,
  portable accepted-chain provenance, and immutable fitted-input snapshots;
- `results/validation/main/`: Chapter 5 diagnostics and sensitivity checks;
- `results/counterfactual/main/`: posterior counterfactual summaries and the
  precision-policy calculations;
- `results/robustness/chapter6/reports/pair_L1_L2/`: the matched alternative-pair,
  L1-scale, and L2-scale robustness results used by the paper;
- `results/synthetic/optimal_hourly/`: the nonlinear-policy synthetic design;
- `results/theory/numerical_joint_plane_moderate_dense/three_class_comparison/`:
  the numerical joint-design classification used by the paper.

The many named directories immediately below `results/estimation/` are not
competing publication versions. They are resumable estimation shards and
refinements referenced by `results/estimation/main/accepted_chain_sources.csv`.
Do not move them unless that manifest is regenerated and verified. The
incomplete `Linf` Chapter 6 experiment is exploratory and is not part of the
paper release.

## Tested environment

- macOS with Python 3.13.15;
- packages pinned in `requirements-lock.txt`;
- CmdStan 2.36.0;
- TeX Live 2025, pdfTeX 1.40.28, and latexmk 4.86a.

Create the environment with:

```sh
make setup
```

The optional raw-data downloader in `metakaggle/api.py` additionally requires
the Kaggle API package and credentials. It is not imported by the analysis
pipeline. The local raw archive is expected at
`metakaggle/__rawdata__/meta-kaggle.zip`; processed publication inputs are the
73 JSON files in `metakaggle/__jsondata__/`.

## Verification and reproduction levels

Run the fast, non-estimation verification first:

```sh
make test
make verify-release
make paper
```

`verify-release` checks the matched 146-fit panel, every accepted input hash,
the presence of all accepted chains, convergence thresholds, derived-result
dimensions, sensitivity-input provenance, paper-local figures, and the saved
compact release hashes in `results/release_manifest.json`.

To regenerate all paper-facing outputs from already accepted chains, without
rerunning Bayesian sampling:

```sh
make reproduce-derived
make release-manifest
make verify-release
```

The expensive estimation stages are intentionally separate:

```sh
make models
make estimate
make finalize-estimation
make build-chapter6-data
make estimate-chapter6
```

Estimation is resumable. Finalization refuses inputs that differ from the
saved Stan data or fits that fail the declared convergence checks. Rebuilding
the raw Meta-Kaggle inputs is a separate provenance operation; do not run it
inside an ordinary paper rebuild because it deliberately replaces processed
inputs after validation.

## Updating a release

1. Regenerate the affected derived outputs.
2. Run `make sync-paper-assets` and `make paper`.
3. Run the full test suite.
4. Run `make release-manifest` once the intended outputs are final.
5. Run `make verify-release`; it must pass without modifying files.

The release manifest hashes source code, tests, processed contest inputs,
compact result tables, every paper-facing figure, and every accepted CmdStan
chain. The large chain files are not duplicated in Git; publish them as a
separate artifact bundle when third parties need to regenerate the derived
tables without rerunning estimation. Their saved hashes allow that bundle to
be verified file by file.
