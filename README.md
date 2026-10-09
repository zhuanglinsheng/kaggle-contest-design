# Kaggle contest design

Code and paper artifacts for the current PaperJK7 model of dynamic contest
design. The repository has one canonical execution path: modules under
[`pipeline/`](pipeline/). Historical notebooks, exploratory scripts, and old
paper versions are retained under [`archive/`](archive/) but are not part of
the current results.

## Repository layout

| Path | Purpose |
| --- | --- |
| `pipeline/` | Current synthetic, empirical, validation, and robustness workflows |
| `metakaggle/` | Data preparation utilities, processed contest data, and the empirical Stan model |
| `synthetic_data/` | Current simulator and synthetic Stan models |
| `results/` | Current estimation, counterfactual, and validation outputs |
| `tests/` | Fast tests for the current model and data transformations |
| `paper/` | Current PaperJK7 source, figures, tables, and compiled PDF |
| `archive/` | Superseded paper versions, notebooks, and exploratory code |
| `ryvkin/` | Reference implementation of the benchmark contest model |
| `prediction/` | Separate predictive exercises; not required to reproduce PaperJK7 |

## Environment

The tested environment is Python 3.13.15 with the exact packages in
[`requirements-lock.txt`](requirements-lock.txt), CmdStan 2.36.0, and TeX Live
2025. Create or refresh the local environment with `make setup`. Raw-data
download is optional and requires separate Kaggle API credentials; the
analysis pipeline itself does not import that downloader.

## Current workflow

Commands are run from the repository root. For an existing release, the normal
verification path is:

```sh
make test
make verify-release
make paper
```

To rebuild all paper-facing tables and figures from the already accepted
posterior chains, run `make reproduce-derived`. Bayesian estimation and
Chapter 6 fitting remain separate, resumable stages because they are costly.
After intentionally updating a release, run `make release-manifest` followed
by `make verify-release`.

See [`pipeline/README.md`](pipeline/README.md) for inputs, outputs, and the
purpose of each stage, and [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)
for the exact release boundary and commands. Prize units are defined once in
[`pipeline/config.py`](pipeline/config.py): source JSON files use thousands of
U.S. dollars, while the current estimation uses units of USD 100,000.

## Current versus historical outputs

Only `results/` is used by the current pipeline. The publication subset is
listed in [`results/README.md`](results/README.md) and pinned by
`results/release_manifest.json`. Estimation shards referenced by the accepted
manifest remain active provenance even when their directory names describe an
earlier batch. Superseded paper versions, notebooks, and results live under
`archive/`.
