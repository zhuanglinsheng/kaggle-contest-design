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

Python 3.11 or later is recommended. Create an isolated environment and
install the declared dependencies:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The estimation workflows also require a working CmdStan installation and the
compiled executables `metakaggle/real_data`, `synthetic_data/synthetic_data`,
and `synthetic_data/synthetic_data_90_replicated`. Compile all three with
`make models`.

## Current workflow

Commands are run from the repository root. The stages are intentionally
separate because estimation is expensive and resumable.

```sh
make test
make figure3
make synthetic-recovery
make estimate
make finalize-estimation
make counterfactuals
make validate-main
make build-robustness-data
make estimate-robustness
make finalize-robustness
make validate-robustness
make paper
```

See [`pipeline/README.md`](pipeline/README.md) for inputs, outputs, and the
purpose of each stage. Prize units are defined once in
[`pipeline/config.py`](pipeline/config.py): source JSON files use thousands of
U.S. dollars, while the current estimation uses units of USD 100,000.

## Current versus historical outputs

Only `results/` is used by the current pipeline. Outputs from superseded
estimators, prize-unit experiments, and earlier counterfactual exercises are
retained under `archive/results/` for provenance.
