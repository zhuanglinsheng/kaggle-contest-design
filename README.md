# Kaggle contest design — PaperJK9

The active manuscript is [paper/PaperJK9.tex](paper/PaperJK9.tex).
`current_release.toml`, the Makefile and the evidence index all select JK9.

## Start here

Use Python 3.12 or newer (the source uses PEP 695 syntax), a virtual environment,
and the pinned scientific dependencies in `requirements.txt` (Python 3.14 uses
`requirements-py314.txt`). CmdStan 2.36.0,
a C++17 compiler and LaTeX are separate prerequisites for the corresponding stages.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --only-binary=:all: -r requirements.txt
make PYTHON=.venv/bin/python verify-release
make PYTHON=.venv/bin/python reproduce-derived
make paper
```

`reproduce-derived` reaggregates both nonlinear design tables from final
per-contest evidence and compares the numerical rows with the manuscript.
It does not rerun estimation, solve HJBs, or require point caches.

[Reproduction guide](docs/REPRODUCIBILITY.md) distinguishes aggregation from
fresh fitting and numerical search. Git archaeology recovered the original
prediction sources: the historical prediction table can now be reaggregated,
and both nonstrategic baselines have been independently rerun for all 73 contests
and both splits. Portable C/D fitting, scoring and importance integration are
available. Historical candidate chains were also recovered into a separate local
bundle; their old acceptance/integration certificates remain unverified. A
complete fresh C/D rerun remains unexecuted.

## Structure

- `paper/`: manuscript, appendices, bibliography and publication assets.
- `pipeline/`: maintained entry points, solvers, model sources and cache backend.
- `tests/`: numerical and implementation checks; ordinary tests do not require
  deleted empirical experiment directories.
- `results/releases/paperjk9-20261029/`: compact final evidence, 73-case portable
  posterior index, provenance and source/input byte lock. These files belong in Git.
- `metakaggle/`: data preparation source and the immutable raw ZIP; working input
  copies live in `tmp/inputs/`. Frozen inputs remain under
  `results/reestimation/lambda100_sd225/input_snapshots/`.
- `docs/review/`: retained methods, research narrative and frozen source snapshots.
- `tmp/`: all intermediate runs, posterior chains, hourly outputs, caches, builds
  and superseded experimental outputs; excluded from Git.

[Storage layout](docs/STORAGE_LAYOUT.md) defines each role and the migration map.

## Intermediate results

New nonlinear evaluations use `JK9_CACHE_MODE=sqlite` by default: one database
per cache directory instead of one JSON file per point. `memory` uses bounded
in-process memoization; `off` writes no point cache; `files` explicitly selects
the legacy layout. Final selections, random-draw indices, uncertainty records,
input configurations and posterior diagnostics remain ordinary evidence files.

All retained intermediate data now resides under `tmp/`; the exact relocation
map is `docs/storage_layout.json`. Existing point data and verified archives were
moved without deletion or further compression.
See [maintenance decisions](docs/REPOSITORY_MAINTENANCE.md) for storage and
submission policy. No Git history rewriting is part of this maintenance.
