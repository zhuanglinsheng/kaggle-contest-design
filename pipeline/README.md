# Maintained pipeline entry points

- `make verify-reproducibility`: retained inputs/source/results; absent generated historical draws are optional.

- `python -m pipeline.jk9 verify`: current compact source/input/evidence lock selected by `current_release.toml`.
- `python -m pipeline.jk9 tables`: reaggregate current nonlinear tables and compare TeX rows.
- `python -m pipeline.jk9_fit --help`: fresh fits from frozen inputs; no historical-chain dependency.
- `python -m pipeline.jk9 prepare-inputs`: materialize matching main working inputs.
- `python -m pipeline.jk9 compile-models`: compile portable Stan sources.
- `python -m pipeline.figures.joint_design`: rebuild the current theory display from frozen CSVs.
- `pipeline.empirical.nonlinear_design`: nonlinear solver/search API.
- `python -m pipeline.jk9_design --help`: isolated original nonlinear search/refinement/fixed-precision controllers.
- `python -m pipeline.jk9_prediction --help`: A/B scoring, C/D fitting, independent integration via `check`, rare-hour integration via `repair`, and fresh-score registration and bootstrap via `summary-fresh`.
- `python -m pipeline.empirical.recovery --help`: regenerate the 40 fixed nonlinear-policy recovery datasets, fit them, and report error and coverage.
- `python -m pipeline.empirical.cost_sensitivity --help`: fit the two alternate cost priors and evaluate the four nonlinear redesign directions.
- `python -m pipeline.figures.variance --help`: regenerate the conditional-variance illustration.
- `python -m pipeline.organize --audit`: check storage classification and output boundaries.
- `pipeline.storage`: one definition of `tmp/` runs, caches, builds and legacy path resolution.
- `pipeline.solver_cache`: SQLite/memory/off point storage; final evidence stays separate.

Paper-specific controllers are in `docs/review/PaperJK9/`. Historical controller versions are available in Git; publication commands use the selected release inputs and retained evidence.
See `../docs/REPRODUCIBILITY.md` for the difference between table aggregation,
fresh estimation, historical controller replay and recovered prediction provenance.
