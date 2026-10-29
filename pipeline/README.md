# Maintained pipeline entry points

- `python -m pipeline.jk9 verify`: JK9 compact source/input/evidence lock.
- `python -m pipeline.jk9 tables`: reaggregate current nonlinear tables and compare TeX rows.
- `python -m pipeline.jk9_fit --help`: fresh fits from frozen inputs; no historical-chain dependency.
- `python -m pipeline.jk9 prepare-inputs`: materialize matching main working inputs.
- `python -m pipeline.jk9 compile-models`: compile portable Stan sources.
- `python -m pipeline.figures.joint_design`: rebuild the current theory display from frozen CSVs.
- `pipeline.empirical.nonlinear_design`: nonlinear solver/search API.
- `python -m pipeline.jk9_design --help`: isolated original nonlinear search/refinement/fixed-precision controllers.
- `python -m pipeline.jk9_prediction --help`: recovered prediction summary, baselines, C/D fitting, scoring and independent integration.
- `python -m pipeline.organize --audit`: check storage classification and output boundaries.
- `pipeline.storage`: one definition of `tmp/` runs, caches, builds and legacy path resolution.
- `pipeline.solver_cache`: SQLite/memory/off point storage; final evidence stays separate.

Frozen paper-specific controllers remain in `docs/review/PaperJK9/` to preserve
historical source identities. They are not implicitly run by publication commands.
See `../docs/REPRODUCIBILITY.md` for the difference between table aggregation,
fresh estimation, historical controller replay and recovered prediction provenance.
