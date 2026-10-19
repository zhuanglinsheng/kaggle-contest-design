# Repository maintenance decisions, 2026-10-16

This cleanup establishes an active publication entry point without relocating
scientific code, accepted runs or raw chains. It changes no manuscript prose,
parameter estimates or table values.

- `current_release.toml` selects the current version. The older
  `results/release_manifest.json` and scientific runner defaults retain their
  historical meanings.
- `docs/PAPER_EVIDENCE.csv` uses LaTeX labels instead of mutable table numbers.
  Keep its evidence, producer and scope columns current when adding a table.
- The publication manifest pins source, compact results and frozen inputs.
  The portable fit index keeps paths relative while preserving original hashes.
  Neither file certifies the underlying statistical conclusions.
- Prior/sampler modules referenced by frozen protocols must not be moved or
  reformatted during writing cleanup. Their exact source hashes are evidence.
- Do not classify entire trial directories as obsolete. The final recovery
  table and some accepted fits depend on earlier experiments.
- Small publication metadata is explicitly allowed by `.gitignore`. Large
  draws, arrays and executables remain excluded. An ignore-rule exception makes
  a file eligible for Git; it does not stage or commit it.

The workspace also contains pre-existing modified and untracked manuscript,
research-code and result files. They have been preserved. No commit is made by
this cleanup. Before creating a publication commit, review the full status,
include the intended new source files, evidence, protocols and input snapshots,
and retain the chain bundle separately. Use `make verify-release` after the
review; use `make verify-chains` when the bundle is available.

A subsequent computational maintenance task can extract the complete Table 7
scenario/comparator builder and test it against the frozen final CSV. That
requires checking scientific calculations and is intentionally separate from
this writing-preparation cleanup. No directory relocation or deletion is needed
to begin editing the manuscript now.
