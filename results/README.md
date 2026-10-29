# Retained publication results

This tree contains retained formal results and the frozen scientific input
snapshots used by the current release. Regenerable work products live in `tmp/`.

- `releases/paperjk9-20261029/`: current per-contest results, tables, sensitivity
  checks, uncertainty records, provenance and the finite publication file lock.
- `releases/paperjk7-review-20261016/`: historical release and frozen-input index.
- `reestimation/lambda100_sd225/input_snapshots/`: immutable, hash-addressed inputs.
- Other retained compact CSV/JSON/PDF files: current manuscript evidence selected
  by the publication lock; they are not scratch output directories.

New fits, nonlinear searches and predictions write to `tmp/runs/`;
other experiments write to `tmp/work/results/`. Historical work products moved
to `tmp/history/`. Generated code snapshots that are useful provenance are kept
with the source records under `docs/review/source_snapshots/`, not in this tree.

Use `make reproduce-derived` and `make verify-release` for compact publication
checks. `docs/STORAGE_LAYOUT.md` and `docs/storage_layout.json` describe all roles
and old-to-new paths. No full scientific rerun is required for this organization.
