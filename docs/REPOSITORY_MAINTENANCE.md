# JK9 maintenance and submission policy — 29 October 2026

Priority: make sources, inputs, evidence and rebuild commands explicit before
judging old caches disposable. No automatic deletion, history rewrite, staging
or commit is part of this maintenance.

Acceptance is dependency and workflow completeness, with proportionate static,
build and small regression checks. A complete scientific rerun is optional;
its absence alone is not a reproducibility gap. Cache retention decisions follow
whether its contents are reproducible from retained inputs and declared code,
rather than requiring every expensive result to be recomputed during cleanup.

## Storage contract

| Layer | Keep in Git? | Policy |
| --- | --- | --- |
| Maintained Python/Stan/C++ and tests | Yes | No scientific dependency on archive/ |
| Frozen inputs/configuration/seeds | Yes | Changes are scientific revisions, not cache cleanup |
| Compact final results/diagnostics/unresolved summaries | Yes | `results/releases/paperjk9-20261029/evidence/` |
| Immutable raw data | No | Retain as a source input with SHA-256; it is not temporary |
| Full posterior draws and hourly outputs | No | Generated intermediate artifacts under `tmp/`; keep their provenance/index with formal evidence |
| Point memoization | No | Under `tmp/`; default SQLite, optionally memory or off |
| Executables/build products | No | Build under `tmp/build/` from retained source |
| Frozen historical controllers | Source only | Preserve old source versions; do not silently update old hash certificates |

The `.gitignore` rules expose source and compact release evidence, while hiding
review-run binaries, chains and per-point data. Do not force-add entire research
folders. The publication manifest is a finite file allowlist, not a recursive
copy of the working directory. Changes to the evidence lock must be explicit.

## Before committing

1. Run `make test`, `make reproduce-derived`, `make paper`, and review diffs.
2. If the changed publication bytes are intended, run `make release-manifest`,
   then `make verify-release` (a hash refresh is not a scientific revalidation).
3. Review new files individually. Commit source/dependency fixes, then the JK9
   selector/evidence, then documentation; review older deletions separately.
4. Check a clean exported tree. Include every file in the compact release lock,
   even if it was initially ignored; never include a large chain merely because
   its filename escaped an old ignore rule.
5. Keep prediction reaggregation separate from numerical revalidation. Original
   sources and scores were recovered; historical C/D chain/integration bundles
   are not certified by source recovery or by a fresh baseline rerun. Recovered
   candidate chains are retained in a separate ignored bundle with a compact
   content index; keep that bundle before garbage-collecting old Git objects.

## Current code changes

- JK11 is the current default build/release entry point; scientific evidence retains JK9 provenance.
- Compact final evidence is copied without changing original records.
- Fresh posterior fits start from declared values and frozen inputs, not old chains.
- Default point storage is SQLite; memory/off modes avoid disk point artifacts.
- Active theory modules use `pipeline.theory.dimensionless`, recovered byte-for-byte
  from the deleted implementation in Git history.
- Numerical/cache changes preserve pre-maintenance solver source snapshots.
- Ordinary likelihood tests use explicit small fixtures instead of deleted research
  outputs. Independent Stan compilation is an opt-in integration check.

A partial initial cache compaction was stopped. Completed archives were verified
before removing their originals; their inventory is outside Git, with aggregate
counts in the maintenance report. Remaining caches and chains are untouched.

## Consolidated work root

All local intermediate output now lives under `tmp/`. New maintained commands
use `tmp/runs/`, auxiliary experiments use `tmp/work/results/`, and old work was
moved to `tmp/history/`. `docs/storage_layout.json` records each relocation;
`docs/STORAGE_LAYOUT.md` is the current human-readable classification. Source,
raw input and formal publication evidence are retained outside `tmp/`.
Run `make storage-audit` to check this boundary. Files were moved without
compression, deletion, staging or history rewriting. Scientific reruns are not
part of this storage acceptance criterion.
