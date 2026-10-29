# JK9 maintenance verification — 29 October 2026

## Implemented

- One current JK9 build/release selector and a compact, finite publication lock.
- Byte-preserving final evidence copies and a relative-path index for the 73
  main posteriors used by the nonlinear design tables.
- Cache-independent table aggregation, frozen-input preparation, standalone
  theory display reconstruction, and a fresh same-target posterior runner.
- Missing active theory dependencies recovered into pipeline.theory.dimensionless.
- SQLite point memoization by default, with bounded memory and no-persistence
  modes. Legacy point JSON remains readable. Final results/checkpoints are not
  disposable cache entries.
- Tests no longer need deleted historical empirical fixtures for ordinary
  likelihood/kernel checks; a compiler-backed Stan parity test is explicit.

## Actual validation

A separate temporary export was populated only from the compact release file
list, without raw Meta-Kaggle ZIP, historical chains, point caches or archives.
In that exported tree, compact verification, both design-table aggregations,
154 ordinary tests (one compiler integration test intentionally excluded),
and a fresh LaTeX build passed. The independent Stan parity tests were then run
explicitly in the working tree and both passed. Both retained Stan model sources
compiled successfully. The rebuilt theory display preserved 14,553 points and
all 1,179 unresolved cells without new equilibrium solves.

The validation used Python 3.14.2 and requirements-py314.txt, plus CmdStan 2.36.0.
This is separate from the historical pinned scientific environment. No complete
397-fit posterior rerun or 73-case nonlinear search was executed during maintenance.

## Cache handling status

Initial compaction was stopped when the user clarified the desired order.
36 completed archives contain 831,188 original cache files, with independently
checked content fingerprints before originals were removed. The local aggregate
inventory is results/releases/paperjk9-20261029/artifact_inventory.json.
An interrupted partial archive authorizes no deletion. No posterior chains,
final result files, configuration or seed records were removed. Remaining
historical caches have not been judged disposable or further cleaned.

## Archaeology and boundary repairs

Git tree/blob archaeology recovered 99 original prediction text sources
(693,976 bytes), including the exact free-kernel Stan, A/B probability code,
C/D prediction code, C++ full-path evaluator, independent importance sampler,
and rare-tail repair methods. Source identities are retained in the compact
prediction recovery manifest; a new clone needs no unreachable Git objects.

Original hourly score artifacts yielded 584 compact contest/model/fold records.
All 146 split inventories and every observed hourly count were checked against
frozen inputs. The exact original 500,000-replicate paired bootstrap reproduces
the original means and interval endpoints. All 146 A/B case-splits were then
rerun independently: every numerical check passed, maximum mean discrepancy
1.35e-9. Fresh C/D model training, chain scoring and independent full-path
importance integration now have maintained entry points with the original gates.
The recovered free-kernel Stan compiles. Native training densities agree with
independent CmdStan; native forecasts agree with independent Python for both
kernels. Future-readout timing and strict nesting are regression tested.

The nonlinear search/refinement/assembly/fixed-precision controllers now run via
`pipeline.jk9_design` in isolated outputs with a checked runtime posterior adapter.
Case 2445 quadrature was actually initialized from its four original chains.
Fresh fit chain paths are accepted-record-relative, fixing the previous mismatch
with the generic draw loader. A full 73-case nonlinear rerun was not executed.

## Historical provenance and optional validation

User clarification: maintenance requires complete reproducible sources, inputs,
dependencies and instructions; it does not require a full scientific rerun.
Unexecuted full runs below are validation-scope notes, not unfinished maintenance
work merely because they have not been executed. Missing indispensable inputs
or executable stages must still be treated as actual reproducibility gaps.

- Historical C/D acceptance identities and integration certificates remain
  unverified. Recursive tree inspection subsequently recovered complete candidate
  chain sets for all 292 targets, preserving 1,400 unique blobs (1.26 GB) outside
  Git tracking. Failed attempts may be present. Restored scores and bootstrap
  prove table reaggregation, not the numerical validity of every old integral. Complete fresh C/D fitting/integration
  remains unexecuted; specialized original tail methods are preserved, and a new
  run must obtain its own acceptance rather than reuse old repair certificates.
- Full 397-fit estimation, 73-case nonlinear search, and fresh complete C/D runs
  are substantial scientific computations. The maintained commands retain their
  failure and uncertainty flags; no full-run success is claimed.
- Local caches and chain bundles are not remote backups. No upload, Git commit,
  staging or history rewrite was performed.

No further historical-cache deletion was performed during archaeology. Decisions
about remaining artifacts should follow their dependency role: distinguish
regenerable memoization from indispensable inputs and historical evidence, and
check the generator/configuration/output contract. A full fresh scientific run
is not required before classifying a cache as regenerable. Preserve any unique
historical evidence or indispensable input in a verified separate bundle.

## Final archaeology validation

The first updated compact export contained 1,020 source/input/evidence files;
the final lock additionally includes the chain recovery utility and inventories. No locked
dependency is hidden by Git ignore rules. In a fresh temporary export without
raw ZIPs, historical chains or point caches, release verification, both design
tables, source-verified historical prediction reaggregation and the design plan
passed. All 165 tests passed (two optional compiler tests skipped); the nine
prediction tests including the independent C/D CmdStan integration check also
passed separately. Full C/D scientific reruns remain outside this validation.

## Subsequent storage organization

The current storage contract is `docs/STORAGE_LAYOUT.md`, with exact old-to-new
paths in `docs/storage_layout.json`. Runtime output, posterior chains, hourly
scores, caches, compiler products, auxiliary files and superseded experimental
results were moved into `tmp/` by rename. Formal result evidence, frozen inputs,
source/method records and immutable raw data remain retained outside `tmp/`.
Historical generated source snapshots were separated from results. Two theory
figure input CSVs were added to formal evidence so figure reconstruction does
not depend on a historical working directory. The 73-fit index retains original
acceptance bytes and chain hashes and points to the moved chains. New default
output paths, compilation paths, legacy record resolution and LaTeX paths now
follow the same contract. No full scientific rerun was requested or required.

The storage migration inventory records approximately 829,127 intermediate files
(2.069 GiB) under `tmp/`, with unchanged raw-input and formal-evidence roles.
All 397 frozen fit-input references and all 292 main-chain hashes were checked.
The 168-test suite passed (two optional compiler integrations skipped), the
current design quadrature adapter initialized successfully after relocation,
and the manuscript built with auxiliary files confined to `tmp/latex/`.
Theory display reconstruction read only the retained CSVs and performed zero
new equilibrium solves. This validates the structural changes without a new
scientific estimation/search/integration run.
