# Kaggle contest design

Research code and replication materials for dynamic crowdsourcing contests with
noisy public feedback. The project combines a Bayesian dynamic game, structural
estimation using Kaggle submission and leaderboard data, and numerical comparisons
of prizes, deadlines, and feedback policies.

## Project entry points

- [Publication configuration](current_release.toml): selects the active manuscript,
  release manifest, empirical evidence, posterior index, and publication assets.
- [Reproduction guide](docs/REPRODUCIBILITY.md): requirements, computation stages,
  commands, and the scope of completed numerical validation.
- [Evidence index](docs/PAPER_EVIDENCE.csv): maps manuscript figures and tables to
  retained results and their producing code or methods.
- [Storage guide](docs/STORAGE_LAYOUT.md): defines permanent materials and local
  work products.

Manuscript filenames and dated evidence directories preserve version history.
Use the publication configuration to identify the active materials; a historical
name does not imply that the underlying evidence is obsolete.

## Repository structure

| Location | Purpose |
| --- | --- |
| [`paper/`](paper/) | Manuscript sources, appendices, bibliography, figures, and compiled papers. |
| [`pipeline/`](pipeline/) | Maintained estimation, simulation, prediction, numerical design, and publication commands. |
| [`tests/`](tests/) | Numerical and implementation regression checks. |
| [`results/`](results/) | Retained scientific results, diagnostics, and frozen estimation inputs. |
| [`results/releases/`](results/releases/) | Dated publication manifests, compact evidence, and provenance indexes. |
| [`metakaggle/`](metakaggle/) | Data preparation source and local raw-data storage; raw data are separate from generated work products. |
| [`synthetic_data/`](synthetic_data/) | Synthetic experiment source and model specifications. |
| [`docs/`](docs/) | Reproduction instructions, evidence mapping, methods, and maintenance policies. |
| [`docs/review/`](docs/review/) | Dated research records and frozen source snapshots for historical provenance. |
| [`archive/`](archive/) | Historical research and analytical derivations; see its README for scope. |
| `tmp/` | Local runs, posterior draws, hourly predictions, numerical caches, builds, and superseded outputs; excluded from Git. |

For maintained computation interfaces, see the [pipeline guide](pipeline/README.md).
The [storage migration map](docs/storage_layout.json) records historical path changes.

## Environment

Use Python 3.12 or newer and a virtual environment. Install the scientific
requirements with:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --only-binary=:all: -r requirements.txt
```

Python 3.14 uses `requirements-py314.txt` instead. Stan estimation additionally
requires CmdStan 2.36.0 and a C++17 compiler. Building the manuscript requires
LaTeX and `latexmk`. See the reproduction guide for stage-specific dependencies
and historical environment details.

## Common workflows

Run these commands from the repository root:

| Command | Purpose |
| --- | --- |
| `make verify-release` | Check retained publication files against the selected manifest. |
| `make verify-reproducibility` | Check retained sources, frozen inputs, final results, and provenance; report optional historical draws separately. |
| `make reproduce-derived` | Reaggregate both nonlinear design tables and compare their numbers with the selected manuscript. |
| `make prediction-summary` | Reaggregate retained submission-prediction scores and paired bootstrap comparisons. |
| `make test` | Run the ordinary numerical and implementation checks. |
| `make paper` | Build the active manuscript; auxiliary files go to `tmp/latex/`. |
| `make storage-audit` | Check that classified work products follow the storage policy. |

Set `PYTHON=/path/to/python` when needed; Make uses `.venv/bin/python` when
available. Table aggregation and publication checks use compact evidence and do
not require historical posterior chains or point caches. Fresh fitting,
nonlinear search, and predictive integration are separate, potentially expensive
stages documented in the reproduction guide.

## Evidence and generated files

Keep source code, frozen scientific inputs, configurations, seeds, final results,
diagnostics, and provenance records as permanent materials. Store generated
chains, hourly outputs, caches, and compiled artifacts under `tmp/`.
Some raw datasets and historical local bundles are not distributed through Git;
consult the reproduction guide before running a stage that requires them.

Generated historical draws are optional for compact publication checks. When the
original local chains are available, `make verify-chains` checks their recorded
hashes. Otherwise, fresh estimates can be generated from frozen inputs and
maintained code. New runs require their own numerical acceptance checks and need
not reproduce historical Monte Carlo bytes.

File completeness, table reaggregation, and scientific validation establish
different things. Current validation coverage and unresolved historical
integration records are documented in the reproduction guide; a successful file
check does not certify every numerical integral.

## Maintaining a publication

Update `current_release.toml` when selecting a new manuscript or release. Keep
the evidence index aligned with its figures and tables, preserve the original
provenance of reused results, and document changes to inputs or methods.

After reviewing intended publication changes, run the relevant checks and
explicitly refresh the file manifest with `make release-manifest`, then run
`make verify-release`. Refreshing hashes records approved file contents; it does
not rerun or validate the scientific analysis. See the
[maintenance policy](docs/REPOSITORY_MAINTENANCE.md) for details.
