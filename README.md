# Kaggle contest design

The current manuscript is [`paper/PaperJK7.tex`](paper/PaperJK7.tex).
[`current_release.toml`](current_release.toml) selects its empirical results,
figure sources, evidence index, and publication snapshot. Start there when
checking which version a file belongs to.

## Working on the paper

```sh
make verify-release   # check the saved publication snapshot; read-only
make paper            # compile the manuscript, including its appendix inputs
```

After intentional edits, review the differences, compile and check the affected
results, then run `make release-manifest` and `make verify-release` to record the
new snapshot. Refreshing the manifest records bytes; it does not establish
statistical validity. `make sync-paper-assets` copies figures only from the
selected, hash-checked sources. `make verify-chains` additionally checks all
397 accepted fit records and their four chains when the full artifact bundle
is available.

[`docs/PAPER_EVIDENCE.csv`](docs/PAPER_EVIDENCE.csv) maps each figure/table label
to its evidence and producing module. Labels remain useful when table numbers
change. [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) distinguishes paper
compilation, snapshot verification, derived calculations, and re-estimation.

## Repository map

| Path | Purpose |
| --- | --- |
| `paper/` | Current manuscript, appendix fragments, local figures and PDF |
| `pipeline/` | Scientific runners and publication checks; see its README for roles |
| `results/reestimation/lambda100_sd225/` | Current empirical estimates and dependent analyses |
| `results/releases/` | Publication file hashes and portable accepted-fit indexes |
| `results/synthetic/`, `results/theory/` | Retained synthetic and theory evidence |
| `metakaggle/`, `synthetic_data/` | Data utilities and model sources |
| `tests/` | Numerical, data-processing and publication checks |
| `docs/` | Reproduction guide, evidence map and repository maintenance notes |
| `archive/` | Previously archived materials; some other old paths remain active dependencies |
| `ryvkin/`, `prediction/` | Reference model and separate predictive exercises |

## Environment and conventions

The tested environment is Python 3.13.15, the packages in
[`requirements-lock.txt`](requirements-lock.txt), CmdStan 2.36.0 and TeX Live
2025. `make setup` installs the Python environment; `make test` runs the tests.
The empirical prize unit is USD 100,000; source JSON prizes are in thousands
of dollars. Primary estimates use posterior medians.

Historical empirical Make targets require an explicit `LEGACY=1` selection
and should be run in an isolated checkout. Their paths in `pipeline/config.py`
are retained for scientific provenance. They do not define the current release.
The old `reproduce-derived` shortcut stops with guidance because it would
combine old empirical outputs with the current manuscript.
