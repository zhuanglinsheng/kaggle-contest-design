# Paper

`PaperJK7.tex` and `PaperJK7.pdf` are the current manuscript. PaperJK3--6 are
stored under `archive/paper_versions/`.

Chapter 5 and its dependent robustness and appendix results use the unified
empirical re-estimation in `results/reestimation/lambda100_sd225/`. Its
`release_summary_zh.txt` documents the estimates, and
`paper_integration/current.json` records the manuscript update and backup.
Parameter tables and regressions use posterior medians; precision
counterfactuals integrate the joint posterior draws. Chapter 4 retains its
estimation experiments and adds a direct player-i effort comparison after the
baseline data figure. Its 2-by-3 panels vary theta over 1, 2, 4, then independently
change lambda to 0.1 or 10 or opponent cost c_j to 0.6, retaining baseline
primitives otherwise. It is a controlled comparison rather than an
empirical-sample approximation guarantee.
Run `python -m pipeline.synthetic.effort_comparison` to regenerate its figure
and error summary in `results/synthetic/effort_comparison/controlled_incentives/`.

Table 7 reports fixed-initial-distribution, asymmetric first-order design
comparisons. Initial variance is calibrated from each observed-regime posterior
draw and held fixed across candidate policies. Results, grid/quadrature checks,
and the repeatable producer are in
`results/reestimation/lambda100_sd225/fixed_initial_design/` and
`pipeline/empirical/fixed_initial_design.py`. Run that module, then
`pipeline.empirical.fixed_initial_manuscript` to update the design passages.
Zero-precision results are labeled no-feedback limits (unattained under the
positive-feedback restriction). Original estimation, validation, and historical
design outputs remain available. Score-scale robustness now reports parameter
rankings; historical policy classifications are not evidence for this revision.
The design passages in `release_summary_zh.txt` describe the earlier calibration;
the current design tables are documented by `fixed_initial_design/summary.json`.

Every image referenced by the TeX source is copied into this directory, so
the manuscript compiles without paths outside `paper/`. Run `make paper`.
The current `sync-paper-assets` target reads `../current_release.toml` and checks
source hashes before copying. `verify-release` checks a saved source snapshot;
`paper` compiles edits independently. See `../docs/PAPER_EVIDENCE.csv` for the
figure/table map and `../docs/REPRODUCIBILITY.md` for updating a snapshot.
