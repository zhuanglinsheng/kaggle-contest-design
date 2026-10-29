# PaperJK9

`PaperJK9.tex` is current. From the repository root, `make paper` compiles JK9;
`make reproduce-derived` checks the two nonlinear tables against compact final
evidence without updating prose. `current_release.toml` selects JK9.

JK7/JK8 are historical sources. Numerical calculations must not run manuscript
integration patches over later writing edits. See `../docs/REPRODUCIBILITY.md`.

`make paper` builds auxiliary files under `tmp/latex/` and copies only the
finished `PaperJK9.pdf` into this directory. Manuscript source and publication
figures stay here; work products and review renders belong under `tmp/`.
