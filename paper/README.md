# PaperJK12

`PaperJK12.tex` is current. From the repository root, `make paper` compiles JK12;
`make reproduce-derived` checks the two nonlinear tables against compact final
evidence without updating prose. `current_release.toml` selects JK12.

JK7/JK8/JK9/JK11 are historical manuscript sources. Scientific evidence retains its original JK9 paths. Numerical calculations must not run manuscript
integration patches over later writing edits. See `../docs/REPRODUCIBILITY.md`.

`make paper` builds auxiliary files under `tmp/latex/` and copies only the
finished `PaperJK12.pdf` into this directory. Manuscript source and publication
figures stay here; work products and review renders belong under `tmp/`.
