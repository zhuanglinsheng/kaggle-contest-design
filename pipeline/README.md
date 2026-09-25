# Current PaperJK7 pipeline

All modules are invoked from the repository root with `python -m`. Shared paths
and the common USD 100,000 prize unit are defined in `pipeline/config.py`.
Run `make models` after changing any Stan source file.

| Stage | Module | Main output |
| --- | --- | --- |
| Synthetic baseline | `pipeline.synthetic.figure3` | `paper/synthetic_data.pdf` |
| Synthetic recovery | `pipeline.synthetic.recovery` | `results/synthetic/recovery/` |
| Main estimation | `pipeline.empirical.estimate` | `results/estimation/main/` |
| Failed-fit reruns | `pipeline.empirical.rerun_failed` | `results/estimation/main/reruns/` |
| Accepted-fit manifest | `pipeline.empirical.finalize` | final summary and accepted-chain manifest |
| Counterfactuals | `pipeline.empirical.counterfactual` | `results/counterfactual/main/` |
| Chapter 5 validation | `pipeline.validation.chapter5` | `results/validation/main/` and paper figures |
| Robustness data | `pipeline.robustness.build_data` | `metakaggle/__jsondata_robust23__/` |
| Robustness estimation | `pipeline.robustness.estimate` | `results/estimation/robustness/` |
| Robustness finalization | `pipeline.robustness.finalize` | accepted robustness summaries |
| Chapter 6 validation | `pipeline.validation.robustness` | `results/validation/robustness/` and paper figures |

The main and robustness estimators use the same Stan likelihood. The robustness
module changes only the processed-data directory and output target. Raw
CmdStan chains are resumable; finalization records exactly which run is
accepted for every contest.

Historical notebooks are under `archive/notebooks/`. They are useful for
provenance but are not authoritative implementations of the current model.
