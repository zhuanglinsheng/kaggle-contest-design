# Results map

The PaperJK7 release uses the following authoritative outputs:

| Path | Status |
| --- | --- |
| `estimation/main/` | Accepted 73-contest full/early summary and portable chain provenance |
| `counterfactual/main/` | Current counterfactual and precision-policy outputs |
| `validation/main/` | Current Chapter 5 diagnostics and sensitivity checks |
| `robustness/chapter6/reports/pair_L1_L2/` | Current alternative-pair and score-scale robustness results |
| `synthetic/optimal_hourly/` | Current nonlinear-policy synthetic design |
| `theory/numerical_joint_plane_moderate_dense/three_class_comparison/` | Current joint-design numerical classification |
| `release_manifest.json` | Machine-checkable hashes for the compact release |

Other named directories under `estimation/` are resumable shards,
refinements, or sensitivity fits. Some are still referenced by the accepted
manifest and therefore must not be moved merely because their names describe
earlier batches. Outputs explicitly marked `SUPERSEDED.md`, and material under
`archive/results/`, are not publication inputs.

Raw CmdStan chains are large intermediate artifacts. The compact summary and
accepted-chain manifests identify the exact accepted run and fitted-input hash
for each contest. See `docs/REPRODUCIBILITY.md` for the release procedure.
