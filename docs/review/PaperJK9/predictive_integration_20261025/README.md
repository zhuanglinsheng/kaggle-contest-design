# Task3 shared dependency retained after cleanup

Despite this directory's historical name, its remaining three original
files are retained for Task3's accepted same-target cost-floor posterior
check: `density_components.so`, `build.json`, and `parity.json`.

`../task3_global_cost_floor_check_method_20261025.md` loads the verified
likelihood backend from `../task4_predictive_density_parity_method_20261025.md`.
That backend uses this binary and checks its build hash. The full source is
embedded in `../task4_fast_density_evaluator_method_20261025.md`.
`../task4_final_protocol_training_method_20261025.md` also remains as shared
sampling-protocol provenance referenced by Task3 reconstruction reports.

The original bytes are preserved. All Task4-only subdirectories, draws,
integration arrays and prediction results from this directory were deleted.
