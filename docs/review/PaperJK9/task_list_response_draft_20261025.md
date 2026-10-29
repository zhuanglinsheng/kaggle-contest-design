# Tasks 1–3: internal response draft

Updated 27 October 2026. Internal draft only; not sent. The earlier draft was
superseded by the original-model table recalculation and focused sensitivity
exercise. Current evidence is indexed in [INDEX.md](INDEX.md).

**Task 1.** I have checked the manuscript against the implementation. The
empirical submission intensities and cumulative drift use the same
left-point effort values. The data-dependent initialization and the different
feedback/readout conventions across the empirical and simulation exercises
are described consistently. The fixed-input identification result retains
its qualification concerning empirical initialization.

**Task 2.** I have rechecked the uniform-in-precision comparison and global
competitor argument under their stated assumptions and did not identify a
further substantive error. Additional numerical repairs do not certify the
complete partition. The revised figure explicitly marks the 1,179 unresolved
grid points, including 926 with potentially competitive failed candidates.
The notes distinguish numerical uncertainty from infeasibility and explain
that smoothing does not certify regime boundaries.

**Task 3.** Both counterfactual tables have been recomputed using the original
nonlinear model for all 73 contests; the leading-order approximation is used
only for Bayesian parameter estimation. Unresolved alternatives remain
unranked, and the unresolved full-disclosure comparison for contest 6322 is
retained in the reported subgroup-median bounds.

I also checked four selected contest/scenario pairs under a wider common
cost-scale prior and a lower cost bound, using the corresponding accepted
directly refitted joint posteriors. All eight comparisons support a
cost-saving design along the original joint adjustment direction. Seven
support the original design; one requires a smaller prize reduction, with
savings falling from about 39.59% to 19.80%. The very small-saving case was
checked with direct fine-grid integration. These findings support directional
robustness for the selected cases and settings, rather than invariance of
optimal magnitudes or robustness throughout the sample. Joint improvements
are distinguished from the isolated feedback effect; contest 2551 does not
provide evidence that increasing feedback precision itself improves quality.

## Evidence

- [Task1 implementation correspondence](task1_implementation_response_20261024.md)
- [Task2 theory/numerics review](task2_theory_numerics_progress.md)
- [Task2 figure update](task2_manuscript_figure_update_20261025/change_manifest.json)
- [Task3 original-model calculations](nonlinear_table_rebuild_20261025/README.md)
- [Task3 focused prior/bound sensitivity](task3_nonlinear_prior_sensitivity_20261026/README.md)
