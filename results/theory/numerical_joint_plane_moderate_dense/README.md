# Dense linear-axis joint-design panel

Run `python3 -m pipeline.theory.joint_design_moderate_dense --neighbor-polish --full-island-check` from the
repository root. Preferences and the equilibrium model are unchanged:
`S0=1`, `u(B)=3*(1-exp(-2*B))`, `D(T)=T**2/2`.

The risk mesh uses increments of 0.05 plus the original measured slices
and sigma=0.03. The effort-cost mesh is 0.03 to 0.5 in increments of 0.0025.
Both figure axes are linear, displayed over sigma=0 to 3 and c=0 to 0.5.
The strips below the measured positive minima are hatched, not extrapolated.
Neither zero endpoint is evaluated using formulas that divide by it.

Each new risk slice gets independent nonlinear output solves. Designs from
neighboring measured slices initialize the adaptive prize/deadline/precision
search; local design probes then reoptimize at the new slice. There is no
interpolation or smoothing of regime labels. These are numerical comparisons
over adaptive candidate sets, not certificates over the continuous design
domain. The numerical alpha cap remains 12.

Finite and full-disclosure families are compared using independently optimized
optimistic/pessimistic loss envelopes. Their relevant candidates receive
resolution checks. Failed competitive no-signal candidates are transferred
alongside successful designs, rather than silently removed. An independent
backward-Euler/CFL audit retries competitive failures on three meshes, retaining
the energy and no-signal payoff-identity checks from the previous experiment.
Mesh differences are numerical error diagnostics, not rigorous PDE bounds.
An unresolved cell cannot be filled merely to make the boundary look smooth.

`output_samples.csv` and `stable_audit_samples.csv` store new output solves;
previous preference-independent samples remain in their original directories.
`regions.csv` records the dense comparisons; `metadata.json` records the
actual mesh and counts. Generated PDF/PNG figures use the existing black/gray
style. The original sparse-grid experiment is preserved for comparison.
`regions_initial_density.csv` retains the first dense pass before neighboring
design cross-checks. The latter transfer both families' candidates and recompute
their outputs at the neighboring risk value; isolated labels are not deleted
or relabeled for graphical appearance.
At remaining isolated finite cells, an additional two-start continuous
full-disclosure prize/deadline search supplies more candidates. These outputs
are still resolution-checked and enter the original interval comparison.
The component test selects diagnostic locations only; it does not change labels.
`linear_area_comparison` weights midpoint cells on the same rectangle
sigma=0.06 to 3 and c=0.03 to 0.5, rather than comparing raw cell counts
across different grid densities.

Before the targeted left audit, the pass had 77 measured risk slices and 189 effort-cost points
(14,553 comparisons): 4,958 finite, 2,191 full-disclosure, 6,773 zero-prize,
and 631 unresolved. On the common measured rectangle the unresolved
linear area decreases from 2.84% to 2.56%. The targeted continuous
full-disclosure optimization checks 17 isolated finite cells; the isolated
finite components disappear after recomparison, not through label smoothing.
The broad low-risk unresolved band is retained. More sampled parameter
points alone do not establish convergence of the strong no-signal branch.

## Targeted direct comparisons

Run `python3 -m pipeline.theory.joint_design_left_audit` after the dense
experiment. It focuses on unresolved low-risk/high-cost cells, directly
reoptimizes full-disclosure prizes and deadlines at three representative
costs per slice, and searches additional positive precisions down to 1e-6.
Competitive designs are refined through levels 4/5/6. The original endpoint
and all potentially competitive failures stay in the comparison.

The stable solver now advances the organizer and players on the same
backward steps without saving all policies. Reference cases reproduce the
previous two-pass output to 12 decimal places. The optional physical-policy
check also solves each player's linear payoff equation using the computed
efforts, physical diffusion, and actual effort costs (without the HJB's
artificial diffusion). Its initial value must agree with the nonlinear HJB
value within 0.002 in prize-normalized units. This is a necessary numerical
consistency check, not an equilibrium-existence or uniqueness theorem.

`verified_policy_samples.csv` preserves the finest checked results, including
failures, and takes precedence on resume. `regions_before_left_audit.csv`
preserves the earlier classification. Current counts, area changes and
targeted-search qualifications are in `metadata.json` under
`targeted_left_audit`. A successful direct comparison can settle a cell;
a failed potentially superior design cannot be called inferior.
The optional `--coverage` pass checks the optimistic envelope candidates
left after failed-value tests. Candidate selection uses the same analytical
loss floors as classification. A single numerical mesh has no estimated
resolution error yet and cannot certify a winning family; its analytical
output bound remains as an optimistic screen.
`left_direct_comparisons.csv` distinguishes the point-estimate preference
from the conservative classification and its unresolved reason.

The targeted pass and two coverage rounds perform 232 finest physical-policy
checks: 109 pass and 123 fail the numerical consistency tests. They do not
settle the strong weak-signal branch. The updated panel has 1,179 unresolved
cells, occupying 4.95% of the common linear comparison rectangle versus
2.56% before this audit. This increase reflects previously unexcluded
competitive designs and the single-mesh error correction, not evidence for
a new economic regime. The updated figure retains the uncertainty instead
of treating solver failure as inferior payoff. Further progress requires a
more accurate weak-signal equilibrium/payoff calculation or tighter valid
payoff bounds, not more pointwise prize/deadline comparisons alone.
