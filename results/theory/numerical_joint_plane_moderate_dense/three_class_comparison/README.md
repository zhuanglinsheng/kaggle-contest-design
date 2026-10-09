# Three-class decisions for previously unresolved points

This is a separate exploratory comparison, not a replacement for the guarded
classification or the paper figure. The preferences, delay cost, output cache,
parameter grid and rejected numerical solves are unchanged. This reporting pass
performs no new equilibrium solves.

For each of the 1,179 formerly unresolved points, compare the exact optimized
zero-prize payoff with the lowest point-estimate losses among successfully
evaluated positive-prize finite-precision and full-disclosure candidates. Loss
is `3 - net payoff`; smaller is better. Assign the corresponding one of three
labels. The point-estimate decisions are 1,144 zero-prize, 19 full-disclosure and
16 finite-precision points. At ties the ordering prefers zero prize, then full
disclosure, then finite precision; margins are retained in the table.

These are best-evaluated-design decisions, not global optimum certificates.
Failed designs are not assumed inferior. At 926 points a failed candidate's
analytical payoff screen still allows it to beat the comparison upper envelope.
At 16 points ranking conservative upper loss envelopes chooses a different
family from ranking point estimates. All 1,179 points remain unverified under
the original guarded criteria, which are preserved in the parent `regions.csv`.
Even that guarded search is a numerical design-grid comparison, not a proof
over all continuous designs.

`unresolved_point_decisions.csv` records the three-way decisions, payoff margins,
conservative-envelope decisions and competitive-failure flags separately.
`three_class_regions.csv` keeps previously classified cells unchanged and uses
these point-estimate decisions only for previously unresolved cells. The panel
includes an explicit unverified-results note.

Update: the paper now uses this three-class panel. Its two displayed region
envelopes use smoothing splines, with target RMS fitting residual 0.0035 in
effort cost. Fills and outlines use the same curves; no monotonicity is imposed.
Tiny isolated grid islands are not represented by this envelope illustration.
The finite envelope is clipped below the prize-exit envelope and extended at
its right endpoint by its nonpositive tangent only until the measured cost
floor. The CSV decisions and original guarded classifications remain unchanged.
Smoothing is presentation only, not a new boundary estimate or certification.
The initial segment without full-disclosure cells uses exactly one shared
finite/zero boundary, rather than independent fits. Separation is smoothly
blended across the first risk-grid interval with full-disclosure cells. The
dashed line is drawn only after separation, preventing overprinting artifacts.
