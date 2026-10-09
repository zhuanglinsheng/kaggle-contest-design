# Optimal-policy generation checks

- All 46 paths (23 contests at each S0) retain the variance-only experiment's
  initial draws, variance paths, and diffusion shocks. The original hourly
  belief recursion and the original readout noise were checked step by step.
- Replacing the policy callback by the original first-order rule reproduces
  every returned path and submission time exactly (unit test).
- The nonlinear callback is evaluated at each pre-update perceived state;
  it returns a genuinely different policy from the first-order kernel.
- All six HJB policy solutions passed spatial/time grid refinement from
  (0.2, 1/48 day) to (0.1, 1/96 day). The largest error divided by reference
  peak effort was 0.004258386 (less than 0.43%) on the evaluation grid.
- The HJB nonlinearity and terminal conditions match the coupled normalized
  value equations in the paper. Policies are derived from value gradients.
- All simulated perceived states remain inside the solved state grid.
  The original thinning bound remains valid for every policy/horizon.
- All 19 repository unit tests passed. No Bayesian sampling was performed.

These are numerical checks of the theoretical HJB policy, not a claim that
the preserved held-readout approximation is an exact continuous-signal filter.
