Direct effort-policy comparison for PaperJK7

Run: .venv/bin/python -m pipeline.synthetic.effort_comparison
For figure styling only, append --plot-only to reuse curves.npz and retain
all numerical metrics without resolving the equilibria. Summary.json records
the rendering source hash separately from the original calculation source hash.
Then copy effort_policy_comparison.pdf from this directory into paper/ and
run make paper. Figures 4 and 5 float independently; page positions are
determined by LaTeX rather than a local float barrier.

Only player i is displayed, in a 2-by-3 figure. Top-row theta is 1, 2, 4;
bottom-row panels independently change only lambda to 0.1 or 10, or c_j
to 0.6, with theta=1. All remaining primitives use the Chapter 4 synthetic
baseline, including non-steady S0=4. Both players remain in the nonlinear
equilibrium calculation. There is no re-estimation.

Curves show effort at t/T=0.5 on z=y_tilde/sqrt(Q_t) in [-3,3]. The manuscript
error metric takes, at each time, max_z|m_i^NL-m_i^(1)| / max_z m_i^NL,
then maximizes over the 378 evaluation times. The 601 state points and
extra near-terminal times are saved in theta_*_audit.npz. Summary.json
also retains midpoint errors and whole-grid peak-normalized errors, which
have different denominators and should not be confused.

Grid checks halve dimensionless state spacing from .00625 to .003125
(from about .004191 to .002096 for lambda=10 to resolve its narrower
terminal reward), and double base time resolution from 2048 to 4096.
Adaptive points additionally resolve the initial filtering layer. Maximum
grid differences are under 0.40% with time-specific peak normalization.

The original empirical upper-decile trial in the parent directory was not
accepted: its policy failed grid convergence. The user then authorized a
moderate-theta experiment. This controlled comparison does not validate the
full empirical parameter range or optimized counterfactual designs. Parent
directory exploratory files are not manuscript evidence.
