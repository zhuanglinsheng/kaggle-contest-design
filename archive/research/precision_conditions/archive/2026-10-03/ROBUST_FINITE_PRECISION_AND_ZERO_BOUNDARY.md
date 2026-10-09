# Uniform exclusion of high precision and the zero-precision boundary

> 历史研究存档（2026-10-03）。保留原推导以便恢复和核查；本文中的“已证明”“已闭合”不代表当前审定状态。当前方向与结果状态以上级目录 CASE2_AND_REGION_GAP.md 和 PROOF_LEDGER.md 为准。不得单独引用本存档宣布精确联合设计已经完成。

Research continuation; manuscript unchanged. This note separates results that
follow from the exact positive-precision equilibria from a still-unproved
extension to a no-signal Markov-perfect equilibrium.

## 1. Maintained results and hypotheses

Use the exact positive-precision output, energy bound, continuous dependence
on positive parameters, and compact-uniform high-precision limit from the
manuscript. Use the corrected upper constant kappa < 8 and the primitive
low-cost/low-noise comparison in GENERAL_DELAY_LOW_COST_LOW_NOISE.md.

Assume symmetric costs, zero initial posterior mean, S0 > 0, u C2 with
u' > 0 and u'' <= 0, and D C1, nondecreasing, convex, D(0) = 0,
D not identically zero. For coercivity assume either u'(B) -> 0 or
D(T)/T -> infinity. These assumptions are external to the design problem.

For sufficiently small positive c and sigma, the preceding comparison gives
a feasible design (theta_f,T_f,lambda_f) with lambda_f > 0, whose payoff J_f
exceeds the entire optimized full-disclosure value H. Write delta = J_f-H > 0
only in this proof. H is a benchmark value, not an assumed primitive
condition or a point at which a new utility restriction is imposed.

For theta,T >= 0 and lambda > 0 write J for exact net payoff, and

\[
J^*=\sup_{\theta,T\ge0,\lambda>0}J(\theta,T,\lambda).
\]

This domain intentionally does not assume a no-signal equilibrium.

## 2. Robust finite-precision conclusion

There are positive finite constants epsilon0, theta_min, theta_max,
T_min, T_max, and Lambda such that every positive-precision design with
J >= J* - epsilon0 satisfies

\[
\theta_{\min}\le\theta\le\theta_{\max},\qquad
T_{\min}\le T\le T_{\max},\qquad 0<\lambda<\Lambda.
\]

Consequently all sufficiently accurate maximizing sequences have uniformly
bounded precision. This is stronger than a pointwise comparison at infinity
and does not assume attainment at lambda = 0. Constants can depend on fixed
primitives, including c and sigma, but never on a chosen maximizing sequence.

### Proof

The coercivity proof in the preceding note gives a compact prize/deadline
box containing every design with J >= J_f-delta/4, uniformly over lambda.
It also proves J* finite. Let epsilon0 = delta/4. Any epsilon0-optimal design
has J >= J* - epsilon0 >= J_f-delta/4 = H+3delta/4.

On the compact box, uniform high-precision convergence implies that for
lambda >= Lambda,

\[
J(\theta,T,\lambda)\le J_\infty(\theta,T)+\delta/4
\le H+\delta/4.
\]

Outside the box the near-optimal payoff threshold is already impossible.
Thus no epsilon0-optimal design has lambda >= Lambda.

For explicit separation from zero prize and deadline, write
b = sqrt(S0/(2 pi)), v = u'(b), and b_T = sqrt((S0+sigma^2 T)/(2 pi)).
Monotonicity and concavity, and the exact energy output bound, imply

\[
J\le u(b_T)-D(T)+v\sqrt{2\theta T/c}-\theta
\le H+v\sqrt{2\theta T/c}.
\]

The second inequality holds because theta = 0 has the same payoff at full
disclosure and at finite precision. Hence every epsilon0-optimal design has

\[
\theta T\ge\frac{9c\delta^2}{32v^2}>0.
\]

Combining this product lower bound with the two compact upper bounds yields
strictly positive lower bounds for theta and T. This proves the claim.

## 3. Exact attainment dichotomy on the positive-precision domain

Either the supremum J* is attained at a design with positive finite precision,
or every maximizing sequence has lambda tending to zero. This statement is
exhaustive, although in the first case there might also be maximizing
sequences tending to zero.

Indeed, all maximizing sequences eventually lie in the set of Section 2.
If one has a subsequence with lambda bounded away from zero, compactness
gives a further subsequence converging to positive theta,T,lambda. Continuity
of exact positive-precision output implies attainment of J*. If no positive
precision maximizer exists, such a subsequence is impossible; therefore
lambda must tend to zero along every maximizing sequence.

This is not a proof of the original problem's attainment on lambda >= 0.
Assigning a limsup payoff to lambda = 0 would close a relaxed optimization
problem by definition, but would not construct a no-signal equilibrium.
Likewise, imposing an arbitrary positive precision floor would change the
feasible set. Neither workaround is used here.

## 4. Why zero precision is not an automatic transport limit

With p = U_i,x and r = -U_j,x, the corrected reversed-time equations are

\[
p_\tau=\frac a2p_{xx}+e\partial_x(p^2/2-pr),\qquad
r_\tau=\frac a2r_{xx}+e\partial_x(pr-r^2/2).
\]

Here e = theta T/(c S0) in the scaled coordinates and a is the scaled
filtering diffusion variance. At a smooth point with p,r > 0, freeze the
coefficients of the linearized principal part. Its first-derivative matrix is

\[
e\begin{pmatrix}p-r&-p\\r&p-r\end{pmatrix}.
\]

Its eigenvalues are e[(p-r) +/- i sqrt(pr)]. For a spatial Fourier mode
with wavenumber k, the larger real part of the temporal growth rate is

\[
-\frac a2k^2+e\sqrt{pr}|k|.
\]

For a > 0 its maximum is e^2 pr/(2a); for a = 0 it grows without bound
as |k| increases. Thus a routine hyperbolic-characteristic continuation or
a diffusion-independent Sobolev stability estimate is not available. The
local analytic proof handles this growth by spending analytic radius, which
explains its small-e restriction.

This frozen-coefficient calculation is a diagnostic of the principal
system, not a proof that the particular Gaussian-terminal problem lacks an
equilibrium or that its scalar output fails to converge. Those stronger
conclusions remain unproved. Finite-grid numerical stability, especially
with artificial viscosity, cannot substitute for the missing boundary proof.

## 5. What can be stated now and what cannot

Established conditional on the manuscript's exact positive-precision bounds:
- a primitive, nonempty low-cost/low-noise region strictly beating the entire
  optimized full-disclosure boundary;
- uniformly finite precision for every sufficiently near-optimal design;
- compactness away from zero prize and zero deadline;
- attainment at positive finite precision unless all maximizing sequences
  approach no disclosure.

Not established:
- a global no-signal Markov-perfect equilibrium for arbitrary prize/deadline;
- continuity or sufficient upper semicontinuity of its payoff at lambda = 0;
- strictly positive optimal precision;
- an exact parameter regime boundary or a converse perfect-disclosure region.

The rigorous interim main result is therefore uniform suboptimality of high
precision under joint design. Do not state unconditional existence of the
original unrestricted global optimum until the zero-boundary issue is resolved.
