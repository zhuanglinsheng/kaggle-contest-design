# General delay costs: an exogenous low-cost, low-noise region

> 历史研究存档（2026-10-03）。保留原推导以便恢复和核查；本文中的“已证明”“已闭合”不代表当前审定状态。当前方向与结果状态以上级目录 CASE2_AND_REGION_GAP.md 和 PROOF_LEDGER.md 为准。不得单独引用本存档宣布精确联合设计已经完成。

This research note does not modify the manuscript. It establishes strict
dominance of the optimized full-disclosure boundary, not unconditional
attainment at zero precision. It uses the exact output bounds developed in
EXOGENOUS_UTILITY_JOINT_DESIGN.md, including its corrected constant kappa < 8.

## Statement

Fix S0 > 0 and symmetric contestants with initial perceived gap zero. Let u
be C2, strictly increasing and concave on nonnegative output. Let D be C1,
nondecreasing and convex, with D(0) = 0 and D not identically zero.
Convexity then implies unboundedness. Neither utility boundedness, superlinear
delay growth, nor a restriction to linear delay is needed for the strict
comparison below. For compactness of prize and deadline, additionally assume
either D(T)/T tends to infinity OR u'(B) tends to zero as B tends to infinity.
The second alternative allows arbitrary positive linear delay costs.

There exist positive constants c0 and sigma0, depending only on the given
u, D and S0, such that, for every 0 < c <= c0 and 0 < sigma <= sigma0, some
positive finite-precision design has payoff strictly exceeding

\[
\sup_{\theta,T\ge0}
\{u(\mathcal B_\infty(\theta,T))-\theta-D(T)\}.
\]

Thus full disclosure cannot be a global joint optimum. If the unrestricted
joint optimum is attained, every maximizer has positive prize and deadline
and finite precision; zero precision is not excluded.

If additionally D'(0) = 0, the optimized full-disclosure value itself
strictly exceeds the immediate-termination payoff whenever sigma > 0.
The result therefore does not require the full-disclosure benchmark to
collapse to immediate termination. It does not prove a positive prize at
that benchmark.

## Proof: auxiliary quantities are used only here

Write b = sqrt(S0/(2 pi)), v = u'(b), and

\[
g=u(51b/50)-u(b)>0.
\]

Choose t > 0 satisfying D(t) = g/4. Such a point exists by continuity and
unboundedness of D; all these objects depend only on primitives. Set

\[
c_0=\frac{25gt}{4S_0},\qquad
\sigma_0^2=
\min\left\{\frac{S_0}{225t},
\frac{g\sqrt{2\pi S_0}}{34vt}\right\}.
\]

These are conservative sufficient thresholds, not exact regime boundaries.
They are not assumptions on utility evaluated at an optimized design.

### 1. A feasible finite-precision design

Take theta = c S0/(25t), T = t, and lambda = 1/(S0 t). The hypotheses of
the previously established exact finite-precision output lower bound hold:
theta T/(c S0) = 1/25 and sigma^2 T/S0 <= 1/225. Consequently

\[
\mathcal B(0,\theta,t,\lambda;S_0)
\ge b+\frac{3b}{100\sqrt{226/225}}
>\frac{51}{50}b.
\]

Since theta <= g/4 and D(t) = g/4, its payoff is strictly greater than
u(b) + g/2. This is an inequality for exact equilibrium output, not a
truncated prize expansion.

### 2. The entire optimized full-disclosure boundary

The exact upper bound with kappa < 8 and square-root concavity imply

\[
\mathcal B_\infty(\theta,T)
\le b+\frac{17\sigma^2 T}{2\sqrt{2\pi S_0}}.
\]

Applying the tangent upper bound for concave u gives

\[
u(\mathcal B_\infty)-\theta-D(T)
\le u(b)-\theta+
\frac{17v\sigma^2T}{2\sqrt{2\pi S_0}}-D(T).
\]

The coefficient of T is at most g/(4t). For T <= t, the last two terms
are at most g/4, since D >= 0. For T >= t, convexity and D(0) = 0 imply
D(T)/T >= D(t)/t = g/(4t), so they are at most zero. Therefore

\[
\sup_{\theta,T\ge0}
\{u(\mathcal B_\infty)-\theta-D(T)\}
\le u(b)+g/4.
\]

The finite candidate beats this bound by more than g/4, even after prize
and deadline have been reoptimized at full disclosure. No assumption on
the boundary-optimal prize or deadline has been used.

### 3. Nondegenerate waiting under full disclosure

At theta = 0, effort vanishes and output is sqrt((S0 + sigma^2 T)/(2 pi)),
independently of precision. If D'(0) = 0, the right derivative of payoff
at T = 0 is

\[
\frac{u'(b)\sigma^2}{2\sqrt{2\pi S_0}}>0.
\]

Thus the optimized full-disclosure payoff is above u(b), despite being
strictly below the finite candidate. The proof does not establish that
the full-disclosure optimizer uses a positive prize.

### 4. Compactness in prize and deadline, and the remaining boundary issue

The exact energy output bound, the tangent inequality for u, and Young's
inequality yield, uniformly over precision,

\[
\begin{aligned}
u(\mathcal B)-\theta-D(T)
&\le u(b)+v\left(\sqrt{\frac{S_0+\sigma^2T}{2\pi}}-b\right)
+v\sqrt{\frac{2\theta T}{c}}-\theta-D(T)\\
&\le u(b)+v\left(\sqrt{\frac{S_0+\sigma^2T}{2\pi}}-b\right)
+\frac{v^2T}{c}-\frac\theta2-D(T).
\end{aligned}
\]

For each fixed positive c and sigma, superlinear D bounds both prize and
deadline on any upper payoff level set. If the previously established
compact-uniform high-precision convergence is applied on that set, the
strict gap above also excludes lambda tending to infinity along any
maximizing sequence.

This does not by itself prove attainment: a maximizing sequence may have
lambda tending to zero. The existing analytic zero-precision continuation
argument only covers theta T/(c S0) <= 1/25. Our constructed candidate
satisfies that inequality, but the entire upper level set need not. A
general extension/upper-semicontinuity result at lambda = 0 is still needed
to assert unconditional existence of an unrestricted global maximizer.
Do not restore a cost-slope restriction merely to hide this gap.

There is a more natural alternative to superlinear delay. Suppose u'(B)
tends to zero. Concavity gives u(B)/B tending to zero (additive utility
normalizations do not affect this limit). Since D is convex and nonzero,
there are fixed d > 0 and C >= 0 with D(T) >= d T - C. Write n = theta + T
for this proof only. The output bound implies B <= b + C1 sqrt(n) + C2 n,
where the constants are finite for each fixed positive c and sigma. Thus
u(B) = o(n), whereas theta + D(T) >= min(1,d) n - C. Payoff tends to minus
infinity uniformly in precision whenever theta + T tends to infinity.
This proves compactness without a superlinear-delay assumption or any
lower/upper interval restricting the given cost slope.

## A less conservative, computable certificate (appendix only)

The fixed 2-percent output increment and quarter-budget allocation above
are convenient for the existence of an exogenous rectangle, not optimal
constants. They can be avoided without solving the equilibrium again.
For fixed sigma > 0, define an upper certificate depending only on primitives:

\[
H_\sigma=\sup_{s\ge0}
\left\{u\!\left(\sqrt{\frac{S_0+\sigma^2s}{2\pi}}
+\frac{8\sigma^2s}{\sqrt{2\pi S_0}}\right)-D(s)\right\}.
\]

This number is finite under either compactness alternative: the argument
of u grows at most linearly in s; either u is sublinear and D eventually
at least linear, or u is at most linear and D is superlinear. It bounds
the entire optimized full-disclosure payoff after dropping the nonnegative
prize cost. Define the sufficient cost threshold

\[
\begin{split}
C_\sigma=\sup_{\substack{0<t\le S_0/(225\sigma^2)\\0<e\le1/25}}
\frac{t}{eS_0}\Bigg[
u\!\left(\sqrt{\frac{S_0+\sigma^2t}{2\pi}}
+\frac{3eS_0}{4\sqrt{2\pi(S_0+\sigma^2t)}}\right)
-D(t)-H_\sigma\Bigg].
\end{split}
\]

If 0 < c < C_sigma, a pair (t,e) in this supremum supplies a strict
comparison: use theta = c e S0/t and lambda = 1/(S0 t). The finite-precision
lower bound exceeds H_sigma after subtracting prize and delay. Supremum
attainment in this auxiliary test is not required, since the inequality
is strict. The threshold depends on u,D,S0,sigma, not on c or an unknown
joint optimizer. The variables s,t,e are eliminated by the computation;
no assumption is imposed on u at their associated outputs.

This is a low-dimensional sufficient-region calculation, not an exact
regime boundary, and it belongs in the proof/research material rather than
the main proposition. Using the exact concave utility in H_sigma is at
least as favorable as the earlier tangent bound. Allowing t and e to vary
includes the earlier constructed candidate, so the new certificate recovers
its strict-dominance region and may enlarge it. Do not describe ordinary
grid maximization as a rigorous upper certificate for H_sigma; numerical
implementation requires a verified upper bound there and feasible lower
values for the candidate optimization.

## Audit of the no-signal gradient equations

The manuscript's no-signal subsection currently defines p = U_i,x and
r = -U_j,x but writes plus cross-terms in both reversed-time gradient
equations. Differentiating the displayed original HJB instead gives

\[
p_\tau=\frac a2p_{xx}+e\partial_x(p^2/2-pr),\qquad
r_\tau=\frac a2r_{xx}+e\partial_x(pr-r^2/2).
\]

The signs in the manuscript must be corrected when editing is authorized.
The local analytic estimate is not invalidated by this correction: product
norms give (P+2R)Np+(2P+R)Nr <= 2(P+R)(Np+Nr), exactly the same
2 e M N bound as before. Positivity follows, where a smooth solution
exists, from the component equations with zeroth-order coefficients.

However, at a = 0, the first-order coefficient matrix divided by e is

\[
\begin{pmatrix}p-r&-p\\r&p-r\end{pmatrix},
\]

whose eigenvalues are p-r plus/minus i sqrt(pr) when p,r > 0. Thus this
is not a strictly hyperbolic transport system to which a routine global
characteristic argument applies. This fact does not prove that the specific
Gaussian-terminal game lacks an equilibrium, or that its particular
vanishing-diffusion limit fails. It does establish that global no-signal
continuation cannot simply be asserted from the local analytic estimate.
The existence/selection issue at zero precision remains substantive.

## Interpretation and limits

The sufficient regime is low effort cost and low innovation noise, holding
preferences, initial uncertainty and delay costs fixed. At finite precision,
a prize proportional to c sustains a fixed positive output improvement.
At full disclosure, the uniform exact gain bound shrinks with sigma^2.
General convex delay costs prevent the full-disclosure organizer from
offsetting that loss by arbitrarily extending the deadline.

This is a genuine joint-value comparison, not a fixed-design comparison.
It gives a nonempty exogenous region for every admissible u and D, but not
a tight boundary or an interior positive-precision result. The special
u-curvature bound discussed in chat is unnecessary for this comparison.
