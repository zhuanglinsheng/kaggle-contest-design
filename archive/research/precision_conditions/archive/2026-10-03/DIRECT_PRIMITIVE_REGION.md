# A direct primitive region for finite optimal feedback precision

> 历史研究存档（2026-10-03）。保留原推导以便恢复和核查；本文中的“已证明”“已闭合”不代表当前审定状态。当前方向与结果状态以上级目录 CASE2_AND_REGION_GAP.md 和 PROOF_LEDGER.md 为准。不得单独引用本存档宣布精确联合设计已经完成。

## 1. Why an additional restriction is necessary

With unrestricted increasing concave utility `u` and unrestricted convex
delay cost `D`, the joint-design boundary depends on the global shapes of both
functions. This dependence cannot be summarized by `c`, `S0`, and `sigma`
alone. For example, multiplying `u` by an arbitrarily small positive constant
makes every positive prize unprofitable without changing any technological
parameter. Likewise, rescaling `D` changes the cheapest deadline associated
with a given cumulative incentive exposure.

The objects `C(e)` and `bar e` in the current draft encode these two functional
optimizations exactly, but they do not deliver a finite-dimensional regime.
To obtain a direct parameter region, impose the convex power specification

\[
D(T)=dT^p,\qquad d>0,\quad p\geq1.
\tag{1}
\]

This contains linear delay cost as `p=1` and permits strictly convex delay
cost for `p>1`.

## 2. Direct sufficient region

Maintain symmetric effort cost `c>0`, symmetric initial belief, and initial
uncertainty `S0>0`. Let

\[
B_0=\sqrt{\frac{S_0}{2\pi}},\qquad
R=\sup_{B\geq0}u(B)-u(B_0),
\tag{2}
\]

where `u` is continuously differentiable, strictly increasing, concave, and
bounded above. The quantity `R` is the remaining value available above the
immediate-termination benchmark. Let

\[
G=u\!\left(\frac{76}{75}B_0\right)-u(B_0).
\tag{3}
\]

Thus `G` is the value of a 1.33 percent improvement over baseline output.
Both `R` and `G` are exogenous features of the primitive utility function at
the primitive initial environment; neither contains a design choice.

Suppose

\[
d\geq
\frac{25^p p^p}{(p+1)^{p+1}}
\frac{R^{p+1}}{(cS_0)^p}
\tag{4}
\]

and

\[
G>
(p+1)p^{-p/(p+1)}d^{1/(p+1)}
\left(\frac{cS_0}{50}\right)^{p/(p+1)}.
\tag{5}
\]

To state an explicit innovation-risk bound, let the certified zero-risk
surplus be

\[
M:=G-(p+1)p^{-p/(p+1)}d^{1/(p+1)}
\left(\frac{cS_0}{50}\right)^{p/(p+1)}>0
\tag{6}
\]

and let `v=u'(B0)`. If `p=1`, suppose

\[
\sigma^2<\min\left\{
S_0\left(\frac{50d}{cS_0}\right)^{1/2},
\frac{2d\sqrt{2\pi S_0}}{17v}
\right\}.
\tag{7a}
\]

If `p>1`, suppose

\[
\sigma^2<\min\left\{
S_0\left(\frac{50pd}{cS_0}\right)^{1/(p+1)},
\frac{2p\sqrt{2\pi S_0}}{17v}
d^{1/p}\left(\frac{M}{p-1}\right)^{(p-1)/p}
\right\}.
\tag{7b}
\]

Then the unrestricted exact joint-design problem attains a global maximum and
every maximizer has positive prize, positive deadline, and finite feedback
precision. Zero feedback precision remains admissible.

Conditions (4)--(7) are a direct primitive region. They contain no optimized
prize, deadline, precision, cumulative exposure, implementation-cost
function, or affordability frontier.

## 3. Derivation

### 3.1 All payoff-relevant designs remain in the regular equilibrium region

Any design capable of improving on immediate termination must satisfy

\[
\theta+dT^p<R.
\tag{8}
\]

The maximum cumulative exposure under this inequality is

\[
\max_{T\geq0}T(R-dT^p)
=\frac{pR}{p+1}
\left(\frac{R}{(p+1)d}\right)^{1/p}.
\tag{9}
\]

Raising (9) to the power `p` shows that condition (4) is exactly sufficient
for

\[
\frac{\theta T}{cS_0}\leq\frac1{25}
\tag{10}
\]

throughout the payoff-relevant set. Hence the analytic no-signal equilibrium
and continuity result applies everywhere a global maximizer could lie.

### 3.2 A finite-precision design beats immediate termination

Choose the proof candidate's cumulative exposure at one half of the certified
regularity bound,

\[
e_0:=\theta_0T_0=\frac{cS_0}{50}.
\tag{11}
\]

For the power delay cost, the least costly prize--deadline implementation is

\[
T_0=\left(\frac{e_0}{pd}\right)^{1/(p+1)},
\qquad
\theta_0=\frac{e_0}{T_0},
\tag{12}
\]

and its total cost is

\[
\theta_0+dT_0^p
=(p+1)p^{-p/(p+1)}d^{1/(p+1)}e_0^{p/(p+1)}.
\tag{13}
\]

Set `lambda_0=1/(S0*T0)`. The exact finite-precision lower bound applies
because `e0/(c*S0)=1/50<1/25` and, for all sufficiently small `sigma`,
`sigma^2*T0<=S0`. It gives

\[
\mathcal B(0,\theta_0,T_0,\lambda_0;S_0)
\geq
\sqrt{\frac{S_0+\sigma^2T_0}{2\pi}}
+\frac{2e_0}{3c\sqrt{2\pi(S_0+\sigma^2T_0)}}.
\tag{14}
\]

As `sigma` decreases to zero, the right-hand side converges to

\[
B_0+\frac{2e_0}{3c\sqrt{2\pi S_0}}
=\frac{76}{75}B_0.
\tag{15}
\]

For `delta=sigma^2*T0/S0`, the right-hand side of (14), divided by `B0`, is

\[
\sqrt{1+\delta}+\frac{1}{75\sqrt{1+\delta}}.
\tag{16}
\]

This expression is increasing for `delta>=0`. The first bound in (7a) or
(7b) ensures `delta<1`, as required by the exact finite-precision estimate,
and equations (5)--(6) therefore imply that the candidate payoff is at least
`u(B0)+M`.

### 3.3 Full disclosure and attainment

The exact high-precision bound implies

\[
\limsup_{\lambda\to\infty}\mathcal B
\leq
\sqrt{\frac{S_0+\sigma^2T}{2\pi}}
+\frac{8\sigma^2T}{\sqrt{2\pi S_0}}.
\tag{17}
\]

Concavity of the square root and of `u` gives the full-disclosure payoff bound

\[
u(B_0)+\sup_{T\geq0}\left\{
\frac{17v\sigma^2}{2\sqrt{2\pi S_0}}T-dT^p
\right\}.
\tag{18}
\]

For `p=1`, the supremum in (18) is zero under the second bound in (7a). For
`p>1`, it equals

\[
(p-1)p^{-p/(p-1)}
\left(\frac{17v\sigma^2}{2\sqrt{2\pi S_0}}\right)^{p/(p-1)}
d^{-1/(p-1)}.
\tag{19}
\]

The second bound in (7b) makes (19) strictly smaller than `M`. Hence the
finite-precision candidate strictly exceeds the entire reoptimized
full-disclosure boundary.

Condition (4) supplies the no-signal equilibrium and continuity on the entire
payoff-relevant compact set. Uniform high-precision convergence excludes a
finite upper tail of precisions. Weierstrass' theorem then gives global
attainment. The strict surplus also excludes zero prize and zero deadline.

## 4. When the region is nonempty

For a fixed utility function, the interval of admissible delay scales is

\[
\frac{25^p p^pR^{p+1}}
{(p+1)^{p+1}(cS_0)^p}
\leq d<
\frac{50^p p^pG^{p+1}}
{(p+1)^{p+1}(cS_0)^p}.
\tag{20}
\]

This interval is nonempty exactly when

\[
G>2^{-p/(p+1)}R.
\tag{21}
\]

Equation (21) exposes the substantive economic restriction hidden by the
composite conditions in the earlier draft. Utility must realize a sufficiently
large share of its remaining value from a 1.33 percent output improvement at
the initial benchmark. The required share is `1/sqrt(2)` under linear delay
cost and decreases toward one half as delay cost becomes more convex.

The region is nonempty. For example, take

\[
c=1,\qquad S_0=2\pi,\qquad p=2,\qquad d=3,
\qquad u(B)=1-e^{-100(B-1)}.
\]

Then `B0=R=1`, `G=1-e^{-4/3}=0.736402...`, and the interval in
(20) is

\[
2.345397\ldots\leq d<3.746475\ldots .
\]

The certified surplus is `M=0.0525727...`; equation (7b) permits, for
example, `sigma=0.05`. Thus this is an analytical nonempty example, not a
numerical optimization claim.

This is a strong saturation condition. It is not implied by ordinary
concavity. Its severity comes from the currently proved no-signal regularity
bound (8). A materially broader primitive region requires a stronger global
existence result at zero precision, not another change of notation.

## 5. Implication for the manuscript

The direct result is mathematically cleaner than the current proposition, but
it reveals a tradeoff that should not be hidden:

1. Keeping completely general `D` requires functional objects such as `C(e)`
   and `bar e`.
2. Eliminating those objects requires a quantitative cost specification such
   as (1), or comparable global upper and lower shape bounds on `D`.
3. Under the presently proved zero-precision equilibrium bound, the resulting
   utility-saturation condition is strong.

Accordingly, the current composite statement should not be presented as a
simple regime boundary. A defensible main-text proposition must either adopt
the explicit power-delay region above and acknowledge its conservative
saturation condition, or first improve the zero-precision equilibrium theorem.
