# An economic regime boundary for finite feedback precision

> 历史研究存档（2026-10-03）。保留原推导以便恢复和核查；本文中的“已证明”“已闭合”不代表当前审定状态。当前方向与结果状态以上级目录 CASE2_AND_REGION_GAP.md 和 PROOF_LEDGER.md 为准。不得单独引用本存档宣布精确联合设计已经完成。

## 1. Purpose

This note replaces the candidate-specific conditions in the current
Proposition 1 with a primitive economic condition. The result concerns the
exact equilibrium output and the unrestricted joint choice of prize,
deadline, and feedback precision. It does not use a precision cost or a
truncated approximation.

The boundary is in innovation risk. When innovation risk is sufficiently
small, full disclosure reveals the likely winner early and destroys the value
of continued competition. If buying cumulative incentives under noisy
feedback is profitable, the joint optimum must then have finite precision.

## 2. Primitive objects

Maintain symmetric effort costs \(c_i=c_j=c>0\), initial perceived gap zero,
and initial uncertainty \(S_0>0\). Let \(u\) be continuously differentiable,
strictly increasing, concave, and bounded above. Let \(D\) be continuously
differentiable, nondecreasing, convex, satisfy \(D(0)=0\), and not be
identically zero. Write

\[
b_0:=\sqrt{\frac{S_0}{2\pi}},
\qquad
R:=\sup_{B\geq0}u(B)-u(b_0).
\tag{1}
\]

For \(e>0\), define

\[
C(e):=\min_{T>0}\left\{\frac eT+D(T)\right\}.
\tag{2}
\]

Here \(e=\theta T\) is cumulative incentive exposure. Thus \(C(e)\) is the
lowest combined prize and delay cost capable of delivering exposure \(e\).
The minimum exists: \(e/T\to\infty\) as \(T\downarrow0\), while convexity,
nonnegativity, and \(D\not\equiv0\) imply \(D(T)\to\infty\) as
\(T\to\infty\).

The largest exposure that the organizer could rationally finance relative
to immediate termination is

\[
\bar e
:=\max_{\substack{\theta,T\geq0\\ \theta+D(T)\leq R}}\theta T
=\max_{T\geq0}T\,[R-D(T)]_+.
\tag{3}
\]

Both \(C\) and \(\bar e\) are determined before the design is chosen. They
contain no optimized prize, deadline, or precision.

## 3. Exact low-innovation theorem

### Theorem

Suppose

\[
\bar e\leq\frac{cS_0}{25}
\tag{4}
\]

and cumulative incentives are profitable under noisy feedback in the
zero-innovation limit:

\[
\sup_{e>0}
\left\{
u\!\left(
b_0+\frac{2e}{3c\sqrt{2\pi S_0}}
\right)-u(b_0)-C(e)
\right\}>0.
\tag{5}
\]

Then there is a primitive threshold
\(\bar\sigma=\bar\sigma(u,D,c,S_0)>0\) such that, for every
\(0<\sigma<\bar\sigma\), the unrestricted joint problem

\[
\max_{\theta,T,\lambda\geq0}
\left\{
u\!\left(\mathcal B(0,\theta,T,\lambda;S_0)\right)
-\theta-D(T)
\right\}
\tag{6}
\]

attains a global maximum. Every global maximizer satisfies

\[
\theta^*>0,\qquad T^*>0,\qquad 0\leq\lambda^*<\infty.
\tag{7}
\]

The conclusion allows no disclosure. Its substantive content is that full
disclosure cannot be jointly optimal, even after the organizer reoptimizes
both the prize and the deadline.

### Interpretation

Condition (4) says that the largest cumulative incentive exposure worth
financing is small relative to initial uncertainty and effort cost. It is
the exact economic object that the current slope restriction on \(D'(0)\)
only bounds indirectly.

Condition (5) says that there is some cumulative exposure whose output value
under noisy feedback exceeds the cheapest prize--deadline cost of providing
it. Unlike the current condition (10), it does not select a prize, deadline,
or precision inside the assumption.

The regime boundary is \(\sigma=\bar\sigma\). Below it, full disclosure
resolves the ranking mainly through the initial gap, while finite feedback
can preserve a profitable amount of competition.

## 4. Proof

### Step 1: condition (5) supplies a primitive finite-feedback candidate

Choose \(e_0>0\) and \(\eta>0\) such that

\[
u\!\left(
b_0+\frac{2e_0}{3c\sqrt{2\pi S_0}}
\right)-C(e_0)
\geq u(b_0)+3\eta.
\tag{8}
\]

Because the utility gain in (8) is at most \(R\), it follows that
\(C(e_0)<R\). Let \(T_0\) attain the minimum in (2) and set

\[
\theta_0:=\frac{e_0}{T_0},
\qquad
\lambda_0:=\frac1{S_0T_0}.
\tag{9}
\]

Then \(\theta_0+D(T_0)=C(e_0)<R\), so equation (3) gives
\(e_0\leq\bar e\). By (4),

\[
\frac{\theta_0T_0}{cS_0}
=\frac{e_0}{cS_0}\leq\frac1{25}.
\tag{10}
\]

For sufficiently small \(\sigma>0\), also
\(S_0\geq\sigma^2T_0\). The exact finite-precision lower bound at
\(\lambda_0=1/(S_0T_0)\) therefore yields

\[
\mathcal B(0,\theta_0,T_0,\lambda_0;S_0)
\geq
\sqrt{\frac{S_0+\sigma^2T_0}{2\pi}}
+\frac{2e_0}
{3c\sqrt{2\pi(S_0+\sigma^2T_0)}}.
\tag{11}
\]

The right-hand side converges to the output argument in (8) as
\(\sigma\downarrow0\). Hence, after reducing \(\bar\sigma\) if necessary,
the exact payoff \(J_0(\sigma)\) of the design in (9) satisfies

\[
J_0(\sigma)\geq u(b_0)+2\eta.
\tag{12}
\]

### Step 2: the optimized full-disclosure boundary converges to immediate
termination

The exact high-precision analysis gives, uniformly on compact
prize--deadline sets,

\[
\limsup_{\lambda\to\infty}
\mathcal B(0,\theta,T,\lambda;S_0)
\leq
\sqrt{\frac{S_0+\sigma^2T}{2\pi}}
+\frac{8\sigma^2T}{\sqrt{2\pi S_0}}.
\tag{13}
\]

Dropping the nonnegative prize cost, the optimized full-disclosure payoff is
therefore bounded above by

\[
H_\sigma
:=
\sup_{T\geq0}
\left\{
u\!\left(
\sqrt{\frac{S_0+\sigma^2T}{2\pi}}
+\frac{8\sigma^2T}{\sqrt{2\pi S_0}}
\right)-D(T)
\right\}.
\tag{14}
\]

We claim that

\[
H_\sigma\longrightarrow u(b_0)
\qquad\text{as }\sigma\downarrow0.
\tag{15}
\]

The lower bound follows by taking \(T=0\). For the upper bound, convexity,
\(D(0)=0\), and \(D\not\equiv0\) imply that there are \(d>0\) and
\(T_d<\infty\) such that \(D(T)\geq dT\) for \(T\geq T_d\). Since \(u\) is
bounded above, deadlines outside a sufficiently large compact interval give
payoff strictly below \(u(b_0)\), uniformly in all sufficiently small
\(\sigma\). On that compact interval, the output argument in (14) converges
uniformly to \(b_0\). Continuity of \(u\) proves (15).

Equations (12) and (15) imply that, for all sufficiently small \(\sigma\),

\[
J_0(\sigma)>H_\sigma+\eta.
\tag{16}
\]

Thus the finite-precision candidate strictly dominates every
full-disclosure design after prize and deadline have been reoptimized.

### Step 3: endogenous compactness and the zero-precision boundary

Any design with payoff at least \(J_0(\sigma)\) satisfies

\[
\theta+D(T)
\leq
\sup_{B\geq0}u(B)-J_0(\sigma)
<R.
\tag{17}
\]

All potentially optimal prize--deadline pairs therefore lie in the compact,
primitive set

\[
K_R:=\{(\theta,T)\geq0:\theta+D(T)\leq R\}.
\tag{18}
\]

By the definition of \(\bar e\) and condition (4),

\[
\frac{\theta T}{cS_0}
\leq\frac{\bar e}{cS_0}
\leq\frac1{25}
\qquad\text{throughout }K_R.
\tag{19}
\]

The analytic zero-precision continuation result consequently applies to
every payoff-relevant design, not merely to the candidate. It supplies the
selected no-signal equilibrium and continuity of exact output at
\(\lambda=0\) throughout \(K_R\). Positive finite precisions are covered by
the parabolic equilibrium construction.

Uniform high-precision convergence on \(K_R\), together with the strict gap
in (16), gives a finite \(\bar\lambda\) such that no
\((\theta,T)\in K_R\) with \(\lambda>\bar\lambda\) can attain
\(J_0(\sigma)\). Designs outside \(K_R\) are already excluded by (17).
The objective is therefore continuous on the compact set
\(K_R\times[0,\bar\lambda]\) and attains its global maximum there.

### Step 4: the optimal prize and deadline are positive

When \(\theta=0\), effort vanishes and the best-output payoff is bounded by

\[
\sup_{T\geq0}
\left\{
u\!\left(\sqrt{\frac{S_0+\sigma^2T}{2\pi}}\right)-D(T)
\right\}.
\tag{20}
\]

The same compactness argument used for (15) shows that (20) converges to
\(u(b_0)\) as \(\sigma\downarrow0\). It is therefore below the candidate
payoff in (12) for sufficiently small \(\sigma\). A zero deadline gives at
most \(u(b_0)\). Hence every global maximizer has
\(\theta^*>0\) and \(T^*>0\). Equation (16) and compactness give
\(\lambda^*<\infty\), completing the proof.

## 5. Nonemptiness

The conditions describe a nonempty region. Take

\[
S_0=100,\qquad c=1,\qquad D(T)=T/4,
\]

and normalize \(u(b_0)=0\) with

\[
u(B)=1-\exp[-110(B-b_0)].
\]

Then \(R=1\),

\[
\bar e=\max_{0\leq T\leq4}T(1-T/4)=1
<\frac{cS_0}{25}=4,
\]

and direct minimization gives \(C(e)=\sqrt e\). At \(e=1/4\),

\[
u\!\left(
b_0+\frac{2e}{3c\sqrt{2\pi S_0}}
\right)-u(b_0)
\approx0.519
>0.5=C(e).
\]

Thus both primitive conditions hold strictly. The theorem then applies for
all sufficiently small positive innovation risks. The earlier numerical
example \(\sigma=10^{-3}\) lies in this qualitative regime, although this
note does not claim that the largest admissible \(\bar\sigma\) equals that
particular value.

## 6. What this improves and what remains

This theorem makes three substantive improvements over the current
Proposition 1.

1. It replaces the lower bound on \(D'(0)\) by the exact maximum affordable
   cumulative exposure \(\bar e\).
2. It replaces the candidate-specific utility condition by a comparison
   between the benefit of exposure and its minimum economic cost \(C(e)\).
3. It identifies a genuine technology regime: sufficiently low innovation
   risk implies finite jointly optimal feedback precision.

The theorem is one-sided. It does not prove that full disclosure is optimal
above \(\bar\sigma\). Establishing that converse requires a uniform
weak-incentive result for exact output; the existing second-order calculation
is not yet sufficient for that claim.
