# Finite optimal precision when zero precision is feasible

> 历史研究存档（2026-10-03）。保留原推导以便恢复和核查；本文中的“已证明”“已闭合”不代表当前审定状态。当前方向与结果状态以上级目录 CASE2_AND_REGION_GAP.md 和 PROOF_LEDGER.md 为准。不得单独引用本存档宣布精确联合设计已经完成。

> Current audit notice. This file preserves an earlier sufficient-region argument;
> it must not be read as a proof of unconditional attainment over every prize and
> deadline in the original zero-precision model. The cumulative-incentive budget
> restriction in the argument is substantive, not merely notation. The current
> model-specific analysis is in `CASE2_AND_REGION_GAP.md`, Sections 37--45:
> a full-state no-signal Nash branch, including its critical initial endpoint, is
> verified, but continuation beyond its critical exposure is not established.
> Section 45 proves that a uniformly bounded, uniformly spatially Lipschitz
> family of positive-precision feedback equilibria is impossible in that
> supercritical range. Neither a positive precision floor nor a weak/relaxed
> equilibrium has been silently substituted for the original problem.

## 1. Objective and current status

The designer chooses

\[
(\theta,T,\lambda)\in[0,\infty)^3.
\]

The objective is to prove that a global maximizer exists and satisfies

\[
0\leq\lambda^*<\infty.
\]

The result does not require \(\lambda^*>0\). A no-disclosure design may be
optimal. Consequently, the proof must admit the boundary \(\lambda=0\), but it
does not need to show that this boundary is suboptimal.

The exact positive-precision analysis rules out escape through
\(\lambda\to\infty\) once a finite-precision design strictly dominates the
full-disclosure benchmark. The zero boundary can also be closed without
imposing prize or deadline caps: bounded output utility and a coercive delay
cost first restrict every competitive design to a compact set derived from the
primitives. If cumulative incentives are small throughout this derived set,
the analytic no-signal construction applies there. The resulting theorem is
an unrestricted joint-design result.

The earlier positive-floor construction solved a different problem with
\(\lambda\geq\lambda_{\min}>0\). It is not used here.

## 2. Maintained primitive conditions

Fix \(S_0>0\). Contestants are symmetric, \(c_i=c_j=c>0\), and the initial
perceived gap is zero. Let \(u\) be twice continuously differentiable,
strictly increasing, and concave. Let \(D\) be continuously differentiable,
nondecreasing, and convex, with \(D(0)=0\) and \(D\not\equiv0\).

To prevent prize or deadline from diverging, assume either

\[
u'(B)\longrightarrow0\quad\text{as }B\to\infty,
\]

or

\[
\frac{D(T)}T\longrightarrow\infty\quad\text{as }T\to\infty.
\]

These are alternative coercivity conditions. In particular, the first permits
any given positive linear delay cost. Neither condition depends on a design
chosen by the organizer.

For \(\lambda>0\), write

\[
J(\theta,T,\lambda)
=u\!\left(\mathcal B(0,\theta,T,\lambda;S_0)\right)-\theta-D(T).
\]

The preceding exact analysis establishes a nonempty primitive region---small
positive \(c\) and \(\sigma\), holding \(u,D,S_0\) fixed---in which a feasible
positive finite-precision design has payoff \(J_f\) satisfying

\[
J_f>
H_\infty:=\sup_{\theta,T\geq0}
\left\{u\!\left(\mathcal B_\infty(\theta,T)\right)
-\theta-D(T)\right\}.
\tag{1}
\]

This comparison uses exact equilibrium output. The prize and deadline on the
right-hand side are reoptimized, so equation (1) is a joint-design result.

## 3. What is already proved

### Compactness in prize and deadline

The equilibrium energy bound gives

\[
\mathcal B(0,\theta,T,\lambda;S_0)
\leq
\sqrt{\frac{S_0+\sigma^2T}{2\pi}}
+\sqrt{\frac{2\theta T}{c}},
\tag{2}
\]

uniformly over \(\lambda>0\). Concavity of \(u\), together with either
coercivity condition above, implies that every design with payoff at least
\(J_f\) lies in a compact set

\[
K\subset[0,\infty)^2
\tag{3}
\]

of prizes and deadlines. This set is derived from primitives and the feasible
payoff \(J_f\); it is not an imposed design cap.

### Uniform exclusion of high precision

The exact high-precision analysis gives

\[
\mathcal B(0,\theta,T,\lambda;S_0)
\longrightarrow\mathcal B_\infty(\theta,T)
\quad\text{as }\lambda\to\infty,
\tag{4}
\]

uniformly for \((\theta,T)\in K\). Let
\(\delta=J_f-H_\infty>0\). Equations (1) and (4) imply that some finite
\(\bar\lambda\) satisfies

\[
\sup_{(\theta,T)\in K,\,\lambda\geq\bar\lambda}
J(\theta,T,\lambda)
\leq H_\infty+\frac\delta2<J_f.
\tag{5}
\]

Thus every maximizing sequence eventually lies in

\[
K\times(0,\bar\lambda].
\tag{6}
\]

Equations (3)--(6) rule out all unbounded directions. They do not by themselves
give a maximizer, because the set in (6) is not closed at zero precision.

## 4. The exact closure lemma that remains

The unrestricted existence proof requires only the following result.

**Zero-boundary closure lemma.** For every \((\theta,T)\in K\), the no-signal
game admits a selected Markov-perfect equilibrium with output
\(\mathcal B(0,\theta,T,0;S_0)\). Moreover, the resulting objective is upper
semicontinuous on \(K\times[0,\bar\lambda]\):

\[
\limsup_{n\to\infty}
J(\theta_n,T_n,\lambda_n)
\leq J(\theta,T,0)
\tag{7}
\]

whenever
\((\theta_n,T_n,\lambda_n)\to(\theta,T,0)\) with \(\lambda_n>0\).

Continuity at zero would be sufficient but is stronger than necessary. The
lemma does not require the no-signal design to be inferior, nor does it require
\(\lambda^*>0\).

### Completion of the joint-design proof from the lemma

Extend \(J\) to \(\lambda=0\) using the no-signal equilibrium in the lemma.
Positive-precision equilibrium theory gives continuity on
\(K\times(0,\bar\lambda]\), while equation (7) supplies upper semicontinuity at the
remaining boundary. Hence \(J\) is upper semicontinuous on the compact set

\[
K\times[0,\bar\lambda].
\]

It therefore attains a maximum there. Designs outside \(K\) cannot beat
\(J_f\), and equation (5) excludes all \(\lambda\geq\bar\lambda\). The resulting
maximizer is global and satisfies

\[
0\leq\lambda^*<\bar\lambda<\infty.
\]

This argument fully proves existence and finite optimal precision once the
zero-boundary closure lemma is available.

## 5. The no-signal system

Let \(p=U_{i,x}\) and \(r=-U_{j,x}\) after scaling state by \(\sqrt{S_0}\),
time by \(T\), and reversing time. Put

\[
\epsilon=\frac{\theta T}{cS_0}.
\]

The correct gradient equations are

\[
p_\tau=\frac{a(\tau)}2p_{xx}
+\epsilon\partial_x\!\left(\frac{p^2}{2}-pr\right),
\qquad
r_\tau=\frac{a(\tau)}2r_{xx}
+\epsilon\partial_x\!\left(pr-\frac{r^2}{2}\right),
\tag{8}
\]

where \(a=0\) at zero precision. The signs in equation (8) follow directly
from \(r=-U_{j,x}\); the earlier plus-plus version was incorrect.

For the analytic Fourier norms of \(p\) and \(r\), the corrected nonlinear
terms still satisfy

\[
\frac{dM}{d\tau}
\leq\bigl(\rho'+2\epsilon M\bigr)N.
\tag{9}
\]

Indeed, writing \(P=\|p\|_\rho\), \(R=\|r\|_\rho\),
\(N_p=\|p_x\|_\rho\), and \(N_r=\|r_x\|_\rho\), the nonlinear contribution is
bounded by

\[
(P+R)N_p+P N_r+R N_p+(P+R)N_r
\leq2(P+R)(N_p+N_r).
\]

After correcting the signs, the existing Fourier-cutoff and shrinking-radius
argument constructs an analytic no-signal solution and gives uniqueness and
continuous dependence within the controlled analytic class whenever

\[
\epsilon\leq\frac1{25}.
\tag{10}
\]

The resulting gradients are bounded, so the induced state dynamics are
well posed; the usual completion-of-the-square argument then verifies the
associated strategies. These bounds also permit passage to expected output and
prove equation (7) on any compact prize--deadline set satisfying (10). Hence
the zero boundary is controlled in the small-cumulative-incentive region.

The primitive low-cost/low-noise result of Sections 2--3 does not imply (10)
throughout its competitive set \(K\). It imposes (10) on the constructed
finite-precision candidate only. Other designs in \(K\) may have much larger
\(\theta T/(cS_0)\). This is why the local analytic result does not yet close
the unrestricted primitive theorem.

## 6. Why the general zero boundary is nontrivial

At a smooth point with \(p,r>0\), the first-derivative matrix in equation (8)
has eigenvalues

\[
\epsilon\left[(p-r)\pm i\sqrt{pr}\right].
\]

The zero-diffusion system is therefore not hyperbolic in the interior region.
Scalar Hamilton--Jacobi vanishing-viscosity theory does not apply directly.
This observation does not prove that the Gaussian-terminal no-signal game lacks
an equilibrium or that its positive-precision outputs fail to converge. It
only shows that global continuation beyond condition (10) needs a separate
argument.

Numerical calculations near zero precision are also inconclusive. Several
coarse-grid solutions that satisfied the effort-sign diagnostic failed it after
grid refinement. They should not be used either to assert or to reject boundary
continuity.

## 7. Resolution by endogenous compactness

Assume in addition that

\[
u_\infty:=\sup_{B\geq0}u(B)<\infty,
\qquad
D(T)\longrightarrow\infty.
\]

Let

\[
B_0=\sqrt{\frac{S_0}{2\pi}},\qquad
R=u_\infty-u(B_0),\qquad
T_R=\sup\{T\geq0:D(T)\leq R\}.
\tag{11}
\]

These quantities are not design constraints. They are functions of the
primitives. The no-design option has payoff \(u(B_0)\). If a design has payoff
\(J>u(B_0)\), then

\[
\theta+D(T)
\leq u_\infty-J
<u_\infty-u(B_0)=R.
\tag{12}
\]

Consequently, every design that can improve on the no-design option satisfies

\[
0\leq\theta<R,
\qquad
0\leq T\leq T_R.
\tag{13}
\]

Thus

\[
\frac{RT_R}{cS_0}\leq\frac1{25}
\tag{14}
\]

implies \(\theta T/(cS_0)\leq1/25\) for every payoff-relevant design. The
zero-boundary closure lemma in Section 5 therefore applies on the entire set
that can contain an optimizer. Unlike an imposed pair
\((\bar\theta,\bar T)\), the compact set in equation (13) follows from the
objective itself.

It remains to exclude \(\lambda\to\infty\). For any \(T\geq0\), define the
explicit high-precision output bound

\[
\overline B_\infty(T)
=\sqrt{\frac{S_0+\sigma^2T}{2\pi}}
+\frac{8\sigma^2T}{\sqrt{2\pi S_0}}.
\tag{15}
\]

Suppose there are \(\theta_f,T_f>0\) such that
\(S_0\geq\sigma^2T_f\) and

\[
\begin{aligned}
&u\!\left(
\sqrt{\frac{S_0+\sigma^2T_f}{2\pi}}
+\frac{2\theta_fT_f}
{3c\sqrt{2\pi(S_0+\sigma^2T_f)}}
\right)-\theta_f-D(T_f)\\
&\qquad>
\sup_{T\geq0}\{u(\overline B_\infty(T))-D(T)\}.
\end{aligned}
\tag{16}
\]

The right-hand side is at least \(u(B_0)\). Since \(u\leq u_\infty\),
equation (16) first implies \(\theta_f+D(T_f)<R\). Equations (13)--(14) then
give

\[
\frac{\theta_fT_f}{cS_0}\leq\frac1{25}.
\]

The exact finite-precision lower bound is therefore valid at
\(\lambda_f=1/(S_0T_f)\), and equation (16) proves that this design strictly
dominates every full-disclosure design, including after reoptimizing the prize
and deadline. Uniform high-precision convergence on the compact set in
equation (13) then supplies a finite \(\bar\lambda\) beyond which no design can be
optimal. Continuity at zero follows from Section 5 on the same set. Hence the
objective attains a global maximum and

\[
0\leq\lambda^*\leq\bar\lambda<\infty.
\tag{17}
\]

This closes the unrestricted joint-design proof under equations (14)--(16).
The optimizer may have \(\lambda^*=0\); no positive precision floor is used.

### A simpler, stronger certificate

The paper uses a stronger condition that avoids the two optimization operators
in equation (16). Assume that \(D\) is convex and

\[
D'(0)\geq
\max\left\{
\frac{25R^2}{cS_0},
\frac{17\sigma^2}{2\sqrt{2\pi S_0}}
u'\!\left(\sqrt{\frac{S_0}{2\pi}}\right)
\right\}.
\tag{18}
\]

Convexity and \(D(0)=0\) give \(D(T)\geq D'(0)T\). Hence the first
term in equation (18) implies \(T_R\leq R/D'(0)\) and therefore equation
(14). It also makes a separate coercivity assumption on \(D\) unnecessary.
The high-precision output bound in equation (15) has derivative at most
\(17\sigma^2/(2\sqrt{2\pi S_0})\). Concavity of \(u\) and convexity of \(D\)
and the second term in equation (18) therefore imply that its payoff is
nonincreasing in \(T\). Its supremum is the no-design payoff \(u(B_0)\).

Suppose there is a \(T>0\) such that

\[
S_0\geq\sigma^2T
\tag{19}
\]

and

\[
u\!\left(
B_0+\frac{RT}{6c\sqrt{2\pi(S_0+\sigma^2T)}}
\right)-u(B_0)>\frac R4+D(T).
\tag{20}
\]

Choose \(\theta=R/4\) and \(\lambda=1/(S_0T)\). Equations (14) and (19)
do not yet directly locate the test duration in the payoff-relevant set.
Instead, the left-hand side of equation (20) is at most \(R\), so equation
(20) implies \(D(T)<3R/4<R\), and hence \(T\leq T_R\). Equation (14) now gives
\(\theta T/(cS_0)\leq1/100\), so the exact finite-precision lower bound applies.
Retaining two thirds of the leading term, its guaranteed output increment over \(B_0\) is
\(RT/[6c\sqrt{2\pi(S_0+\sigma^2T)}]\). Equation (20) says that this
utility gain covers the candidate prize and its actual delay cost, so the design strictly exceeds
\(u(B_0)\), while equation (18) bounds every full-disclosure design by
\(u(B_0)\). Thus equations (18)--(20) imply equation (16).

The sufficient region is nonempty. For example, take
\(S_0=100\), \(c=1\), \(\sigma=10^{-3}\),
\(D(T)=T/4\), and normalize \(u(B_0)=0\) with

\[
u(B)=1-\exp[-110(B-B_0)].
\]

Then \(R=1\), \(T_R=4\), and the first term in equation (18) holds with
equality; the second is slack. The test
design \((\theta_f,T_f)=(1/4,1)\) satisfies the variance condition. Its
finite-precision lower-bound payoff is approximately \(0.0188\), while the
right-hand side of equation (16) equals zero. Thus the assumptions do not
describe an empty parameter set.

## 8. Conclusions for Proposition 1

There are two distinct results.

1. Under bounded increasing concave utility, coercive delay cost, and
   equations (14)--(16), the original unbounded joint-design problem has a
   global optimizer with \(0\leq\lambda^*<\infty\). Prize and deadline bounds
   are consequences of the objective, not additions to the feasible set.

2. Without equation (14), the special no-signal Nash system remains
   uncontrolled for large \(\theta T/(cS_0)\). The broader unrestricted theorem
   would still require a global zero-boundary closure result. This limitation
   does not affect the sufficient theorem in Section 7.
