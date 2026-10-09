# Exact joint design with an exogenously specified, bounded concave utility

> 历史研究存档（2026-10-03）。保留原推导以便恢复和核查；本文中的“已证明”“已闭合”不代表当前审定状态。当前方向与结果状态以上级目录 CASE2_AND_REGION_GAP.md 和 PROOF_LEDGER.md 为准。不得单独引用本存档宣布精确联合设计已经完成。

This derivation treats the utility function and all technology/cost parameters
as fixed before prize, deadline, or precision is chosen. It does not assume
anything about a full-disclosure optimizer's prize. The manuscript has not
been changed by this derivation.

## 1. Corrections to the preceding route

1. A fixed-design improvement is not sufficient to exclude the entire
   full-disclosure boundary. Bounds depending on a candidate deadline are not
   uniform bounds over all deadlines in an unbounded design problem.
2. Coercivity in prize and deadline alone does not establish attainment at
   zero precision. The existing analytic continuation argument requires
   theta*T/(c*S0) <= 1/25. This condition must hold on the relevant optimizing
   set, not only at the comparison candidate.
3. The previous research note's purported upper Mills bound
   2/(z+sqrt(z^2+4)) for (1-Phi(z))/phi(z) has the wrong direction.
   At z=1 these quantities are approximately .6180 and .6557, respectively.
   Consequently, that argument does not certify kappa < 33/5. The new result
   uses the independently verified, weaker kappa < 8 bound below. This does
   not disprove the earlier numerical value of kappa; it repairs the bound
   actually used in this proof.

## 2. Statement using only exogenous preference and technology features

Maintain symmetric players c_i=c_j=c>0, initial perceived gap zero, sigma>0,
S0>0, and the exact equilibria constructed in the manuscript. Let u be fixed,
C^2, strictly increasing, concave, and bounded above on the relevant
nonnegative output domain. Set

\[
u_\infty:=\lim_{B\to\infty}u(B)<\infty,
\qquad b_0:=\sqrt{\frac{S_0}{2\pi}},
\qquad R:=u_\infty-u(b_0),\qquad v:=u'(b_0).
\]

Thus R>0 is the remaining utility gain above immediate termination, and v>0
is the marginal value at that exogenous baseline. Define h>0 uniquely by

\[
u(b_0+h)-u(b_0)=\frac34 R.
\tag{1}
\]

This is a definition of a feature of the given utility function at the given
initial environment. It is not a restriction on utility at an optimized or
candidate output. None of b0,R,v,h depends on theta,T,lambda. In particular
h is the output increment realizing three quarters of the remaining utility.

For an explicit sufficient region, take the exogenous delay-cost function
D(T)=dT, with d>0. Suppose
\[
\boxed{
\max\left\{
\frac{9v\sigma^2}{\sqrt{2\pi S_0}},
\frac{25R^2}{4cS_0}
\right\}
\le d\le
\frac{R^2}{22c\sqrt{2\pi S_0}\,h}.
}
\tag{3}
\]

Then the unrestricted joint problem

\[
\sup_{\theta,T,\lambda\ge0}
\left\{u\bigl(\mathcal B(0,\theta,T,\lambda;S_0)\bigr)-\theta-dT\right\}
\tag{4}
\]

attains its value. Every global maximizer satisfies

\[
\theta^*>0,\qquad T^*>0,\qquad 0\le\lambda^*<\infty.
\tag{5}
\]

In fact the best full-disclosure boundary value equals u(b0), and the joint
value is at least u(b0)+R/4. This is a sufficient region; it does not classify
its complement or establish strictly positive optimal precision.

## 3. Proof

### Exact output ingredients

The finite-precision lower bound from the current proof is

\[
\mathcal B(0,\theta,T,1/(S_0T);S_0)
\ge \sqrt{\frac{S_0+\sigma^2T}{2\pi}}
+\frac{3\theta T}{4c\sqrt{2\pi(S_0+\sigma^2T)}}
\tag{6}
\]

whenever theta*T/(c*S0)<=1/25 and S0/(sigma^2*T)>=225.
It follows from the gradient contraction estimate, not from a truncated
prize expansion. The energy budget and full-disclosure value limit imply

\[
\mathcal B_\infty(\theta,T)
\le\sqrt{\frac{S_0+\sigma^2T}{2\pi}}
+\frac{\kappa\sigma^2T}{\sqrt{2\pi S_0}},
\quad
\kappa=2\int_{\mathbb R}\sqrt{-\log|2\Phi(z)-1|}\,dz<8.
\tag{7}
\]

This bound is valid for arbitrary fixed positive prize and deadline, not
only for the candidate used below. Its compact-uniform high-precision
version follows from the value-limit and energy arguments in the existing
proof. At zero prize or zero deadline use

\[
0\le\mathcal B-\sqrt{\frac{S_0+\sigma^2T}{2\pi}}
\le\sqrt{\frac{2\theta T}{c}}.
\tag{8}
\]

For completeness, a valid bound for the constant in (7) follows without the
incorrect sharpened Mills inequality. On 0<z<1,

\[
2\Phi(z)-1\ge\sqrt{\frac2\pi}z(1-z^2/6),
\quad -\log(1-z^2/6)\le z^2/5.
\]

Jensen's inequality for the concave square-root function yields

\[
\int_0^1\sqrt{-\log(2\Phi(z)-1)}\,dz
\le\sqrt{1+\tfrac12\log(\pi/2)+\tfrac1{15}}.
\]

On z>=1, the ordinary Mills upper bound 1-Phi(z)<=phi(z)/z gives

\[
\sqrt{-\log(2\Phi(z)-1)}
\le \sqrt{\frac{2}{2\Phi(1)-1}}(2\pi)^{-1/4}
z^{-1/2}e^{-z^2/4}.
\]

Bound z^{-1/2} by 1 on [1,2] and by 1/sqrt(2) on [2,infinity).
Consequently kappa is at most

\[
4\left[
\sqrt{1+\tfrac12\log(\pi/2)+\tfrac1{15}}
+\sqrt{\frac{2}{2\Phi(1)-1}}(2\pi)^{-1/4}\sqrt\pi
\left\{\operatorname{erf}(1)-\operatorname{erf}(1/2)
+\frac{\operatorname{erfc}(1)}{\sqrt2}\right\}
\right]<8.
\tag{9}
\]

The displayed explicit bound is approximately 7.86954. The inequality also
follows from the conservative bounds for its two bracketed terms, 1.137 and
0.831, respectively, which give 4*(1.137+.831)=7.872<8. These bounds can be
checked by integrating the convergent alternating Taylor series for exp(-x^2)
on [0,1], with its next-term remainder.

### Bounding the entire full-disclosure boundary

Concavity of the square root and (7) imply

\[
\mathcal B_\infty(\theta,T)
\le b_0+\frac{17\sigma^2}{2\sqrt{2\pi S_0}}T.
\]

Concavity of u gives u(B)<=u(b0)+v(B-b0) for B>=b0. Therefore

\[
u(\mathcal B_\infty(\theta,T))-\theta-dT
\le u(b_0)-\theta
+\left(\frac{17v\sigma^2}{2\sqrt{2\pi S_0}}-d\right)T
\le u(b_0)-\theta-\frac{v\sigma^2}{2\sqrt{2\pi S_0}}T.
\tag{10}
\]

Hence the full-disclosure boundary maximum is u(b0), attained only at
theta=T=0. This explicitly shows why a large boundary-optimal prize is
unnecessary for a finite-precision joint optimum.

### A strictly better finite-precision design

Choose, only inside the proof,

\[
\theta_f:=\frac R4,\qquad T_f:=\frac R{4d},\qquad
\lambda_f:=\frac1{S_0T_f}=\frac{4d}{S_0R}.
\tag{11}
\]

The candidate is a function of primitives; no condition on u is imposed
after choosing it. Concavity first gives 3R/4<=v*h. The information lower
bound and the upper bound in (3) imply R^2>=198*c*v*sigma^2*h. Together,

\[
R\ge\frac{297}{2}c\sigma^2>36c\sigma^2.
\tag{2}
\]

Thus no additional large-prize or utility-scale condition is needed. These
inequalities and the compactness lower bound in (3) imply

\[
\frac{\theta_f}{c\sigma^2}\ge9,
\qquad
\frac{\theta_fT_f}{cS_0}=
\frac{R^2}{16dcS_0}\le\frac1{100},
\qquad
\frac{\sigma^2T_f}{S_0}\le\frac1{900}.
\]

Thus (6) applies, and

\[
\begin{aligned}
\mathcal B(0,\theta_f,T_f,\lambda_f;S_0)-b_0
&\ge\frac{3R^2}{64dc\sqrt{2\pi S_0}
\sqrt{1+\sigma^2T_f/S_0}}\\
&\ge\frac{R^2}{22dc\sqrt{2\pi S_0}}\ge h.
\end{aligned}
\tag{12}
\]

For the second inequality, use 3/(64*sqrt(901/900))>1/22.
By (1) and monotonicity of u,

\[
u(\mathcal B(0,\theta_f,T_f,\lambda_f;S_0))
-\theta_f-dT_f
\ge u(b_0)+\frac34R-\frac12R
=u(b_0)+\frac R4.
\tag{13}
\]

This exceeds every full-disclosure design, even after its prize and deadline
are reoptimized.

### Compactness and boundary attainment

Any design with theta+dT>=R has objective at most
u_infinity-R=u(b0), so it cannot compete with (13). Restrict attention to
the compact prize-deadline set

\[
K:=\{(\theta,T)\ge0:\theta+dT\le R\}.
\]

This is a consequence of the given utility and cost, not an imposed design
cap. On all of K,

\[
\frac{\theta T}{cS_0}
\le\frac{R^2}{4dcS_0}\le\frac1{25}.
\tag{14}
\]

The no-signal analytic construction in the current appendix consequently
applies uniformly on the relevant set, including sequences with lambda->0.
In detail, for lambda*S0*T<=1, the scaled terminal variance is at least
1/2. The analytic Fourier-norm estimate has

\[
M'\le(\rho'+2\epsilon M)N,\quad
M(0)\le4e/\sqrt\pi<8,\quad
\rho(\tau)=1-16\epsilon\tau\ge9/25>0.
\]

The uniform analytic radius gives continuation and continuous dependence to
lambda=0. Positive-precision continuity follows from the finite-horizon
parabolic construction. Vanishing prize or deadline is covered by (8).
The high-precision limit is uniform on K: its interior cutoff estimates are
uniform on positive compact prize-deadline sets, while (8) handles their
endpoints. Applying (10) then excludes all sufficiently large lambda,
uniformly on K, from attaining the value of (13).

The remaining design set is compact and the exact output is continuous
there, so a global maximum is attained. A zero-prize or zero-duration design
has the same objective at any precision and at full disclosure, and hence
cannot reach (13). All maximizers have positive prize and deadline, and
finite precision. Zero precision itself is not excluded.

The existence argument uses the equilibrium existence, uniqueness, and
boundary estimates developed in the manuscript; it does not assert a
general no-signal equilibrium theorem outside the controlled set (14).

## 4. The parameter region is nonempty

Fix the utility function first:

\[
u(B)=400\left[1-\exp\{-500\log(4)(B-1)\}\right].
\]

It is strictly increasing and strictly concave, bounded above by 400, and
its derivative tends to zero. Its formula contains no design variable or
technology parameter. Next take

\[
c=\sigma=1,\qquad S_0=2\pi,\qquad d=500000.
\]

Then b0=1, R=400, h=.002, and v=200000*log(4). The bounds in (3) are

\[
\max\{397144.081,159154.944\}<500000<578745.247.
\]

The candidate is theta_f=100, T_f=.0002,
lambda_f=795.774715459..., and (6) gives B>=1.0024032015.
Its certified objective is at least 124.38216 above u(b0)=0; the general
proposition alone guarantees a gain of R/4=100. All full-disclosure designs
have objective at most zero by (10). This is an analytical comparison of
proved bounds, not a numerical global optimization claim.

As a separate diagnostic, the existing exact-HJB solver at the single
candidate gives B=1.0031991903 with dy=.04 and 600 time steps, and
B=1.0031992533 with dy=.02 and 1200 time steps. These calculations are
consistent with the bound but are not used in the proof.

## 5. Interpretation and remaining scope

All hypotheses concern u,c,sigma,S0,d. The preference features R,v,h are
determined by the exogenous pair (u,S0). Neither a candidate output nor an
optimal prize appears in them. The comparison candidate (11) occurs only
in the proof. No exogenous prize/deadline cap is required.

The condition on d has three distinct roles: the first lower bound excludes
profitable full-disclosure duration; the second keeps all potentially
profitable designs inside the known zero-precision regularity region; the
upper bound leaves enough time for the finite-precision candidate to achieve
the utility increment. This is a strong sufficient condition, not a sharp
regime characterization.

The interval in (3) is nonempty precisely when

\[
h\le\frac{2}{275}\sqrt{\frac{S_0}{2\pi}},\qquad
R^2\ge198cv\sigma^2h,
\]

The large-utility-scale inequality (2) follows automatically. Thus the utility must approach its upper bound over a
relatively small output increment above the initial baseline. This is an
economically substantive saturation assumption, not ordinary concavity.
The numerical constants are conservative.

The theorem does not cover unbounded utility functions such as logarithmic
utility. Boundedness, rather than a global positive marginal utility floor,
provides endogenous cost bounds and closes the joint-design existence proof.
The conclusion still permits lambda*=0. No exact weak-incentive region
with lambda*=infinity is established here.

The argument extends to an exogenous nonlinear D satisfying
d_minus*T<=D(T)<=d_plus*T for T>=0: impose the two lower bounds in (3) on
d_minus, impose its upper bound on d_plus, use the candidate T=R/(4*d_plus),
and use theta+d_minus*T<=R for compactness. The linear specification uses
only one cost parameter and is the clearest statement of this sufficient
region.
