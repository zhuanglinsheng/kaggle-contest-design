# Exact finite-precision design: simpler sufficient conditions

> 历史研究存档（2026-10-03）。保留原推导以便恢复和核查；本文中的“已证明”“已闭合”不代表当前审定状态。当前方向与结果状态以上级目录 CASE2_AND_REGION_GAP.md 和 PROOF_LEDGER.md 为准。不得单独引用本存档宣布精确联合设计已经完成。

This note strengthens the current Proposition 1 using its exact finite-precision
equilibria and its high-precision **value-function** limit. It does not use a
small-prize approximation of output. The constants below are sufficient, not
estimated regime boundaries. In particular, the numerical interval
`1.5 <= theta/(c sigma^2) <= 2` remains unproved.

## Fixed prize and deadline

Assume symmetric costs `c_i=c_j=c`, initial public mean zero, and positive
`sigma,S0,theta,T`. If

\[
\boxed{\theta\ge9c\sigma^2,\qquad S_0\ge25\theta T/c,}
\tag{1}
\]

then, with `lambda_L=1/(S0 T)`, the exact output satisfies

\[
\mathcal B(0,\theta,T,\lambda_L;S_0)
>\limsup_{\lambda\to\infty}\mathcal B(0,\theta,T,\lambda;S_0).
\tag{2}
\]

Furthermore, the maximum over `lambda>=0` is attained at a finite precision.
The result allows zero precision to maximize output; it does not establish an
interior precision optimum. Every positive interior maximizer satisfies
`B_lambda=0` when output is differentiable.

### 1. Finite-precision lower bound

Use `alpha=theta/c`, `w=alpha/sigma^2`, `s=S0/(sigma^2 T)` and
`epsilon=alpha T/S0` only in the proof. Condition (1) gives
`epsilon<=1/25` and `s>=225`.

At `lambda_L=1/(S0 T)`, scale time by `T` and the posterior mean by `sqrt(S0)`.
Writing `v(tau)=S(T tau)/S0`,

\[
v'=s^{-1}-v^2,\qquad (1+\tau)^{-1}\le v(\tau)\le1.
\]

The terminal
normalized value gradients have sup norm at most `1/sqrt(pi)`. Work with their
sup norms in the ball of radius `R=61/100`; value levels are recovered from the
mild equation. Its quadratic nonlinearity obeys

\[
\|F(U)\|_\infty\le\tfrac32\epsilon R^2,
\qquad
\|F(U)-F(\widehat U)\|_\infty
\le3\epsilon R\|U_y-\widehat U_y\|_\infty.
\]

For `0<=tau<r<=1`, the diffusion variance is bounded below by

\[
\int_\tau^r v(z)^2dz
\ge\frac{r-\tau}{(1+\tau)(1+r)}.
\]

Consequently, the integrated spatial-derivative Gaussian-semigroup bound is

\[
K\le\sqrt{\frac2\pi}\max_{0\le\tau\le1}
\sqrt{1+\tau}\int_\tau^1\sqrt{\frac{1+r}{r-\tau}}dr<2.
\]

For completeness, the integral equals

\[
(1+\tau)\operatorname{arsinh}\sqrt{\frac{1-\tau}{1+\tau}}
+\sqrt{2(1-\tau)}.
\]

Using `arsinh(x)<=x`, its contribution after multiplication by
`sqrt(1+tau)` is bounded by `(1+tau)sqrt(1-tau)`, whose maximum is
`(4/3)sqrt(2/3)`. The remaining term is at most `sqrt(2)`. Hence
`K<sqrt(2/pi){(4/3)sqrt(2/3)+sqrt(2)}<2`.

Thus the gradient map stays in the ball and is a contraction, because

\[
\pi^{-1/2}+K\tfrac32\epsilon R^2<0.609<R,
\qquad 3K\epsilon R<0.147<1.
\]

Bounded-gradient uniqueness identifies this solution with the exact equilibrium.
Let `p=U_{i,x}`, `r=-U_{j,x}` and

\[
h(x,\tau)=\frac{\varphi(x/\sqrt{Q(\tau)})}{\sqrt{Q(\tau)}},
\qquad Q(\tau)=v(\tau)+(1-\tau)/s.
\]

Both `p` and `r` differ from `h` by at most

\[
\delta=K\tfrac32\epsilon R^2<0.044652.
\]

The strategic state drift is `epsilon(p-r)` and is bounded by
`2 epsilon delta`. Couple the equilibrium state `Y` and the driftless state `X`
using the same Brownian motion. Then

\[
\sup_{\tau\le1}|Y_\tau-X_\tau|\le2\epsilon\delta,
\quad \|h_x\|_\infty\le1,
\quad E h(X_\tau,\tau)=h(0,0).
\]

In the exact output drift identity, the average-effort error is bounded by
`delta` and the gap-drift error by `delta`. Consequently,

\[
\frac{\mathcal B(\lambda_L)-\sqrt{(S_0+\sigma^2T)/(2\pi)}}{\sqrt{S_0}}
\ge\epsilon\{h(0,0)-2\delta(1+\epsilon)\}.
\]

Here `h(0,0)>=1/sqrt(2 pi (226/225))>0.3980`, whereas
`2 delta(1+epsilon)<0.09288<h(0,0)/4`. It follows that

\[
\mathcal B(\lambda_L)
\ge\sqrt{\frac{S_0+\sigma^2T}{2\pi}}
+\frac{3\theta T}{4c\sqrt{2\pi(S_0+\sigma^2T)}}.
\tag{3}
\]

### 2. Sharper high-precision upper bound

The useful change to the existing proof is to apply the energy estimate
**conditionally**, before averaging over the initial revealed gap.

Write `W=U_i+U_j`. The exact conditional budget identity is

\[
E\left[\int_\eta^T(m_i^2+m_j^2)dt\mid I_\eta\right]
=2\alpha[1-W(\tilde y_\eta,\eta)].
\tag{4}
\]

Couple terminal scores with the same initial scores and innovation paths but
zero effort. For the pre-`eta` effort, unconditional Cauchy--Schwarz gives
`sqrt(2 alpha eta)`. For the remaining effort, apply conditional
Cauchy--Schwarz and then average. This yields

\[
\mathcal B(\lambda)-\sqrt{\frac{S_0+\sigma^2T}{2\pi}}
\le\sqrt{2\alpha\eta}
+\sqrt{2\alpha(T-\eta)}\,
E\sqrt{1-W(\tilde y_\eta,\eta)}.
\tag{5}
\]

The current Proposition's value-function limit gives, for fixed `0<eta<T`,

\[
W(y,\eta)\to f_w(d_w(y,\eta)),\qquad
f_w(d)=1+\frac2w\log\frac{\cosh(wd/2)}{\cosh(w/2)},
\]

\[
K_w(d_w(y,t))=K_w(1)[2\Phi(y/(\sigma\sqrt{T-t}))-1],
\qquad K_w(d)=\frac{d+\sinh(wd)/w}{2}.
\]

As in that proof, for `y!=0`,

\[
1-f_w(d_w(y,0))
\le\frac2w\{-\log|2\Phi(y/(\sigma\sqrt T))-1|\}.
\tag{6}
\]

To pass to the initial distribution in (5), the martingale part of
`tilde y_eta` is Gaussian with variance `S0+sigma^2 eta-S_eta`, and its drift
has expected absolute value at most `sqrt(2 alpha eta)`. The value convergence
is uniform in space at fixed `eta`. The function `sqrt(1-f_w(d_w(y,eta)))`
is bounded, continuous, and converges uniformly as `eta` decreases to zero.
First let precision increase to infinity, then let `eta` decrease to zero.
Equation (5) therefore bounds the limiting gain by

\[
\sqrt{2\alpha T}\,E\sqrt{1-f_w(d_w(Y,0))},\qquad Y\sim N(0,S_0).
\]

Using the maximum of this Gaussian density and (6), the gain is at most

\[
\frac{\kappa\sigma^2T}{\sqrt{2\pi S_0}},
\qquad
\kappa:=2\int_{\mathbb R}\sqrt{-\log|2\Phi(z)-1|}\,dz
\approx6.42947.
\tag{7}
\]

The approximation is only interpretive; the integral defines the exact
universal constant. It is finite, and the following elementary split gives
`kappa<33/5`. On
`(0,1)`, integrating `exp(-x^2/2)>=1-x^2/2` gives

\[
2\Phi(z)-1\ge\sqrt{\frac2\pi}\left(z-\frac{z^3}{6}\right).
\]

Apply Jensen's inequality separately on the intervals with endpoints
`0,1/10,1/4,1/2,3/4,1`. Direct integration of the logarithm on the right gives

\[
\int_0^1\sqrt{-\log(2\Phi(z)-1)}dz<1.077.
\]

The five interval contributions, rounded upward, are respectively
`0.1879, 0.2125, 0.2796, 0.2196, 0.1769`; their sum is below `1.077`.

On `[1,infinity)`, use `-log(1-x)<=x/(1-x)` and the sharpened Mills bound

\[
2[1-\Phi(z)]\le
\frac{4\varphi(z)}{z+\sqrt{z^2+4}}.
\]

Partition at `5/4,3/2,7/4,2,5/2,3,4,6`. On each interval, bound
`2\Phi(z)-1` at the left endpoint and
`1+sqrt(1+4/z^2)` at the right endpoint. Direct Gaussian integration gives

\[
\int_1^\infty\sqrt{-\log(2\Phi(z)-1)}dz<0.573.
\]

The nine interval bounds, again rounded upward, are
`0.1550, 0.1161, 0.0869, 0.0645, 0.0817, 0.0399, 0.0246, 0.0035, 0.0001`;
their sum is below `0.573`. The final entry also covers the tail beyond `20`.

Both displayed decimal bounds follow from elementary logarithms and Gaussian
integrals; rounding them upward gives rational bounds `1077/1000` and `573/1000`.
By symmetry, the integral in (7) is below `2(1077/1000+573/1000)`;
hence `kappa<4(1077/1000+573/1000)=33/5`.

Combining (5)--(7) proves the exact bound

\[
\limsup_{\lambda\to\infty}\mathcal B(\lambda)
\le\sqrt{\frac{S_0+\sigma^2T}{2\pi}}
+\frac{\kappa\sigma^2T}{\sqrt{2\pi S_0}}
<\sqrt{\frac{S_0+\sigma^2T}{2\pi}}
+\frac{33\sigma^2T}{5\sqrt{2\pi S_0}}.
\tag{8}
\]

Unlike the old bound, the gain in (8) decays as `S0^{-1/2}`, the same order as
the finite candidate's gain. This eliminates the old `s^{1/4}` prize threshold.

### 3. Strict comparison and attainment

Subtract (8) from (3). The difference is bounded below by

\[
\frac{\sigma^2T}{\sqrt{2\pi S_0}}
\left\{\frac34 w\sqrt{\frac{s}{s+1}}-\kappa\right\}.
\]

Thus the sharper sufficient condition is

\[
\epsilon\le\frac1{25},\qquad
\frac34w\sqrt{\frac{s}{s+1}}>\kappa.
\tag{C}
\]

The primitive conditions in (1) are a calculation-free corollary, because
`w>=9`, `s>=225`, `kappa<33/5`, and
`(27/4)sqrt(225/226)>33/5`.

For zero precision, the analytic-gradient argument in the current appendix
also applies with the improved condition `epsilon<=1/25`. In its notation,
`M(0)<8`, and choosing `rho(tau)=1-16 epsilon tau` gives
`M(tau)<=M(0)` and `rho(tau)>=9/25>0` over the full horizon. Thus the analytic
no-signal equilibrium exists, its verification remains valid, and the exact
positive-precision equilibria converge to it as precision decreases to zero.
On positive compact precision intervals, the existing parabolic estimates give
continuous dependence. Output is therefore continuous on finite closed precision
intervals. Strict inequality (2) excludes a sufficiently distant tail, and the
maximum is attained on the remaining compact interval.

## Joint design: explicit sufficient conditions with abstract u

Keep the manuscript's bounded prize and deadline set, and assume `u` is
continuously differentiable and increasing and `D` is continuous, nonnegative,
nondecreasing, and satisfies `D(0)=0`. Define the certified output margin

\[
\Delta_\kappa:=\frac1{\sqrt{2\pi S_0}}
\left\{
\frac{3\bar\theta\bar T}{4c}
\sqrt{\frac{S_0}{S_0+\sigma^2\bar T}}
-\kappa\sigma^2\bar T
\right\}.
\tag{J0}
\]

The sharp version of the sufficient conditions is

\[
S_0\ge25\bar\theta\bar T/c,\qquad
\Delta_\kappa>0,\qquad
\inf_b u'(b)>\frac{\bar\theta+D(\bar T)}{\Delta_\kappa},
\]

where the infimum is over the attainable output interval displayed below.
For a statement requiring no numerical evaluation of `kappa`, the following
slightly stronger primitive conditions are sufficient:

\[
\boxed{\bar\theta\ge13c\sigma^2,\qquad
S_0\ge25\bar\theta\bar T/c,}
\tag{J1}
\]

\[
\boxed{u'(b)>
\frac{5c\sqrt{2\pi S_0}}{\bar T}
\left(1+\frac{D(\bar T)}{\bar\theta}\right)
\quad\text{throughout the attainable output range}.}
\tag{J2}
\]

For a completely explicit interval on which to impose (J2), use

\[
\sqrt{S_0/(2\pi)}\le b\le
\sqrt{(S_0+\sigma^2\bar T)/(2\pi)}
+\sqrt{2\bar\theta\bar T/c}.
\]

The energy bound places every feasible exact output in this interval, and all
comparison bounds used below also belong to it. For concave `u`, it suffices
to check (J2) at the interval's upper endpoint. Linear utility is permitted but
is not required.

**Conclusion.** The exact joint problem has a global maximizing design with
positive prize, positive deadline, and finite precision. Every global maximizer
has those properties. Precision may be zero, and prize or deadline may lie at
their imposed upper limits. If all coordinates of a maximizer are interior and
the exact output is differentiable, it satisfies the manuscript's three exact
first-order conditions.

**Proof.** Use the candidate `(bartheta,barT,1/(S0 barT))`. The fixed-design
proof supplies (3) at this candidate. The high-precision estimate (8) is uniform
over the bounded prize--deadline set: along sequences with positive limiting
prize and deadline the value estimates are uniform; if either tends to zero,
the vanishing-effort bound applies. Its right-hand side increases with deadline,
so (8) at `barT` bounds the output of every high-precision design.

For this paragraph only, denote these lower and upper output bounds by `b_l`
and `b_u`. Under (J1), their difference satisfies

\[
b_l-b_u\ge
\frac{\bar\theta\bar T}{5c\sqrt{2\pi S_0}}.
\tag{J3}
\]

Indeed, after division by `sigma^2 barT/sqrt(2 pi S0)`, this reduces to

\[
\frac34\bar w\sqrt{\frac{s}{s+1}}-\frac{33}{5}
\ge\frac{\bar w}{5},
\qquad \bar w\ge13,\quad s\ge25\bar w\ge325,
\]

which follows from `(3/4)sqrt(325/326)-1/5>33/65`.
The fundamental theorem of calculus, (J2), and (J3) give

\[
u(b_l)-u(b_u)>\bar\theta+D(\bar T).
\]

Thus the finite candidate's net payoff strictly exceeds `u(b_u)`, while every
high-precision design has limiting net payoff at most `u(b_u)` because its costs
are nonnegative. Uniformity excludes all sufficiently large precisions. The
analytic-gradient argument at zero and parabolic continuity on positive compact
precision intervals give continuity on the remaining compact design box.
Weierstrass' theorem gives a maximizing design. Zero-prize and zero-deadline
designs have output at most `sqrt((S0+sigma^2 barT)/(2 pi))<b_u`; neither can
maximize. The first-order conditions follow by differentiating the exact
objective at an interior optimum.

**A nonempty example.** Let `c=sigma=barT=1`, `bartheta=13`, `S0=325`,
`D(T)=T`, and `u(b)=300 b`. Then (J1) holds, and the threshold in (J2) is
`5 sqrt(650 pi) (1+1/13)<244<300`. The finite candidate's proved lower output
is `7.4185203`, while the uniform high-precision upper output is `7.3491441`.
Its net objective advantage over the high-precision upper objective is at least
`300(7.4185203-7.3491441)-13-1>6.81`. These numbers only evaluate proved
inequalities; they do not substitute a numerical optimizer for the proof.

These are simple, nonempty, conservative primitive conditions. The first is a
strong-incentive condition; the second keeps initial uncertainty large relative
to the strategic effort scale; the marginal-utility condition ensures the
organizer values the output improvement enough to pay for the contest. They do
not identify the sharp regime boundary suggested by the numerical experiments.

## Additional benchmark corollary

Keep the manuscript's bounded prize and deadline set, increasing continuous
`u`, and nondecreasing continuous `D` with `D(0)=0`. Assume

\[
\boxed{S_0\ge25\bar\theta\bar T/c.}
\tag{9}
\]

The high-precision output has a continuous limit on the compact prize--deadline
set. A proof from the existing value-function limit is given below. Define an
optimal fully revealing benchmark by

\[
(\theta_\infty,T_\infty)\in
\operatorname*{argmax}_{0\le\theta\le\bar\theta,\,0\le T\le\bar T}
\{u(\mathcal B_\infty(\theta,T))-\theta-D(T)\}.
\]

If **one** such optimum satisfies

\[
\boxed{\theta_\infty\ge9c\sigma^2,}
\tag{10}
\]

then the joint problem attains a global maximum at finite precision, with
positive prize and deadline. Unlike (J1)--(J2), no nonempty example meeting
this benchmark-prize threshold has been established. The organizer may prefer
a moderate prize under full disclosure even at high output valuations. This
conditional corollary is therefore not the recommended main theorem.

For a formulation that does not use the output-limit lemma, it is enough that
some sequence of high-precision designs approaching

\[
L=\limsup_{\lambda\to\infty}
\sup_{0\le\theta\le\bar\theta,\,0\le T\le\bar T}
\{u(\mathcal B(\theta,T,\lambda))-\theta-D(T)\}.
\tag{11}
\]

has a subsequential prize limit at least `9 c sigma^2`. Condition (10) implies
this property by keeping its benchmark prize and deadline fixed.

Proof: let `(theta_n,T_n,lambda_n)` be the sequence in (11), and pass to a
subsequence with `(theta_n,T_n)->(theta_infty,T_infty)` and
`theta_infty>=9 c sigma^2`. Its deadline limit is positive. Otherwise the
energy bound makes its limiting payoff
`u(sqrt(S0/(2 pi)))-theta_infty`, strictly less than the feasible zero-prize,
zero-duration value, contradicting (11). The estimates leading to (8) are
uniform on compact positive prize--deadline sets, and therefore bound `L` by

\[
u\left(\sqrt{\frac{S_0+\sigma^2T_\infty}{2\pi}}
+\frac{33\sigma^2T_\infty}{5\sqrt{2\pi S_0}}\right)
-\theta_\infty-D(T_\infty).
\]

At this same limiting prize and deadline, (9) and (3) provide a finite
candidate with strictly higher output and exactly the same costs. Its objective
thus strictly exceeds `L`. All sufficiently high precisions are excluded.
The analytic and parabolic continuity arguments above are uniform on finite
design boxes under (9); at zero prize or deadline, use the vanishing-effort
energy bound. Compactness gives a finite joint maximizer. A zero-prize or
zero-duration objective is independent of precision and cannot exceed `L`;
such designs are therefore excluded as well.

If a maximizing design is differentiable and interior in all three coordinates,
its necessary conditions are

\[
u'(\mathcal B)\mathcal B_\theta=1,\qquad
u'(\mathcal B)\mathcal B_T=D'(T),\qquad
\mathcal B_\lambda=0.
\]

Existence of a maximizer does not imply an interior prize, deadline, or precision,
nor that every solution of these first-order equations is optimal.

### Output convergence at high precision

This argument uses the bounded-solution interior parabolic estimates already
used to construct the equilibria in Theorem 1. It does not verify an independent
game with singular terminal prizes, which is unnecessary for the design limit.

Fix positive prize and deadline. On any time strip separated from zero and `T`,
`a_lambda=lambda S_lambda^2` converges uniformly to `sigma^2` and is uniformly
elliptic and bounded. The normalized values are in `[0,1]`. The local gradient
and subsequent regularity estimates therefore have constants independent of
large `lambda` on a smaller strip. Together with the already established uniform
value convergence, interpolation yields uniform gradient convergence there.
The limiting policies are

\[
m_{i,\infty}=\frac\alpha2(1+f_w'(d_w))\partial_y d_w,
\qquad
m_{j,\infty}=\frac\alpha2(1-f_w'(d_w))\partial_y d_w.
\tag{12}
\]

For each terminal cutoff `b<T`, these functions and their spatial derivatives
are bounded on `[0,b]`, by their explicit formulas. Thus the limiting state
diffusion with drift `m_i,infty-m_j,infty`, volatility `sigma`, and initial law
`N(0,S0)` is uniquely defined on `[0,T)`.

Here is how to remove the initial layer without assuming policy convergence
there. At an initial cutoff `a>0`, couple the finite-precision state with the
Gaussian martingale component at time `a`; their expected distance is at most
`sqrt(2 alpha a)`. That component has variance
`S0+sigma^2 a-S_lambda(a)`, tending to `S0+sigma^2 a`. Its future Brownian
increments are independent. The limiting diffusion's state at time `a` differs
from the corresponding driftless state by `O(a)` for fixed parameters, since
its drift is bounded near zero. On `[a,b]`, policy and volatility convergence
and the Lipschitz property of the limiting drift give state convergence by
coupling and Gronwall. First send precision to infinity, then `a` to zero.
The error from the initial cutoff is bounded by a constant depending on `b`
times `sqrt(a)`. The energy budget gives uniform second-moment bounds for the
states and allows convergence of expectations of Lipschitz functions.

The output can be evaluated at a terminal cutoff `b=T-e` using

\[
E\int_0^b\frac{m_i+m_j}{2}\,dt+\frac12 E|\tilde y_b|.
\]

Its difference from exact terminal output is bounded by the remaining effort
and filtering/innovation terms. The effort terms are `O(sqrt(e))` by the energy
budget. The remaining state martingale has variance
`sigma^2 e+S_lambda(b)-S_lambda(T)`, and the posterior error at `T` contributes
at most `sqrt(S_lambda(T))/2`. After precision tends to infinity the latter
errors are also `O(sqrt(e))`. The same effort bound passes to the limiting
diffusion on truncated intervals by Fatou, giving an integrable limiting policy
and a terminal state. Taking `e` to zero proves

\[
\mathcal B(\theta,T,\lambda)\longrightarrow
\mathcal B_\infty(\theta,T)
=E\int_0^T\frac{m_{i,\infty}+m_{j,\infty}}2dt
+\frac12E|Y_{T,\infty}|.
\tag{13}
\]

All cutoff estimates and interior convergence are uniform on positive compact
prize--deadline sets. If prize or deadline tends to zero, the basic gain bound
`sqrt(2 theta T/c)` applies instead. Sequential compactness therefore gives
uniform convergence on the full bounded prize--deadline set and continuity of
the limit. This justifies the benchmark used in (10).

## Scope of the advance

The fixed-design result and (J1)--(J2) are explicit exact theorems with simple
primitive conditions and a demonstrated nonempty joint-design example.
The utility condition concerns marginal output value, leaving the functional
form abstract. Increasingness or concavity of an otherwise unrestricted `u`
alone cannot force a positive prize. The separate benchmark corollary removes
that marginal-value condition but has not been shown to have a nonempty
strong-incentive region; it should not replace the explicit joint theorem.

This does not prove the numerical threshold `3/2`, strict positivity of optimal
precision, or optimality of full disclosure for weak prizes. Those remain
separate questions. The source manuscript is not changed by this research note.
