# A nonempty positive-prize full-disclosure window

## 1. Scope and proof status

The current target is ONE positive-prize full-disclosure joint-optimal point,
with the existing ordinary u,D kept fixed. A window, a precision-boundary
formula, and a single-crossing theorem are not required.
The manuscript now includes the fixed-primitives joint-existence result in
Proposition 1(ii), in addition to the general necessary prize condition.
The fixed-primitives mathematical existence result is now closed in Sections
40--42, using the previously audited exact weak-prize comparison of Section
19. The uniform weak-prize comparison, global competitive-output estimate,
and joint-existence argument have now been integrated, in blue, into the
manuscript appendix. The conclusion and limitations have been reconciled
with the precise scope. Local draft-mode LaTeX checks pass, with resolved
cross-references and no overfull boxes. The
built-in preview compiler cannot load the companion informs3.cls, so its
check fails before reaching the source changes. The witness has
extremely large risk; it is not a validation of Figure 3's plotted range.

The following paragraph records the earlier stage of the argument, not its
current proof status. Section 14 proves full-disclosure benchmark prize exit;
Section 17 proves an exact, uniform high-precision output envelope with
coefficient 2; Section 18 proves global payoff exclusion conditional on a
sharper coefficient-2/3 envelope; Section 19 consolidates the exact uniform
weak-prize precision comparison using the original small-prize analytic
estimates. The remaining joint-design obstruction is the sharp global
equilibrium-output bound, or an alternative exclusion of remote competitors.
At that stage the fixed-primitives existence theorem was NOT closed. Section 20 closes a weaker
existence-in-the-admissible-class result: one fixed external utility and delay
cost support an open positive-prize full-disclosure joint-design window.
This uses the specialized small-headroom construction, not the missing sharp
global output bound.

Sections 2--7 retain the historical small-headroom construction and its
reverse audit. That construction hides extreme local utility and delay-cost
restrictions and is not the proposed moderate-shape manuscript extension.
Neither a numerical classification nor the new conditional payoff proof
supplies the missing equilibrium bound.

Latest fixed-primitives progress: Section 21 excludes all high-precision
competitive prizes and deadlines at an explicit risk/entry-cost point for the
figure's unchanged exponential utility and quadratic delay cost. Coefficient 2
suffices there; the coefficient-2/3 requirement is not necessary for this
fixed-utility point. The remaining unrestricted joint-design obstacle is its
low-precision competitors. The user's single-point target is NOT closed.

Section 22 now independently closes zero precision on the complete competitive
prize range and extends exclusion to a small positive precision neighborhood.
The remaining gap is a finite intermediate-precision interval; no overlap of
the low- and high-precision certificates has been established.

Section 23 extends the upper certificate to lambda>=18 sigma^2/25 by retaining
the actual-gradient constants and checking the slightly larger payoff
coefficient. The currently uncovered interval is
lambda_0<lambda<18 sigma^2/25. The fixed-primitives single-point target remains
open.

## 2. The equilibrium-comparison input (consolidated in Section 19)

For the selected symmetric equilibria, Section 19 establishes a constant
`w_* > 0` such that, for fixed prize and deadline,

\[
0<\frac{\theta}{c\sigma^2}<w_*,\qquad
\frac12\leq\frac{S_0}{\sigma^2T}\leq2
\quad\Longrightarrow\quad
\mathcal B(0,\theta,T,\infty;S_0)
>\mathcal B(0,\theta,T,\lambda;S_0)
\quad(0\leq\lambda<\infty).
\tag{W}
\]

This must compare exact equilibrium outputs, uniformly over all finite
precisions, including arbitrarily large precision and the selected weak-prize
no-signal equilibrium. A positive second-order coefficient alone is not enough.
Bounded-precision derivative estimates and a separate high-precision-tail
comparison can cover the entire range; the latter must control the remainder
relative to the vanishing disclosure gap. No global continuity at zero precision
is needed if its weak-prize equilibrium is verified separately.

We also use the full-disclosure first-order formula, at fixed positive T,

\[
\mathcal B(0,c\alpha,T,\infty;S_0)
=\sqrt{\frac{S_0+\sigma^2T}{2\pi}}
+\frac{\alpha T}{\sqrt{2\pi(S_0+\sigma^2T)}}+o(\alpha).
\tag{F}
\]

The full-disclosure equilibrium is explicitly verified in the current paper.
Its energy bound implies the same common output screen as for a verified finite
precision equilibrium:

\[
\mathcal B\leq\sqrt{\frac{S_0+\sigma^2T}{2\pi}}+
\sqrt{2\alpha T},\qquad \alpha=\theta/c.
\tag{E}
\]

For finite precision, (E) follows from each player's outside option of zero
effort. Write the two expected equilibrium payoffs as V_i,V_j>=0. Since exactly
one prize is awarded and instantaneous effort cost is c q_k^2/2,

\[
\frac c2\mathbb E\int_0^T(q_i^2+q_j^2)dt
=\theta-V_i-V_j\le\theta.
\]

Thus the SUM of the two effort energies is at most 2 alpha; the individual
energy bound is 2 alpha, not alpha. Coupling private scores with zero-effort
scores gives max(x_i,x_j)-max(x_i^0,x_j^0)<=max(A_i,A_j)
<=sqrt(A_i^2+A_j^2), where A_k=int_0^T q_k dt. Cauchy--Schwarz and Jensen
then give (E). This does not accept an unverified numerical policy as an
equilibrium.

## 3. External sufficient conditions and joint localization

Fix primitive S0,sigma>0. For this proof only, write

\[
\tau=\frac{S_0}{\sigma^2},\qquad b=\sqrt{\frac{S_0}{\pi}},
\qquad a=u'(b)>0.
\]

Neither tau nor b is a design choice. Assume the paper's increasing concave C2
utility and increasing convex C1 delay cost, with D(0)=D'(0)=0. Additionally,
utility has a finite upper limit and

\[
D'(\tau)=\frac{a\sigma^2}{4\sqrt{\pi S_0}},\qquad
u_\infty-u(b)+D(\tau)<\varepsilon ab.
\tag{A}
\]

The derivative equality merely calibrates this particular external environment;
it does not assert anything about a joint optimum. It makes tau the unique
zero-prize optimal deadline, because the zero-prize objective is strictly
concave and its derivative vanishes there. Let its payoff be V0=u(b)-D(tau).
In particular V0 exceeds the zero-deadline payoff: the zero-prize objective has
strictly positive derivative at zero and its maximum is at tau>0.

Define, only in the proof,

\[
c_e=\frac{a\sqrt{S_0}}{2\sqrt\pi\,\sigma^2},\qquad
\delta=1-\frac{\sqrt3}{2}>0,
\quad
0<\varepsilon<\min\left\{\frac{w_*}{8},\frac18,
\frac{\delta}{8},\frac{\delta^2}{64\pi}\right\}.
\]

For every c in [ce/2,ce), any design not worse than V0 has

\[
\theta+D(T)\leq u_\infty-V_0<\varepsilon ab.
\tag{1}
\]

Since c sigma^2>=ab/4, this gives

\[
0\leq\frac{\theta}{c\sigma^2}<4\varepsilon<w_*.
\tag{2}
\]

The supporting line of D at tau and (1) give

\[
D'(\tau)(T-\tau)<u_\infty-u(b)<\varepsilon ab.
\]

As tau D'(tau)=ab/4, we obtain T<(1+4 epsilon)tau.
For T<=tau/2, (E) and (2) instead give

\[
\mathcal B\leq\frac{\sqrt3}{2}b+
2\sqrt{\pi\varepsilon}\,b
<\left(1-\frac{3\delta}{4}\right)b.
\]

Concavity implies u'(B)>=a for 0<=B<=b, hence

\[
u(b)-u(\mathcal B)>\frac{3\delta}{4}ab
>D(\tau).
\]

Such a design has payoff below V0, contradicting its selection. Therefore all
potential winners satisfy

\[
\frac\tau2<T<(1+4\varepsilon)\tau,
\qquad \frac12<\frac{S_0}{\sigma^2T}<2.
\tag{3}
\]

Equations (1)--(3) control all potentially winning joint designs, not just local
stationary points. A distant high-prize competitor is excluded by the budget,
and a short-deadline competitor by its output bound. This replaces the
unproved continuous-exit step.

## 4. Full-disclosure window and attainment, using (W)

At the zero-prize deadline tau, (F) yields the prize right derivative

\[
\left.\partial_\theta
\{u(\mathcal B)-\theta-D(\tau)\}\right|_{0+}
=\frac{c_e}{c}-1>0.
\]

Thus some full-disclosure positive prize beats V0 whenever c<ce.
Its maximum is attained: every design beating V0 lies in the bounded prize
range (1) and the compact closure of the positive deadline range (3), where the
explicit full-disclosure objective is continuous, including theta=0. Designs
outside this set cannot improve V0. No continuity in lambda at zero is invoked.

By (W), every positive-prize finite-precision design with payoff at least V0 is
strictly improved by retaining its prize and deadline and using full disclosure.
Designs below V0 are already inferior. Zero-prize designs are inferior to the
positive-prize candidate; zero deadlines are inferior to V0. Consequently the
full-disclosure optimum is a global optimum of the three-parameter problem,
and every global optimum satisfies theta*>0, T*>0, lambda*=infinity.

This is a genuine joint-design comparison, using (W), rather than a
fixed-prize/fixed-deadline optimum. It establishes a cost interval of positive
length, not the actual regime boundary or its ordering for every risk level.

## 5. The external environment is nonempty

Fix S0,sigma,a>0 and tau,b as above. Let eta>0 and define utility by its marginal:

\[
u_\eta'(B)=\frac{2a}{1+\exp((B-b)/(\eta b))},\qquad u_\eta(0)=0.
\]

This is a bounded, C-infinity, strictly increasing, strictly concave utility,
with u_eta'(b)=a and u_eta,infinity-u_eta(b)=2a eta b log2.
Put h=a sigma^2/(4 sqrt(pi S0)) and q=exp(-1/eta). Define

\[
D_\eta(T)=\frac{h}{1-q}
\left\{\eta\tau\left[e^{(T/\tau-1)/\eta}-q\right]-Tq\right\}.
\]

Then D_eta(0)=D_eta'(0)=0, D_eta is C-infinity, strictly increasing on positive
T and strictly convex, and D_eta'(tau)=h. Moreover

\[
0<D_\eta(\tau)=h\tau\left[\eta-\frac{q}{1-q}\right]
<\eta h\tau=\frac{\eta ab}{4}.
\]

The strict inequalities use exp(1/eta)-1>1/eta. Consequently

\[
u_{\eta,\infty}-u_\eta(b)+D_\eta(\tau)
<\eta ab\left(2\log2+\frac14\right).
\]

Choose eta<epsilon/(2 log2+1/4). Conditions (A) then hold strictly and the
positive-prize full-disclosure window, using (W), is nonempty. The construction
uses only external primitives; it contains no optimized prize, deadline or
precision in its utility or cost definitions.

This construction proves nonemptiness, not mildness. Utility saturates sharply
near b and marginal delay cost rises sharply near tau as eta decreases. It does
not establish a window for the figure's u(B)=3(1-exp(-2B)), D(T)=T^2/2. It also
does not supply an open set in all primitive function spaces: persistence under
risk perturbations requires a separate continuous-reference-deadline argument,
now supplied in Section 20.

## 6. Historical outstanding tasks (status superseded by Sections 19--20)

1. The relative remainder and selected no-signal equilibrium input in (W)
   have now been consolidated in Section 19 using the original analytic proof.
2. Decide whether the local sharpness in (A) is economically acceptable; do not
   describe it as a consequence of ordinary concavity/convexity.
3. A moderate-utility/quadratic-delay window remains an unsolved stronger target.
   The new localization removes one logical gap for this specialized external
   family, but does not prove the general three-region theorem.

## 7. Reverse audit: small headroom conceals strong local shape

Historical audit: the shape warning remains valid. The outstanding precision
input listed below has since been consolidated in Section 19; Section 20 uses
the certificate only for an explicitly weakened existence-in-the-class theorem,
not as a moderate-shape sufficient region.

### Issues, in order of severity

1. The precision comparison (W) remains an independent nontrivial input; its
   historical proof depends on long high-precision spectral and propagation
   estimates, not merely a positive second-order coefficient.
2. Conditions (A) do not deliver mild shape assumptions. They force extreme
   local utility curvature and delay elasticity, even without naming them.
3. Compact-precision approximation cannot establish infinity as an exact optimum.
   Shrinking the budget separately for each finite precision cutoff changes the
   external utility/cost environment; it does not close the original problem.

### Direct derivation of the concealed restrictions

From D(tau)<epsilon ab and tau D'(tau)=ab/4,

\[
\frac{\tau D'(\tau)}{D(\tau)}>\frac1{4\varepsilon}.
\tag{4}
\]

Furthermore u_infinity-u(b)<epsilon ab. If u'(B)>=a/2 throughout
[b,b+2 epsilon b], its integral on this interval would be at least epsilon ab,
contradicting this strict tail bound. Some B1 in that interval therefore has
u'(B1)<a/2. The mean-value theorem applied to log u' gives a point xi in (b,B1)
with

\[
-\frac{\xi u''(\xi)}{u'(\xi)}
>\frac{\log2}{2\varepsilon}.
\tag{5}
\]

Strict positivity of u' follows from concavity and strict increase on the whole
domain. Thus the logarithm is legitimate. Equations (4)--(5) are necessary
consequences of the certificate, not additional assumptions.

Our conservative bound epsilon<delta^2/(64 pi) is about 8.93e-5. It consequently
forces reference-date delay elasticity above approximately 2,800 and relative
utility curvature above approximately 3,880 at some nearby output. These
numeric values summarize an analytic implication; they are not calibrated
economic thresholds. Improving these crude localization constants would not
remove the structural 1/epsilon blow-up when epsilon is sent to zero to absorb
an unspecified weak-incentive comparison threshold.

### Why bounded precision is not enough

Suppose the exact weak-prize outputs admit, uniformly on a bounded precision
range, expansions B(w,s,ell)=b0(s)+w b1(s)+w^2 b2(s,ell)+O(w^3).
Even granting b2(s,infinity)>b2(s,ell) for every finite ell, this proves exact
dominance only on fixed compact ell ranges: the positive second-order gap
approaches zero as ell increases. An absolute O(w^3) bound cannot settle its
sign for all ell at one positive w. A bound for the *difference* relative to
this vanishing gap, or a direct exact comparison, is indispensable.

In logical terms,

\[
\forall L<\infty\ \exists\varepsilon_L>0
\]

does not imply a positive epsilon working for all L. Choosing different
u_eta,D_eta for each L likewise does not construct one environment with exact
optimal lambda=infinity. A large optimal numerical precision is not a proof
of a full-disclosure window.

### Revised research structure and unresolved requirement

Do not use this certificate in the main paper, and do not revive the long
strong-shape proof under a new label. A satisfactory extension still requires
either a direct, short exact weak-precision comparison combined with modest
global prize localization, or an independently proven continuous-exit point
under mild external primitives. Neither has been closed here. No unconditional
full-disclosure theorem or counterexample to its existence is claimed.

## 8. Modest shapes and deliberately imprecise relative placement

### Proposed change of proof direction

Abandon (A) as an economic certificate. The useful historical local calculation
uses ordinary quadratic delay cost rather than a small-headroom/high-elasticity
budget. The remaining global issue must be solved directly, not assumed away.

Let T0 be the unique zero-prize optimum at a fixed sigma, and only for this
calculation put b=sqrt((S0+sigma^2 T0)/(2 pi)), A=b u'(b),
q=1+S0/(sigma^2 T0), eta=-b u''(b)/u'(b),
xi=T0 D''(T0)/D'(T0). Delay is C2 locally for this calculation. The zero-prize
first-order condition gives D'(T0)=A/(2qT0) and
ce=u'(b)T0/sqrt(2 pi(S0+sigma^2 T0)).

For every fixed precision with the selected weak-prize expansion, the exact
objective derivatives at theta=0,c=ce are

\[
J_{\theta\theta}=\frac{-\eta+H_\lambda}{A},\quad
J_{\theta T}=\frac{1-(\eta+1)/(2q)}{T_0},\quad
J_{TT}=-\frac{A(\eta+1+2q\xi)}{4q^2T_0^2},
\]

where the previously computed exact Gaussian second-order coefficient satisfies
0<=H_lambda<=H_infinity<q/3. The theta--T cross derivative includes the
deadline response; it cannot be omitted. For clarity, the first-order output
derivative used here is B_theta=T/[c sqrt(2 pi(S0+sigma^2 T))], not a
steady-state derivative.

For eta>=1, xi>=1 and 2S0<=sigma^2 T0<=4S0, we have 5/4<=q<=3/2.
Since JTT<0, completing the quadratic form gives

\[
A\left(J_{\theta\theta}-\frac{J_{\theta T}^2}{J_{TT}}\right)
< -\eta+\frac q3+
\frac{(2q-\eta-1)^2}{\eta+1+2q\xi}.
\]

The right side decreases in eta and xi: its derivatives are respectively
-4q^2(1+xi)^2/(eta+1+2qxi)^2 and
-2q(2q-eta-1)^2/(eta+1+2qxi)^2. At eta=xi=1 it is
(7q^2-14q+3)/(3(q+1)), increasing on [5/4,3/2] and at most -3/10.
Thus the exact local Hessian is negative definite at every fixed precision.

Relative utility curvature one is a familiar, moderate shape restriction;
xi=1 holds for D(T)=dT^2/2. This is not the thousands-level restriction in
Section 7. The statement is a local derivative result, using the earlier exact
coefficient calculation; it is NOT global exclusion of positive prizes.
Nor is a common neighborhood in unbounded precision asserted without uniform
continuity of the relevant second derivatives. Different fixed-precision local
neighborhoods cannot simply be intersected and assumed to have positive size.

### Why approximate relative placement needs no new shape condition

Suppose a positive-prize full-disclosure open window has actually been proved
near some (c_d,sigma_d) with sigma_d>0. Shrink that window so sigma>sigma_d/2.
The existing finite-feedback result remains valid after replacing its risk
threshold by any smaller positive number, in particular one below sigma_d/2.
Then the two sufficient windows are risk-separated, with the displayed
finite-feedback window at lower risk. This observation uses neither a
single-crossing theorem nor monotonicity of a true precision boundary.

This describes sufficient subsets only: it does not say every finite-precision
optimum has lower risk than every full-disclosure optimum. It does not prove
the full-disclosure window exists. At fixed sigma, once zero prize is globally
optimal, it remains globally optimal as c increases, since a fixed effective
incentive alpha costs c alpha. This cost-order observation likewise does not
establish continuous prize exit or the existence of a positive-prize
full-disclosure interval immediately below exit.

### What remains before Proposition 1 can be expanded

The modest local calculation is insufficient: an intermediate-precision
positive-prize design outside its local neighborhood may still outperform it.
The ledger's Section 117 proposes a global output envelope to exclude such
designs, but that envelope is unproved for intermediate precision. It must not
be treated as a new utility/delay assumption or as an established bound.

Accordingly no moderate-shape global window is claimed. Relaxing the requested
relative placement does remove a separate ordering task, but not the two
essential exact comparisons: global prize/deadline competition, and all-finite-
precision competition against full disclosure. The paper remains unchanged.

## 9. Sharper moderate-shape local calculation

This is a new simplification of Section 8's Schur calculation, not a completed
global window. Keep xi>=1 (ordinary quadratic delay gives equality), but remove
the upper innovation-variance limit. Suppose sigma^2 T0>=2 S0, so 1<q<=3/2.
The right-hand Schur bound at xi=1 simplifies exactly to

\[
-\eta+\frac q3+\frac{(2q-\eta-1)^2}{\eta+1+2q}
=\frac{14q^2-11q+3-(17q-3)\eta}{3(\eta+1+2q)}.
\tag{6}
\]

Since 17q-3>0, it is nonpositive whenever
eta>=(14q^2-11q+3)/(17q-3). This ratio is increasing for q>=1: its derivative is
(238q^2-84q-18)/(17q-3)^2>0. Its maximum on [1,3/2] is exactly 4/5. The exact
Hessian's Schur complement is *strictly* below (6), because H_lambda<q/3.
Thus eta>=4/5 and xi>=1 suffice for strict local negative definiteness at every
fixed precision. There is no upper bound on sigma^2 T0 in this local statement.

This makes the local requirement milder than logarithmic relative curvature.
It does not provide uniform higher derivatives in unbounded precision, exclude
remote positive-prize designs, or turn local maxima into joint global optima.

## 10. A diagnostic alternative envelope, not an established inequality

The existing intermediate-precision envelope is still unproved. To avoid
recycling a false prior-independent bound, a diagnostic checked the candidate
logarithmic screen

\[
\log\!\left(\frac{\mathcal B}{\sqrt{(S_0+\sigma^2T)/(2\pi)}}\right)
\leq\frac{\alpha T}{S_0+\sigma^2T}
\quad\text{when}\quad S_0\leq\sigma^2T,
\tag{7?}
\]

on 7,141 successful cached calculations with theta/(c sigma^2)<=1. No violation
was observed. The checks use existing numerical outputs, some coarse; they
prove neither (7?) nor a joint-design classification. The physical-policy
failure guards must not be bypassed on the strength of this diagnostic.

Even if (7?) were proved, it does not automatically exclude global jump entry.
For log-shaped utility and quadratic delay, its linear incentive term can
increase when the deadline is moved above T0. The deadline cross effect is
essential; a strict negative quadratic correction or another global inequality
is needed. Therefore (7?) must not be presented as a closed proof shortcut.

Current conclusion: local shape thresholds are now genuinely mild, but there
is still no closed global positive-prize full-disclosure theorem under these
conditions. No theorem, assumption or paper text is changed to conceal this.

## 11. A closed local prize-entry lemma, and its precise limitation

The negative Hessian can be used to prove more than a derivative sign. It
gives an actual positive-prize branch of exact full-disclosure designs. This
section deliberately does not call those designs global joint optima.

Fix sigma>0 and the unique zero-prize optimal deadline T0. Suppose D is C2
near T0, sigma^2 T0>=2 S0, and the local relative utility and delay curvatures
eta,xi in Section 8 satisfy eta>=4/5, xi>=1. Let J denote the *exact*
full-disclosure objective. Its explicit full-disclosure equilibrium gives a
C2 objective near zero prize, including the one-sided prize derivatives.
The expression can be extended mathematically to small negative prizes to
apply the implicit function theorem; only its positive-prize branch is used
as a contest. Equivalently, one can use the one-sided version of that theorem.

At c=ce, theta=0,T=T0, both first derivatives vanish. Write h11,h12,h22 for
the exact theta--T Hessian entries in Section 8 and

\[
S=h_{11}-h_{12}^2/h_{22}<0.
\]

Its determinant is h22 S>0, so the stationary equations have a unique C1
branch theta(c),T(c) near this point. Since
J_theta(0,T0,c)=ce/c-1 and J_T(0,T0,c)=0, differentiation of these equations
at ce gives

\[
\begin{pmatrix}h_{11}&h_{12}\\h_{12}&h_{22}\end{pmatrix}
\begin{pmatrix}\theta'(c_e)\\T'(c_e)\end{pmatrix}
=\begin{pmatrix}1/c_e\\0\end{pmatrix}.
\]

Consequently, as delta=ce-c decreases to zero through positive values,

\[
\theta(c)=\frac{\delta}{c_e(-S)}+o(\delta)>0,\qquad
T(c)=T_0-\frac{h_{12}}{h_{22}}\theta(c)+o(\delta)>0.
\tag{8}
\]

By continuity, the Hessian remains negative definite along the branch. These
are strict local maxima in prize and deadline *within the full-disclosure
benchmark*. They strictly improve on every zero-prize design: expanding the
exact objective around (0,T0) gives

\[
J(\theta(c),T(c),c)-J(0,T_0,c)
=\frac{\delta^2}{2c_e^2(-S)}+o(\delta^2)>0.
\tag{9}
\]

The linear prize term is (delta/ce+o(delta)) theta. Optimizing the deadline
changes the quadratic prize term to S theta^2/2, which produces (9). Omitting
that deadline response would give the wrong coefficient. Zero-prize output
is independent of c and precision, so its globally best payoff remains
J(0,T0,c).

Thus a moderate-shape local positive-prize full-disclosure branch exists and
beats all zero-prize designs. The proof uses no small utility-headroom budget
and no thousands-level curvature condition. Its full-disclosure second
derivative input is the exact Gaussian integral in ledger Section 66, not a
truncated policy used as the data-generating process.

Two statements are still missing for the requested paper theorem:

1. No distant positive-prize design beats this branch after joint optimization
   of prize, deadline and precision. Negative definiteness and (9) do not
   establish this.
2. For this one environment and one positive prize neighborhood, full disclosure
   beats *every* finite precision, including arbitrarily large precision. A
   different valid neighborhood for each fixed finite precision is insufficient.

These are not additional assumptions to insert in Proposition 1. They are
the remaining mathematical tasks. Accordingly, (8)--(9) are research-note
results only, and do not justify a full-disclosure region in the manuscript.

## 12. A short exact full-disclosure output envelope

This section closes an endpoint estimate by direct HJB verification. It does
not use a truncated equilibrium or impose any utility/delay curvature. For
0<theta/(c sigma^2)<=2, the exact full-disclosure output satisfies

\[
\mathcal B(0,\theta,T,\infty;S_0)
\leq \sqrt{\frac{S_0+\sigma^2T}{2\pi}}
+\frac{\theta T}{c\sqrt{2\pi(S_0+\sigma^2T)}}
\left(1+\frac{\theta}{2c\sigma^2}\right).
\tag{10}
\]

The bound is global in prize/deadline within its stated incentive range, not
an asymptotic expansion. This simplifies the full-disclosure input to the
conditional global comparison in ledger Section 117. Intermediate precision
is still not covered by (10).

### Direct verification

Only in this proof, put alpha=theta/c, w=alpha/sigma^2, tau=T-t, Q=sigma^2 tau,
G_Q(y)=exp(-y^2/(2Q))/sqrt(2 pi Q), and g=2 Phi(y/sqrt(Q))-1. The exact
full-disclosure equilibrium in the paper has R=f(d) and

\[
K(d)=K(1)g,\quad
K(d)=\frac{d+\sinh(wd)/w}{2},\quad
f'(d)=\tanh(wd/2).
\]

Let H=d_y and A=R_y. Thus H=2 K(1)G_Q/K'(d), A=f'(d)H, and the equilibrium
gap drift and total effort are alpha A and alpha H respectively. Since
K'>0 and K is odd, y,d,A have the same sign.

Define the zero-effort conditional best output

\[
\beta(y,Q)=\sqrt Q\,\phi(y/\sqrt Q)+y\,[\Phi(y/\sqrt Q)-1/2].
\]

It satisfies beta_y=g/2 and the backward heat equation, as does G_Q. Consider
the candidate upper bound

\[
U(y,t)=\beta(y,Q)+\alpha\tau(1+w/2)G_Q(y).
\]

For L=partial_t+(sigma^2/2)partial_yy+alpha A partial_y, differentiation gives
the exact residual

\[
LU+\frac{\alpha H}{2}
=\alpha\left\{\frac H2-(1+w/2)G_Q
+A\left[\frac g2+\alpha\tau(1+w/2)(G_Q)_y\right]\right\}.
\tag{11}
\]

The last term involving (G_Q)_y is nonpositive because A y>=0. It remains
to bound H/2+A g/2. By symmetry it suffices to take 0<=d<=1. Convexity of
K on that interval gives K(d)<=d K(1), hence 0<=g<=d. Also
f'(d)<=wd/2 and K'(d)=(1+cosh(wd))/2>=1+w^2 d^2/4. Consequently,

\[
\frac{H/2+A g/2}{G_Q}
=\frac{K(1)}{K'(d)}[1+f'(d)g]
\leq K(1)\frac{1+wd^2/2}{1+w^2d^2/4}
\leq K(1)\frac{1+w/2}{1+w^2/4}
\leq1+w/2.
\tag{12}
\]

The middle inequality follows by differentiating the ratio with respect to
d^2: its derivative has the sign of w/2-w^2/4, nonnegative for w<=2.
For the last inequality, the positive series for sinh gives, when 0<w<=2,

\[
K(1)-1=\frac12\sum_{n\ge1}\frac{w^{2n}}{(2n+1)!}
\leq\frac{w^2}{8}\left(\frac{\sinh2}{2}-1\right)<\frac{w^2}{4}.
\]

Here sinh(2)<6, for example from e<3. Thus (11) is nonpositive everywhere
before the deadline. The candidate U tends uniformly to |y|/2 as tau tends
to zero: both the Gaussian smoothing error and tau G_Q are O(sqrt(tau)).
Along the verified full-disclosure equilibrium, Dynkin's formula, localized
in space and before the deadline, now bounds terminal best output plus the
accumulated average effort by U. The equilibrium's finite expected effort
energy bounds E sup|y_t|, and U has at most linear growth, so the localization
limits are legitimate. This proves the conditional-on-initial-gap upper bound.

Finally average over Y0~N(0,S0). Gaussian convolution gives
E beta(Y0,sigma^2 T)=sqrt((S0+sigma^2 T)/(2 pi)) and
E G_(sigma^2 T)(Y0)=1/sqrt(2 pi(S0+sigma^2 T)), proving (10).

### Why the same calculation does not yet prove the finite-precision bound

For finite positive lambda, Q_t=S_t+sigma^2(T-t) has derivative
Q'_t=-a_t, where a_t=lambda S_t^2. Therefore beta(y,Q_t) and G_(Q_t)(y)
again solve the appropriate backward heat equation. The designer HJB has
the same drift/source form, so its candidate residual is again (11).
However, its equilibrium generally does NOT satisfy R=f(d) or
K(d)=K(1)g. Inequality (12), the decisive gradient comparison, has therefore
only been proved for full disclosure. Substituting those identities into a
finite-precision proof would change the equilibrium being compared.

A sufficient new task is to establish the finite-precision analogue of the
residual inequality (11), or another scalar supersolution giving (10).
Outward drift and (12) must not be assumed for that game. The original coupled
gradient equations have cross spatial derivatives and are not a cooperative
scalar system, so the maximum principle does not supply them automatically.

Status: the short exact full-disclosure envelope is closed. The requested
global full-disclosure window remains unclosed because the intermediate-
precision competitors have not yet been bounded or excluded.

### Diagnostic of the finite-precision residual, not a proof

The exact residual (11), rather than only aggregate output, was tested using
the existing adaptive-time HJB solver. With sigma=T=c=1, the 27 combinations
S0 in {1/2,1,2}, w in {0.1,0.45,0.875}, and lambda in {0.01,1,100} used dy=0.04
and 500 base time steps. Over |y|<=3 sqrt(Q_t) and T-t>0.005, every computed
normalized residual (11)/(alpha G_Q) was negative. The largest was about
-0.0205883, at S0=1/2,w=0.1,lambda=100,t=0,y=-3.64.

These are coarse-grid diagnostic values, without a certified discretization
error or coverage of the spatial tails and terminal layer. They do not verify
the residual on the continuum, or justify substituting the full-disclosure
identities into finite precision. They do, however, identify a concrete short
proof target: one scalar supersolution inequality for the *actual* coupled HJB
gradients. No synthetic or empirical data, numerical region labels, figure,
or manuscript assumption was changed by this diagnostic.

## 13. Exact winner-weighted effort representation

The weights below are the zero-future-effort winning probabilities, NOT the
equilibrium winning probabilities. No information-ranking claim follows just
from this identity. For every verified finite-precision equilibrium of finite
expected effort energy, including a separately verified no-signal equilibrium,

\[
\mathcal B-b(T)
=\mathbb E\int_0^T\left[
m_i(\tilde y_t,t)\Phi\!\left(\frac{\tilde y_t}{\sqrt{Q_t}}\right)
+m_j(\tilde y_t,t)\Phi\!\left(-\frac{\tilde y_t}{\sqrt{Q_t}}\right)
\right]dt.
\tag{13}
\]

Here Q_t=S_t+sigma^2(T-t). To prove it, use beta from Section 12 with Q=Q_t.
Since Q'_t=-lambda S_t^2, beta_t+(lambda S_t^2/2)beta_yy=0 and
beta_y=Phi(tilde y/sqrt(Q_t))-1/2. Ito's formula therefore gives
E beta(tilde y_T,S_T)=beta(0,Q_0)+E int(m_i-m_j)beta_y dt.
Best terminal output is the expected average score plus this terminal beta;
the average-score effort is int(m_i+m_j)/2 dt. Adding the terms proves (13).
The beta_y martingale has finite second moment because |beta_y|<=1/2 and
int lambda S_t^2 dt=Q_0-S_T<=Q_0. Finite effort energy controls the drift
integral. Thus the expectation step does not require a conjectured state law.

For full disclosure, use Q_t=sigma^2(T-t) and the observed gap, then average
the initial beta over N(0,S0); its average is again b(T). The same argument
passes to T using bounded beta_y and finite effort energy.

A sufficient route to the finite-precision version of (10) is to prove both
outward strategic drift and the weighted-gradient bound

\[
y(m_i-m_j)\ge0,\qquad
m_i\Phi(y/\sqrt{Q_t})+m_j\Phi(-y/\sqrt{Q_t})
\le\alpha(1+w/2)G_{Q_t}(y).
\tag{14?}
\]

Indeed G_(Q_t)(tilde y_t) is then a positive local supermartingale, because
its heat terms cancel and its drift is -(m_i-m_j)y G/Q_t<=0. Its expectation
is at most G_(Q_0)(0). Integrating (13) proves the envelope. These are proof
targets for the actual equilibrium, not new assumptions on the primitives.
The same 27 coarse solves described in Section 12 support (14?) on the sampled
interior spatial/time domains; the largest weighted-gradient excess normalized
by alpha G was approximately -0.0205327. This is not a continuum certificate.

To record why scalar comparison is not automatic, use reverse physical time
rho=T-t, a=lambda S_(T-rho)^2, A=R_y,H=d_y,g=2 Phi(y/sqrt(Q))-1, and N=H+gA.
The actual gradient equations give

\[
N_\rho-\frac a2N_{yy}
=\frac\alpha2(A-gH)N_y
+\left\{\frac\alpha2[(1+g^2)H+2gA]-a g_y\right\}A_y
-\frac\alpha2(A-gH)g_y A.
\tag{15}
\]

The uncontrolled A_y term prevents applying a scalar maximum principle to N.
The full-disclosure identities eliminate this problem there; substituting them
in a finite-precision equilibrium would be an invalid comparison.

## 14. Global full-disclosure prize exit with moderate eventual utility curvature

This is a global result for the full-disclosure benchmark, not yet the original
three-parameter problem. It replaces the previous purely local prize-entry
argument, without using the unproved finite-precision envelope.

### External shape and parameter conditions

In addition to increasing concave C2 utility, suppose there is a fixed output
threshold b_u>0, determined by u alone, such that

\[
-B u''(B)\ge2u'(B)\quad(B\ge b_u).
\tag{16}
\]

This is only an eventual tail condition, not a large curvature requirement
over every reachable output. It implies bounded utility above. Ordinary
exponential utility satisfies it above an external output threshold.
Suppose D is C2 and its marginal cost is convex, with D'(0)=0; more generally
it suffices that D'(T)/T is nondecreasing. Quadratic delay satisfies equality
and is not a limiting steep-delay case.

Let T0 be the unique zero-prize optimum, put b=b(T0), and assume
sigma^2 T0>=2 S0 and b>=4 b_u/3. These concern only the zero-prize reference
computed from u,D,sigma,S0, not the jointly chosen prize/deadline. Set

\[
c_e=\frac{u'(b)T_0}{\sqrt{2\pi(S_0+\sigma^2T_0)}}.
\]

These reference conditions hold for all sufficiently large sigma for fixed
S0,u,D of the stated shapes. If b remained bounded along sigma->infinity,
then T0=(2 pi b^2-S0)/sigma^2->0, but its first-order condition
D'(T0)=sigma^2 u'(b)/(4 pi b) would have left side tending to zero and right
side diverging, a contradiction. Thus b->infinity and sigma^2 T0->infinity.

### Global comparison at the entry cost

Only for this proof put A=b u'(b), a=S0/(sigma^2 T0), q=1+a, x=T/T0 and
w=theta/(ce sigma^2). Then 0<a<=1/2 and ce sigma^2=A/q. The zero-prize
first-order condition and increasing D'(T)/T give

\[
D(T)-D(T_0)\ge\frac{A}{4q}(x^2-1),\qquad
D(T_0)\le\frac{A}{4q}.
\tag{17}
\]

To check the first inequality, integrate D'(t)>=t D'(T0)/T0 above T0 and
D'(t)<=t D'(T0)/T0 below T0. The second follows by integration from zero.

If B<=3b/4, concavity gives u(b)-u(B)>=A/4>D(T0), so such a design is strictly
worse than the zero-prize benchmark even before subtracting its own costs.
For B>3b/4, condition (16) gives the integrated utility bound

\[
u(B)-u(b)\le A(1-b/B).
\tag{18}
\]

The inequality holds on either side of b: integrate
u'(z)<=u'(b)(b/z)^2 above b, and the reverse derivative bound below b.
It also gives u_infinity-u(b)<=A. Hence any design competitive with the
zero-prize payoff must have theta+D(T)<=A+D(T0), and therefore
w<=q+1/4<=7/4<2. The exact full-disclosure envelope (10) consequently covers
every potentially winning full-disclosure design; no prize upper bound is
imposed on the original problem.

Write m=a+x and N=m+xw+xw^2/2, only within the following algebra. The envelope
gives B/b<=N/sqrt(mq). Combining (17)--(18), the exact net-payoff difference
divided by A is at most

\[
1-\frac{\sqrt{mq}}{N}-\frac{x^2-1+4w}{4q}
=\frac{P}{4qN},\qquad
P=(4q+1-x^2-4w)N-4q\sqrt{mq}.
\tag{19}
\]

For x>0, expand P=c0+c1 w+c2 w^2-2x w^3. Its coefficients are

\[
c_1=(x-1)(4a-x^2-x),\qquad
c_2=-\frac x2(3-4a+x^2)<0.
\]

To retain the exact zero-prize gap, put z=sqrt((a+x)/q). Direct factorization
gives

\[
c_0=-q^2 z(z-1)^2L,\qquad
L=qz^3+2qz^2+(q+2)z+4\ge4.
\tag{20}
\]

If c1<=0, all terms show P<0 except x=1,w=0. If c1>0, necessarily
0<x<1 and 4a<x^2+x. In particular x<z<1. The exact discriminant comparison
is

\[
4(-c_0)(-c_2)-c_1^2
=q^2(z-1)^2\left[
2zx(3-4a+x^2)L-(z+1)^2(x^2+x-4a)^2\right]>0.
\tag{21}
\]

The bracket is positive without a long polynomial expansion: its first term
is greater than 16zx>16x^2, since 3-4a+x^2>3-x>2 and L>=4. Its second term
is at most 4(2x)^2=16x^2. Thus c0+c1 w+c2 w^2<0 for every real w in this
case; subtracting 2x w^3 preserves strict negativity for w>=0.
Zero deadline is separately strictly worse than the positive zero-prize
deadline. We have therefore proved that at c=ce the unique optimal prize and
deadline *within full disclosure* are theta=0,T=T0, globally over all theta,T.

### Higher and lower costs, and continuity of benchmark prize exit

Holding alpha=theta/c fixed keeps the full-disclosure equilibrium output
fixed. At c>ce every positive-prize design loses an additional (c-ce)alpha,
so zero prize with T0 is again the unique optimal prize/deadline. At c<ce,
the prize right derivative at theta=0,T0 is ce/c-1>0; a positive prize strictly
beats all zero-prize designs.

Full-disclosure optimization attains its value near ce. Bounded utility
localizes competitive theta and D(T), hence theta and T, to a common compact
set. Its output is continuous there: use the explicit smooth policies before
T-epsilon, and bound the remaining effort and Brownian increments by a uniform
constant times sqrt(epsilon) on bounded alpha. The endpoint theta=0 is also
controlled by the energy bound, and T=0 by the same bound. Consequently,
compactness and the unique global benchmark maximizer at ce imply that every
full-disclosure optimal prize tends to zero and every optimal deadline tends
to T0 as c increases to ce from below. All these lower-cost benchmark optima
have positive prize and deadline.

### What this advances, and what it does not prove

The full-disclosure global exit is now supported by a short algebraic proof
and moderate *eventual* curvature two, with ordinary quadratic delay allowed.
There is no 33/51-level curvature condition, no strongly convex delay
requirement, and no endogenous restriction on u at a joint optimum.

The same global argument would cover finite precision if its version of (10)
were proved on the competitive domain. That is precisely the missing step;
(14?) is one sufficient route. Full-disclosure benchmark continuity is NOT
continuity of the original joint optimum's prize: a remote finite-precision
branch could still beat that benchmark. Neither (19) nor numerical evidence
allows dropping this qualification. Proposition 1 therefore remains unchanged.

## 15. Manuscript incorporation of the weaker closed result

After the user accepted a weaker result, the full-disclosure benchmark theorem
in Section 14 was added to Proposition 1 as a benchmark prize threshold and
a necessary condition for any positive-prize full-disclosure *joint* optimum.
The existing low-cost, low-risk finite-feedback advantage was retained.
The appendix now contains the direct supersolution bound and the global
zero-prize comparison, with all additions blue. No finite-precision envelope
or sufficient full-disclosure joint-optimum region is asserted in the paper.

For the manuscript, the additional delay assumption was stated directly as
D'(T)/T nondecreasing, requiring only the existing C1 delay regularity. The
utility tail condition was stated equivalently as B^2 u'(B) eventually
nonincreasing, without a globally large curvature parameter or a design-
dependent utility assumption. The numerical figure was not changed.

## 16. Direct endpoint ordering at second order, without a variance-ratio restriction

This is progress toward the sufficient full-disclosure region, not its proof.
The manuscript now treats the two shape conditions as common assumptions and
states only the necessary condition in Proposition 1(ii).

### Information-path comparison

Use the proof-only normalization sigma=T=1, initial variance s>0, and
q=s+1. For finite positive precision ell, let

\[
S'_t=1-\ell S_t^2,\qquad S_0=s,\qquad A_t=s+t-S_t.
\]

Full disclosure has A_t^infty=s+t. The following direct comparison avoids
the high-precision expansion in ledger Section 93. Define
rho_t=A_t/(s+t), solely within this calculation. Then 0<rho_t<1 for t>0,
and

\[
\rho'_t=\frac{S_t-(s+t)S'_t}{(s+t)^2}>0.
\]

To check the strict sign, put E_t=(s+t)S'_t-S_t. If S'_t<=0, E_t<0
immediately. If S'_t>0, the autonomous Riccati equation implies S'_r>0
throughout 0<=r<=t, and

\[
E_0=-\ell s^3<0,\qquad
E'_r=-2\ell(s+r)S_rS'_r<0.
\]

Thus, for 0<t<r<1, rho_t<rho_r<1. This holds for every s>0;
in particular it does not require S0/(sigma^2 T) to stay above 1/4.

### Gaussian second-order coefficient

Use the previously derived Gaussian coefficient (ledger Section 83):

\[
b_2(s,\ell)=\int_0^1\int_t^1 K_q(A_t,A_r)\,dr\,dt,
\qquad
K_q(A,B)=\frac{A}{\pi\sqrt{2\pi}\sqrt{q-B}(q+B)
\sqrt{q^2+qB-2A^2}}.
\]

On 0<A<=B<q, K_A>0. Its radial derivative is also strictly positive.
With x=B/q and z=A/q, direct differentiation gives

\[
\frac{A K_A+B K_B}{K_q}
=\frac{1+x^3+xz^2-3x^2z^2}
{(1-x^2)(1+x-2z^2)}>0.
\]

Both denominator factors are positive. If x<=1/3, the numerator is
positive term by term. Otherwise its minimum over z^2<=x^2 is
(1-x)(3x^3+x^2+x+1)>0. Consequently,

\[
K_q(A_t,A_r)
<K_q(\rho_r(s+t),\rho_r(s+r))
<K_q(s+t,s+r).
\]

The first comparison uses rho_t<rho_r and K_A>0; the second integrates
the positive radial derivative. The full-disclosure kernel is integrable:
writing u=1-t and v=1-r, with 0<v<=u<=1, its last squared denominator is
q(4u-v)-2u^2>=qu. Hence its kernel is bounded by a constant times
1/sqrt(uv), integrable over this triangle. Integrating the strict comparison
therefore proves

\[
0<b_2(s,\ell)<b_2(s,\infty)\qquad(s>0,\ 0<\ell<\infty).
\tag{20}
\]

This establishes the endpoint ordering of the second-order coefficient for
every positive variance ratio. It does NOT establish an exact all-precision
ordering at one common positive prize.

### What Taylor expansion actually permits

For any fixed positive finite precision, on the selected weak-prize branch
where the existing C2 prize expansion applies, the two exact outputs have
the same zero-order and first-order terms. Thus

\[
\widehat{\mathcal B}(w,s,\infty)
-\widehat{\mathcal B}(w,s,\ell)
=w^2[b_2(s,\infty)-b_2(s,\ell)]+o(w^2)>0
\]

for sufficiently small w>0. On a fixed compact positive-precision interval,
this comparison is uniform provided the existing joint C2 dependence of the
weak-prize solution is used. Continuity of b2 and (20) give a positive minimum
coefficient gap there. Since u is increasing, switching to full disclosure
at the same prize and deadline strictly increases the organizer's exact
payoff in that neighborhood. No extra utility or delay curvature enters this
comparison. It also covers the small variance ratios associated with the
large-sigma benchmark prize-exit result.

### Two remaining requirements, not additional model assumptions

1. The neighborhood cannot depend on a finite precision cutoff. As ell tends
   to infinity, the coefficient gap tends to zero. An absolute o(w^2) estimate
   cannot sign the difference uniformly. A relative remainder estimate, or a
   direct exact comparison, is still required. Taking the intersection of the
   cutoff-dependent neighborhoods is not valid. Zero precision needs its own
   verified equilibrium comparison if it is a feasible competitor.
2. The full-disclosure global prize-exit proof does not localize joint finite-
   precision competitors. Even a uniform weak-prize comparison would not
   exclude a distant positive-prize finite-feedback design at costs just below
   cP. That exclusion must be proved, not inferred from a local Hessian.

Accordingly (20) removes a variance-ratio limitation from the second-order
endpoint comparison, but does not close existence of a positive-prize
full-disclosure joint optimum. Proposition 1 is not strengthened on this basis.

## 17. A closed exact output envelope for the whole high-precision range

### Result and scope

The next bound is for the actual finite-precision Nash policies, not the
full-disclosure policies substituted into that game. For

\[
\lambda\ge\sigma^2/S_0^2,\qquad
0<w:=\theta/(c\sigma^2)\le1/2,
\]

the symmetric classical weak-prize branch satisfies

\[
\mathcal B(0,\theta,T,\lambda;S_0)
\le\sqrt{\frac{S_0+\sigma^2T}{2\pi}}
+\frac{\theta T}{c\sqrt{2\pi(S_0+\sigma^2T)}}
\left(1+\frac{2\theta}{c\sigma^2}\right).
\tag{21}
\]

This closes a wider version of the output-envelope task throughout this
high-precision interval, uniformly as lambda increases. No new condition on
u or D is used. The coefficient is 2, not the unproved finite-precision 1/2
in Section 12. Consequently the old payoff algebra using 1/2 cannot simply
be reused. The remaining low-precision competitors are not excluded by (21).

### 17.1 Sharpen the existing actual-gradient bootstrap

Use p=(V_i/theta)_y, n=-(V_j/theta)_y, H=p+n, A=p-n, and
Q_t=S_t+sigma^2(T-t). Let

\[
E_Q(y)=Q^{-1/2}e^{-y^2/(4Q)},\qquad
\|f\|_G=\sup_{t,y}|f(t,y)|/E_{Q_t}(y).
\]

Ledger Sections 9 and 90 supply the genuine high-precision gradient-ratio
comparison |A|<=wH/2 and the bilinear heat-kernel bound
C_G=2(sqrt(2/e)+sqrt(2/pi))<10/3. In the original coupled HJB the exact
sum-gradient mild equation is

\[
H=2G_{Q_t}+\frac\alpha2\mathscr T(A,H),\qquad \alpha=\theta/c.
\]

If M is its Gaussian norm on any backward continuation interval, these
estimates give

\[
M\le\frac2{\sqrt{2\pi}}+\frac{C_Gw^2}{4}M^2
<\frac45+\frac56w^2M^2.
\]

Use the barrier Mbar=51/50 instead of the older 4/3. At a first crossing,
w<=1/2 would imply

\[
M<\frac45+\frac5{24}(51/50)^2
=\frac{4067}{4000}<\frac{51}{50},
\]

a contradiction. The terminal norm is 2/sqrt(2 pi)<4/5. The local classical
continuation and Nash verification are those in ledger Section 90; this
argument sharpens their bound without assuming that the solution stays small.
Therefore throughout the horizon

\[
\|H\|_G<\frac{51}{50},\quad
\|A\|_G\le\frac w2\frac{51}{50},\quad
\|H-2G_{Q_t}\|_G\le\frac56(51/50)^2w^2.
\tag{22}
\]

### 17.2 Keep the Gaussian first-order term sharp

Let g=2Phi(y/sqrt(Q_t))-1. The exact winner-weighted effort source in
Section 13 is alpha F, where

\[
F=p\Phi(y/\sqrt{Q_t})+n\Phi(-y/\sqrt{Q_t})
=\frac{H+gA}{2}.
\]

Since |g|<=1, (22) gives the pointwise bound

\[
F\le G_{Q_t}
+\left[\frac{51}{200}w
+\frac5{12}(51/50)^2w^2\right]E_{Q_t}.
\tag{23}
\]

In particular the leading term is the exact narrow Gaussian G, not the
wider Gaussian envelope E. Replacing both by E would lose the correct entry
cost and would not solve the localization problem.

### 17.3 Bound the actual strategic state law

Ledger Section 118 proves, by the time-dependent exponential transform of
the actual value sum, that its Markov propagation is bounded by
cosh^2(w/2) times the heat propagation. It does not assume a Gaussian
strategic state law or a sign for A. Thus, for every nonnegative f and t,

\[
\mathbb E f(\tilde y_t)
\le\cosh^2(w/2)\,P_{Q_0-Q_t}f(0).
\]

Here the equilibrium horizon stays T: the propagation refers to its same
time-dependent drift alpha A. The variance Q_0-Q_t is the accumulated
innovation variance, not sigma^2 t. Gaussian convolution gives

\[
P_{Q_0-Q_t}G_{Q_t}(0)=G_{Q_0}(0),\qquad
P_{Q_0-Q_t}E_{Q_t}(0)
=\frac{\sqrt2}{\sqrt{Q_0+Q_t}}
\le\sqrt{4\pi}\,G_{Q_0}(0).
\]

Integrating (23) in the exact winner-weighted identity yields

\[
\mathcal B-b(T)
\le\alpha T G_{Q_0}(0)\cosh^2(w/2)
\left\{1+\sqrt{4\pi}
\left[\frac{51}{200}w+\frac5{12}(51/50)^2w^2\right]\right\}.
\tag{24}
\]

All integrands are nonnegative or bounded by the displayed Gaussian
envelopes. Finite expected effort energy supplies the expectation identity;
the propagation comparison is first applied to bounded truncations and
then extended monotonically. No uniform terminal derivative bound is needed.

### 17.4 Reduce the constants

For 0<w<=1/2, the positive cosh series gives

\[
\cosh^2(w/2)\le1+\frac{w^2}{4(1-w^2/12)}
\le1+\frac{12}{47}w^2,
\qquad \sqrt{4\pi}<\frac{71}{20}.
\]

The series estimate uses (2j)!>=2*12^(j-1). With d=12/47 and
Mbar=51/50, the factor in (24), minus 1, divided by w is bounded by

\[
dw+(1+dw^2)\frac{71}{20}
\left(\frac{\bar M}{4}+\frac5{12}\bar M^2w\right)
\le\frac d2+(1+d/4)\frac{71}{20}
\left(\frac{\bar M}{4}+\frac5{24}\bar M^2\right)
=\frac{143577}{75200}<2.
\]

Every term in the bounding polynomial is nondecreasing in w. This proves
(21), including arbitrarily high finite precision. The numbers are proof
constants, not additional economic parameters.

### 17.5 Extension below the stationary-initialization precision

There is also a useful low-precision version, but it is not uniform at zero.
For any fixed lambda>0 put, only in this proof,

\[
a_{\min}=\lambda\min(S_0,S_T)^2>0,\qquad
K=\frac{\theta}{c\,a_{\min}}.
\]

If K<=1/2, the same argument gives

\[
\mathcal B\le b(T)
+\frac{\theta T}{c\sqrt{2\pi(S_0+\sigma^2T)}}(1+2K).
\tag{25}
\]

To justify rather than assume the ratio input, use the original variance
clock Q and k(Q)=alpha/(lambda S_t^2). This k is monotone. When k_Q<=0,
the half-line comparisons in ledger Section 90 bound |log(p/n)| by max k=K.
When k_Q>=0, apply its exact scalar equation for p-exp(kd+b)n with b=0
for the upper bound and b=-k for the lower bound. The source signs are
-n exp(kd) k_Q d<=0 and -n exp(k(d-1)) k_Q(d-1)>=0 on y>=0.
Initial and y=0 boundary signs agree with these comparisons. Hence
k(d-1)<=log(p/n)<=kd, and again |log(p/n)|<=K, with the other half-line
handled by reflection. Therefore |A|/H<=tanh(K/2)<=K/2.

The Gaussian bilinear bound now has 1/a_min instead of 1/sigma^2:
Q_t-Q_r>=a_min(r-t) and Q_t>=a_min(T-t). The value-sum propagation bound
in ledger Section 118 is cosh^2(K/2). Replacing w by K in (22)--(24) thus
proves (25). For the high-precision range, a_min>=sigma^2 and (21) follows.
For lambda<sigma^2/S0^2, a_min=lambda S0^2, so K grows without bound
as lambda decreases at a fixed positive prize. Equation (25) does not
exclude all low-precision competitors or settle the no-signal case.

### Remaining tasks for the requested existence theorem

The coefficient-2 envelope is now closed in a substantial exact-equilibrium
range. Two distinctions remain essential:

* The Section 14 global payoff comparison used coefficient 1/2. Proving
  coefficient 2 is not a proof of that same payoff comparison under unchanged
  utility assumptions. A new comparison must use (21) as stated.
* The full feasible set still includes finite precisions for which K>1/2 and
  the selected no-signal equilibrium. The high-precision result cannot be
  turned into a lower bound on feasible precision to conceal those competitors.

No positive-prize full-disclosure joint-optimum region is asserted yet, and
the manuscript and numerical classification remain unchanged.

## 18. Remote-design exclusion needs coefficient 2/3, not 1/2

### Audit and statement

The coefficient-2 envelope in Section 17 is a genuine equilibrium estimate,
but it is not sharp enough for the current utility assumptions. This section
improves the separate payoff argument: coefficient 2/3 suffices for global
prize exit under the existing eventual curvature two and delay assumptions.
It is a conditional payoff theorem, NOT an assumption added to the model.

Keep the reference quantities from Section 14, with 0<a<=1/2, q=1+a,
b>=4 b_u/3, and c=ce. Suppose the actual output of each competitive design
satisfies

\[
\mathcal B\le b(T)
+\frac{\theta T}{c_e\sqrt{2\pi(S_0+\sigma^2T)}}
\left(1+\frac{2\theta}{3c_e\sigma^2}\right).
\tag{26?}
\]

Then the global best prize and deadline are uniquely theta=0,T=T0,
irrespective of precision. It is enough to establish (26?) for
0<w=theta/(ce sigma^2)<=7/4, since the utility and cost budget already
excludes every potentially winning design outside that range. The question
mark marks the unproved equilibrium input, not the payoff algebra below.

### Complete payoff comparison

As in Section 14, outputs B<=3b/4 are already strictly inferior. For larger
outputs, the utility tail inequality u(B)-u(b)<=A(1-b/B) applies, where
A=b u'(b). The delay bound is unchanged. With x=T/T0 and m=a+x, put

\[
N=m+xw+\frac23xw^2,
\qquad
P=(4q+1-x^2-4w)N-4q\sqrt{mq}.
\]

The original net-payoff difference is bounded above by A P/(4qN).
For x>0 expand P=c0+c1 w+c2 w^2-(8/3)xw^3. Its coefficients are

\[
c_1=(x-1)(4a-x^2-x),\qquad
c_2=-\frac23x(1-4a+x^2).
\]

With z=sqrt((a+x)/q), the same exact zero-prize factorization gives

\[
c_0=-q^2z(z-1)^2L,\qquad
L=qz^3+2qz^2+(q+2)z+4\ge4.
\]

Unlike coefficient 1/2, c2 need not be negative everywhere. It cannot be
dropped on that basis. The following two cases account for its sign.

**Case c1<=0.** If c2<=0, P<0 for w>0 directly, including x=1 because
the cubic term is strictly negative. If c2>0, write ell=4a-1-x^2>0 only
within this paragraph. Since 4a<=2, we have 0<x<1 and
ell<=1-x^2. Here

\[
c_1=-(1-x)(\ell+1-x),\qquad c_2=\frac23x\ell.
\]

Completing the quadratic in w bounds the three incentive terms by

\[
c_1+c_2w-\frac83xw^2
\le c_1+\frac{x\ell^2}{24}
\le-(1-x)\ell+\frac{x\ell^2}{24}<0.
\]

The final strict sign follows from ell<=2(1-x) and x<1. Multiplication
by w>0 and addition of c0<=0 give P<0.

**Case c1>0.** Necessarily 0<x<1 and t=x^2+x-4a>0. Then
1-4a+x^2=1-x+t>0, so c2<0, and c0<0. The exact discriminant is

\[
4c_0c_2-c_1^2=q^2(z-1)^2
\left[\frac83zx(1-x+t)L-(z+1)^2t^2\right].
\]

Because a>=0 and x<1, sqrt(x)<=z<1. Also 0<t<=x^2+x. The function
t^2/(1-x+t) is increasing for t>0, and therefore

\[
\frac{t^2}{1-x+t}
\le\frac{x^2(1+x)^2}{1+x^2}\le2x^2.
\]

Consequently the first term in the discriminant bracket is at least
(32/3)x^(3/2)(1-x+t), whereas the second is at most
8x^2(1-x+t). The first is strictly larger since 32/3>8 sqrt(x).
Thus 4c0c2-c1^2>0, so c0+c1w+c2w^2<0 for every real w.
The negative cubic term gives P<0 for w>=0.

Zero deadline is separately worse than the positive zero-prize deadline.
At w=0, the strict zero-prize factorization selects x=1 uniquely.
This completes the global payoff comparison, conditional on (26?).

### Why coefficient 2 cannot replace 2/3 under unchanged assumptions

This is a counterexample to that proof shortcut, not to the original model.
An admissible utility is

\[
u(B)=\begin{cases}3B-B^2,&0\le B\le1,\\3-1/B,&B\ge1.\end{cases}
\]

It is C2, strictly increasing, concave, bounded above, and satisfies
B^2 u'(B)=1 for B>=1. Choose S0=8 pi/3, sigma^2=16 pi/3, and
D(T)=T^2/12. The zero-prize optimum is T0=1, b=2, a=1/2, q=3/2;
all Section 14 reference conditions hold. At the same deadline, a putative
envelope with coefficient C gives B_upper/b=1+(w+Cw^2)/q. Its utility
minus prize, relative to the zero-prize value and divided by A, is

\[
\frac{w+Cw^2}{q+w+Cw^2}-\frac wq
=\frac{w^2(Cq-1-Cw)}{q(q+w+Cw^2)}.
\]

For every C>2/3 this is positive at sufficiently small w>0. Thus such
an envelope alone cannot certify zero prize at ce for the current class
of utilities, even with the deadline fixed. C=2/3 is the largest coefficient
compatible with this uniform envelope-based approach over 0<a<=1/2.
This does not bound the actual equilibrium output from below.

### What the new comparison gives once its HJB input is proved

The bound P<0 is independent of lambda. Bounded utility gives a common
compact set of competitive prize/deadline choices near ce. On every closed
subset outside a neighborhood of theta=0,T=T0, the continuous upper payoff
bound has a strictly negative maximum. Hence (26?) would exclude remote
competitors uniformly over precision, not merely pointwise. Lowering c by
delta while holding alpha=theta/c fixed changes payoff by exactly delta alpha;
this uniform negative gap consequently forces all competitive choices back
to the weak-prize neighborhood for sufficiently small delta>0.

There is one range detail in this argument. For c>=ce/2 the utility budget
bounds alpha and T uniformly. The common energy screen
B<=b(T)+sqrt(2 alpha T) then gives a finite Bmax independent of precision.
Strictly increasing bounded utility has u(Bmax)<u_infinity. Put
epsilon=(u_infinity-u(Bmax))/A>0 solely for this verification. Every
competitive design near ce consequently satisfies

\[
\frac\alpha{\sigma^2}
\le\frac{c_e}{c}\left[q(1-\epsilon)+\frac14\right]
\le\frac{c_e}{c}\left[\frac74-q\epsilon\right].
\]

This remains below 7/4 when c is sufficiently close to ce from below.
Thus (26?) is not silently applied outside its incentive range to newly
competitive designs. If the budget bracket is negative, there are no such
competitive positive-prize designs and the range issue is vacuous.

No continuity of equilibria at lambda=0 is needed for this localization if
the same output inequality is verified for its separately selected equilibrium.
After localization, an exact weak-prize full-disclosure comparison uniform
over all finite precisions would complete the joint-optimum argument.

### Remaining proof obligations

The required sharp input is now (26?), with coefficient 2/3 rather than 1/2,
for the actual equilibria on the competitive domain. Section 17 proves
coefficient 2 in a smaller incentive range and therefore does not supply it.
The exact high-precision tail comparison also needs independent completion
of its uniform remainder audit. Neither requirement is replaced by a model
assumption, a numerical classification, or the polynomial comparison above.

## 19. Consolidating the exact weak-prize comparison, including the unbounded tail

### Scope and audited inputs

For the selected weak-prize equilibria, fix any 0<s_minus<s_plus<infinity.
There exists w_star>0, depending on this interval but not on precision, such that

\[
\widehat{\mathcal B}(w,s,\infty)
>\widehat{\mathcal B}(w,s,\ell)
\quad(s\in[s_-,s_+],\ 0\le\ell<\infty,\ 0<w<w_*).
\tag{27}
\]

This uses the original small-prize argument in ledger Sections 12--16, not
the expanded 9/20-prize claim in Sections 90--93. The audit checked the
background spectrum and response, the nonlinear rather than initial-only
amplitude, the common first-order cancellation, the original Riccati-path
transfer, and the physical-time analytic construction at zero precision.
The direct endpoint coefficient proof in Section 16 above replaces reliance
on the old coefficient ordering. The following consolidates the estimates
and the order of quantifiers; it does not replace their analytic proofs with
a numerical check.

### 19.1 High-precision input with the prize factors preserved

Let eta=ell^(-1/2). The auxiliary constant-posterior-variance problem is used
only for comparison, not substituted for the original signal process.
The scalar full-disclosure background and its linearized generator have a
simple slow exponent

\[
\kappa(w)=\frac12+\frac{w^2}{2\pi\sqrt3}+O(w^4),
\qquad \frac12<\kappa(w)<\frac{11}{20}
\]

after shrinking a fixed positive prize neighborhood. In particular the
exponent is slightly ABOVE 1/2, not below it. Its designer response and
actual nonlinear amplitude satisfy

\[
\chi_w=-wG_1\beta+O(w^2),\qquad
A(w)=\frac{w}{\sqrt\pi}+O(w^2)>0.
\]

Here beta(z)=G_1(z)+z[Phi(z)-1/2]. The amplitude includes the integral
of the nonlinear forcing, not just the initial projection. In the fixed
Gaussian weighted space its forcing is quadratic, of size O(w^3 e^(-4v/5));
its weighted time integral converges since kappa<3/5. Spectral separation
from the remaining modes gives a fast designer remainder after subtracting
the common first-order term of size O(w^2 e^(-7v/10)). Thus Gaussian prior
averaging, with eta<s_minus/2, gives positive constants d,C and a single
prize neighborhood such that

\[
\widehat{\mathcal B}_{\infty}-\widehat{\mathcal B}_{\rm aux}
\ge\frac{w^2}{\sqrt{1+s}}
\left[d\eta^{\kappa(w)}-C(\eta^{7/10}+\eta)\right].
\]

The auxiliary/full background change and the fast term -w e^(-v)G have
equal and opposite first-order contributions. Without this cancellation,
an O(w eta) error would not give a uniform statement as w tends to zero.

The original Riccati-path comparison, in the common variance clock, gives

\[
|\widehat{\mathcal B}_{\rm original}
-\widehat{\mathcal B}_{\rm aux}|
\le\frac{Cw^2\eta}{\sqrt{1+s}}.
\]

Its two disclosure weights have mass one, lie between zero and one, and
their L1 difference is O(eta). On the low-variance layer the difference is
exponentially small. The high-layer derivative-kernel estimate and its
common first-order cancellation supply the displayed w^2 factor. These
are the original small-prize estimates in ledger Section 15; the larger-
prize continuation is unnecessary for (27).

Consequently, for one fixed sufficiently small eta_star<min(1,s_minus/2),

\[
\widehat{\mathcal B}_{\infty}-\widehat{\mathcal B}_{\rm original}
\ge\frac{w^2}{\sqrt{1+s}}
\left[d\eta^{11/20}-C'(\eta^{7/10}+\eta)\right]>0
\quad(0<\eta<\eta_*).
\]

This choice is uniform in small positive w and in the fixed s interval.
It does not expand eta^kappa(w) into a power series in w; doing so introduces
unbounded log(eta) terms and invalidates a naive relative Taylor remainder.

### 19.2 The remaining compact precision range

Choose L>eta_star^(-2), once and for all. On
[s_minus,s_plus] x [0,L], the terminal posterior variance has a strictly
positive minimum. The physical-time analytic construction in ledger
Section 16 therefore gives a common complex-prize neighborhood for the
actual gradients and output, including the separately selected analytic
no-signal equilibrium. It never divides by the diffusion coefficient at zero.
The Gaussian Fourier data have a uniform analytic norm; a decreasing strip
radius absorbs the one-derivative quadratic flux. Diffusion is dissipative,
including when it vanishes. Nash verification uses the real-prize branch.

The exact output difference has equal zero-order and first-order terms and
therefore satisfies, on this compact parameter set,

\[
\widehat{\mathcal B}(w,s,\infty)
-\widehat{\mathcal B}(w,s,\ell)
=w^2\Delta_2(s,\ell)+O(w^3),
\qquad |O(w^3)|\le C_Lw^3.
\]

Section 16 above proves Delta2>0 for every positive finite ell. At ell=0
the coefficient is b2(s,infinity)>0, since the no-information coefficient
is zero. The coefficients are continuous on this compact set. Its strictly
positive minimum d_L consequently exists. Choosing w<d_L/(2 C_L), also
inside the analytic radius, proves exact dominance throughout 0<=ell<=L.

Finally choose w_star as the minimum of this compact-set threshold and the
previous fixed tail threshold. L was selected before shrinking w. This
proves (27) and avoids an intersection of infinitely many prize neighborhoods
whose radius might vanish. No zero-precision continuity is presumed for
large prizes or for arbitrary equilibrium selections.

### Present obstruction to joint existence

Within this audited weak-prize branch, the unbounded-precision comparison
can now be consolidated without a new utility or delay assumption. The
remaining substantive moderate-shape joint-design task is the sharp global output bound
(26?), or another global payoff exclusion achieving the same result.
Section 18 shows exactly how it would localize all competitive prizes and
deadlines into the domain of (27). Section 17 does not yet imply (26?).
Until that global exclusion is established, a distant finite-precision
positive-prize design can still defeat the full-disclosure weak-prize branch.
No sufficient full-disclosure joint-optimum region is inserted in the paper.
The moderate-shape obstruction here is distinct from the specialized
existence construction closed below.

## 20. Closed weaker result: an open joint-design window for fixed external primitives

### 20.1 What was missing, and what is being weakened

Sections 3--5 already excluded every remote prize/deadline competitor, but
relied on (W) and did not establish persistence when risk varies. Section 19
supplies (W); the continuity argument below supplies persistence. The earlier
energy explanation also had a factor-of-two error for an individual player;
Section 2 now derives the unchanged output screen from the SUM of payoffs.

The weakened claim is existential over the admissible external utility and
delay-cost class. It is NOT a full-disclosure window for every concave utility
and convex delay cost, and NOT a verification of the numerical figure's
exponential utility/quadratic delay calibration. Its equilibrium selection is
the verified weak-prize branch in Section 19, including the separately verified
no-signal equilibrium. It does not assert a ranking across arbitrary additional
equilibrium selections at zero precision.

### 20.2 Proposition

For each fixed S0>0 and each reference risk sigma_0>0, there exist fixed external
functions u,D satisfying all the manuscript's utility and delay assumptions,
and a nonempty open rectangle R in the positive (c,sigma) plane, such that the
exact three-parameter problem attains its global maximum for every (c,sigma)
in R. Every maximizer satisfies

\[
\theta^*>0,\qquad T^*>0,\qquad\lambda^*=\infty.
\tag{28}
\]

The functions are chosen ONCE before (c,sigma) varies over R. The proof adds
neither a precision cost, a precision floor, nor exogenous design bounds.
It makes no assertion about a universal boundary or its direction. The open
rectangle is not obtained by recalibrating u or D separately at each risk.

### 20.3 Fix the external environment before optimizing

Use (27) on the fixed interval s in [1/3,3], and call its positive threshold
w_star, reduced if necessary so that w_star<=1. Set tau=S0/sigma_0^2,
b=sqrt(S0/pi), and choose an external a>0.
Choose epsilon as in Section 3 with this w_star, and choose
eta<epsilon/(2 log(2)+1/4). Fix u=u_eta and D=D_eta from Section 5, with
sigma=sigma_0 in their definitions. No later risk perturbation changes them.

The two additional global shape assumptions also hold. For the utility,

\[
\frac d{dB}\log(B^2u'(B))
=\frac2B-\frac{1}{\eta b}
\frac{e^{(B-b)/(\eta b)}}{1+e^{(B-b)/(\eta b)}}<0
\]

for all sufficiently large B. For the delay cost, with the fixed q=e^(-1/eta),

\[
\frac{D'(T)}{T}
=\frac{hq}{(1-q)\eta\tau}
\frac{e^x-1}{x},\qquad x=\frac{T}{\eta\tau}>0.
\]

The last factor is strictly increasing: its derivative has numerator
e^x(x-1)+1, zero at x=0 and strictly increasing thereafter. Thus the
construction belongs to the manuscript's admissible class; these are not
unannounced alterations to that class.

Put c_0=3c_e/4, using the external reference c_e in Section 3. Sections 3--4
give a positive-prize full-disclosure global optimum at (c_0,sigma_0).
The remaining step is to prove that this is a region, not one isolated point.

### 20.4 Strict localization survives risk and cost perturbations

For the now FIXED u,D,S0 define only in this proof

\[
b_\sigma(T)=\sqrt{\frac{S_0+\sigma^2T}{2\pi}},\qquad
V_0(\sigma)=\max_{T\ge0}\{u(b_\sigma(T))-D(T)\},\qquad
H(\sigma)=u_\infty-V_0(\sigma).
\]

The zero-prize objective is strictly concave: b_sigma is strictly concave,
u is increasing and concave, and D is convex. Its derivative at zero is
positive since sigma>0 and D'(0)=0, and it tends to minus infinity since
D is unbounded. It therefore has a unique positive maximizer T0(sigma).
All these maximizers lie in a common compact time interval, because
D(T)>u_infinity-u(sqrt(S0/(2 pi))) rules out T uniformly in sigma.
Continuity of the objective on that interval and uniqueness imply continuity
of both T0(sigma) and V0(sigma). This is a zero-prize comparison, not an
assumption about continuity of three-parameter optimizers.

Set the FIXED proof barriers T_minus=tau/2 and T_plus=(1+4 epsilon)tau.
At (c_0,sigma_0), the strict inequalities in Sections 3--4 give

\[
\frac{H(\sigma_0)}{c_0\sigma_0^2}<w_*,\qquad
D(T_+)>H(\sigma_0),
\]

\[
u\!\left(b_{\sigma_0}(T_-)
+\sqrt{\frac{2H(\sigma_0)T_-}{c_0}}\right)<V_0(\sigma_0).
\tag{29}
\]

Indeed H(sigma_0)<epsilon ab. The supporting line of D gives
D(T_plus)>=D(tau)+epsilon ab>H(sigma_0). At the lower barrier, the
energy screen is at most (sqrt(3)/2+2 sqrt(pi epsilon))b, while concavity
and the choice of epsilon give the last strict inequality, as in Section 3.
Also S0/(sigma_0^2 T_plus)>1/3 and S0/(sigma_0^2 T_minus)<3.

All quantities in (29) are continuous in (c,sigma). Shrink an open rectangle
around (c_0,sigma_0) so that these strict inequalities and variance-ratio
bounds persist. A design with payoff at least V0(sigma) satisfies
theta+D(T)<=H(sigma). The second inequality excludes T>=T_plus.
For T<=T_minus, (E) and theta<=H(sigma) give the upper output in (29),
so its payoff is strictly below V0(sigma), a contradiction. Consequently,
every competitive design in the entire rectangle satisfies

\[
0\le\theta\le H(\sigma),\qquad T_-<T<T_+,\qquad
\frac{\theta}{c\sigma^2}<w_*,\qquad
\frac13<\frac{S_0}{\sigma^2T}<3.
\tag{30}
\]

These are bounds derived from payoffs, not restrictions on feasible designs.

### 20.5 Entry, global comparison, and attainment

At fixed sigma, the full-disclosure prize right derivative at its zero-prize
deadline is

\[
\left.\partial_\theta J_\infty(\theta,T_0(\sigma))\right|_{0+}
=\frac{u'(b_\sigma(T_0(\sigma)))T_0(\sigma)}
{c\sqrt{2\pi(S_0+\sigma^2T_0(\sigma))}}-1.
\tag{31}
\]

Formula (F) justifies this derivative, with sigma and T held fixed.
At the reference point it is c_e/c_0-1=1/3>0. By continuity of T0,
u', and (31), shrink the rectangle once more so that (31) remains positive.
Thus some positive-prize full-disclosure design beats V0(sigma) at every
point of the rectangle.

The full-disclosure objective is continuous on the compact set
[0,H(sigma)] x [T_minus,T_plus], including theta=0; see the explicit
full-disclosure formula and terminal energy argument in the manuscript's
benchmark-attainment proof. It attains its maximum there. By (30), no
full-disclosure design outside this compact set can beat V0(sigma), so this
maximum is the unrestricted full-disclosure maximum and is strictly above V0.

Finally take ANY selected finite-precision design. If its payoff is below V0,
it is inferior to the attained full-disclosure maximum. Otherwise (30)
applies. A zero prize cannot beat V0. A positive prize is strictly improved
at the SAME prize and deadline by full disclosure, by (27). This includes
lambda=0 on the specified verified weak-prize branch. Hence no finite-
precision design reaches the full-disclosure maximum. The latter is an
attained GLOBAL maximum over theta,T,lambda, and (28) follows.

### 20.6 Reverse audit and precise limitation

- Quantifiers: w_star on [1/3,3] is fixed first; then u,D are fixed; only then
  is the open cost/risk rectangle chosen. No primitive is designed using an
  optimal prize, deadline, or precision.
- Global rather than local: (30) localizes every potentially winning design
  via its payoff. There is no unproved exclusion of remote prizes.
- Attainment: only the explicit full-disclosure objective needs compact-set
  continuity; the comparison rules out every finite precision separately.
  No continuity at lambda=0 or lambda=infinity is used for arbitrary prizes.
- Exactness: (27) compares exact equilibrium outputs, not only second-order
  coefficients. Its analytic inputs remain in Section 19 and ledger
  Sections 12--16; this existence proof does not replace those inputs.
- Weakening: the theorem establishes existence within the admissible class,
  not a sufficient simple (c,sigma) window for an arbitrary preselected u,D.
  Section 7's strong local shape warning still applies. The moderate-shape
  window, in particular the figure's calibration, remains unproved.

No mathematical assumption remains to be supplied for this specialized
selected-equilibrium theorem. Whether its sharply saturating utility and
rapidly rising delay cost are appropriate for the main paper is a substantive
modeling choice, not a missing step in the joint-design proof.

### 20.7 The same fixed environment also has a high-cost no-prize region

The proof can distinguish two attained regions without changing u,D.
Keep a sufficiently small open risk interval around sigma_0. The localization
inequalities (29)--(30), established at c=c_0, remain valid for ALL c>=c_0:
the prize budget divided by c and the short-deadline energy screen only
decrease when c increases. For any competitive positive-prize design, (27)
therefore strictly favors full disclosure at its own prize and deadline.

The exact full-disclosure bound already verified in the manuscript gives,
for 0<w<=1 and T<=T_plus,

\[
\mathcal B_\infty-b_\sigma(T)
\le\frac{\theta T}{c\sqrt{2\pi(S_0+\sigma^2T)}}(1+w/2)
\le\frac{3T_+}{2\sqrt{2\pi S_0}}\frac\theta c.
\]

Concavity gives u(B)-u(b_sigma(T))<=u'(0)(B-b_sigma(T)). Hence

\[
J_\infty(\theta,T)
\le V_0(\sigma)+\theta\left[
\frac{3u'(0)T_+}{2c\sqrt{2\pi S_0}}-1\right].
\tag{32}
\]

For c>max(c_0,3u'(0)T_plus/(2 sqrt(2 pi S0))), every competitive
positive-prize full-disclosure design is strictly below V0; every other
positive-prize design is either below V0 already or improved by full
disclosure. Thus all joint optima have theta=0 and the unique deadline
T0(sigma)>0. Precision is payoff-irrelevant when the prize is zero.

In fact, for each risk in this interval there is a finite TRUE prize-exit
threshold, not just the sufficient bound (32). Equilibrium depends on c and
theta through alpha=theta/c. With the same verified equilibrium selection,
write its exact output as B_sigma(alpha,T,lambda) and define, solely in this
proof, the supremum over verified selected equilibrium designs,

\[
C(\sigma)=\sup_{\alpha>0,T\ge0,\lambda\in[0,\infty]}
\frac{u(B_\sigma(\alpha,T,\lambda))-D(T)-V_0(\sigma)}{\alpha}.
\tag{33}
\]

This is independent of the organizer's design choices after taking the
supremum. The high-cost exclusion proves C is finite; the strict positive-
prize candidate at c_0 proves C>c_0. A prize strictly beats V0 if and only if
its ratio in (33) exceeds c. No unverified large-prize zero-signal branch is
introduced by this definition; all competitive choices for c>=c_0 lie in
the verified weak-prize domain. Thus, on c>=c_0, the full-disclosure compact
maximization and (27) imply

\[
c_0\le c<C(\sigma):\quad
\theta^*>0,\ T^*>0,\ \lambda^*=\infty;
\qquad
c>C(\sigma):\quad\theta^*=0,\ T^*=T_0(\sigma)>0.
\]

At c=C, zero prize is optimal; positive-prize full-disclosure maximizers may
tie. No continuity of C or single switching of precision below c_0 is asserted.
This threshold definition is a consequence of proved finite bounds and
attainment, not a replacement for proving existence of either region.

For these same fixed primitives, manuscript part (i) also gives the low-cost,
low-risk strict finite-feedback advantage. Its separate global-attainment
qualification remains unchanged. Consequently the construction is compatible
with all three numerical design types, but does not certify the figure's
particular primitives, border locations, or an ordered three-region partition.

## 21. Fixed ordinary primitives: a single-point high-precision exclusion

### 21.1 Problem and scope

The current user target keeps u,D fixed and asks for ONE positive-prize
full-disclosure joint-optimal parameter point. Section 20 does not meet that
target because its functions were specially constructed. Here we instead fix
the numerical figure's primitives ONCE:

\[
S_0=1,\qquad u(B)=3(1-e^{-2B}),\qquad D(T)=T^2/2.
\tag{34}
\]

The coefficient-2/3 payoff argument in Section 18 covered an entire utility
class. Its sharp coefficient requirement need not apply to (34). This section
proves a new global prize-exclusion result for (34) using the already proved
coefficient-2 output envelope. It excludes ALL high-precision competitors,
not yet low precision; no joint-optimum point is claimed on this basis.

### 21.2 Explicit external parameter choice

Define proof constants and choose risk by

\[
b=\sqrt{\frac{1001}{2\pi}},\qquad
\sigma=\left(\frac{1000\,4\pi b}{6}e^{2b}\right)^{1/4},\qquad
T_0=\frac{1000}{\sigma^2},\qquad
c_e=\frac{2T_0^2}{\sigma^2}.
\tag{35}
\]

These specify sigma and c_e directly from fixed primitives; T0 is a proof
reference, not a restriction on the organizer's deadline. The zero-prize
first-order condition is

\[
T_0=\frac{\sigma^2u'(b)}{4\pi b},\qquad
b^2=\frac{1+\sigma^2T_0}{2\pi}.
\]

Equation (35) satisfies both. Strict concavity of the zero-prize objective
makes T0 its unique global maximizer. The prize right derivative at T0 is
c_e/c-1, so c_e is the marginal-entry cost. Approximate magnitudes, ONLY to
identify the point, are sigma=7020.57, T0=2.02888e-5, c_e=1.67031e-17.
No claim of a practically calibrated magnitude is intended.

### 21.3 Result

At c=c_e in (35), every selected high-precision design with positive prize is
strictly worse than the zero-prize optimum, uniformly over

\[
\lambda\ge\sigma^2/S_0^2=\sigma^2,
\]

and also in the explicit full-disclosure game. The unique optimal prize and
deadline within this precision range are theta=0,T=T0. The exact output
envelope (21) applies to the verified symmetric branch. The full-disclosure
bound in the manuscript is sharper and implies the same envelope.

### 21.4 Global proof, with prizes and deadlines unrestricted

Only in the proof put

\[
a=1/1000,\quad q=1+a,\quad A=bu'(b),\quad
x=T/T_0,\quad w=\theta/(c_e\sigma^2),\quad
V_0=u(b)-D(T_0).
\]

Then c_e sigma^2=A/q and D(T0)=A/(4q). A design with payoff at least V0
satisfies theta+D(T)<=u_infinity-V0. Consequently

\[
w+x^2/4\le q/(2b)+1/4<1/2,
\tag{36}
\]

since b>12 and q<21/20. Thus ALL competitive prizes lie in the valid
w<=1/2 range of (21); this range is derived rather than imposed.
Also u(B)>=u(b)-D(T0) implies

\[
\mathcal B\ge b-\tfrac12\log(1+b/(2q))
>b-b/(4q)>3b/4>9.
\tag{37}
\]

At any output z>=9, the fixed exponential utility satisfies
-z u''(z)/u'(z)=2z>18>17. Integrating its derivative on the interval between
b and B therefore gives the valid power upper bound

\[
u(B)-u(b)\le\frac A{16}\left[1-(b/B)^{16}\right]
\quad(B\ge9).
\tag{38}
\]

For B<b, the inequality follows by integrating
u'(z)>=u'(b)(b/z)^17; for B>b, integrate the reversed derivative inequality.
This is a consequence of the FIXED utility on outputs already localized by
(37), not an additional curvature assumption or a replacement utility.

The coefficient-2 output envelope, after division by b, is

\[
\frac{\mathcal B}{b}\le r(x,w)
:=\sqrt{\frac{a+x}{q}}\left[1+\frac{x}{a+x}(w+2w^2)\right].
\]

For a competitive design its upper output is at least B>=9. The increasing
right side of (38), evaluated at this upper output, bounds the actual payoff:

\[
\frac{J-V_0}{A}\le F(x,w)
:=\frac{1-r(x,w)^{-16}}{16}
-\frac{w+(x^2-1)/4}{q}.
\tag{39}
\]

We prove F<=0 for all x,w>=0, with equality ONLY at x=1,w=0. In particular
this excludes all positive-prize competitive high-precision designs.

First, F<=1/16+1/(4q)-w/q-x^2/(4q). This is strictly negative when w>=1/2
or x>=2. On x=0, r=sqrt(a/q), so F<0, independently of w. On w=0, F is
strictly concave in x: it is an increasing strictly concave function of the
strictly concave square root, minus a strictly convex quadratic. Its derivative
vanishes at x=1, where F=0. Thus a positive maximum, or any further zero
maximum, would have to be an interior stationary point of
0<x<2,0<w<1/2.

To exclude such a point, write only in this calculation

\[
y=(a+x)/q,\qquad k=x/(a+x),\qquad C=1+w+2w^2,\qquad L=1+4w.
\]

For x>=1, y>=1 and k>=1/q. Bernoulli's inequality gives
y^9>=1+9(x-1)/q>=x and (1+k(w+2w^2))^17>1+4w=L. Hence

\[
qF_w=\frac{xL}{\sqrt y\,r^{17}}-1
=\frac{x}{y^9}\frac{L}{[1+k(w+2w^2)]^{17}}-1<0.
\]

No stationary point has x>=1. For 0<x<1, simultaneous stationarity gives

\[
r^{17}=\frac{xL}{\sqrt y},\qquad
x^2L=1+(2-k)(w+2w^2).
\tag{40}
\]

The second identity follows by differentiating (39) in x and substituting
the first: r_x=[(a+x)+(2a+x)(w+2w^2)]/[2 sqrt(q)(a+x)^(3/2)]. Since 0<k<1,
it implies x^2>=C/L. Moreover

\[
25C-16L=9-39w+50w^2>0,
\]

whose discriminant is -279. Thus x>4/5, and since a<=1/20,
k>=(4/5)/(4/5+a)>=16/17. Concavity of log yields
log(1+k(C-1))>=k log C. Also y>=x for x<1. Combining these facts,

\[
\frac{r^{17}}{xL/\sqrt y}
\ge\frac{x^8[1+k(C-1)]^{17}}L
\ge\frac{C^{4+17k}}{L^5}
\ge\frac{C^{20}}{L^5}>1.
\]

The final strict inequality follows from C>=1+w and
(1+w)^4>1+4w=L for w>0. This contradicts the first equation of (40).
There are no interior stationary points. The boundary comparisons prove
F<=0 with the stated unique equality, and complete the global exclusion.

### 21.5 What this closes, and the remaining single-point task

The payoff comparison is now closed for the entire high-precision domain at
one explicitly chosen risk, with unchanged u,D,S0. Unlike Section 18, it needs
no unproved coefficient-2/3 envelope. Unlike Section 20, it constructs no new
utility or delay cost.

The strict upper-payoff gap also localizes every competitive high-precision
prize/deadline choice near theta=0,T=T0 as c increases to c_e from below.
Indeed (36) gives a common compact set for nearby costs; outside any
neighborhood of the unique zero, the continuous upper payoff has a strictly
negative maximum. Changing c at fixed alpha=theta/c changes payoff by exactly
(c_e-c)alpha, uniformly bounded on that set. The full-disclosure entry derivative
then supplies a positive-prize candidate. Combining localization with (27)
proves a positive-prize full-disclosure optimum against ALL high-precision
competitors for sufficiently small c_e-c>0.

This is NOT the unrestricted three-parameter conclusion: the remaining
competitors are 0<=lambda<sigma^2. Bound (21) has not been proved for them.
The lower output localization (37) and the prize budget (36) DO hold for all
verified precision choices, but do not themselves bound their actual output
by r(x,w). Substituting that envelope at low precision would reintroduce the
original missing proof. No manuscript update or full-disclosure joint-optimum
point is justified yet.

### 21.6 Checks and outstanding conditions

The algebraic constants, derivative identities in (40), and strict polynomial
inequalities are checked by verify_single_point_high_precision.py. This checks
the new payoff proof, not the low-precision HJB envelope. The analytic input
is precisely the already verified high-precision envelope (21) and the
weak-prize comparison (27), with the same equilibrium selection. No new
condition on u,D remains to be supplied for this restricted-domain result.

## 22. Closing the zero-precision endpoint and a positive neighborhood

### 22.1 Scope and proof structure

Keep EXACTLY the primitives and risk/entry-cost point (34)--(35). The
high-precision envelope must not be extrapolated to zero. Instead, construct
the exact analytic no-signal equilibrium on the full competitive prize range,
bound its central gradient, and use the payoff comparison already proved in
Section 21. Then combine a uniform local quadratic bound with compact-set
continuity to obtain a SMALL positive precision neighborhood. Plain pointwise
continuity alone would not preserve the sign as the prize tends to zero.

This does not yet cover intermediate positive precision or close the original
single-point joint-design target.

### 22.2 Exact no-signal gradient construction up to exposure 3/10

Put Q=1+sigma^2 T, alpha=theta/c, e=alpha T/Q. Normalize state by sqrt(Q)
and reverse time by z=alpha(T-t)/Q. Write the two normalized value gradients
as p(x,z),n(x,z), where the physical gradients are p/sqrt(Q),n/sqrt(Q).
At lambda=0 they obey EXACTLY

\[
p_z=(p-n)p_x-pn_x,\qquad
n_z=(p-n)n_x+np_x,\qquad
p(x,0)=n(x,0)=g(x)=\frac{e^{-x^2/2}}{\sqrt{2\pi}}.
\tag{41}
\]

Here n denotes the negative derivative of player j's normalized value; it is
not a negative density. We construct the selected analytic branch, rather
than asserting uniqueness among arbitrary weak solutions of this nonhyperbolic
system.

At any real center x0, the absolute Taylor coefficients of both terminal
gradients are dominated coefficientwise by the positive-coefficient series

\[
h_{x_0}(u)=g(0)e^{-x_0^2/2}\exp(|x_0|u+u^2/2).
\]

The absolute coefficients of each right side in (41) are consequently
dominated by 3 h h_u when both unknown series are dominated by h.
Recursion in the time Taylor coefficients gives the scalar majorant equation
h_z=3h h_u with initial h_x0. Its positive-coefficient solution is the
formal branch of

\[
h(u,z)=h_{x_0}(u+3zh(u,z)).
\tag{42}
\]

This is a CONVERGENT majorant, not a finite-order series assertion. For
0<=u<=1/100, 0<=z<=3/10 and M=1/2, put v=u+3zM<=23/50. Completing
the square gives

\[
h_{x_0}(v)
\le g(0)e^{v^2}
<\frac25 e^{529/2500}<\frac12=M.
\tag{43}
\]

The exponential inequality follows from
e^t<=1+t+t^2/[2(1-t/3)] for 0<=t<3, applied at t=529/2500; the last
rational expression is less than 5/4. Iterating the positive-coefficient
right side of (42) from the initial series fixes each time coefficient after
finitely many iterations and remains bounded by M at the indicated positive
u,z. Its coefficient sum is thus finite. This proves absolute convergence
of the majorant, hence of both actual time/space Taylor series, in the
interior of that polydisc.

The bound is uniform over real x0. Local solutions agree on overlaps by the
uniqueness of the analytic recursion, giving a bounded solution and bounded
spatial derivatives in a common strip for |z|<3/10. The sharper form before
completing the square, with |u+3zh|<=23/50, gives Gaussian real-state tails.
Consequently both gradients have integrable tails. Each equation in (41)
is a scalar transport equation with bounded drift and potential once the
constructed other gradient is inserted. Characteristics preserve strict
positivity. The conservation fluxes
(p^2/2-pn)_x and (pn-n^2/2)_x preserve each gradient's unit mass.

Integrating the gradients with the prescribed tail award values reconstructs
bounded normalized values satisfying the original HJB and terminal CDFs.
The policies alpha p/sqrt(Q),alpha n/sqrt(Q) are nonnegative and bounded
on the finite horizon. Completing the square in each player's control,
followed by the finite-horizon verification identity, verifies the Nash
equilibrium. Thus (41) is an exact verified analytic no-signal branch throughout
0<=e<3/10, not merely a local formal solution at the public mean.

### 22.3 Central bound and exact organizer output

At x0=0,u=0, (42) reduces to h=g(0)exp(9z^2h^2/2). For 0<=z<=3/10,
the positive supersolution M_z=g(0)/(1-z^2) satisfies

\[
\frac{9z^2M_z^2}{2}
=\frac{9}{4\pi}\frac{z^2}{(1-z^2)^2}
<z^2\le-\log(1-z^2)
\quad(z>0).
\]

Indeed pi>3 and (1-z^2)^2>=8281/10000>3/4. Positive iteration of (42)
therefore yields the EXACT bound

\[
0<p(0,z)\le\frac{g(0)}{1-z^2}.
\tag{44}
\]

Reflection symmetry gives p(0,z)=n(0,z); the public mean remains zero
throughout a no-signal contest. Equal efforts leave the true gap's terminal
variance Q unchanged. Hence the exact output is

\[
\mathcal B_0=b(T)+\frac{\alpha T}{\sqrt Q}
\int_0^1p(0,er)\,dr
\le b(T)+\alpha T G_Q(0)\frac1{1-e^2}.
\tag{45}
\]

At the Section 21 point, every competitive design has w<29/100, since
w<=q/(2b)+1/4<29/100 (b^2=1001/(2pi), pi<22/7 gives the strict bound).
Also e=w x/(a+x)<=w. Thus (41)--(45) apply to EVERY competitive
zero-precision design, and

\[
\frac1{1-e^2}\le\frac1{1-w^2}\le1+2w,
\qquad 0<=w<=29/100.
\]

The last inequality is equivalent to w<=2(1-w^2). The same coefficient-2
output screen used in Section 21 now holds at lambda=0, by an INDEPENDENT
proof. Its global payoff argument excludes every positive-prize no-signal
design at c=c_e. The unique optimal prize/deadline there are theta=0,T=T0.

### 22.4 Uniform analytic construction for small positive precision

For continuity we must cover all competitive designs, not just arbitrarily
small prizes. A common positive deadline interval follows without a precision
assumption. The energy screen in Section 2 gives

\[
\mathcal B/b\le\sqrt{(a+x)/q}+\sqrt{4\pi wx/q}.
\]

For x<=1/20,w<=29/100, its right side is less than 23/100+43/100=66/100,
contradicting (37), which requires B/b>3/4. The prize budget also excludes
x>=2. Thus all competitive designs lie in T0/20<T<2T0,w<29/100.

Normalize terminal gradients by Q and physical time by T. Write v=S_T/Q.
If lambda<=1/(4002000 T0), then

\[
1-v=Q^{-1}\int_0^T\lambda S_t^2dt
\le\lambda QT\le4002\lambda T_0\le1/1000
\]

uniformly on that deadline interval. In the Fourier analytic norm of ledger
Section 28, choose initial strip radius 1. The total terminal norm is

\[
M_0=\frac{4e^{1/(2v)}\Phi(1/\sqrt v)}{\sqrt{2\pi v}}<9/4
\quad(v\ge999/1000).
\]

For example pi>157/50, exp(500/999)<33/20, and
Phi(1/sqrt(v))<169/200 give this bound. The last CDF bound follows by
integrating 1-t^2/2+t^4/8 on [0,1] and bounding the remaining interval
to 1/sqrt(v)<1001/1000 by its length times g(0)<2/5.

Use radius rho(r)=1-99r/100 and complex exposure |e|<=291/1000.
Reflection symmetry and the original two conservation fluxes give the
same symmetric norm estimate as ledger Section 28,

\[
M'\le[\rho'+(3/2)|e|M]N,
\qquad
\rho'+(3/2)|e|M_0
<-99/100+(3/2)(291/1000)(9/4)
=-63/8000<0.
\tag{46}
\]

Real nonnegative diffusion only dissipates in this estimate. Fourier cutoffs
and the smaller-radius difference estimate from that construction give a
unique bounded analytic branch, uniform in these parameters, with terminal
radius at least 1/100. Gradient positivity, mass, and Nash verification follow
as in ledger Section 28. It includes every competitive exposure e<29/100
with a strict complex-parameter margin. Thus continuity at zero is justified
in THIS prize/deadline domain; it is not presumed for arbitrary large prizes.

In a smaller strip the second spatial derivatives are uniformly bounded.
The difference estimate for two signal intensities absorbs the diffusion-
coefficient difference as a bounded forcing there. Terminal variance and
coefficients converge uniformly as lambda decreases to zero. Gronwall gives
uniform convergence of the gradients on compact parameter boxes. For real
prizes, their Lipschitz drift and the vanishing signal diffusion imply state-
path convergence in mean square, so the exact output drift identity gives
uniform output continuity. Near zero prize, a sufficiently small complex
exposure disc keeps the complex state path inside the strip; bounded analytic
source integration then gives continuous Taylor coefficients and uniform
third-order remainders by Cauchy estimates. These are the same output-
analyticity qualifications as ledger Section 16, now with the enlarged
competitive prize domain provided by (46).

### 22.5 Global exclusion persists on a positive precision interval

At lambda=0, p_z(0,0)=0 in (41), so (45) has no alpha^2 term. At c=c_e
the exact objective Hessian at theta=0,T=T0 is therefore

\[
J_{\theta\theta}=-\eta/A,\qquad
J_{\theta T}=\frac{1-(\eta+1)/(2q)}{T_0},\qquad
J_{TT}=-\frac{A(\eta+1+2q)}{4q^2T_0^2},\quad\eta=2b>24.
\]

Its dimensionless Schur complement is

\[
-\eta+\frac{(2q-\eta-1)^2}{\eta+1+2q}
=\frac{(2q-1)^2-(6q-1)\eta}{\eta+1+2q}<0.
\]

By the continuous Taylor coefficients just proved, a uniform negative
quadratic bound holds near this zero-prize point for all sufficiently small
lambda>=0. On the complementary compact prize/deadline set, the zero-signal
objective is strictly below V0 by Sections 21 and 22.3. Uniform continuity
supplies a strictly negative gap there for sufficiently small lambda as well.
Hence there exists lambda_0>0 such that, at c=c_e, every positive-prize
design with 0<=lambda<=lambda_0 is strictly worse than V0, and the unique
optimal prize/deadline are theta=0,T=T0.

The strict budget margin w<=q/(2b)+1/4<29/100 persists when c approaches
c_e from below, so the same analytic domain still covers all competitors.
The cost-perturbation argument in Section 21.5 now also applies to this low-
precision interval. For c<c_e sufficiently close, all competitive choices
localize into the uniform weak-prize precision-ranking domain (27). A
positive-prize full-disclosure candidate beats V0 and strictly improves every
competitive low-precision choice at its own prize and deadline. Together
with Section 21 this verifies full-disclosure superiority on

\[
[0,\lambda_0]\ \cup\ [\sigma^2,\infty],
\]

after taking a common sufficiently small cost perturbation.

### 22.6 Remaining gap and verification scope

The unresolved competitors are now in the finite INTERMEDIATE precision
interval lambda_0<lambda<sigma^2. Neither zero-signal analytic convergence nor
the high-precision envelope supplies its output bound. There is no demonstrated
overlap between the two covered intervals. A parameter continuation argument
cannot cross this interval without a valid sign-preserving bound. Accordingly
the unrestricted single-point joint-optimum theorem is STILL NOT closed.

verify_zero_signal_majorant.py checks the new scalar recursion dominance
constants, central supersolution, Fourier-bootstrap margin, and Schur
identity using exact rational arithmetic. It does not compute or certify
the missing intermediate-precision equilibria. No paper statement is changed.

## 23. Extending the upper certificate into intermediate precision

### 23.1 Issue and precise improvement

The cutoff lambda>=sigma^2 in Section 21 came from a convenient sufficient
condition, not an economic boundary. Keeping the constants of Section 17.5
now proves global prize exclusion at the SAME point for every
lambda>=18 sigma^2/25. The user target is still not closed: the remaining
interval is lambda_0<lambda<18 sigma^2/25.

### 23.2 Actual output envelope on the enlarged range

Set r0=18/25. Every potentially winning design at c=c_e has w<29/100.
For lambda>=r0 sigma^2 put a_min=lambda min(S0,S_T)^2 and K=alpha/a_min.
Below lambda=sigma^2/S0^2, posterior variance increases, so
a_min=lambda S0^2>=r0 sigma^2. Above that threshold it decreases toward
sigma/sqrt(lambda), so a_min>=sigma^2. Hence K<=w/r0<29/72<1/2.
The actual-gradient ratio and Gaussian estimates in Section 17.5 give

\[
M<4/5+(5/6)K^2M^2.
\]

Use the smaller barrier Mbar=23/25. Its first-crossing right side is at most
4/5+(5/6)(29/72)^2(23/25)^2<23/25. Thus no crossing occurs.
Sections 17.2--17.3, with this barrier and K, give the exact bound

\[
\mathcal B\le b(T)+\alpha T G_{Q_0}(0)
\cosh^2(K/2)\left[1+\sqrt{4\pi}
\left(\frac{\bar M K}{4}+\frac{5\bar M^2K^2}{12}\right)\right].
\]

Put d=12/47 and K_star=29/72. Since cosh^2(K/2)<=1+dK^2 and
sqrt(4pi)<71/20, its output factor minus one, divided by w, is at most

\[
\frac1{r_0}\left\{dK_*+(1+dK_*^2)\frac{71}{20}
\left[\frac{\bar M}{4}+\frac{5\bar M^2K_*}{12}\right]\right\}
<207/100.
\]

All terms increase in K. Consequently the actual finite-precision policies,
not full-disclosure policies substituted into their game, satisfy

\[
\mathcal B\le b(T)+\alpha T G_{Q_0}(0)(1+207w/100).
\tag{47}
\]

### 23.3 The global payoff argument tolerates this coefficient

Keep Section 21's valid power upper utility and replace coefficient 2 in its
output ratio by C=207/100. The auxiliary payoff F and its boundary comparisons
are unchanged. For x>=1, 17k>=17/q>2C and y^9>=x give F_w<0 as before.
For an interior stationary point with x<1 put B=w+Cw^2,H=1+B,L=1+2Cw.
The same differentiation and elimination give

\[
r^{17}=xL/\sqrt y,\qquad x^2L=1+(2-k)B.
\]

Now 25H-16L=9+(25-32C)w+25Cw^2>0: its discriminant is
-101414/625<0. Therefore x>4/5 and k>=800/801>167/170. Logarithmic
concavity and y>=x imply

\[
\frac{r^{17}}{xL/\sqrt y}
\ge\frac{H^{4+17k}}{L^5}
\ge\frac{H^{10C}}{L^5}>1.
\]

Here 4+17k>10C, and H^(2C)>=(1+w)^(2C)>1+2Cw=L for w>0.
This contradicts stationarity. The payoff upper bound is globally
nonpositive, with unique equality x=1,w=0. Every positive-prize competitor
with lambda>=18 sigma^2/25 is strictly inferior at c=c_e.

### 23.4 Cost perturbation and remaining gap

The strict margin w_budget<29/100 allows the same estimates for c sufficiently
close to c_e from below. Uniform compact localization and (27) then give
positive-prize full-disclosure superiority on this enlarged range. Combining
with Section 22 gives coverage of

\[
[0,\lambda_0]\ \cup\ [18\sigma^2/25,\infty].
\]

No overlap with the first interval has been established. This is a rigorous
extension into the missing interval, not its elimination. Further work needs
estimates not controlled only by minimum diffusion a_min, which degenerates
as precision decreases. No new u,D condition is imposed and no manuscript
theorem is strengthened. verify_intermediate_extension.py checks the rational
margins; it does not certify the remaining precision interval.

## 24. Alternative route: exact precision sensitivity, not minimum diffusion

The target remains one globally optimal positive-prize full-disclosure point
for the fixed ordinary primitives, including S0=1. This section does not
change those primitives or claim that the target is closed. It replaces the
minimum-diffusion estimate by an exact sensitivity identity. The identity is
proved conditional on classical differentiability of the selected equilibrium
and its designer propagation at the finite precision being differentiated;
it is not an existence or regularity theorem for that equilibrium family.

### 24.1 The actual coupled equations

Fix alpha=theta/c>0 and T>0 while changing a positive finite lambda. Write
Ui=Vi/theta, Uj=Vj/theta, p=Ui_y and r=Uj_y (thus r=-n in Section 22).
Let B(y,t) be the expected best terminal score measured relative to the
current average score. With a(t)=lambda S(t)^2, the three backward equations
are

\[
Ui_t+\tfrac a2 Ui_{yy}+\alpha(\tfrac12p^2+pr)=0,
\qquad
Uj_t+\tfrac a2 Uj_{yy}+\alpha(\tfrac12r^2+pr)=0,
\]

\[
B_t+\tfrac a2 B_{yy}+\alpha(p+r)B_y+\tfrac\alpha2(p-r)=0.
\tag{48}
\]

The terminal vector is

\[
z(y,T)=\bigl(\Phi(y/\sqrt{S_T}),
1-\Phi(y/\sqrt{S_T}),\beta(y,S_T)\bigr),
\quad \beta(y,S)=\tfrac12\mathbb E|y+\sqrt S Z|,
\]

where Z is standard normal. The initial designer output is B(0,0).
Every terminal component satisfies the heat identity
\(\partial_S z(y,T)=z_{yy}(y,T)/2\).

Put z=(Ui,Uj,B) and

\[
F(p,r,b)=\alpha\bigl(\tfrac12p^2+pr,
\tfrac12r^2+pr,(p+r)b+\tfrac12(p-r)\bigr).
\]

Then (48) is z_t+(a/2)z_yy+F(z_y)=0. Its linearized operator is
\(L_t h=(a/2)h_{yy}+M h_y\), where the ACTUAL solution supplies

\[
M=\alpha
\begin{pmatrix}
p+r&p&0\\
r&p+r&0\\
B_y+1/2&B_y-1/2&p+r
\end{pmatrix}.
\tag{49}
\]

Let P_{s,t} denote the backward solution operator for h_s+L_s h=0,
mapping data at t to s. It is a vector operator, not a positive scalar
Markov operator. In particular r<0 makes the system noncooperative.

### 24.2 Cancellation of the terminal-variance effect

Let s(t)=partial_lambda S(t), with alpha and T held fixed. Differentiating
the Riccati equation gives

\[
s'=-S^2-2\lambda S s,\quad s(0)=0,\qquad
s(t)=-\int_0^t S(v)^2
\exp\{-2\lambda\int_v^t S(w)dw\}\,dv<0\quad(t>0).
\tag{50}
\]

Also partial_lambda a=-s'. Write v=partial_lambda z. Direct
linearization, including the terminal condition, gives

\[
v_t+L_t v=\tfrac{s'}2z_{yy},\qquad
v(T)=\tfrac{s(T)}2z_{yy}(T).
\]

Consequently

\[
v(0)=\tfrac{s(T)}2P_{0,T}z_{yy}(T)
-\tfrac12\int_0^T s'(t)P_{0,t}z_{yy}(t)dt.
\]

Integrating by parts cancels the terminal term exactly, since s(0)=0.
The derivative of the propagated vector is

\[
\frac d{dt}\{P_{0,t}z_{yy}(t)\}
=P_{0,t}(\partial_t+L_t)z_{yy}(t)
=-P_{0,t}F_{vv}(z_y)[z_{yy},z_{yy}].
\]

Here the last equality follows by differentiating (48) twice in y; no
Gaussian-envelope differentiation or gradient ordering has been assumed.
For a classical family with the spatial derivatives, growth bounds and
parameter dependence needed for these operations, the exact result is

\[
\partial_\lambda\mathcal B(0,c\alpha,T,\lambda;S_0)
=-\frac\alpha2\int_0^T s(t)
\left[P_{0,t}
\begin{pmatrix}
p_y^2+2p_y r_y\\
r_y^2+2p_y r_y\\
2(p_y+r_y)B_{yy}
\end{pmatrix}\right]_3(0)dt.
\tag{51}
\]

All fields and the propagator in (51) are evaluated in the same exact
finite-precision equilibrium. The subscript 3 means the designer component.
For positive finite precision S_T>0; the terminal smoothing in this identity
is retained. Passing to infinity or to zero is a separate limit, not part of
this calculation.

### 24.3 What the new route would need, and what it avoids

Because -s(t)>0, a nonnegative weighted integral on the right of (51)
would prove precision monotonicity at that alpha,T. It is enough to establish
this on the designs capable of beating one full-disclosure candidate, not
on every prize and deadline. Together with a verified full-disclosure limit
and attainment in its explicit two-variable problem, that would certify one
unrestricted joint optimum. A precision derivative sign is stronger than
necessary: an integrated comparison against infinity would suffice.

Unlike Section 23, (51) contains no division by min_t a(t), no precision
floor and no new condition on u or D. It isolates the endogenous information
effect without separately bounding all gradients by an absolute Gaussian
constant. Its remaining sign issue is genuine: the first two sources are
indefinite, and P is not positivity preserving. Nonnegativity of the third
source alone is insufficient because (49) couples it to both value sources.
No sign of (51) for the entire competitive prize range has yet been proved.
The identity therefore opens a different route but is not a full-disclosure
existence theorem or a new assumption to insert into Proposition 1.

### 24.4 Why allowing a tie at prize exit is not yet a shortcut

For fixed sigma let V0=max_T{u(b(T))-D(T)} and, for alpha>0, write
R(alpha,T,lambda)=u(B(0,c alpha,T,lambda;S0))-D(T).
Under the common equilibrium selection used here B depends on alpha rather
than c separately, so R is independent of c. Define the research-only ratio

\[
C_{exit}=\sup_{\alpha>0,T\ge0,\lambda\in[0,\infty]}
\frac{R(\alpha,T,\lambda)-V0}{\alpha}.
\]

If this supremum is finite, then zero prize is globally optimal exactly
when c>=C_exit: this follows directly by subtracting c alpha from R.
If the ratio is attained at positive alpha, positive T and full disclosure,
that design is a joint optimum at c=C_exit, tied with zero prize. This would
meet the single-point target without an open window. The important word is
ATTAINED, not merely approached as alpha decreases to zero.

At Section 21's high-risk reference, the proved full-disclosure-only result
is strict: every positive-prize full-disclosure design loses to V0 at ce,
while its ratio approaches ce as alpha decreases to zero at T0. Hence its
ratio supremum is ce and is not attained at positive alpha. The joint ratio
has C_exit>=ce. If C_exit>ce, full disclosure cannot furnish a positive-prize
optimizer at that exit; if C_exit=ce, its positive-prize designs still lose
strictly there. Thus a tie-at-exit proof cannot close the present reference
point. A successful point at this risk must have c<ce. Establishing local
negative curvature or the common zero-prize entry derivative does not exclude
a remote finite-precision design attaining a larger ratio.

Status: the exact finite-precision sensitivity identity and the exit-ratio
reduction are established with their stated differentiability qualifications.
The weighted response sign in (51), and the original single-point global
existence claim, remain open. The manuscript is unchanged.

## 25. Coarse placement only: a short high-cost zero-prize existence proof

Superseded proof input: Section 26 removes this section's dependence on the
strict all-precision weak-prize ranking. The high-cost conclusion is retained,
but its shorter independent proof is now in the manuscript.

The user target is three nonempty sufficient regions with rough placement,
not exact boundaries, a partition or single switching. Here is a short proof
of the upper zero-prize region under the current ordinary utility and delay
assumptions. It uses the selected equilibria and the audited uniform exact
weak-prize comparison (27). That comparison is an input: this is not a new
proof of it, nor a statement about all possible equilibrium selections.
No lower bound on utility relative curvature is imposed.

### 25.1 Claim

For any compact risk interval 0<sigma_minus<=sigma<=sigma_plus<infinity,
there exists a finite C_Z such that, for c>C_Z, zero prize and the unique
positive zero-prize deadline jointly maximize the exact designer objective.
Every positive-prize design is strictly inferior. At zero prize precision
is irrelevant. Thus an open high-cost strip exists above any chosen bounded
positive-risk interval, including an interval inside the low-risk range of
Proposition 1(i).

### 25.2 Proof at a fixed positive risk

Write b(T)=sqrt((S0+sigma^2 T)/(2pi)) and
f0(T)=u(b(T))-D(T). The manuscript assumptions imply u has a finite upper
limit u_infinity, D is unbounded, and f0 is strictly concave. Moreover
f0'(0)=sigma^2 u'(b(0))/(2 sqrt(2pi S0))>0. Consequently f0 has a unique
maximizer T0>0. Put V0=f0(T0), R=u_infinity-V0>0, and let T_max be the
finite solution D(T_max)=R. Any design with payoff at least V0 satisfies

\[
\theta+D(T)\le R,\qquad 0\le T\le T_{max},\qquad
\alpha:=\theta/c\le R/c.
\tag{52}
\]

The actual equilibrium energy bound, with the corrected sum-energy constant,
gives B<=b(T)+sqrt(2 alpha T). By concavity and monotonicity of u, with
K=u'(b(0)), the corresponding payoff is at most

\[
f0(T)+K\sqrt{2RT_{max}/c}.
\tag{53}
\]

Discarding the negative prize term only weakens this bound. It is valid
uniformly in precision; no zero-precision continuity is required here.

Strict uniqueness of T0 supplies a positive delta such that
f0(T)<=V0-delta on the compact set
[0,T_max] outside (T0/2,2T0). For sufficiently large c, the last term
in (53) is smaller than delta. All potentially competitive designs therefore
have T in (T0/2,2T0). Their variance ratio S0/(sigma^2 T) belongs to a
fixed compact subset of (0,infinity), and their effective prize ratio
w=alpha/sigma^2 tends to zero uniformly by (52).

Choose c still larger so that w is inside the common neighborhood in (27),
and w<=2. Every positive-prize finite-precision competitor can then be
strictly improved by full disclosure at exactly the same alpha and T.
For full disclosure, the exact bound already proved in the manuscript yields

\[
B_{infinity}-b(T)
\le\frac{\alpha T}{\sqrt{2\pi(S0+\sigma^2T)}}(1+w/2)
\le\frac{2\alpha T_{max}}{\sqrt{2\pi S0}}.
\]

It follows that

\[
u(B_{infinity})-c\alpha-D(T)
\le V0+\alpha\left[
\frac{2KT_{max}}{\sqrt{2\pi S0}}-c\right]<V0
\tag{54}
\]

when c exceeds the displayed finite constant. A competitive finite-precision
positive-prize design would be improved by a full-disclosure design that
itself lies strictly below V0, a contradiction. Positive-prize full-disclosure
designs are excluded directly by (54). Designs not competitive with V0
were excluded at the start. Finally theta=0,T=T0 attains V0 and has no
precision-dependent equilibrium issue. This proves joint attainment here,
rather than assuming attainment because lambda=0 is allowed.

### 25.3 Uniformity and coarse placement

On a compact positive-risk interval, T0 and V0 vary continuously by strict
concavity and uniform coercivity. T0 is bounded away from zero, T_max is
uniformly bounded, and the strict deadline gap used in (53) has a positive
common lower bound. The relevant variance ratios lie in one compact subset
of (0,infinity), so (27) supplies a common positive weak-prize threshold.
Also sigma is bounded away from zero in (52). Each required large-c bound
can therefore be chosen uniformly, proving the strip in 25.1.

The proof establishes the TOP region, not a positive-prize full-disclosure
region. Combining it with Proposition 1(i) establishes two coarse locations:
low risk and low cost support finite-feedback advantage, whereas sufficiently
high cost supports zero prize at the same low risks and at any other fixed
bounded risk interval. The former still carries Proposition 1(i)'s attainment
qualification. No c_F=c_P identification or exact boundary is needed.

To obtain all three existence claims under the fixed ordinary primitives,
one positive-prize full-disclosure joint-optimum point remains necessary.
Continuity of designer values alone does not force such a point between
finite feedback and zero prize: finite feedback can in principle persist
until prize exit. Thus this short upper-region proof removes the high-cost
existence task but does not resolve the remaining middle-region task.

## 26. Paper-ready high-cost exit without the long precision-ranking proof

### 26.1 Audit and replacement

Section 25 was logically sufficient given (27), but citing that result would
import a long, delicate unbounded-precision comparison merely to prove zero
prize. It is unnecessary. The new manuscript proof uses only a uniform LINEAR
upper response to small effective incentives, not a strict precision ranking.
The ordinary u,D,S0 remain fixed. No global utility curvature lower bound,
precision floor, precision cost or external design cap is added.

### 26.2 Two short estimates cover all precisions

Fix a finite horizon cap Tbar, set rho0=sqrt(S0) and
M0=2 exp(1/2) sqrt(2/(pi S0)). For
alpha<=rho0/(8 M0 Tbar), precisions lambda<=sigma^2/S0^2 have S_T>=S0.
The actual gradient pair p=Ui_y,n=-Uj_y has Gaussian terminal Fourier data.
In the analytic Fourier norm of radius rho0(1-tau/(2Tbar)), its sum M and
derivative sum N satisfy M'<=(rho'+2 alpha M)N. At M<=M0 the coefficient
is at most -rho0/(4Tbar). Thus the gradients are uniformly bounded by M0,
including at zero signal, and their designer output increment is at most
alpha M0 T. Fourier truncations give the analytic equilibrium; positivity,
mass preservation, bounded-gradient uniqueness at positive precision and
Nash verification are included in the manuscript proof. At zero precision
this specifies the small-prize analytic selection, not all possible weak
no-signal equilibria.

For lambda>=sigma^2/S0^2, actual diffusion a=lambda S_t^2>=sigma^2.
With w=alpha/sigma^2, the constant-w value-sum barrier gives
Ui+Uj>=1-2 log cosh(w/2)/w>=1-w/4. The exact sum-payoff identity therefore
bounds total squared effort by alpha^2/(2 sigma^2). The winner-weighted
output identity and Cauchy--Schwarz give the increment bound
alpha sqrt(T/(2 sigma^2)). The explicit full-disclosure values give the
same bound after averaging over the initial prior.

Consequently, on T<=Tbar, the exact increment is bounded uniformly for all
precisions by alpha L, where
L=max(M0 Tbar,sqrt(Tbar/(2 sigma^2))). There is no minimum-diffusion
division on the low-precision range and no spectral-tail ranking input.

### 26.3 Global optimization and manuscript change

Let V0 be the optimized zero-prize payoff and
Gamma=u_infinity-u(sqrt(S0/(2pi))). Since V0>=u(sqrt(S0/(2pi))),
all potentially competitive designs satisfy theta+D(T)<=Gamma. Choosing
Tbar by D(Tbar)=Gamma thus gives an endogenous PROOF localization, not an
exogenous feasible-design restriction. For c large, alpha=theta/c<=Gamma/c
lies in the uniform response range. Concavity then yields

    J<=V0+alpha[u'(sqrt(S0/(2pi))) L-c]<V0

for every positive-prize competitor. Zero prize at its unique positive
deadline attains V0. Since Gamma and Tbar are independent of sigma, a
uniform upper strip exists over all risks bounded below by any positive
number; no upper risk bound is needed.

This result and its proof have been added as Proposition 1(iii) and proof
Step 5. Changes remain blue. Part (ii) is still explicitly NECESSARY only;
the paper does not claim that all three numerical types have now been
proved to exist. The selected small-prize no-signal equilibrium is stated
where signal endpoints are defined. verify_high_cost_prize_exit.py checks
the exact value-barrier algebra, norm coefficient and effort-weight inequality;
it does not substitute for the analytic convergence or equilibrium argument.

### 26.4 What is still missing

For the fixed ordinary Figure 3 primitives, one positive-prize full-disclosure
JOINT optimum remains unproved. The new upper strip does not fill that gap:
a finite-precision positive-prize branch may persist all the way to prize exit.
Continuity, existence of a full-disclosure-only optimum and weak-prize
precision dominance cannot by themselves exclude such a first-order exit.
The exact response identity in Section 24 remains a separate route to that
missing comparison. No complete three-region existence theorem is claimed.

## 27. Active single-point goal: joint-budget exposure and a time-layer route

### 27.1 Target and why the old rectangular bounds were wasteful

The active long-term goal is a positive-prize, positive-deadline,
full-disclosure JOINT optimum for the ordinary Figure 3 primitives
S0=1, u(B)=3(1-exp(-2B)), D(T)=T^2/2. No functions, design restrictions
or equilibrium payoff objective are changed. Only the external risk point
used in a proof may be selected differently. This section proves construction
and timing estimates, not the missing global precision comparison.

Earlier bounds treated w and x=T/T0 independently, even though every
competitive design satisfies their joint budget. Keeping the joint budget
reduces the exposure alpha T/S_T substantially at precisely those deadlines
that would otherwise have caused the analytic construction to fail.

### 27.2 An explicit alternative external reference

Choose b=251 and set

\[
L=2\pi b^2-1,\quad
\sigma=\left(\frac{4\pi bL}{6}e^{2b}\right)^{1/4},\quad
T0=L/\sigma^2,\quad c_e=2T0^2/\sigma^2.
\tag{55}
\]

As in Section 21, this satisfies the zero-prize first-order condition for
the FIXED ordinary utility and quadratic delay. T0 is its unique zero-prize
maximizer, not a constraint. The magnitudes are a theoretical construction,
not an empirical calibration. Here L>1000 and q=1+1/L<1001/1000.
At c=c_e put x=T/T0 and w=alpha/sigma^2. The same exact utility budget
as (36) gives, for every potentially competitive design,

\[
w+x^2/4\le1/4+q/(2b)<h:=63/250.
\tag{56}
\]

The smaller value of h is obtained by choosing a risk point, not by imposing
a smaller feasible prize or deadline. The old high-precision payoff argument
uses only q<=1001/1000 and b>12, so its algebraic bounds remain available
at this reference. They still do not exclude all low precisions.

### 27.3 An entire low-to-intermediate precision range has verified equilibria

For this proof only, let ell=lambda sigma^2 T0^2; it is a rescaled design
precision, not a new primitive or a proposed condition in the main text.
For 0<ell<=32/5, the Riccati formula gives

\[
S_T=\frac L{\sqrt{\ell}}
\tanh\!\left(\sqrt{\ell}x+
\operatorname{atanh}(\sqrt{\ell}/L)\right).
\]

Thus e=alpha T/S_T satisfies
e<=w z coth(z), z=sqrt(ell)x. The elementary inequality
z coth(z)<=1+z^2/3 follows by expanding
(1+z^2/3)sinh(z)-z cosh(z): its coefficient of z^(2n+1), multiplied
by (2n+1)!, is 4n(n-1)/3>=0. Combining with (56) gives

\[
e\le(h-x^2/4)(1+\ell x^2/3)
\le h+\frac3{32/5}
\left(\frac{(32/5)h}3-\frac14\right)^2
=\frac{11630883}{40000000}<\frac{291}{1000}.
\tag{57}
\]

The second inequality first increases ell to 32/5 and then maximizes the
resulting concave quadratic in x^2. Only x^2<=4h is relevant, so the first
factor is nonnegative when ell is increased. At lambda=0, S_T=1+Lx and
e=wLx/(1+Lx)<=w<h, which also satisfies (57).

Normalize the terminal posterior variance to one and reverse time to a unit
interval. The terminal gradients are both the unit Gaussian density. Their
symmetry p(-y)=n(y) gives equal Fourier norms and equal derivative norms.
Consequently the analytic-norm inequality is the sharper

\[
M'\le[\rho'+(3/2)|e|M]N,
\]

not the coefficient-2 inequality sufficient for the high-cost proof in
Section 26. Positive diffusion is dissipative and does not enter its upper
bound. At radius one the initial total norm is
4 exp(1/2) Phi(1)/sqrt(2pi)<9/4. Taking rho=1-99 tau/100 and
|e|<=291/1000 gives

\[
\rho'+(3/2)|e|(9/4)\le-63/8000<0.
\]

Fourier truncations and smaller-strip difference estimates therefore
construct a common analytic gradient branch through every real competitive
design with ell<=32/5. Positivity, conserved mass and Nash verification
follow as in manuscript proof Step 5. Positive-precision bounded-gradient
uniqueness identifies these solutions with the actual classical equilibria;
zero precision uses the analytic selection. This is a uniform construction
on the WHOLE competitive prize--deadline set in this precision range, not
continuity asserted at one zero-prize point. It does not by itself bound the
designer payoff relative to full disclosure, or establish a corresponding
complex-parameter designer-propagation radius.

### 27.4 The complementary range has an early maturity time

When lambda<18 sigma^2/25, define delta as the first time a(t)=lambda S_t^2
reaches 18 sigma^2/25. The initial variance is below the stationary variance,
so a increases. Its exact time is

\[
\frac\delta{T0}=
\frac{\operatorname{atanh}(\sqrt{18/25})
-\operatorname{atanh}(\sqrt{\ell}/L)}{\sqrt{\ell}}.
\tag{58}
\]

For lambda>=18 sigma^2/25 set delta=0. A rational bound on the positive
atanh series gives 4 atanh(sqrt(18/25))^2<32/5. Hence ell>=32/5
implies delta<T0/2. Once t>=delta, actual diffusion is at least
18 sigma^2/25, so the already proved minimum-diffusion estimates apply to
the FUTURE subgame, with its actual posterior at delta. No subgame before
delta is silently replaced by a full-disclosure game.

There are now overlapping technical cutoffs at ell=32/5: below it the entire
competitive game is analytically constructed; above it the future game
reaches adequate diffusion by T0/2. These are NOT overlapping global PAYOFF
certificates. For a chosen deadline T<=delta there is no mature future
interval; that case must be treated separately rather than discarded.

### 27.5 Remaining tasks and reverse audit

The new construction removes a substantial equilibrium-regularity obstacle
without changing u,D,S0. The single-point joint-optimum proof still needs:

1. A sufficiently sharp real designer-output or direct payoff comparison
   for the constructed competitive designs with ell<=32/5.
2. Actual backward propagation of the mature-subgame gradient/output bounds
   across [0,delta], retaining the exact common first-order Gaussian term.
3. An exclusion or verified comparison for competitive deadlines T<=delta.

The target cannot be declared achieved by (57) or (58): existence of Nash
equilibria is not designer optimality, and a useful future diffusion bound
is not a bound on the initial information layer. The current manuscript
still states only a necessary condition for positive-prize full disclosure.
verify_competitive_exposure.py checks the exact budget, exposure maximum,
analytic-norm margin and maturity-cutoff overlap; it does not certify any
of these remaining payoff comparisons. The long-term goal remains active.

## 28. Actual payoff comparison, and exclusion of an empty mature subgame

This section advances items 1 and 3 of Section 27.5. It does NOT close item 2.
All primitives and the external reference in Section 27 remain unchanged.
The constants below are proof bounds, not restrictions on designer choices.

### 28.1 A convergent gradient majorant retaining the first-order term

Scale the terminal posterior variance to one. For the unit Gaussian density,
the coefficients of its derivative-supremum generating function are bounded
by the nonnegative coefficients of

\[
\Psi(v)=\frac{3991}{10000}e^{v^2/2}
+\frac{50}{157}\sum_{j\ge0}\frac{2^j j!}{(2j+1)!}v^{2j+1}
-\frac3{40}v.
\]

Fourier absolute moments give the even and odd coefficients. For the linear
coefficient use the sharper actual derivative maximum
\(\|G_1'\|_\infty=1/\sqrt{2\pi\exp(1)}<243/1000\).
The bounds \(157/50<\pi<22/7\) and \(\exp(1)>163/60\) verify these
inequalities with rational arithmetic. All coefficients of Psi are positive.

Write H=p+n and A=p-n. Expand their Duhamel equations in the normalized
effort exposure z=alpha(T-t)/S_T and take spatial derivative supremum norms
coefficient by coefficient. Heat evolution contracts every such norm.
The resulting nonnegative majorants satisfy the formal recursive relations

\[
\mathsf H_z=\tfrac12(\mathsf A\mathsf H)_v,\qquad
\mathsf A_z=\tfrac14(3\mathsf A^2+\mathsf H^2)_v,\qquad
\mathsf H(0,v)=2\Psi(v),\quad\mathsf A(0,v)=0.
\]

These are bounds on Taylor coefficients, not a comparison theorem for a
vector HJB system. The minus sign in the actual A equation has been replaced
by a plus sign solely when bounding absolute coefficients. With eta=7/4,

\[
\tfrac12\mathsf A\mathsf H+
\tfrac\eta4(3\mathsf A^2+\mathsf H^2)
\preceq\tfrac\eta4(\mathsf H+\eta\mathsf A)^2.
\]

Here the order means coefficientwise order of nonnegative series; the
inequality follows from eta^2>=3. Recursive induction therefore bounds both
p and n by the scalar series h defined by

\[
h(z,v)=\Psi(v+\eta z h(z,v)),\qquad
h_k:=[z^k]h(z,0)
=\frac{\eta^k}{k+1}[v^k]\Psi(v)^{k+1}.
\]

Set E=18/35 and M=5/8. Direct rational upper bounds on the exponential and
odd-series tails give Psi(eta E M)<M. Monotone scalar iteration is thus bounded
at (E,0), with a strict margin permitting a positive spatial radius and a
slightly larger exposure. The nonnegative power series converge; contraction
of the heat semigroup and the product estimates reconstruct the bounded
analytic gradient solution. This also justifies the Duhamel expansion rather
than treating it as a formal perturbation. At positive precision uniqueness
identifies the actual bounded classical branch; at zero use the analytic
selection already specified in the manuscript.

Let D_a(z)=h(z,0)-h_0. Keeping the zeroth-order Gaussian exactly, rather than
replacing it by h_0, gives, for each actual p,n,

\[
|p-G_{Q_t}|,\ |n-G_{Q_t}|
\le S_T^{-1/2}D_a\bigl(\alpha(T-t)/S_T\bigr).
\tag{59}
\]

### 28.2 Actual state propagation and designer output

Couple the actual state with the uncontrolled public state using the same
innovation Brownian motion. Their drift difference is alpha A. By (59),

\[
|Y_t-Y_t^0|\le\frac{2\alpha}{\sqrt{S_T}}
\int_0^tD_a\bigl(\alpha(T-s)/S_T\bigr)ds.
\]

In the exact output identity, the winner weights are nonnegative and sum to
one. Also \(\|G_{Q_t}'\|_\infty\le(243/1000)/S_T\), and
\(\mathbb E G_{Q_t}(Y_t^0)=G_{Q_0}(0)\). Consequently, with e=alpha T/S_T,

\[
\mathcal B-b(T)\le\alpha T G_{Q_0}(0)
+\frac{\alpha T}{\sqrt{S_T}}g(e),\qquad
g(e)=\int_0^1D_a(ev)dv+
\frac{486}{1000}e\int_0^1vD_a(ev)dv.
\tag{60}
\]

This is a bound for the strategic state, not a Gaussian approximation to it.
It preserves the common exact first-order output term.

The sharper Riccati inequality z coth(z)<=sqrt(1+z^2) follows from sinh(z)>=z.
For every competitive design with ell<=17, it implies

\[
\frac e w\le\sqrt{1+\ell x^2},\quad
\frac{Q_0}{S_T}\le\sqrt{1+\ell(a+x)^2},\quad
e^2\le(h-x^2/4)^2(1+17x^2)<E^2.
\]

The second bound uses atanh(sqrt(ell)/L)>=sqrt(ell)/L and
Q_0=1+Lx. For ell=0 take the explicit posterior, or the continuous formula;
no arbitrary zero-signal equilibrium is being included.

The script verify_low_precision_payoff_envelope.py certifies, by an exact
rational interval cover of 0<=x<=251/250, that (60) gives

\[
\mathcal B\le b(T)+\alpha T G_{Q_0}(0)(1+2.07w)
\qquad(\ell\le17).
\tag{61}
\]

For reproducibility it uses the first 24 coefficients h_k. If
C_N=M-sum_{k=0}^N h_k E^k>0, the omitted contribution to g(e)/e is bounded by

\[
\frac{C_N e^N}{E^{N+1}(N+2)}+
\frac{486}{1000}\frac{C_N e^{N+1}}{E^{N+1}(N+3)}.
\]

Nonnegative coefficients justify this tail bound for 0<=e<=E. The script
compares squared rational bounds with (207/100)^2; floating-point values are
printed only for diagnostics. Its 16 covering intervals have largest squared
upper bound below 4.254, whereas (207/100)^2=4.2849. Thus this is a finite
algebraic certificate, not an HJB grid or sampled maximization.

Substituting (61) into the already verified payoff algebra of Section 23
excludes every positive-prize competitor with ell<=17 at c=c_e. Together
with the existing minimum-diffusion certificate, this covers both ends of
the precision range. It does not cover their intervening interval.

### 28.3 A deadline ending before maturity cannot compete

Any design with J>=V_0 must have
B>=b-0.5 log(1+b/(2q))>.99b. Indeed b=251,q>=1 and
exp(5)>253/2, verified by a finite positive Taylor sum.

Suppose ell>=32/5 and T<=delta, with delta as in Section 27.4. Then
x<=1/2 and Q_0/S_T<3/2. To see the latter, put
y=sqrt(ell)x+atanh(sqrt(ell)/L). Since y<=atanh(sqrt(18/25)),
Q_0/S_T<y coth(y)<=atanh(sqrt(18/25))/sqrt(18/25)<3/2.
The last strict inequality follows already from the maturity-series bound
in Section 27.4. Hence e<=3h/2<E and the scalar majorant gives
p,n<=M/sqrt(S_T). The exact winner-weight output identity now yields

\[
\frac B b\le\sqrt{a+x}
\left[1+\frac58\frac{251}{100}\frac{49}{40}
\left(h-\frac{x^2}{4}\right)\right]<.99,
\qquad 0\le x\le\tfrac12.
\tag{62}
\]

We used q>=1, sqrt(2pi)<251/100, sqrt(3/2)<49/40 and a<1/1000.
The right side increases on this interval: after multiplying its derivative
by 2sqrt(a+x), its sign is bounded below by
1+C h-C(5/16+1/2000)>0, where C=(5/8)(251/100)(49/40).
At x=1/2 replace sqrt(a+x) by 71/100; the resulting rational upper bound
is strictly below .99. The same script verifies every stated rational sign.
This contradicts the necessary competitive-output bound. Thus a competitive
deadline cannot end before maturity in the complementary precision range.

### 28.4 Exact status

Two of Section 27.5's three tasks now have genuine payoff certificates:
the low-precision comparison extends to ell<=17, and the empty mature
subgame is excluded. What remains is the strategic propagation across
[0,delta] for ell>17, T>delta, and lambda<18 sigma^2/25. Neither convergence
of a gradient series nor a future-subgame output bound automatically controls
that initial layer. The current certificates therefore do not yet establish
the requested positive-prize full-disclosure joint optimum. The goal stays
active; Proposition 1 must not be strengthened on their basis alone.

## 29. A strict mature-time buffer and a small early-layer exposure

The previous payoff certificate leaves ell>17, lambda<18 sigma^2/25.
This section narrows that gap without changing the primitives or imposing
a deadline restriction. The restrictions derived below apply only to designs
whose actual payoff competes with the no-prize optimum. They cannot be
silently assumed for arbitrary designs.

### 29.1 A competitor must satisfy T>3 delta/2

Retain r0=18/25, h=63/250, a=1/L<1/1000 and the exact maturity time (58).
If ell>17 then

\[
\frac{\delta}{T_0}
<\sqrt{\frac{8}{5\ell}}<\frac{31}{100}.
\]

For a deadline T<=3 delta/2 put
y=sqrt(ell)x+atanh(sqrt(ell)/L). Formula (58) gives
y<=3 atanh(sqrt(r0))/2<19/10. The Riccati solution therefore satisfies

\[
\frac{Q_0}{S_T}\le y\coth y<2,
\qquad e=\frac{\alpha T}{S_T}\le w\frac{Q_0}{S_T}<2h<\frac{18}{35}.
\tag{63}
\]

For the second inequality note that sigma^2 T<Q_0. The function y coth(y)
is increasing for y>0, since its derivative has the sign of
sinh(y)cosh(y)-y>0. At y=19/10 it is below 2 because
exp(19/5)>39; the latter has a finite positive rational Taylor certificate.

Consequently the convergent majorant of Section 28 applies to the WHOLE
game in this case, not just a future subgame. Both p and n are bounded by
(5/8)/sqrt(S_T). The exact output identity yields

\[
\frac B b\le\sqrt{a+x}
\left[1+C\left(h-\frac{x^2}{4}\right)\right],\qquad
C=\frac58\frac{251}{100}\frac{283}{200},\quad
0\le x\le\frac{93}{200}.
\]

We used sqrt(2pi)<251/100, sqrt(Q_0/S_T)<sqrt(2)<283/200,
q>=1 and the competitive bound w<h-x^2/4. The derivative of this last
upper bound is positive throughout the displayed interval: after multiplying
by 2sqrt(a+x), its numerator is
1+C h-C(5x^2/4+a x)>0. At x=93/200,
sqrt(a+x)<683/1000, and the resulting rational upper bound is below .99.
This contradicts B/b>.99 from Section 28.3. Thus every competitor in the
remaining range has T>3 delta/2. The conclusion implies a mature future
interval of length T-delta>T/3, not merely a nonempty interval.

Independently, the energy screen already gives T>T_0/8 for ANY competitive
design. For x<=1/8 its normalized output upper bound is less than
9/25+63/100=.99: use a<1/1000, w<h and pi<22/7. This strict lower deadline
bound will be useful when delta is extremely small.

### 29.2 The early exposure uses Q_delta, not the terminal posterior S_T

The exact maturity variance is S_delta=sqrt(r0)*L/sqrt(ell). Hence

\[
\frac{S_\delta}{\sigma^2\delta}
=\frac{\sqrt{r_0}}{
\operatorname{atanh}\sqrt{r_0}-
\operatorname{atanh}(\sqrt\ell/L)}
>\frac{\sqrt{r_0}}{\operatorname{atanh}\sqrt{r_0}}>\frac23.
\]

The last inequality follows from the already verified
atanh(sqrt(r0))/sqrt(r0)<3/2. By definition,
Q_delta=S_delta+sigma^2(T-delta). Combining this identity with T>3 delta/2
gives

\[
Q_\delta>\sigma^2(T-\delta/3)>
\frac76\sigma^2\delta,\qquad
\frac{\alpha\delta}{Q_\delta}<\frac67w<\frac{27}{125}=.216.
\tag{64}
\]

This is the appropriate dimensionless duration for propagating analytic
gradient data of spatial scale sqrt(Q_delta) through the early layer. It
does not assert that those data are exactly Gaussian. In particular the
terminal data for that propagation are the ACTUAL future-game p(delta),
n(delta), not G_{Q_delta} alone. The data's derivative bounds remain a
necessary input before applying any scalar majorant to that propagation.

### 29.3 The future game has a uniform amount of smoothing

Let tau=T-delta. One also has S_delta<(3/2)sigma^2 tau. For ell<=L,
sqrt(ell)/L<=1/sqrt(L)<4/125 and atanh(sqrt(ell)/L)<1/25.
A finite positive atanh series proves atanh(sqrt(r0))>6/5. Therefore
atanh(sqrt(r0))-atanh(sqrt(ell)/L)>29/25>(4/3)sqrt(r0),
so S_delta/(sigma^2 delta)<3/4. Since tau>delta/2, the assertion follows.
For ell>=L use instead tau>T/3>T_0/24 and
S_delta/(sigma^2 T_0)=sqrt(r0/ell)<=sqrt(r0/L). The inequality
24sqrt(r0/1000)<3/2 proves the same assertion. These two cases exhaust the
range and agree at ell=L.

Thus Q_delta=S_delta+sigma^2 tau<(5/2)sigma^2 tau. Throughout the future
game a(t)>=r0 sigma^2, whence its actual accumulated innovation variance is

\[
Q_\delta-S_T=\int_\delta^T a(t)dt
\ge r_0\sigma^2\tau>\frac{36}{125}Q_\delta.
\tag{65}
\]

The strictly positive fraction .288 is not a change in feedback policy.
It proves that the future game is not an arbitrarily short unsmoothed layer
relative to the spatial scale of its gradients at delta.

### 29.4 Reverse audit and remaining work

verify_time_layer_localization.py verifies the rational bounds in (63)--(65),
the strict output exclusion and its derivative sign. The argument uses the
actual output identity and the already convergent Section 28 majorant, not
a local-prize approximation. No new utility, delay-cost or design-domain
condition has been added.

The remaining task is now more specific: obtain quantitative analytic
derivative bounds for the ACTUAL future-game gradients at delta, whose
smoothing fraction is at least .288, and propagate them through an early
exposure below .216. The real Gaussian envelopes in Section 23 alone cannot
be differentiated to supply those bounds: a bound on a function does not
bound its derivatives. Nor does qualitative parabolic analyticity give a
numerical constant small enough for the designer payoff screen. Those steps
must still be proved. Accordingly (64) and (65) are useful proved inputs,
not the missing full-disclosure joint-optimum theorem itself.

## 30. A pointwise analytic-jet bound and a stronger fixed-utility payoff screen

### 30.1 Audit and structure

The unresolved derivative input in Section 29 cannot be obtained by
differentiating the real Gaussian envelope. The coefficientwise scalar
majorant of Section 28 is valid, but taking a separate spatial supremum for
each initial derivative loses the Gaussian's common spatial location. We can
avoid that loss by summing derivatives at a point FIRST and then taking the
spatial supremum. This provides a genuine analytic-norm continuation bound.

Separately, the power-16 utility estimate in Section 23 was chosen for an
earlier, much smaller output reference. At the present fixed reference b=251,
the SAME utility supports power 100. This allows a larger output-error
coefficient and enlarges the strict payoff exclusion to ell<=21. Neither
improvement changes u,D,S0 or asserts a full joint optimum.

The order of proof is: local analytic solution -> derivative-norm estimate
-> continuation on a uniform exposure interval -> stronger deadline exclusion;
and independently, fixed-utility power estimate -> global payoff algebra ->
application to a verified actual output envelope. The two uses of analyticity
and payoff comparison must not be conflated.

### 30.2 A derivative bound, not a differentiated function bound

Normalize S_T=1, and put physical backward time s=T-t. On a local analytic
solution define the pointwise derivative sums

\[
L_H(y,\rho,s)=\sum_{j\ge0}\frac{|\partial_y^j H(y,s)|}{j!}\rho^j,
\quad L_A(y,\rho,s)=\sum_{j\ge0}\frac{|\partial_y^j A(y,s)|}{j!}\rho^j,
\quad W=\tfrac12(L_H+\eta L_A),\quad\eta=\tfrac74.
\]

These nonnegative sums have nonnegative rho derivatives. The Kato inequality
for each derivative and the product rule in the ACTUAL gradient equations give

\[
\partial_s L_H\le\tfrac{a(s)}2\partial_{yy}L_H
+\tfrac{|\alpha|}{2}\partial_\rho(L_A L_H),\qquad
\partial_s L_A\le\tfrac{a(s)}2\partial_{yy}L_A
+\tfrac{|\alpha|}{4}\partial_\rho(3L_A^2+L_H^2).
\]

Absolute values can be regularized before using Kato. Sum first on a strictly
smaller analytic radius and pass to the absolutely convergent series;
all differentiated products converge there. The polynomial flux difference
used in Section 28 has nonnegative coefficients, so its rho derivative is
nonnegative as well. Consequently

\[
W_s\le\tfrac{a(s)}2W_{yy}+\eta|\alpha|W W_\rho.
\tag{66}
\]

This is a SCALAR differential inequality for the derivative norm. It is not
a vector comparison of the original HJB, nor a coefficientwise assertion
that the spatial supremum can be differentiated.

For the initial Gaussian, expanding its translation with absolute Hermite
coefficients yields

\[
W(y,\rho,0)\le G_1(y)e^{|y|\rho+\rho^2/2},\qquad
\sup_y W(y,\rho,0)\le\frac1{\sqrt{2\pi}}e^{\rho^2}.
\tag{67}
\]

The second inequality follows by completing the square in y. It is a
pointwise bound on the WHOLE derivative sum, not a bound on each coefficient
by those of the even function exp(rho^2).

Choose positive r,M such that g0 exp(r^2)<M, where g0=1/sqrt(2pi), and set
rho(s)=r-eta|alpha|M s. As long as rho(s)>0, (66) gives for the function
W(y,rho(s),s)

\[
\frac d{ds}W\le\tfrac{a(s)}2W_{yy}
+\eta|\alpha|(W-M)W_\rho.
\]

On a first-crossing interval W<=M the last term is nonpositive. The scalar
maximum principle thus keeps the spatial supremum below its initial value
and below M. The same argument with arbitrarily small strict margins deals
with a supremum not attained on the real line. This uniform positive analytic
radius and bounded derivative norm continue the local analytic solution:
the usual analytic-scale local construction can be restarted with bounds
independent of the restart time. For fixed positive precision the bounded
derivatives also provide classical continuation; at zero precision this
continues the selected analytic solution, not arbitrary weak solutions.
The nonnegative deterministic diffusion coefficient only contributes the
dissipative Kato term; it is never divided by a positive lower bound.

In particular r=707/1000, M=33/50 and e=|alpha|T<=61/100 satisfy

\[
\tfrac25 e^{r^2}<M,\qquad r-\eta(61/100)M>0.
\]

Exact rational Taylor-tail bounds verify both inequalities. Restoring S_T,
we obtain a verified analytic branch whenever |alpha|T/S_T<=.61, with

\[
\sup_y\sum_{j\ge0}
\frac{\sqrt{S_T}^{\,j+1}}{j!}
\frac{|\partial_y^j H|+\eta|\partial_y^j A|}{2}
\rho_*^j\le\frac{33}{50},\qquad
\rho_*=r-\eta\frac{|\alpha|T}{S_T}M>0.
\tag{68}
\]

Earlier times have a larger remaining radius. For real alpha the scalar
equations for p and n, with the constructed other gradient inserted, have
bounded drift and potential and preserve positivity. Gaussian initial tails
and bounded coefficients give integrable tails; the conservative fluxes
preserve each gradient's unit mass. Reconstructing the values from the tail
award constants and completing the square in each control then verifies
the Nash policies. No complex-prize object is interpreted as an equilibrium.
If complex prize dependence is needed, the same absolute-value estimate
with |alpha| bounds the local holomorphic continuation as well.

More flexibly, a REAL exposure bound e admits any cap M satisfying
g0 exp((eta e M)^2)<M: continuity permits an initial radius r>eta e M
with (67) below M. This flexibility supplies the strict output comparison
in the next step.

Under the joint competitive budget, the Riccati bound also gives
e^2<=(h-x^2/4)^2(1+ell x^2). Its maximum for ell=36 is below .61^2,
by direct rational maximization in x^2. Thus (68) constructs the entire
competitive family through ell<=36, not merely the previous ell<=17.
This last observation is a regularity statement, not a payoff classification.

### 30.3 A competitor must satisfy T>19 delta/10

Continue in the remaining range ell>17. Recall T>T_0/8 from Section 29.1.
If T<=19 delta/10, then delta/T0<307/1000 and
x<=5833/10000. The Riccati argument in (63) now yields

\[
y\le\tfrac{19}{10}\operatorname{atanh}\sqrt{r_0}<2.41,
\quad\frac{Q_0}{S_T}<\frac{49}{20},\quad
e<\left(h-\frac{x^2}{4}\right)\frac{49}{20}<.61.
\]

Indeed 2.41 coth(2.41)<2.45 is equivalent to exp(4.82)>121.5; a finite
positive Taylor sum certifies it. The last exposure bound uses x>1/8.
For each rational interval [l,v] covering 1/8<=x<=5833/10000, take
w_l=h-l^2/4 and an adaptive rational cap M_l satisfying
(2/5)exp((eta w_l(49/20)M_l)^2)<M_l. The exact output identity and (68) imply

\[
\frac B b\le\sqrt{a+x}
\left[1+M_l\frac{251}{100}\frac{783}{500}w_l\right]<.99.
\tag{69}
\]

Here (783/500)^2>49/20. The seven rational covering intervals and the strict
exponential tail tests are reproduced by verify_peak_majorant_and_buffer.py;
the largest upper bound is below .989868. The cover is of this scalar
upper-bound formula, not of numerically computed equilibria. It contradicts
the necessary B/b>.99 and excludes every such deadline.

Combining T>1.9 delta with S_delta>(2/3)sigma^2 delta sharpens (64) to

\[
\frac{\alpha\delta}{Q_\delta}<\frac{30}{47}w<.161.
\tag{70}
\]

For ell<=L, the proof of Section 29.3 gives S_delta<(3/4)sigma^2 delta.
Now tau=T-delta>(9/10)delta, so S_delta<(5/6)sigma^2 tau.
For ell>=L the previous independent bound tau>T0/24 gives the same result
since 24sqrt(r0/1000)<5/6. Therefore

\[
Q_\delta-S_T\ge r_0\sigma^2\tau
>\frac{108}{275}Q_\delta>.39Q_\delta.
\tag{71}
\]

Both the early exposure and the mature smoothing fraction improve without
any new primitive assumption. They still do not themselves bound the
actual mature gradients' derivatives on the scale sqrt(Q_delta).

### 30.4 The fixed utility tolerates output coefficient 12/5

At b=251 every competitive output exceeds .99b. For the SAME fixed utility,
-B u''(B)/u'(B)=2B>101 throughout the interval between b and any competitive
output. Integrating as in (38) now gives, with A=b u'(b),

\[
u(B)-u(b)\le\frac A{100}[1-(b/B)^{100}].
\]

Suppose a proved actual envelope gives the output correction C=12/5,
so B/b<=r=sqrt(y)[1+k(w+Cw^2)], y=(a+x)/q,k=x/(a+x).
Then the payoff upper bound is
F=(1-r^{-100})/100-[w+(x^2-1)/4]/q. We prove F<=0 globally,
with unique equality x=1,w=0.

The far boundaries w>=1/2 or x>=2 are negative from
F<=1/100+1/(4q)-w/q-x^2/(4q). At x=0, r=sqrt(a/q) and F<0.
At w=0 strict concavity in x and F_x(1,0)=0 give the unique maximum zero.
Thus any other nonnegative maximum must be an interior stationary point.

Write B_w=w+Cw^2, H_w=1+B_w,L_w=1+2Cw only in this calculation.
At x>=1, y^51>=x and (1+k B_w)^101>L_w, since 51/q>1 and 101/q>2C.
Hence qF_w=x L_w/[sqrt(y)r^101]-1<0. At x<1, differentiating F and
eliminating r^101 gives exactly

\[
r^{101}=\frac{x L_w}{\sqrt y},\qquad
x^2L_w=1+(2-k)B_w.
\]

The second identity implies x^2>=H_w/L_w. The polynomial
H_w-(3/5)L_w=2/5+(1-6C/5)w+Cw^2 is strictly positive for all real w
because its discriminant is negative. Thus x>sqrt(3/5)>3/4 and
k>750/751. Concavity of log and y>=x give

\[
\frac{r^{101}}{x L_w/\sqrt y}
\ge\frac{H_w^{25+101k}}{L_w^{26}}>1.
\]

For the last strict sign, 25+101(750/751)>52C, and
H_w^{2C}>=(1+w)^{2C}>1+2Cw=L_w for w>0. This contradicts stationarity
and proves the global payoff screen. verify_fixed_utility_payoff_margin.py
checks its rational signs; no changed utility or curvature assumption was used.

Running the Section 28 actual-output certificate with ell=21 and target
coefficient 12/5 succeeds on 41 rational intervals. Together with this payoff
argument it excludes all positive-prize competitors with ell<=21 at c=c_e.
The old ell<=17 certificate remains independently verified. This enlarged
low-precision payoff range still does not meet the high-precision certificate.

### 30.5 Remaining condition and completion audit

The new derivative norm (68) is proved and quantitative, but its scale is
sqrt(S_T), not automatically sqrt(Q_delta). For a mature future game with
very small S_T, its exposure alpha(T-delta)/S_T can exceed .61, so (68)
cannot simply be applied to claim the missing handoff estimate. Likewise
(71) supplies a smoothing fraction, not the numerical derivative bound.

The remaining task is the actual mature-gradient handoff in the intervening
precision range, followed by the real early-layer output comparison. All
three scripts pass; their scopes are explicitly narrower than global joint
optimality. The requested single joint-optimal full-disclosure point is still
unproved, the long-term goal remains active, and the manuscript is unchanged.

Manuscript audit flag for the final integration: the current conclusion still
says the exact analysis distinguishes a weak-incentive region where full
disclosure is optimal, although the present Proposition states only its
necessary condition. That concluding sentence must be reconciled with the
actually proved scope during the paper update; it is not evidence that the
current existence target was already achieved. The figure uses the unchanged
primitives in (34). No source edit has been made in this research step.

## 31. Actual mature-gradient derivatives from a split heat estimate

### 31.1 Problem and proof order

The first genuine handoff derivative can be bounded without complex-state
propagation. Simply differentiating the source everywhere is insufficient:
it gives a logarithm of Q_delta/S_T, which is not uniform as S_T decreases.
Instead differentiate the source only where Q_s is comparable to Q_t, and
differentiate the HEAT KERNEL where the accumulated diffusion is substantial.
All bounds below apply to actual mature-game gradients, not a differentiated
real-function envelope.

Fix the future subgame [delta,T]. Its diffusion is at least
a_min=(18/25)sigma^2. Set K=alpha/a_min<=7/20, as the competitive budget
gives w<63/250. Section 23 supplies, in its established Gaussian norm,
H<=h E_Q and |A|<=a_* E_Q, with h=23/25 and a_*=K h/2.
The new norm is only for the derivatives now being proved:

\[
\|f\|_{G_1}=\sup_{t,y}Q_t e^{y^2/(4Q_t)}|f(y,t)|.
\]

At K=0, H=2G_Q,A=0 and all assertions follow directly. The proof below
uses K>0, first establishing derivatives, then retaining the exact leading
Gaussian source to sharpen their remainder orders. Finally it establishes
an independent local prize-analyticity bound from the mature real solution.

### 31.2 Near and far estimates for the differentiated mild equation

For each t split the s-integral at Q_s=(9/10)Q_t. If that level is outside
the actual integration interval, the corresponding piece is simply empty.
All formulas continue to hold with these shorter intervals.

On the near piece, commute the derivative to the product. If |f|<=M_f E_Q
and |g_y|<=D_g Q^{-1}exp(-y^2/(4Q)), the elementary Gaussian kernel
calculation gives, with v=Q_t-Q_s,

\[
Q_t e^{y^2/(4Q_t)}|\partial_y P_v(f_s g_{y,s})(y)|
\le M_fD_g\left[\frac{\sqrt{2/\exp(1)}}{Q_s}
+\frac{\sqrt{2/\pi}}{\sqrt{v Q_s}}\right].
\]

The two time integrals are at most
log(10/9)/a_min and 2 asin(sqrt(1/10))/a_min, respectively: use
-dQ_s/ds=a(s)>=a_min. Thus the near constant is

\[
c_N=\sqrt{2/\exp(1)}\log(10/9)
+2\sqrt{2/\pi}\arcsin\sqrt{1/10}<.61.
\tag{72}
\]

On the far piece keep both derivatives on the heat kernel, and use only the
PROVED function bound |fg|<=M_fM_g Q_s^{-1}exp(-y^2/(2Q_s)). In the Gaussian
product, x-y has conditional mean -v y/Q_t and variance v Q_s/Q_t.
The kernel's second derivative is ((x-y)^2/v^2-1/v) times its density.
Taking its absolute value and its conditional second moment gives

\[
Q_t e^{y^2/(4Q_t)}|\partial_{yy}P_v(f_sg_s)(y)|
\le\frac{M_fM_g}{\sqrt{Q_sQ_t}}
\left[\frac4{\exp(1)}+\frac{Q_t+Q_s}{Q_t-Q_s}\right].
\]

Integrating with r=Q_s/Q_t<=9/10 yields a far constant

\[
c_F=\left(\frac8{\exp(1)}-2\right)\sqrt{9/10}
+4\operatorname{atanh}\sqrt{9/10}<8.2.
\tag{73}
\]

The appearance of 8.2 is a kernel bound, not a new parameter. Rational
upper bounds for the logarithm, arcsine and atanh tails verify (72)--(73).
Unlike differentiating a pointwise envelope, this argument differentiates
the actual integral equation before applying an envelope to its sources.

Let d_H=||H_y||_{G_1}, d_A=||A_y||_{G_1}. The exact mild equations imply

\[
d_H\le .7+\frac K2[c_F a_*h+c_N(a_*d_H+h d_A)],
\qquad
d_A\le\frac K4[c_F(3a_*^2+h^2)+2c_N(3a_*d_A+h d_H)].
\tag{74}
\]

Here .7 bounds ||2G_Q'||_{G_1}; the actual value is
2 sqrt(2/exp(1))/sqrt(2pi)<.7. A first crossing of the barriers
d_H=1.1 or d_A=2.5K contradicts (74): its right sides at K=.35 are
below 1.017392 and .816234, while the barriers are 1.1 and .875.
After dividing the second bound by K, both right sides increase in K,
so the same strict signs apply throughout 0<K<=.35. Local weighted
derivative estimates start at the Gaussian terminal data; the strict
barriers continue them through the mature interval. This proves finite
uniform bounds even when Q_delta/S_T is arbitrarily large.

### 31.3 Keep the exact first coefficient, and sharpen the derivative remainder

Put A^(1)=-alpha T(G_Q,G_Q), where T is the ORIGINAL bilinear heat operator
on [t,T], and write R_H=H-2G_Q. This is an exact decomposition, not a
replacement of equilibrium by leading-order policies. The Gaussian product
can be propagated explicitly:

\[
A^{(1)}(y,t)=-\frac\alpha{2\sqrt\pi}
\int_t^T\frac{G'_{Q_t-Q_s/2}(y)}{\sqrt{Q_s}}ds.
\tag{75}
\]

For R=Q_t-Q_s/2 put r=R/Q_t in [1/2,1]. Directly from G_R'',

\[
Q_t e^{y^2/(4Q_t)}|G_R''(y)|
=\frac1{\sqrt{2\pi Q_t}}r^{-3/2}
|z-1|e^{-(2-r)z/4},\qquad z=y^2/R.
\]

For 0<=z<=1 the bracket is at most r^{-3/2}<=2sqrt(2). On z>=1 its
maximum is r^{-3/2}4 exp(-1-(2-r)/4)/(2-r). Since
r^{-3/2}/(2-r) decreases on [1/2,1] and exp(1)>8/3, that maximum is
also below 2sqrt(2). Integrating (75) and using
int_t^T ds/sqrt(Q_s)<=2sqrt(Q_t)/a_min therefore proves

\[
\|(A^{(1)})_y\|_{G_1}\le\frac2\pi K<\frac23 K.
\tag{76}
\]

The established function-level estimates give ||R_H||_G<=b_*K^2,
b_*=(5/6)h^2. Subtracting (75) from the exact A equation gives

\[
A-A^{(1)}=\frac\alpha4\int_t^T\partial_yP_{Q_t-Q_s}
\{3A_s^2-R_{H,s}(H_s+2G_{Q_s})\}\,ds.
\]

Apply the SAME near/far estimates to this residual and to
R_H=alpha T(A,H)/2. Suppose
||R_{H,y}||_{G_1}<=U K^2 and ||A_y||_{G_1}<=L K. Then (72)--(76) give

\[
\frac{\|R_{H,y}\|_{G_1}}{K^2}
\le\frac{c_F h^2}{4}
+\frac{c_N}{2}\left[\frac h2(.7+U K^2)+h L\right],
\]

\[
\frac{\|A_y\|_{G_1}}K\le\frac23+
\frac{K^2}{4}\left\{
c_F[3h^2/4+b_*(h+.8)]
+c_N[3h L+U(h+.8)+b_*(1.4+U K^2)]\right\}.
\]

At U=2.3,L=1.4 and K=.35, the two right sides are below 2.265700 and
1.298986. The strict first-crossing argument thus proves the sharper
ACTUAL derivative estimates

\[
\|\partial_y(H-2G_Q)\|_{G_1}<2.3K^2,\qquad
\|A_y\|_{G_1}<1.4K.
\tag{77}
\]

In particular ||H_y||_{G_1}<.7+2.3K^2. This preserves the common Gaussian
leading term and the correct residual orders. It is more useful for payoff
propagation than a prize-independent derivative bound.

### 31.4 A uniform local prize-analyticity neighborhood of the mature solution

There is also a stable linearization at EVERY real mature branch in the
same range. In the product Gaussian Banach norm, the derivative of the
coupled mild map is bounded componentwise by

\[
\frac{K C_G}{2}
\begin{pmatrix}a_*&h\\h&3a_*\end{pmatrix},\qquad C_G<10/3.
\]

Its maximum row sum at K=.35 is below .818417. Thus the linearized
resolvent is a convergent Neumann series; this is not an unproved
nondegeneracy assumption. More explicitly, allow a COMPLEX effective
prize k with |k-K|<.001, holding the future diffusion path and deadline
fixed. In the componentwise radius-.02 ball about the real (H,A),
the nonlinear mild map has Lipschitz bound

\[
\frac{(K+.001)(10/3)}2(h+3a_*+.08)<.867555<1.
\]

Its displacement at the center is bounded by
.001(10/3) max{ha_*/2,(3a_*^2+h^2)/4}, which is smaller than
.02 times one minus this Lipschitz bound. The map preserves the ball and
contracts. Iteration yields a unique holomorphic policy continuation in
that complex prize disc, uniformly across the mature subgames under study.
These complex policies are only an analytic proof device. The result does
not by itself describe a complex strategic-state process or its output.

### 31.5 Reverse audit and missing input

verify_mature_first_derivative.py checks the transcendental-series bounds,
both first-crossing systems, the leading Gaussian derivative constant's
elementary signs, and the complex-prize contraction margins. The underlying
continuous argument starts with the actual mild equations; every source
bound is applied after the relevant derivative is taken.

This completes a genuine first-derivative part of the mature-to-early handoff,
with O(K^2) and O(K) residual orders. It does not yet complete the handoff:
the early system has small diffusion and nonhyperbolic gradient coupling,
so bounded FIRST derivatives alone do not justify analytic continuation
across that whole layer. Higher spatial derivatives, or a different direct
output comparison that avoids them, remain necessary. The prize-analyticity
disc in Section 31.4 is not a spatial analytic strip and cannot replace it.
Accordingly the requested full-disclosure joint-optimum point remains
unproved; no strengthened Proposition or completion claim is made.

## 32. Actual mature spatial analyticity via a moving complex shift

### 32.1 Problem and proof structure

Section 31 proves first derivatives and prize analyticity, neither of which
is a spatial analytic strip. This section obtains such a strip from the
original mature-game mild equations. A fixed imaginary translation of the
terminal Gaussian would give exp(h^2/(2S_T)), destroying uniformity when
S_T is small. Instead let the translation grow with accumulated diffusion
and start it at zero at the terminal date. A sharper Gaussian bilinear
constant supplies the needed contraction margin.

There are three separate steps: sharpen the ORIGINAL heat operator; solve
its moving-shift coupled equations holomorphically; identify that solution
with the actual spatial continuation. The last step is not inferred from
mere boundedness of the auxiliary complex system. The early-layer payoff
comparison is still not proved by this section.

### 32.2 A sharper real Gaussian bilinear constant

For a mature subgame let a(t)>=a_min>0 and retain E_Q=Q^{-1/2}exp(-y^2/(4Q)).
The Gaussian-product kernel calculation in ledger Section 9 gives the exact
absolute-normal-moment upper bound for the bilinear operator. With
u=|y|/sqrt(Q_t), r=Q_s/Q_t it is

\[
a_{\min}\|T(f,g)\|_G/(\|f\|_G\|g\|_G)
\le\sup_{u\ge0}e^{-u^2/4}
\int_0^1\frac{\mathbb E|N(u,r/(1-r))|}{\sqrt r}\,dr.
\tag{78}
\]

Here N has the displayed mean and variance. This bound keeps the same u
throughout the time integral; taking a separate supremum for each kernel
loses the improvement below. Extending the actual lower integration limit
to zero and using -Q_s'=a(s)>=a_min are legitimate because the integrand
is nonnegative.

For a normal variable with standard deviation v>0, the triangle inequality
and the second derivative of its absolute first moment give respectively

\[
\mathbb E|N(u,v^2)|\le .8v+u,
\qquad \mathbb E|N(u,v^2)|\le .8v+.4u^2/v.
\]

The rational constants bound sqrt(2/pi) and 1/sqrt(2pi); the second
inequality integrates the bound sqrt(2/pi)/v on the moment's second
u-derivative from zero. Thus both hold simultaneously. Split their minimum
at r=4u^2/(25+4u^2). The integral in (78) is at most

\[
\frac45\{2+u^2\operatorname{asinh}(5/(2u))\}.
\]

For u=0 take its continuous limit. The cancellation in this integral is
explicit: int_{r0}^1 sqrt(1-r)/r dr=2 atanh(sqrt(1-r0))-2sqrt(1-r0),
and the latter term cancels 2u sqrt(r0) from the other part.

A finite rational interval cover of .1<=u<=4 proves that the last
display, multiplied by exp(-u^2/4), is below 7/3. For 0<=u<=.1 use
asinh(z)<=z, giving an upper bound 1.8. For u>=4 the earlier triangle
bound gives (2u+1.6)exp(-u^2/4), decreasing there and below 7/3 at 4.
verify_sharp_gaussian_bilinear.py reproduces the 58 interior intervals;
logarithms use range reduction and rational atanh tails, square roots use
rational upper bounds, and exponentials in denominators use positive
Taylor lower sums. Consequently

\[
\|T(f,g)\|_G<\frac{7}{3a_{\min}}\|f\|_G\|g\|_G.
\tag{79}
\]

This is an operator bound for arbitrary real or complex source functions
with the stated Gaussian norm, not a numerical equilibrium calculation.

### 32.3 A moving shift avoids terminal amplification

Put u_t=Q_t-S_T, and for a complex scalar zeta replace the heat propagation
P_{Q_t-Q_s} by its kernel evaluated at the displacement
zeta(sqrt(u_t)-sqrt(u_s)). Define T_zeta with this shifted derivative kernel.
The forcing becomes 2G_{Q_t}(y+zeta sqrt(u_t)). The coupled fixed-point system is

\[
H_\zeta=2G_{Q_t}(y+\zeta\sqrt{u_t})+
\frac\alpha2 T_\zeta(A_\zeta,H_\zeta),\qquad
A_\zeta=\frac\alpha4
\{3T_\zeta(A_\zeta,A_\zeta)-T_\zeta(H_\zeta,H_\zeta)\}.
\tag{80}
\]

At the terminal date u_T=0, so the initial data are UNCHANGED. This is a
proof device, not a modified information policy.

For |Im zeta|<=R and |Re zeta|<=d, the identity
sqrt(u_t)-sqrt(u_s)=(Q_t-Q_s)/(sqrt(u_t)+sqrt(u_s)) controls the shifted
kernel. In particular its imaginary amplification is at most exp(R^2/2).
The additional imaginary derivative term has time integral at most
2R/a_min: int ds/[(sqrt(u_t)+sqrt(u_s))sqrt(Q_s)]<=2/a_min.
The additional real-mean derivative term is at most 2d/a_min.

A real shift also perturbs the GAUSSIAN WEIGHT; one must not silently reuse
the sharp constant (79) without accounting for it. Its factor is bounded
by exp(d u), where u=|y|/sqrt(Q_t). Use
exp(d u)-1<=d u exp(d u), d u<=u^2/8+2d^2, and the old triangle moment
bound 2u+1.6. Then

\[
\sup_{u\ge0}u(2u+1.6)e^{-u^2/4+d u}<8e^{2d^2},
\quad\sup_{u\ge0}e^{-u^2/4+d u}=e^{d^2}.
\]

For the first sign, sup u^2 exp(-u^2/8)=8/exp(1)<3 and
sup u exp(-u^2/8)=2/sqrt(exp(1))<5/4. The shifted operator therefore obeys

\[
\|T_\zeta(f,g)\|_G\le\frac{C_{R,d}}{a_{\min}}\|f\|_G\|g\|_G,
\quad C_{R,d}=e^{R^2/2}\{7/3+8d e^{2d^2}+2(R+d)e^{d^2}\}.
\tag{81}
\]

The Gaussian forcing has norm at most
(3991/5000)exp((R^2+d^2)/2). Every estimate remains finite as S_T decreases;
none contains exp(R^2 Q_delta/(2S_T)).

### 32.4 A holomorphic contraction, and its identification with actual gradients

Take R=1/4,d=1/10000,K=alpha/a_min<=7/20. Equation (80) preserves the
rectangle ||H_zeta||_G<=1.03,||A_zeta||_G<=.39. At the two boundary caps
its right sides are below 1.029109 and .388217. In the weighted metric
max{||Delta H||_G,||Delta A||_G/(29/20)}, its derivative has row bounds

\[
\frac{K C_{R,d}}2[.39+1.03(29/20)],\qquad
\frac{K C_{R,d}}2[1.03/(29/20)+3(.39)],
\]

both below .963888. Thus Banach iteration gives a unique holomorphic
fixed point in the open zeta rectangle. The operator and forcing are
holomorphic there by dominated Gaussian integration; near s=t the identity
for the square-root difference removes the apparent shifted-kernel singularity.
At zeta=0 the actual real mature gradients lie within this rectangle by
Section 23, hence are this fixed point.

The spatial identification requires an additional argument. It is NOT
enough to assert that a translated real function belongs to the same
Gaussian Banach space: translating a coarse Gaussian envelope can violate
that norm. Instead use derivatives at zeta=0. Each finite spatial derivative
has a finite Q-weighted Gaussian norm on the mature subgame. To verify this
without assuming an analytic strip, induct on derivative order n using the
near/far split of Section 31. For general n the near operator constant is
bounded by

\[
\beta^{-n/2}\{2\sqrt{2/\exp(1)}(1-\sqrt\beta)
+2\sqrt{2/\pi}\sqrt{1-\beta}\}/a_{\min},
\]

and tends to zero as beta increases to one for fixed n. Choose beta so its
linear highest-derivative terms can be absorbed. All lower product
derivatives are finite by induction. On the far part, Q_t-Q_s>=(1-beta)Q_t;
differentiate the heat kernel n+1 times and use its Gaussian Cauchy bound.
Its remaining time integral is at most 2/a_min times a finite n,beta
constant. Gaussian terminal derivatives start the induction. This proves
finite weighted norms of each order, without an explicit uniform n-bound
or an analytic assumption.

Now differentiate (80) n times at zeta=0. The chain-rule candidate is
u_t^(n/2) partial_y^n(H,A). It belongs to the original Gaussian Banach space
by those weighted derivative bounds. The differentiated mild equations
verify the candidate, with the already identified lower-order candidates;
in every source convolution the coefficients combine by the identity

\[
\sum_{j=0}^n\binom nj
(\sqrt{u_t}-\sqrt{u_s})^j u_s^{(n-j)/2}=u_t^{n/2}.
\]

the remaining highest derivative has the same invertible contraction
linearization. Hence it equals partial_zeta^n(H_zeta,A_zeta) at zero for
every n. Cauchy estimates in a small zeta disc, uniformly for real y, then
bound the actual spatial Taylor remainders and identify the family with
the actual spatial analytic extension near each real point.

For fixed t,y, (80) is holomorphic on the thin horizontal zeta rectangle.
After multiplying by sqrt(u_t), this is a vertical complex-state rectangle
about y, of height R sqrt(u_t). Rectangles based at neighboring real y
agree on their real overlap by the preceding local identification, and
therefore agree throughout their connected complex overlap by the identity
theorem. They glue to the entire actual spatial strip

\[
|\operatorname{Im}z|<\tfrac14\sqrt{Q_t-S_T},\qquad
|H(z,t)|\le1.03E_{Q_t}(\operatorname{Re}z),\quad
|A(z,t)|\le.39E_{Q_t}(\operatorname{Re}z).
\tag{82}
\]

All derivative orders are controlled by Cauchy on this strip. The narrow
real-zeta width is not a limitation on strip height after this gluing step;
it must not instead be treated as the usable spatial radius.

### 32.5 The actual mature strip has width greater than sqrt(Q_delta)/sqrt(32)

The diffusion fraction .39 in Section 30 can also be sharpened at the SAME
fixed reference b=251. In the remaining range ell>21, a competitor has
T>1.9 delta. Let S_stat=L/sqrt(ell), A0=atanh(sqrt(r0)) and
b0=atanh(sqrt(ell)/L), only in this calculation. Then

\[
S_\delta=\sqrt{r_0}S_{\rm stat},\quad
S_T=S_{\rm stat}\tanh(A_0+v),\quad
Q_\delta=S_{\rm stat}(\sqrt{r_0}+v),\quad
v=\sigma^2(T-\delta)/S_{\rm stat}>.9(A_0-b_0).
\]

The ratio tanh(A0+v)/(sqrt(r0)+v) decreases for v>=0. Indeed
g(v)=tanh(A0+v)-(sqrt(r0)+v)sech^2(A0+v) has g(0)=r0 sqrt(r0)>0
and positive derivative, so the ratio's derivative is negative.

For ell<=L, L>600^2 gives b0<1/599. Rational atanh-series bounds give
5/4<A0<63/50. At the minimal v the ratio's numerator is below
tanh(1.9(63/50))<123/125, while its denominator exceeds
106/125+.9(5/4-1/599)>2(123/125). For larger v monotonicity preserves
the strict bound 1/2. For ell>=L use instead T>T0/8 and
T-delta>(9/19)T: v>(9/152)sqrt(ell)>(9/152)600>2 and S_T<S_stat,
again giving S_T/Q_delta<1/2. Thus in both cases

\[
Q_\delta-S_T>Q_\delta/2.
\tag{83}
\]

Equations (82)--(83) provide a genuine mature-gradient spatial analytic
strip of width greater than sqrt(Q_delta)/sqrt(32), uniform over the
remaining competitive family. verify_mature_spatial_strip.py checks the
coupled caps, weighted contraction, transcendental tails and both variance
cases. These are not assumptions on the designer or on a utility evaluated
at an endogenous output.

### 32.6 Completion audit and next missing comparison

This section supplies the previously missing all-order mature spatial
derivative INPUT. It does not complete the EARLY propagation or its payoff
screen. The strip width in (82) must still be compared with the radius
consumed by the actual coupled early equations. A crude Cauchy derivative
sum followed by the scalar majorant of Section 30 can lose too much radius;
its success cannot be presumed from .161<1/sqrt(32), since its loss also
depends on the gradient amplitudes and the coupled flux coefficients.

The next proof must use (82) and the exact leading Gaussian/residual
decomposition to control that consumption and the designer's actual output,
or obtain an alternative direct early-payoff comparison. Until then a gap
remains between the low- and high-precision payoff certificates. No single
positive-prize full-disclosure joint optimum has yet been established.
The goal stays active and the manuscript is not strengthened.

## 33. A double-maturity buffer and the exact nonhyperbolic principal structure

### 33.1 Audit: a later handoff is not automatically a better handoff

Section 32 supplies an actual mature spatial strip, but an absolute
derivative-sum majorant can consume too much of it. Simply choosing a later
handoff is not a resolution: the diffusion coefficient is then larger,
but the remaining future time can be shorter and its accumulated smoothing
fraction smaller. This section first strengthens the actual short-deadline
exclusion, then derives the original system's exact characteristic structure.
The latter identifies where the previous scalar radius bound is wasteful;
it is not by itself a nonlinear propagation theorem.

### 33.2 Keeping one Gaussian Hermite cancellation extends the analytic construction

For the initial unit Gaussian, the derivative-sum estimate (67) threw away
the cancellation in G_1''=(y^2-1)G_1. Keeping this single coefficient gives
for y>=0, rho>=0,

\[
\sum_{j\ge0}\frac{|G_1^{(j)}(y)|}{j!}\rho^j
\le G_1(y)\{e^{y\rho+\rho^2/2}-\rho^2\min(y^2,1)\}.
\tag{84}
\]

Indeed the uncorrected absolute-Hermite coefficient at order two is
(y^2+1)/2, the actual one is |y^2-1|/2, and their difference is min(y^2,1).
All higher coefficients retain their previous valid upper bounds. Reflection
handles y<0. The correction is nonnegative; no cancellation between separate
players or between strategic and Gaussian state laws is being assumed.

At rho=.8, a finite rational interval cover of 0<=y<=4 proves that the
right side of (84) is below .68. For y>=4 discard the correction and use
the decreasing tail exp(rho^2-(y-rho)^2/2). The script
verify_gaussian_jet_cancellation.py reproduces all 15 intervals, using
Gaussian prefactor <=.4 and rational exponential upper/lower tails.

Apply the ACTUAL derivative-norm inequality (66), not a new PDE model,
with initial radius .8, cap M=.68 and eta=7/4. For total exposure e<=.67,
the final radius .8-eta e M is strictly positive. Thus the selected analytic
Nash branch, including the stated no-signal selection, is constructed
through alpha T/S_T<=.67. This improves the sufficient exposure range
without changing the equilibrium or using a formal truncated expansion.

### 33.3 A competitive deadline must exceed twice the original maturity time

The low-precision payoff certificate already covers ell<=21. In the
remaining range ell>21, keep delta defined by a(delta)=.72 sigma^2.
The verified atanh bounds in Section 32 give
delta/T0<1.26/sqrt(21)<.275. If T<=2delta, then x<.55, and the Riccati
formula gives y<=2atanh(sqrt(.72))<2.52 and Q0/S_T<2.56. The last bound
follows from 2.52 coth(2.52)<2.56, equivalently exp(5.04)>127.

Every competitor also satisfies x>1/8. Its total exposure therefore obeys

\[
\alpha T/S_T<2.56(h-x^2/4)<.67,
\qquad h=63/250.
\]

The new analytic construction applies to the WHOLE contest in this case.
For exposures up to .61 the sharper adaptive Gaussian cap from Section 30
can still be used; above .61 use the newly certified cap .68. On each
rational interval [l,v] covering 1/8<=x<=.55 choose the appropriate cap
M_l at exposure 2.56(h-l^2/4). The exact output identity gives

\[
B/b\le\sqrt{a+x}\{1+M_l(251/100)(8/5)(h-l^2/4)\}<.99.
\tag{85}
\]

Here sqrt(2.56)=8/5. verify_double_maturity_buffer.py supplies six intervals;
the largest output upper bound is below .988566. This contradicts the
necessary competitive output B/b>.99. Hence every remaining competitor
has T>2delta. The bound comes from actual output and a convergent analytic
norm, not a fixed-deadline assumption or a comparison of leading-order outputs.

Since S_delta>(2/3)sigma^2 delta, this sharpens the early exposure to

\[
\alpha\delta/Q_\delta<(3/5)w<.1512.
\tag{86}
\]

The actual mature spatial strip from Section 32 is still available; its
variance fraction bound remains valid because T>2delta implies T>1.9delta.

### 33.4 Exact diagonalization of the real principal part

On any smooth positive-gradient part of the selected solution, write the
physical backward gradient system in nonconservative form:

\[
\partial_s\begin{pmatrix}H\\A\end{pmatrix}
=\frac{a(s)}2\partial_{yy}\begin{pmatrix}H\\A\end{pmatrix}
+\alpha J(H,A)\partial_y\begin{pmatrix}H\\A\end{pmatrix},\qquad
J(H,A)=\begin{pmatrix}A/2&H/2\\-H/2&3A/2\end{pmatrix}.
\]

Since p,n>0 on real states, H^2-A^2=4pn>0.
Put C=sqrt(H^2-A^2), Z=C+iA and W=C-iA only for this transformation.
The principal matrix on real states has complex eigenvalues:

\[
\lambda_+=A-iC/2,\quad\lambda_-=A+iC/2,
\qquad \nabla Z=(H/C,-A/C+i),\quad\nabla W=(H/C,-A/C-i).
\]

These gradients are left eigenvectors. The first component of
2C(nabla Z J)=2C lambda_+ nabla Z is 2AH-iCH on both sides; the second
reduces to H^2-3A^2+3iAC versus C^2-2A^2+3iAC, equal because
C^2=H^2-A^2. Conjugation gives the W identity. Thus these are actual
Riemann variables for the first-order part, not an analogy with scalar HJB.

The diffusion cannot be omitted after this nonlinear change of variables.
Its exact Hessian contribution is

\[
C_{HH}=-A^2/C^3,\quad C_{HA}=HA/C^3,\quad C_{AA}=-H^2/C^3,
\quad \mathcal R=\frac{a(s)}{2C^3}(A H_y-H A_y)^2.
\]

Consequently the FULL transformed equations are

\[
Z_s=\tfrac{a(s)}2Z_{yy}-\tfrac{i\alpha}4(3Z-W)Z_y+\mathcal R,
\qquad
W_s=\tfrac{a(s)}2W_{yy}+\tfrac{i\alpha}4(3W-Z)W_y+\mathcal R.
\tag{87}
\]

For example C_s-a C_yy/2 subtracts the Hessian quadratic form, whose sign
is negative, yielding the positive real source displayed above. Equivalently,

\[
\mathcal R=-\frac{a(s)}4
\frac{(Z W_y-W Z_y)^2}{ZW(Z+W)}.
\]

This follows from AH_y-HA_y=C(ZW_y-WZ_y)/(2iH). For real states the
numerator in the latter representation is nonpositive and the denominator
positive, so the two representations agree in sign. In complex states
these formulas require a consistent nonzero square-root branch; the real
positivity statement alone does not establish such a global branch.

### 33.5 Why this structure is useful, and what it does not prove

At the original maturity time, the proved real ratio |A|/H<=K/2 and
H<=.92/sqrt(Q_delta) give

\[
|\lambda_\pm|\le\tfrac H2\sqrt{1+3K^2/4}<.49/\sqrt{Q_\delta}.
\]

Combining this FROZEN coefficient estimate with (86), its corresponding
imaginary travel over delta would be less than .075sqrt(Q_delta), which
is less than half the proved strip width sqrt(Q_delta)/sqrt(32).
The script checks the exact squared inequality. This is evidence that
the earlier scalar derivative-norm coefficient eta=7/4 loses relevant
principal cancellation. It is NOT a statement that the nonlinear
characteristics remain frozen or that their coefficients remain bounded
by these initial values.

A successful early propagation proof must control the evolving coefficients,
the diffusion source R in (87), and possible complex zeros of C, or work
directly with the original matrix system and its analytic energy without
introducing that square root. Ignoring R or replacing the whole system by
a scalar Burgers equation would be incorrect. A later handoff near a=.97
also does not remove these tasks: when T is close to 2delta it can leave
little future smoothing, despite its stronger instantaneous diffusion.

The new exposure construction and deadline exclusion are proved. The
diagonalization and diffusion source are exact identities supplying a
different propagation route, not an unproved assumption about u,D or the
designer's parameters. No early-payoff screen or unrestricted joint-optimum
point has yet been proved. The goal remains active and the manuscript is
not strengthened.

## 34. A time-dependent value barrier: the exact missing input

This section tests an alternative to the complex-characteristic route. It
does not claim the early layer or the requested joint optimum has closed.
Let s denote physical backward time, a(s) the actual information diffusion,
D=U_i-U_j, R=U_i+U_j, H=D_y and A=R_y. For an absolutely continuous k(s)>0,
define only for the proof

\[
f_k(z)=1+\frac2k\log\frac{\cosh(kz/2)}{\cosh(k/2)},
\qquad E=R-f_{k(s)}(D).
\]

The original value equations imply the exact identity

\[
E_s=\frac a2 E_{yy}
+\left(\frac{3\alpha}4E_y+\alpha f_k'(D)H\right)E_y
+\frac{ak-\alpha}4[1-f_k'(D)^2]H^2
-\partial_k f_k(D)k'.
\tag{88}
\]

Indeed A=E_y+f_k'(D)H, and expanding
3A^2-H^2-2f_k'(D)AH gives
3E_y^2+4f_k'(D)HE_y-[1-f_k'(D)^2]H^2. The diffusion chain rule contributes
+a f_k''(D)H^2/2, where f_k''=k[1-f_k'^2]/2. These signs matter.

For |z|<=1, symmetry and integration of f_k' give

\[
-\partial_k f_k(z)
=\frac12\int_{|z|}^1 v\operatorname{sech}^2(kv/2)\,dv,
\quad
\frac{1-z^2}4\operatorname{sech}^2(k/2)
\le-\partial_k f_k(z)\le\frac{1-z^2}4.
\tag{89}
\]

Thus, CONDITIONAL on an actual-gradient bound
H^2<=M(s)(1-D^2), the scalar differential inequality

\[
k'\ge (\alpha-ak)_+M(s)\cosh^2(k/2)
\tag{90}
\]

makes the last two terms in (88) nonnegative. Provided k stays finite, on
a classical branch with suitable bounded comparison data, E initially
nonnegative then remains nonnegative. The bound |D|<=1 follows from
0<=U_i,U_j<=1 for a verified Nash equilibrium: zero effort guarantees a
nonnegative payoff, and the prize probability minus a nonnegative cost is
at most one. This statement uses scalar comparison only AFTER the
source signs are established; it is not scalar comparison for the coupled
gradient system itself. At |D|=1 the displayed ratio must be handled by its
inequality form, rather than division by zero.

This explains why merely replacing alpha/a_min by a time-varying k does not
solve the low-diffusion problem. The term -partial_k f_k k' cannot be dropped,
and (90) needs a bound for the ACTUAL quantity H^2/(1-D^2). The existing
Gaussian bound for H alone does not control this denominator near the
payoff tails. These are proof inputs to establish, not additional assumptions
on the designer's problem. No such uniform early-layer gradient input or
resulting finite-k bound has yet been proved.

A diagnostic at the unchanged external reference checked ell=22,40,100,400
and w=.08,.16,.24 at x=1. The computed B/b values range from 1.080169690 to
1.243120044, with no materially negative computed effort. The reproducible
script is audit_fixed_reference_intermediate_precision.py. This is only a
counterexample search using finite differences: it neither certifies a true
output bound nor searches all prizes and deadlines. It provides no new
joint-optimum existence claim. The goal remains active, and Proposition 1
has not been strengthened on the basis of this diagnostic.

## 35. Time averaging strengthens the actual deadline exclusion

Section 33 bounded every policy by the largest analytic cap over the whole
contest. Keeping its backward-time dependence improves this without a new
equilibrium assumption. Let m(z) be any of the proved caps in Section 30 for
z<=.61, and use m(z)=.68 for .61<z<=.67 as in Section 33. The original
analytic-jet construction, applied at backward elapsed time vT, gives

\[
\max(p,n)(y,T-vT)\le m(ev)/\sqrt{S_T},
\quad e=\alpha T/S_T.
\]

Both weights in the exact output identity are nonnegative and sum to one.
Consequently, for an existing verified branch covered by that construction,

\[
B-b(T)\le\frac{\alpha T}{\sqrt{S_T}}
\int_0^1m(ev)\,dv.
\tag{91}
\]

This is a bound for the actual state law, not an evaluation along an
uncontrolled Gaussian state. Subgame construction identifies the branch
by the same convergent analytic equations used for the whole contest.

Suppose a remaining competitive design has T<=(53/25)delta. Write x=T/T0
as before. Since ell>21 and atanh(sqrt(.72))<1.26, x<.584. The Riccati
bound from Section 33 gives Q0/S_T<2.7: indeed z<=2.12(1.26)<2.672 and
z coth(z)<2.7 at this upper endpoint. Thus

\[
e\le2.7(h-x^2/4)<.67,
\quad
\frac B b\le\sqrt{.001+x}\left[
1+2.51\sqrt{2.7}(h-x^2/4)
\int_0^1m(2.7(h-x^2/4)v)\,dv\right].
\tag{92}
\]

The last integral need not use a monotone choice of m. For each of sixteen
time rectangles choose the cap valid at that rectangle's upper exposure;
the same analytic cap is valid at every smaller exposure. This bounds the
integral by the average of those sixteen caps. The rational script
verify_integrated_maturity_buffer.py bounds (92) on seven x intervals
covering 1/8<=x<=.584. Its largest upper bound is .987933<.99, contradicting
the already necessary competitive output B/b>.99. Hence

\[
T>(53/25)\delta,\qquad
\frac{\alpha\delta}{Q_\delta}
<\frac{h}{53/25-1/3}<.142.
\tag{93}
\]

The last estimate uses S_delta>(2/3)sigma^2 delta as before. This is a
proved stronger localization, not a proof of the remaining early payoff
bound. It does not introduce a deadline constraint in the design problem.

## 36. Exact holomorphic strip energy avoids a square-root change of variables

The principal cancellation of Section 33 can be retained without dividing
by C=sqrt(H^2-A^2). On any existing analytic interval, put U=(H,A) and
F(U)=(AH/2,(3A^2-H^2)/4), so U_s=a U_yy/2+alpha partial_y F(U).
For z=y+i eta, holomorphicity gives U_eta=i U_y. With
K=conjugate(U) dot F(U), the Cauchy--Riemann identities imply

\[
(\partial_y-i\partial_\eta)K
=2\overline U\mathbin{\cdot}\partial_yF(U).
\]

The derivative of conjugate(U) cancels here. Therefore the EXACT energy
identity, including diffusion, is

\[
\partial_s|U|^2
=\frac a2\partial_{yy}|U|^2-a|U_y|^2
+\alpha\{\partial_y\Re K+\partial_\eta\Im K\}.
\tag{94}
\]

Moreover
K=A|H|^2/2+3A|A|^2/4-conjugate(A)H^2/4, whence
|Im K|<=3|A||U|^2/4. No derivative norm occurs in this bound.

For V=U_y, differentiation of the ORIGINAL equation gives
V_s=a V_yy/2+alpha partial_y(J(U)V), where J is Section 33's matrix.
The same identity holds with K_V=conjugate(V) dot J(U)V. Its imaginary
part is the Hermitian form with matrix

\[
\frac{J-J^*}{2i}=
\begin{pmatrix}
\Im A/2&-i\Re H/2\\i\Re H/2&3\Im A/2
\end{pmatrix}.
\]

Its eigenvalues are Im A plus or minus
sqrt((Im A)^2+(Re H)^2)/2. Hence

\[
|\Im K_V|\le\left[|\Im A|+
\tfrac12\sqrt{(\Im A)^2+(\Re H)^2}\right]|V|^2.
\tag{95}
\]

For integrable fields with vanishing horizontal flux, integrate (94) on
the moving strip -r(s)<eta<r(s). If r'<=-alpha beta and the two boundary
fluxes obey |Im K|<=beta |U|^2, then its strip energy is nonincreasing;
the remaining volume term is -a int|U_y|^2<=0. The same argument applies
to V using (95). This calculation has no minimum-diffusion denominator and
no diffusion source omitted by a nonlinear transformation.

At the PROVED mature complex caps |H|<=1.03/sqrt(Q_delta),
|A|<=.39/sqrt(Q_delta), the coefficient in (95) is less than
.941/sqrt(Q_delta). The new exposure .142 would consume less than
.134 sqrt(Q_delta), within the mature strip width sqrt(Q_delta)/sqrt(32).
This comparison is only an INITIAL-cap feasibility check. It does not
prove that those pointwise caps persist: an integral strip-energy bound
alone is not the same as a pointwise bound on its moving boundary. The
remaining task is a quantified bootstrap for that conversion, or a
different bound sufficient for the actual output screen. The exact
identities and the strengthened deadline exclusion are established; the
unrestricted positive-prize full-disclosure joint-optimum point is not.

## 37. A closed Gaussian moment identity simplifies and improves the mature strip

Section 32's real heat-operator bound admits a simpler estimate than its
58-interval scalar cover. Let
I(u)=int_0^1 E|N(u,r/(1-r))| dr/sqrt(r), u>=0, the integral in (78).
Set v=sqrt(r/(1-r)) and c=sqrt(2/pi). The normal absolute moment gives

\[
I(u)=2\int_0^\infty\frac{
cv e^{-u^2/(2v^2)}+u\operatorname{erf}(u/(\sqrt2v))
}{(1+v^2)^{3/2}}\,dv.
\]

For u>0, integrate the second term by parts using the primitive
v/sqrt(1+v^2). The boundary terms vanish. Substitute z=1/v in both terms:

\[
I(u)=2c\left[\int_0^\infty
\frac{e^{-u^2z^2/2}}{(1+z^2)^{3/2}}\,dz
+u^2\int_0^\infty\frac{e^{-u^2z^2/2}}{\sqrt{1+z^2}}\,dz\right].
\]

An integration by parts of the first integral, using
(z/sqrt(1+z^2))'=(1+z^2)^(-3/2), followed by t=uz, now yields

\[
I(u)=2c\int_0^\infty\sqrt{u^2+t^2}e^{-t^2/2}\,dt
=2\mathbb E\sqrt{u^2+Z^2},\qquad Z\sim N(0,1).
\tag{96}
\]

At u=0 both sides equal 2sqrt(2/pi), independently by direct integration,
so no division by u is used there. All integrals are finite; the displayed
boundary limits and integration by parts are justified by Gaussian decay.
Jensen's inequality and maximization over s=u^2 give

\[
e^{-u^2/4}I(u)\le2e^{-s/4}\sqrt{1+s}
\le2\sqrt2e^{-1/4}<2.21.
\tag{97}
\]

The maximum is at s=1 since the logarithmic derivative is
-1/4+1/[2(1+s)]. Thus the entire real Gaussian operator cover of Section
32 can be replaced by this short calculation. This improves its constant
7/3 to 2.21, for arbitrary real or complex sources in the same norm.

Keep the same moving-shift construction and its actual-spatial identification,
but set R=.28, d=.0001, ||H||_G<=1.04, ||A||_G<=.40. With K<=.35, insert
2.21 in place of 7/3 in (81). The two map bounds are below 1.039905 and
.393767, and the weighted derivative row bounds are below .966888.
Hence the original mature fixed point has the wider ACTUAL spatial strip
|Im z|<.28sqrt(Q_t-S_T). The identification proof is unchanged; there is
no claim that auxiliary prize analyticity is spatial analyticity.

At delta the previously proved variance fraction exceeds 1/2, so the width
is greater than (.28/sqrt2)sqrt(Q_delta). For these mature complex caps,
the energy coefficient in (95) is less than .958/sqrt(Q_delta). Combined
with (93), the INITIAL-cap radius calculation leaves more than
.061sqrt(Q_delta). verify_closed_gaussian_constant.py checks the rational
exponential sign, both invariant inequalities and contraction, and this
initial margin. The latter remains conditional on propagation of pointwise
caps; it does not supply that propagation. The proof of the requested
positive-prize full-disclosure joint optimum is still incomplete.

## 38. Gaussian centering gives a wider actual mature strip

The rectangle estimate for A in Sections 32 and 37 discards the explicit
first source -alpha T(H0,H0)/4, where H0=2G_Q. Keep it. In the moving-shift
system let A1=-alpha T_zeta(H0,H0)/4 and E_H=H-H0. Exact Gaussian convolution
gives, with z=y+zeta sqrt(Q_t-S_T),

\[
A_1(z,t)=-\frac{\alpha}{2\sqrt\pi}
\int_t^T Q_s^{-1/2}G'_{Q_t-Q_s/2}(z)\,ds.
\tag{98}
\]

The shift increments compose to the single argument z, so (98) is also
the actual first source of the auxiliary complex system, not a real-only
formula substituted into a complex equation.

Here is a direct bound for this term. Normalize Q_t=1, put v=1-u^2/2,
and first take a pure imaginary displacement of magnitude at most R<1/2.
Since -Q_s'>=a_min, differentiating the Gaussian in (98), maximizing its
Gaussian-weighted absolute value at each integration point, and substituting
Q_s/Q_t=u^2 give

\[
\|A_1\|_G\le\frac K{\pi\sqrt{\mathrm e}}
\int_0^1\frac{\exp\{R^2(1/v-1/4)\}}{v\sqrt{2-v}}\,du,
\qquad K=\alpha/a_{\min}.
\tag{99}
\]

For verification, with beta=(2-v)/(4v), the weighted derivative factor is
sqrt(y^2+R^2) exp(-beta y^2+R^2/(2v)). Its maximum is
exp{R^2(beta+1/(2v))}/sqrt(2 beta e). The maximizer has
y^2=1/(2 beta)-R^2>=0 because beta<=3/4 and R<1/2.

Convexity of the exponential in 1/v supplies its chord between 1 and 2.
The convex function (2-v)^(-1/2) is bounded by a+bv, with
a=.633,b=.367: the line has value one at v=1 and value .8165>sqrt(2/3)
at v=1/2. Write J=int_0^1du/(1-u^2/2)<1.25; its positive series is
sum_{j>=0}2^(-j)/(2j+1). The second integral of v^(-2) is 1+J/2.
Integrating the two nonnegative chord weights therefore gives

\[
\|A_1\|_G\le .194K\left[
\frac{6633}{8000}e^{3R^2/4}
+\frac{2633}{8000}e^{7R^2/4}\right].
\tag{100}
\]

For example the first coefficient bounds
int (2-1/v)/(v sqrt(2-v))du by
(3a/2-b)J+2b-a; the second bounds
int (1/v-1)/(v sqrt(2-v))du by a-b+(b-a/2)J. Their J coefficients are
positive. The prefactor uses 1/(pi sqrt(e))<.194.

A real displacement of magnitude at most d=.0001 adds a factor less than
1.0001. Indeed, after translating x=y+h, the additional exponent is
-hx/2+h^2/4. Bound |hx|/2 by d beta x^2+h^2/(16d beta). Since beta>=1/4,
the resulting factor is at most exp{(d^2+d)/4}/sqrt(1-d)<1.0001.
Replacing beta by (1-d)beta only lowers the R-dependent exponent used
in (99). Thus (100) times 1.0001 holds throughout the original complex
rectangle, including its tiny real width.

Now set R=.45, K<=.35 and retain d=.0001. The equations centered at H0 are

\[
E_H=\frac\alpha2T_\zeta(A,H_0+E_H),\qquad
A=A_1+\frac\alpha4\{3T_\zeta(A,A)
-T_\zeta(E_H,2H_0+E_H)\}.
\tag{101}
\]

Use the same complex operator bound (81), with the improved 2.21 from
Section 37. The forcing has norm h0<.883251 and A1 has norm <.097393.
The region ||E_H||_G<=.15, ||A||_G<=.235 is invariant: its two map bounds
are below .146280 and .233890. In the weighted metric
max{||Delta E_H||_G,||Delta A||_G/(5/4)}, the row bounds are below .922691.
The real mature solution lies in this region at zeta=0 by its earlier
O(K^2) remainder estimate and |A|<=KH/2. Banach iteration and the spatial
identification of Section 32 therefore give the ACTUAL mature strip

\[
|\Im z|<.45\sqrt{Q_t-S_T},\quad
|H(z,t)|\le1.033251E_{Q_t}(\Re z),\quad
|A(z,t)|\le.235E_{Q_t}(\Re z).
\tag{102}
\]

Every rational sign is checked by verify_centered_mature_strip.py; no HJB
numerical integral is an input. At delta, (102) has width greater than
(.45/sqrt2)sqrt(Q_delta). The INITIAL-cap energy coefficient (95) is now
less than .767/sqrt(Q_delta). With the proved early exposure .142, the
corresponding initial radius margin exceeds .209sqrt(Q_delta).

This substantial margin is not yet persistence of the pointwise caps.
The holomorphic energy estimate still needs a justified conversion from
integral energy to evolving boundary values. In particular it would be
incorrect to replace those evolving coefficients by their initial caps
for the whole early layer. The enlarged mature strip is proved; the
requested full-disclosure joint-optimum point remains unproved.

## 39. Closing the early analytic continuation with two real jets

The wider strip from Section 38 allows a direct analytic-jet continuation.
It does not require converting the conditional strip energy into pointwise
boundary values. The decisive improvement is to keep the first two REAL
derivatives of the mature remainder rather than bounding every Taylor
coefficient by the complex-strip supremum.

### Real centered bounds and their first two derivatives

The centered real fixed point has the smaller invariant region
||H-2G_Q||_G<=.04, ||A||_G<=.11 when K<=.35. With C_G<2.21, the real
forcing norm <.8 and the leading-term bound from (100) at R=0, both
invariant inequalities are strict. This real fixed point is inside the
complex region of Section 38 at zeta=0, where uniqueness identifies it with
the actual mature solution. Hence H has norm <.84. The same construction
with remainder bound .04(K/.35)^2 and difference bound .11(K/.35)
preserves these orders throughout 0<=K<=.35.

Differentiate the ORIGINAL mild equations, not the function envelopes.
Retain the near/far split Q_s/Q_t=beta=.9 of Section 31. For one derivative
use c_N1<.61,c_F1<8.2. For two derivatives its general near calculation gives

\[
c_{N2}\le\beta^{-1}\left[
2\sqrt{2/\mathrm e}(1-\sqrt\beta)
+2\sqrt{2/\pi}\sqrt{1-\beta}\right]<.66.
\tag{103}
\]

Here the product's nth derivative has an additional factor Q_s^(-n/2);
on the near interval this contributes at most beta^(-n/2) to the
zeroth-derivative kernel bound. Product-rule terms are retained separately.
On the far interval put all THREE derivatives on the heat kernel. In the
Gaussian product the kernel displacement, at Q_t=1,Q_s=r, has conditional
mean (1-r)y and variance r(1-r). The third derivative satisfies
|P'''_v|<=P_v(|x|^3/v^3+3|x|/v^2). Expand the bound
E(|mean|+|centered normal|)^3 and use
E|Z|<.8,E|Z|^3<1.6. The Gaussian-weighted maxima of y^3,y^2,y are below
3.3,1.5,.86. Integration yields the explicit upper bound

\[
\begin{split}
c_{F2}\le{}&6.6\sqrt\beta
+7.2(1-\sqrt{1-\beta})
+2.58[2\operatorname{atanh}\sqrt\beta-2\sqrt\beta]\\
&+1.6[2/\sqrt{1-\beta}+2\sqrt{1-\beta}-4]
+5.16\operatorname{atanh}\sqrt\beta
+2.4[2/\sqrt{1-\beta}-2]<41.
\end{split}
\tag{104}
\]

The explicit Gaussian source (98) also has sharper REAL derivative bounds.
For z=y/sqrt(v), its first derivative uses
|z^2-1|exp(-(2-v)z^2/4)<=23/20, v in [1/2,1]. Its second derivative uses
|z^3-3z|exp(-(2-v)z^2/4)<2. For the latter it suffices to replace
(2-v)/4 by 1/4; the nonzero critical squared arguments are
(9 plus or minus sqrt(57))/2. Direct evaluation bounds both extrema below
two. Thus, integrating v^(-3/2) and v^(-2), respectively, gives

\[
\|(A_1)_y\|_{G_1}<.367K,\qquad
\|(A_1)_{yy}\|_{G_2}<.735K.
\tag{105}
\]

Indeed int_0^1(1-u^2/2)^(-3/2)du=sqrt(2), and the corresponding
second-power integral is 1+J/2<13/8. The Gaussian leading H0=2G_Q itself
has first and second weighted derivative norms below .7 and .92.

Let R_H=H-H0 and use the real bounds R_H=.04,A=.11,H=.84,H0=.8 only as
upper constants in the next inequalities. The first-derivative comparison is

\[
d_{R_H,1}\le\frac K2[8.2AH+.61\{A(.7+d_{R_H,1})+Hd_{A,1}\}],
\]

\[
d_{A,1}\le .367K+\frac K4[
8.2\{3A^2+R_H(H+H0)\}
+.61\{6Ad_{A,1}+2R_H(.7)+2Hd_{R_H,1}\}].
\]

At barriers d_RH,1=.17,d_A,1=.24,K=.35 the right sides are below
.164331,.228251. A first crossing is impossible. For the second derivatives,
insert those first-derivative bounds and (103)--(105) into the differentiated
product sources. At d_RH,2=d_A,2=1 the two right sides are

\[
\frac K2[41AH+.66\{A(.92+1)+2(.24)(.7+.17)+H\}]<.832617,
\]

\[
.735K+\frac K4[41\{3A^2+R_H(H+H0)\}
+.66\{6A+6(.24)^2+2R_H(.92)+4(.7)(.17)+2(.17)^2+2H\}]<.812988.
\]

Both are strictly below one. The scaled barriers are proportional to K^2
for R_H and K for A; after division by these orders each right side is
nondecreasing in K. Therefore the comparisons at .35 cover the entire
range, starting from zero remainders at the terminal date. This proves
the actual real bounds

\[
\|R_H\|_G\le.04,\quad\|A\|_G\le.11,\quad
\|(R_H)_y\|_{G_1}<.17,\quad\|A_y\|_{G_1}<.24,\quad
\|(R_H)_{yy}\|_{G_2}<1,\quad\|A_{yy}\|_{G_2}<1.
\tag{106}
\]

### A positive analytic radius across the whole early interval

At delta normalize Q_delta=1. By the variance fraction and (102), every
circle of radius r=.318 about a real state is inside the proved actual
strip: r<.45/sqrt(2). The remainder and difference have unweighted complex
caps .15 and .235 on those circles. Cauchy's inequality controls their
Taylor coefficients of order at least three; use (106) for the first three
coefficients instead. For eta=7/4 the initial analytic jet W=(L_H+eta L_A)/2
is bounded, uniformly in real states, by

\[
\Psi(\rho)=.4e^{\rho^2}
+\frac{.04+\eta(.11)}2
+\rho\frac{.17+\eta(.24)}2
+\frac{\rho^2}4(1+\eta)
+\frac{.15+\eta(.235)}2
\frac{(\rho/r)^3}{1-\rho/r}.
\tag{107}
\]

The Gaussian part uses its direct Hermite generating bound, not a derivative
of a Gaussian envelope. Since the early exposure satisfies
alpha delta/Q_delta < h/(53/25-1/3)<.1411, take rho=.173 and M=.7.
Exact rational estimates give

\[
\Psi(.173)<.699107<.7,\qquad
.173-\tfrac74(.1411)(.7)=.0001525>0.
\tag{108}
\]

In physical spatial radius R, the original differentiated gradient equations
give, as in Section 30, W_s<=a W_yy/2+eta alpha W W_R. On the physical radius
R(s)=sqrt(Q_delta)[.173-eta alpha M s/Q_delta], scalar comparison keeps
W<=M/sqrt(Q_delta) through the physical backward interval 0<=s<=delta.
The radius stays strictly positive by (108). Local analytic mild construction and this
a-priori bound continue the solution across that entire interval, since a
nonnegative heat semigroup preserves the analytic norm and the final
positive radius supplies a uniform Cauchy derivative margin. This does
not rely on a lower bound for the early diffusion or on the conditional
energy caps of Section 36.

This also gives a VERIFIED real equilibrium, not just a complex auxiliary
solution. Write p=(H+A)/2,n=(H-A)/2. The original equations imply

\[
p_s=\tfrac a2p_{yy}+\alpha(p-n)p_y-\alpha n_y p,
\qquad
n_s=\tfrac a2n_{yy}+\alpha(p-n)n_y+\alpha p_y n.
\tag{109}
\]

The actual mature initial p,n are positive. The analytic bound supplies
bounded drift and bounded zero-order coefficients, so scalar comparison
in (109) preserves nonnegativity; here every remaining design has finite
positive lambda, and no zero-precision limiting argument is needed.
Reconstruct normalized values by the original mild value equations with
their actual mature data. Their gradients are p,-n by uniqueness of the
continued gradient equations. Efforts alpha p,alpha n are bounded,
nonnegative and maximize each quadratic HJB term. The usual Ito verification
with admissible square-integrable deviations then establishes Nash best
responses. Finite horizon, bounded values and bounded equilibrium policies
supply the required integrability. Spatial tail limits follow by continuation
in the space of bounded analytic gradients vanishing at real infinity,
which is preserved by the heat operator, products and uniform mild limits.

verify_early_analytic_continuation.py checks every new rational kernel sign,
the real invariant bounds, both derivative crossings, and the strict radius
margin. THE EARLY CONTINUATION GAP IS CLOSED for the remaining competitive
designs at the unchanged external reference. The missing step is now their
ACTUAL global payoff comparison: continuation alone does not prove the
coefficient-2.4 output screen used in Section 30, nor superiority of full
disclosure. The requested joint-optimum point and paper strengthening
remain incomplete.

## 40. Actual state coupling and a sharper mature output bound

This section uses the actual equilibrium, not the uncontrolled state as a
substitute. Retain the external reference (55), h=.252 and the remaining
competitive designs. Set f=delta/T<25/53. By (93),

\[
e_\delta:=\alpha\delta/Q_\delta<(75/134)w<.56w,
\qquad Q_0/Q_\delta<1+.72(75/134)=94/67.
\tag{110}
\]

The variance inequality follows by integrating a(t)<=.72sigma^2 before
delta. In particular sqrt(Q0/Qdelta)<1.185. Coupling actual Y and the
uncontrolled innovation Z with the SAME Brownian motion gives
Y_t=Z_t+d_t, d_t=alpha int_0^t A(Y_s,s)ds. Section 39's analytic bound
implies |A|<=.8/sqrt(Q_delta) on this early interval. Thus
|d_delta|/sqrt(Q_delta)<=.8e_delta. No independence between d_delta and
Z_delta is assumed.

On the mature interval, the actual Markov propagation of a nonnegative
function obeys

\[
\mathcal K_{\delta,t}g\le\cosh^2(K/2)
P_{Q_\delta-Q_t}g,\qquad K=w/.72\le.35.
\tag{111}
\]

This is the original-path conditional bound, valid for every starting state.
For completeness its value barrier is R>=1-L with
L=2log cosh(K/2)/K. In the variance clock, the gauge
M=exp(kappa(R-1)), kappa=alpha/a, has potential
kappa_Q(R-1)-kappa^2(H^2-A^2)/4. Riccati monotonicity makes kappa monotone;
retaining its derivative and integrating its possible positive contribution
gives exp(KL)=cosh^2(K/2). This is the argument in ledger Section 118.3,
not a Gaussian-law assumption about Y.

The scaled real bounds of Section 39 give
|H-2G_Q|/2+|A|/2<=m(w)E_Q, with m(w)<=.22w+.32w^2.
The exact flow is H/2+A[2Phi(y/sqrt(Q))-1]/2, hence at most G_Q+m(w)E_Q.
Since P G_Q=G_Qdelta and P E_Q=2sqrt(pi)G_{Q_delta+Q}, (111) yields

\[
\mathbb E\{\text{flow at }t\}
\le F_K\left[G_{Q_0}(0)+\frac{\sqrt2m(w)}{\sqrt{Q_0}}
+\{.243+.862m(w)\}\frac{\sup|d_\delta|}{Q_\delta}\right].
\tag{112}
\]

Here sup|G'_v|<.243/v and sup|2sqrt(pi)G'_v|<.862/v; the two expectations
at Z_delta use its actual Gaussian variance Q0-Qdelta. The Lipschitz errors
are pointwise, so a path-dependent displacement is permitted.
Using G_Q0(0)>.398/sqrt(Q0), (110), and F_K<=1+w^2/2, the right side
divided by G_Q0(0) is bounded by

\[
(1+w^2/2)\left[1+3.556m(w)
+2.978\{.243+.862m(w)\}(.448w)\right]<1+1.8w.
\tag{113}
\]

After subtraction of one and division by w, all coefficients are
nonnegative, so checking w=.252 proves (113) throughout its range. The
exact certificate gives a coefficient below 1.653407. Thus the ACTUAL
mature output increment is at most alpha(T-delta)G_Q0(0)(1+1.8w).

## 41. Closing the remaining actual global output screen

### Weak prizes: preserve the residual orders

First take 0<w<=.12 and write q=w/.12. The complex centered construction
of Section 38 also works with K<=1/6, R=.8, remainder cap .075q^2 and
difference cap .13q. Formula (99) only needs R^2<2/3 for its positive
maximizing squared argument; this includes R=.8. The invariant inequalities
and contraction are checked explicitly by verify_remaining_global_payoff.py.
At delta this supplies the actual unweighted analytic circles of radius
r=.565sqrt(Qdelta), since .565<.8/sqrt2.

Normalize Qdelta=1. The initial residual coefficients of orders 0,1,2
are those of (106), multiplied by (10/21)^2 q^2 for H-2G and (10/21)q
for A. For orders at least three use the wider-circle Cauchy caps above.
For the Gaussian component, an appropriate COEFFICIENTWISE majorant is

\[
\Gamma(\rho)/2=.3991e^{\rho^2/2}
+\frac{50}{157}\frac{\rho}{1-\rho^2/3}-.075\rho.
\tag{114}
\]

All its Taylor coefficients are nonnegative. The even coefficients bound
the Gaussian Fourier moments. The odd moment ratios are at most rho^2/3;
the first derivative permits the displayed subtraction because its actual
supremum is below .243. This is not the pointwise even bound exp(rho^2)
differentiated as though it were a coefficientwise majorant.

Define positive-series residual and difference majorants e,a, initially
with the coefficients just described. With Gamma constant in backward
time, their exact positive comparison system is

\[
e_v=\tfrac12\{a(\Gamma_\rho+e_\rho)
+(\Gamma+e)a_\rho\},\qquad
a_v=\tfrac12\{3aa_\rho+(\Gamma+e)(\Gamma_\rho+e_\rho)\}.
\tag{115}
\]

The clock v=alpha s/Qdelta is at most .0672q. Heat propagation contracts
each derivative sup norm; after subtracting the propagated Gaussian, the
original mild equations are dominated coefficientwise by (115). Picard
iteration preserves nonnegative coefficients. On writing e=q^2 E,a=q A
and v=q z, the equations for E,A have only nonnegative powers q^2.
Therefore their coefficients for q<=1 are bounded by their q=1 values.

At q=1, S=(Gamma+e+eta a)/2, eta=7/4, satisfies coefficientwise
S_v<=eta S S_rho. Let W be the scalar positive Burgers majorant with the
same initial coefficients. Its implicit formula is
W(v,rho)=Psi(rho+eta v W(v,rho)). Direct rational estimates give
Psi(.056)<.475 and eta(.0672)(.475)<.056. Thus W(v,0)<=.475 through this
clock interval. Since both components and their rho derivatives are bounded
coefficientwise by 2W and 2W/eta, respectively, (115) gives

\[
e(v,0)\le e(0,0)+\frac4{\eta^2}[W(v,0)-W(0,0)]<.044,
\]

\[
a(v,0)\le a(0,0)+\frac2\eta(1+3/\eta^2)
[W(v,0)-W(0,0)]<.112.
\tag{116}
\]

These differentiated majorant bounds follow from coefficientwise order,
not differentiation of a pointwise function bound. Restoring q and units,
the ACTUAL early fields satisfy
|H-2G_Qt|<=.044q^2/sqrt(Qdelta), |A|<=.112q/sqrt(Qdelta).
Couple Y_t with its uncontrolled innovation for t<=delta. The common
Gaussian flow has expectation G_Q0(0); the Lipschitz displacement error
integrates with the factor int_0^delta t dt=delta^2/2. Adding (116)'s
residual gives

\[
\frac{\text{early output increment}}{\alpha\delta G_{Q_0}(0)}
\le1+2.978\left[\frac{7}{15}w+
\left\{\frac{55}{36}+\frac{.243}2(.56)\frac{14}{15}\right\}w^2\right]
<1+2w.
\tag{117}
\]

Its coefficient is below 1.958394 on w<=.12. Combining with (113) proves
the desired output screen with coefficient 2 for these weak prizes.

### Larger competitive prizes: average the actual early cap

For .12<=w<=.252 retain the Section 39 initial pointwise analytic jet bound,
but scale each remainder coefficient by (w/.252)^2 and each difference
coefficient by w/.252. Call this increasing bound Psi_q. At exposure
e_delta v, any M with Psi_q(rho)<M and rho>eta e_delta v M bounds the
ACTUAL policy max(p,n) up to that time. Thus, if m(w,v) is such a cap,

\[
\text{early increment}\le\frac{\alpha\delta}{\sqrt{Q_\delta}}
\int_0^1m(w,v)dv.
\tag{118}
\]

Use sixteen time rectangles; on each choose the cap at its upper exposure.
The same cap remains valid earlier, so no monotonicity of a chosen cap
function needs to be assumed. All its defining inequalities use the original
analytic jet, not an HJB numerical solution. Conditional mature bound (113)
and (110) now give a total output factor at most

\[
(1-f)(1+1.8w)+f\,2.978\int_0^1m(w,v)dv.
\tag{119}
\]

The rational certificate covers [.12,.252] by nine w intervals and bounds
the coefficient of w in (119) by 2.376414<2.4, using f<25/53. These are
intervals of a proved upper function, not numerical optimal-design cells.
If the early coefficient is below 1.8, use 1.8 itself instead; otherwise
the upper f endpoint maximizes (119). This exhausts the two possibilities.

Together, (117)--(119) prove for ALL remaining competitive designs

\[
B\le b(T)+\alpha T G_{Q_0}(0)(1+2.4w).
\tag{120}
\]

The fixed weak threshold .12 is a proof partition, not a new parameter
restriction or a proposed economic boundary. The low ell<=21 screen and
the high lambda>=.72sigma^2 screen already cover the complementary precision
sets. Consequently the ACTUAL global output input required by Section 30
is now available on every competitive design. All original primitives,
precision endpoints and the three-variable design domain are unchanged.

## 42. The fixed ordinary primitives admit a true joint optimum with full disclosure

Keep the externally specified sigma in (55) and the unchanged Figure 3
functions. Let b=251,L=2pi b^2-1,T0=L/sigma^2,c_e=2T0^2/sigma^2.
This is the unique zero-prize deadline and the full-disclosure prize-entry
cost; neither T0 nor c_e restricts feasible designs.

At c=c_e suppose any positive-prize design weakly beats V0=u(b)-T0^2/2.
The utility budget gives w+x^2/4<=h0=1/4+(1+1/L)/(2b)<.252.
Also u(B)>=u(b)-T0^2/2 implies
B>=b-log(1+b/[2(1+1/L)])/2>.99b. The energy screen gives x>1/8.
Thus every such design belongs to one of the already verified low, high or
remaining precision ranges. All satisfy (120), with the smaller available
coefficients relaxed to 2.4.

At outputs B>.99b the fixed exponential utility itself has
-Bu''(B)>=101u'(B). Section 30's power-100 payoff comparison therefore
applies to its ACTUAL output, and proves a strictly negative gain for w>0.
Its auxiliary upper function has unique zero (w,x)=(0,1).
This contradicts competitiveness. Hence at c=c_e all joint optima have
zero prize and deadline T0, for every precision.

Now decrease c below c_e, using alpha=theta/c as the effort coordinate.
Equilibrium output then does not depend on c. For c/c_e sufficiently close
to one, every competitive design still has w+x^2/4<.252: indeed
(c/c_e)w+x^2/4<=h0 and h0<.252. The same necessary B>.99b and all strict
analytic/time comparisons remain valid. The scalar payoff upper function
from Section 30 acquires only the additional term (1-c/c_e)w/(1+1/L).
On the compact proof rectangle 0<=w<=.252,1/8<=x<=2, its zero at (0,1)
is unique. Therefore every competitive positive-prize design has w->0 and
x->1 as c increases to c_e. This is UNIFORM IN PRECISION because the
output upper function no longer depends on precision.

Apply the audited exact weak-prize comparison (27) on the fixed external
interval of prior ratios s in [1/(2L),2/L]. It supplies a positive w_star
uniform over every finite precision, including its separately constructed
zero-signal branch and the unbounded high-precision tail. Thus there is
epsilon>0 such that every competitive positive-prize design for
c_e-epsilon<c<c_e has T in [T0/2,2T0], w<w_star, and

\[
B(\alpha,T,\infty)>B(\alpha,T,\lambda)
\quad\text{for every finite }\lambda.
\tag{121}
\]

Strictly increasing u makes the same inequality hold for organizer payoff,
with the prize and deadline held fixed. The bounds used to LOCALIZE these
designs came from their joint payoff; they are not a conditional deadline
or prize maximization and are not imposed feasibility limits.

For any c<c_e a sufficiently small positive prize at T0 under full disclosure
strictly beats V0, since its alpha derivative at zero is c_e-c>0.
The full-disclosure problem attains its maximum: bounded utility and
quadratic delay localize competitors to a compact prize-deadline rectangle,
and the exact scalar full-disclosure equilibrium and output are continuous
there, including zero prize and zero deadline as proved in manuscript
Step 4. Its maximizing prize is positive and its deadline is positive,
because zero-prize designs give at most V0 and a zero-deadline positive
prize is strictly dominated by zero prize at that same deadline.

Every competitive finite-precision design is strictly improved by (121);
every noncompetitive design is already below V0. The attained full-disclosure
maximum is therefore an attained maximum of the ORIGINAL THREE-PARAMETER
problem, and every joint maximizer in this cost interval satisfies

\[
\theta^*>0,\qquad T^*>0,\qquad\lambda^*=\infty.
\tag{122}
\]

This proves at least one external (c,sigma) point for the FIXED ordinary
functions, in fact a one-dimensional cost window at the constructed sigma.
It does not verify Figure 3's plotted parameter range, provide a precise
economic boundary, or claim the window for every concave utility and convex
delay cost. The constructed risk is extremely large and is not a calibration.
There is no new precision cost, precision floor, exogenous design cap or
curvature assumption. The proof uses the actual uniform weak-prize lemma
in Section 19, not second-order comparisons at a fixed finite precision.

Reverse audit: weak-prize localization uses the now-proved GLOBAL actual
payoff exclusion; the weak-prize lemma treats its infinite tail separately;
positive prize follows from an actual first variation; attainment is first
proved on the full boundary and then lifted to joint attainment by dominance.
The numerical Figure 3 is not a proof input. The mathematical single-point
goal is closed by Sections 40--42, conditional only on the previously audited
selected equilibrium convention already used by the manuscript. Paper
integration and the final completion audit are recorded in Section 43.

## 43. Manuscript integration and requirement-by-requirement audit

The previous goal turn made concrete progress by integrating the exact
uniform weak-prize comparison. This turn integrated the actual output
screen and the global joint-existence argument, and strengthened
Proposition 1(ii). The manuscript's conclusion no longer claims a complete
three-region partition or unrestricted attainment in the finite-feedback
region.

The objective is checked against the current PaperJK7.tex as follows.

| Requirement | Authoritative evidence |
| --- | --- |
| Keep the figure's ordinary primitives fixed | Proposition 1(ii) specifies S0=1, u(B)=3(1-exp(-2B)), D(T)=T^2/2; these agree with the unchanged Figure 3 note. |
| No precision cost, floor, or design caps | The organizer's objective still ranges over theta,T>=0 and lambda in [0,infinity]. All auxiliary prize/deadline bounds are derived from competitiveness against the zero-prize optimum. |
| Exact output, not leading-order output | The output-estimate proof starts from the actual nonlinear gradient equations and controlled-state output identity; explicit full-disclosure values are HJB-verified. Brownian coupling and conditional kernel domination handle the actual state. |
| Every precision competitor covered | Low ell<=21, high lambda>=.72 sigma^2, and the remaining initial-layer construction cover every finite precision, including its analytic zero-signal selection. Full disclosure is separately verified. |
| Uniform weak-prize ranking, including an unbounded tail | The auxiliary-comparison proof treats bounded precisions by strict Gaussian kernel ordering and analytic remainder bounds; the unbounded tail uses the exact nonlinear slow mode and the original Riccati-path comparison. |
| True three-variable global optimum | Step 5 derives a precision-independent global payoff upper function with unique zero at prize exit, uniformly localizes all competitive designs as cost approaches exit, and excludes every finite-precision competitor at its own prize and deadline. |
| Positive prize and deadline, plus attainment | The full-disclosure maximum is attained on a payoff-derived compact set and strictly exceeds the zero-prize maximum for c below c_e. Global dominance lifts this attained maximum to the original joint problem; zero prizes and zero deadlines cannot tie it. |
| Exogenous parameter point | The chosen sigma is the explicit expression in eq-design-existence-reference. The external nonempty interval c_e-epsilon<c<c_e supplies at least one positive cost. No utility or delay-cost property is stipulated at an endogenous design. |
| Blue substantive edits and contextual consistency | Proposition, appendix, explanatory paragraph, conclusion and limitations are updated inside blue groups; the original Theorem 1, data, and figures are unchanged. |
| Compilation checked without replacing the document | Built-in compile_latex_document was called on PaperJK7.tex and fails before source processing because its isolated compiler cannot load informs3.cls. Two local pdflatex draft-mode passes succeed using the existing class and project files; no separate PDF or editor tab is created. |

The exact low-precision, maturity-buffer, centered-strip, real-jet/early
continuation, remaining-output, payoff-algebra, variable-gauge and mature
variance certificates all pass again. Their scopes were inspected: these
are checks of analytic inequalities and finite rational interval covers,
not optimizer-grid evidence. The mathematical deductions connecting them
to the theorem appear in the manuscript proof.

The witness sigma is extremely large and its cost interval correspondingly
small. It proves the requested existence point, not the moderate-range
white region in Figure 3, a general boundary, an open two-dimensional
window, or a theorem for every concave utility and convex delay cost.
No unresolved mathematical assumption or new author choice is introduced
within the maintained selected-equilibrium convention. The built-in
preview's companion-file limitation is reported separately from the
successful local source-compilation check.

Final reverse check clarified that the output-screen gradients use the
original score units, and explicitly handles zero prize without dividing
values by zero. The large-prize certificate now takes the maximum of the
early and mature coefficients before using the upper maturity fraction;
its nine-interval bound remains unchanged and passes. The final local
two-pass draft-mode compilation succeeds with no unresolved references,
overfull boxes, or LaTeX errors. All requested mathematical and manuscript
deliverables are complete; the native preview limitation is not a source
compilation failure.
