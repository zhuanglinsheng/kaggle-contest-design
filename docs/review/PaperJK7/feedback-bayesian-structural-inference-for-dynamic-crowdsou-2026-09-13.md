# Bayesian Structural Inference for Dynamic Crowdsourcing Contests

**Date**: 9/13/2026, 12:52:47 PM
**Domain**: social_sciences/economics
**Taxonomy**: academic/research_paper
**Filter**: Active comments

---

## Overall Feedback

Here are some observations and thoughts for further developing the paper.

**Terminal payoff specification**

Section 2 states that the actual prize is determined by the private outcome, but the model imposes the terminal utility $\theta \cdot \mathbf{1}_{\{\tilde{y}_T>0\}}$. Because $y_T|I_T \sim \mathcal{N}(\tilde{y}_T,S_T)$, the corresponding expected terminal payoff is $\theta \cdot \Phi(\tilde{y}_T/\sqrt{S_T})$. Substituting the indicator function creates the discontinuous tug-of-war problem inherited from Ryvkin and, combined with steady-state filtering, causes signal precision to drop out of the equilibrium effort policy entirely. Because noisy feedback and Bayesian learning are the principal theoretical extensions of the paper, this approximation merits deeper exploration. The paper would benefit from either validating the indicator approximation over the estimated empirical parameter range, or reframing the core contribution around an econometric filtering model that intentionally abstracts away from strategic responses to posterior uncertainty.

**Observation system and likelihood construction**

The alignment between the theoretical observation system and the empirical likelihood invites some clarification. Equations (20)–(22) shift between an interval-averaged continuous signal, endpoint observations with variance $1/(\lambda h_k)$, and continuous updating from a carried-forward public gap during intervals without submissions. Treating a stale leaderboard value as a continuing continuous-time signal may overstate learning, which directly alters the perceived states and effort paths entering the likelihood. Additionally, Equation (24) does not supply the explicit predictive recursion for $\mu_{k|k-1}$ and $V_{k|k-1}$. Finally, the identification argument in Lemma 3 relies on an off-diagonal covariance assumption of $\sigma^2 \min(t_i, t_j)$, which is not immediately apparent for the nonlinear, history-dependent continuous-update system actually estimated. Establishing a unified event-time transition and measurement system, alongside evidence quantifying the continuous-update approximation against a discrete-continuous filter, would strengthen the foundation of the likelihood.

**Empirical identification of the cost-scale parameter**

The identification of the structural parameters warrants further discussion regarding the effort-to-submission scale $r$. Lemma 3 establishes local identification strictly conditional on a fixed $r$, yet the empirical application estimates $r$ jointly with $c_i$ and $c_j$. The synthetic check using estimated $r$ reports posterior correlations of 0.709 and 0.656 between $r$ and the two costs, pointing to substantial scale interactions. Furthermore, Table 9 reports posterior means with several costs near the upper prior support and $\sigma$ estimates near the lower prior support, but omits empirical posterior uncertainty, boundary mass, and sampler diagnostics. The identification argument would be more complete if the paper either normalized $r$ to focus on relative costs, or provided comprehensive diagnostics demonstrating that the joint posteriors are driven by the empirical likelihood rather than prior boundaries.

**Interpretation of private-path diagnostics**

The private-leaderboard validation exercises present an intriguing approach to model assessment, but shared scaling and update mechanics currently complicate their interpretation. In Table 7, the association between estimated $\sigma$ and private-gap volatility shares the contest-specific transformed-score scale, while the zeroes and jumps in the outcome variable are dictated by the same submission timing that enters the estimation likelihood. The time-split specification does not fully isolate these persistent contest-level scale effects. Furthermore, the $\lambda$ benchmarks in Section 5.2 rely on a maintained Gaussian discrepancy equation and display extreme inverse-square values in Table 9. The cross-contest regressions also treat posterior means as known regressors without propagating estimation uncertainty. Conducting a scale-normalized posterior predictive evaluation of held-out event-time private paths, and employing uncertainty-aware regression models, would provide firmer support for the structural recovery claims.

**Assumptions in the prize-duration counterfactuals**

Section 5.3 calculates counterfactuals by combining the general asymmetric numerical effort calculation with a symmetric small-$w$ selection term from Lemma 2, where $w=\theta/(\sigma^2 c)$. Applying this asymptotic symmetric term to contests that are asymmetric and not uniformly constrained to a small $w$ introduces unquantified approximation error. Additionally, the counterfactual calculation retains the positive sampling term $\sigma\sqrt{T}/\sqrt{2\pi}$ when $\theta$ and equilibrium effort approach zero. Under Equation (18), zero expected effort generates zero submissions, meaning the platform receives nothing to sample from unless it relies entirely on the unestimated opportunistic baseline attempts proposed in Remark 2. Consequently, the model mechanically substitutes duration for prize spending even when the submission framework predicts an inactive contest. Recomputing the frontier using a coherent submission-output process with full posterior simulation of the selection term would provide a more reliable evaluation of alternative designs.

**Status**: [Pending]

---

## Detailed Comments (30)

### 1. Lemma 3 uses an unsupported covariance decomposition

**Status**: [Pending]

**Quote**:
> First, consider the Gaussian leaderboard component in (24). Equality of Gaussian likelihoods for all sampled paths implies equality of both covariance and mean structures. Hence,

$$
\Sigma_{y}+D_{\lambda}=\Sigma_{y}^{\prime}+D_{\lambda^{\prime}}, \quad \mu_{k}=\mu_{k}^{\prime} \text { for all } k .
$$

Here $D_{\lambda}$ and $D_{\lambda^{\prime}}$ are diagonal matrices with $D_{\lambda, k k}=1 /\left(\lambda h_{k}\right)$ and $D_{\lambda^{\prime}, k k}=1 /\left(\lambda^{\prime} h_{k}\right)$, where the readout intervals $\left\{h_{k}\right\}_{k=1}^{N}$ are fixed by the observation design. With at least two distinct leaderboard observations, the off-diagonal covariance terms equal $\sigma^{2} \min \left(t_{i}, t_{j}\right)$ and are unaffected by the independent signal-noise term. Therefore equality of the off-diagonal covariance terms implies $\sigma=\sigma^{\prime}$. Given $\sigma=\sigma^{\prime}$, equality of the diagonal terms then implies $\lambda=\lambda^{\prime}$.

**Feedback**:
The proof of Lemma 3 does not justify the covariance decomposition used to identify $\sigma$ and $\lambda$. Under (22)–(24), past public-gap innovations alter later perceived states and nonlinear effort paths, so the raw leaderboard vector need not have off-diagonal covariance $\sigma^2\min(t_i,t_j)$. An identification argument must instead use the full collection of history-conditional predictive distributions or rigorously justify a parameter-dependent residual transformation; as written, local identification of $\sigma$ and $\lambda$ is not established.

---

### 2. Section 3.1 gives the public gap incompatible roles

**Status**: [Pending]

**Quote**:
> The public gap is the empirical counterpart of the noisy signal in (2), observed through discrete leaderboard updates. We write this empirical readout as

$$
d Z_{t}=\hat{y}_{t} d t,
$$

which should be interpreted as an observation convention linking the public leaderboard gap to the signal path, not as a second structural definition of $Z_{t}$. Combining (2) and (19) gives the formal white-noise representation

$$
\hat{y}_{t}=y_{t}+\frac{1}{\sqrt{\lambda}} \xi_{t}, \quad \xi_{t} \equiv \frac{d B_{t}}{d t},
$$

where $\xi_{t}$ denotes Gaussian white noise in the generalized-process sense. Because white noise is not pointwise observed, equation (20) is not a pointwise sampling equation. In the empirical data, the public leaderboard is sampled only at submission-triggered update times. At an empirical update time $t_{k}$, averaging the white-noise representation over the observed readout interval $h_{k}=t_{k}-t_{k-1}$ yields a finite-variance noise term $\varepsilon_{k} \sim \mathcal{N}\left(0,1 /\left(\lambda h_{k}\right)\right)$.

**Feedback**:
The empirical observation system does not yet define a single coherent role for $\hat y_t$. As a carried-forward leaderboard path it is piecewise constant, whereas equations (19)–(20) treat it as the generalized derivative of the diffusion signal. Averaging that signal over $(t_{k-1},t_k]$ would define a distinct interval readout, not automatically the new snapshot revealed at $t_k$; equation (22) then carries that new snapshot over the following interval. Because these distinct objects feed the likelihood and filtering recursion, the observation model underlying identification of $\lambda$ is internally unresolved.

---

### 3. Selection bias from ex-post ranking

**Status**: [Pending]

**Quote**:
> We select candidate focal contestants beginning from the top of the final private leaderboard. This focuses the two-player model on rivalry among highperforming contestants. Private path values do not enter the estimation likelihood, but final private ranks determine the focal pair; hence, the validation exercises are path-wise holdout diagnostics conditional on ex post pair selection.

**Feedback**:
Selecting focal contestants by final private rank makes their public histories conditional on terminal outcomes, but the likelihood does not condition on that selection event. This can affect the population interpretation and consistency of the estimated primitives, even though private path values are excluded from the likelihood. The direction and magnitude of any bias are not established: selecting two top contestants does not simply give the relative gap a positive mean shock, and the estimates may instead be interpreted as selection-conditioned focal-rivalry parameters.

---

### 4. Flawed continuous-update approximation for beliefs

**Status**: [Pending]

**Quote**:
> To preserve the tractable one-dimensional state representation used in the equilibrium analysis, we instead employ a continuous-update approximation. Specifically, we carry forward the most recently observed public gap and evolve the perceived gap according to

$$
d \tilde{y}_{t}=\left[m_{i}\left(\tilde{y}_{t}, t\right)-m_{j}\left(\tilde{y}_{t}, t\right)\right] d t+\sqrt{\lambda} \sigma\left(\hat{y}_{t}-\tilde{y}_{t}\right) d t
$$

**Feedback**:
Equation (22) repeatedly applies a carried-forward public gap during intervals with no new leaderboard observation, creating continuous mean reversion toward stale feedback. An exact discrete-observation filter would instead predict between updates and incorporate each score only when it arrives. Unless inactivity is intended to provide repeated information about the unchanged score, this approximation overweights stale observations in a way that depends on gap length and may materially affect perceived states, effort paths, and the estimates of $c_i$, $c_j$, $\lambda$, and $\sigma$.

---

### 5. Systematic bias in asynchronous gap construction

**Status**: [Pending]

**Quote**:
> We construct hourly evaluation grids by carrying forward each contestant's most recently observed transformed score between submissions. Because submissions are asynchronous, only the submitting contestant's score is updated, while the opponent's most recent score is carried forward. The observed public gap may therefore compare scores submitted at different times and only approximates the contemporaneous latent performance gap.

**Feedback**:
Carried-forward asynchronous gaps introduce lag- and submitter-dependent discrepancies relative to the contemporaneous latent gap. Under positive latent drift, this staleness component generally has a nonzero conditional mean and is correlated with effort and submission timing, whereas the endpoint Gaussian readout and continuous-update likelihood do not explicitly model it. The paper acknowledges the approximation, but its implications for parameter recovery remain substantive; the error’s sign alternates with which contestant submits rather than uniformly favoring one fixed gap direction.

---

### 6. Initial-state variance is inconsistent and conflicts with Lemma 2

**Status**: [Pending]

**Quote**:
> Finally, $x_{0}^{i}=x_{0}^{j}=0$ implies $y_{0}=0$. Integrating the gap dynamics and using $W_{T}=\left(W_{T}^{i}-W_{T}^{j}\right) / \sqrt{2}$ gives

$$
y_{T}=J_{T}+\sigma W_{T}, \quad W_{T} \sim \mathcal{N}(0, T) .
$$

Consequently, $\mathbb{E}\left|\sigma W_{T}\right|=\sigma \sqrt{2 T / \pi}$. The pathwise inequality $||a+b|-|a|| \leq|b|$, applied with $a=\sigma W_{T}$ and $b=J_{T}$, now yields

**Feedback**:
The initialization in Lemma 2 is inconsistent with the maintained filtering initialization. Given $S_0=\mathbb{E}[(\tilde y_0-y_0)^2\mid I_0]$ and $\tilde y_0=0$, imposing deterministic $y_0=0$ implies $S_0=0$, not $S_0=\bar S>0$. Starting from $S_0=0$ produces time-varying filtering variance and removes the stated steady-state reduction. If the stationary prior is retained, then $y_T=y_0+J_T+\sigma W_T$; under the stated linear-Gaussian setup, the leading selection term at $w=0$ becomes $\sqrt{\bar S+\sigma^2T}/\sqrt{2\pi}$ rather than $\sigma\sqrt{T}/\sqrt{2\pi}$. The design derivation therefore needs to distinguish the incremental gap $y_t-y_0$ from the latent level gap $y_t$, or otherwise account for initial-state uncertainty. As written, this affects equation (13) and leaves the dependent design and counterfactual formulas without a consistent common initialization, unless the design analysis is explicitly based on a distinct known-initial-state approximation.

---

### 7. Effort observability is unclear in Section 2's filter and HJB

**Status**: [Pending]

**Quote**:
> The information set of both players at time $t \geq 0$ is $I_{t} \equiv\left\{Z_{s}: 0 \leq s \leq t\right\}$. Player $i$ estimates the unknown output gap $y_{t}$ based on $I_{t}$. Let $\tilde{y}_{t} \equiv E\left(y_{t} \mid I_{t}\right)$ denote the filtered mean output gap, and let $S_{t} \equiv E\left[\left(\tilde{y}_{t}-y_{t}\right)^{2} \mid I_{t}\right]$ denote the conditional mean-squared filtering error. We normalize the initial perceived gap to $\tilde{y}_{0}=0$ throughout the theoretical analysis. Under standard linear-Gaussian assumptions, the Kalman-Bucy filter (Chapter 1.2 of Bensoussan 1992; see also Frogerais et al. 2012, Barrau and Bonnabel 2017) implies

$$
\begin{aligned}
d \tilde{y}_{t} & =\left(q_{i, t}-q_{j, t}\right) d t+\lambda S_{t}\left(d Z_{t}-\tilde{y}_{t} d t\right) \\
\frac{d S_{t}}{d t} & =\sigma^{2}-\lambda S_{t}^{2}
\end{aligned}
$$

**Feedback**:
The controlled filter and the subsequent common-state HJB require $q_{i,t}$ and $q_{j,t}$ to be common $I_t$-measurable inputs—for example, publicly observed efforts or actions generated by common-knowledge public-feedback strategies. That restriction is not stated, nor does the paper specify off-path beliefs, and the treatment of unilateral deviations is therefore not established. If effort is privately known or hidden, unilateral deviations may separate the players' posterior states, and the players need not share the posterior represented by the displayed filter. Consequently, the displayed HJB would not follow from the stated information structure.

---

### 8. Counterfactual exceeds the approximation’s domain

**Status**: [Pending]

**Quote**:
> For the counterfactual calculation, we combine the exact decomposition in Lemma 2 with the leading-order selection component in equation (13) and evaluate predicted expected best output as $B(\theta, T)=M\left(\tilde{y}_{0}, 0\right) / 2+\sigma \sqrt{T} / \sqrt{2 \pi}$. At each candidate design $(\theta, T)$, total expected effort $M\left(\tilde{y}_{0}, 0\right)$ is obtained from the general asymmetric ODE in Lemma 1, using the contest-specific estimates of $\left(c_{i}, c_{j}, \sigma, \mu_{0}\right)$ and setting $\tilde{y}_{0}=\mu_{0}$. Thus, the calculation retains the general asymmetric numerical effort prediction while approximating the selection component at leading order.

**Feedback**:
The quantitative prize-duration frontier applies $\mathbb{E}|y_T|/2\approx\sigma\sqrt{T}/\sqrt{2\pi}$ outside the domain established in Lemma 2. Equation (13) is proved only under symmetric costs as $w\downarrow0$, whereas the empirical calculation uses asymmetric estimated costs without showing that the relevant $w_i$ and $w_j$ are small or that the omitted selection effect is negligible. The resulting quantitative frontier is therefore not supported by the stated approximation alone.

---

### 9. Incorrect observation noise variance scaling with time

**Status**: [Pending]

**Quote**:
> At an empirical update time $t_{k}$, averaging the white-noise representation over the observed readout interval $h_{k}=t_{k}-t_{k-1}$ yields a finite-variance noise term $\varepsilon_{k} \sim \mathcal{N}\left(0,1 /\left(\lambda h_{k}\right)\right)$. Approximating the corresponding readout of the latent gap by its endpoint value and substituting the latent state dynamics in (1) gives

$$
\hat{y}_{t_{k}} \approx y_{0}+\int_{0}^{t_{k}}\left[m_{i}\left(\tilde{y}_{s}, s\right)-m_{j}\left(\tilde{y}_{s}, s\right)\right] d s+\sigma W_{t_{k}}+\varepsilon_{k},
$$

**Feedback**:
The variance $1/(\lambda h_k)$ is correct for an interval-average readout of the continuous diffusion signal, but using it for a submission-triggered leaderboard snapshot imposes the substantive restriction that longer gaps produce more precise observations. Waiting longer between submissions does not by itself make a Kaggle score more precise, as the platform uses a discrete, fixed-split evaluation mechanism. That restriction is not implied by the snapshot construction and may materially affect estimation of $\lambda$ and $\sigma$. If public-private discrepancy variance is approximately independent of $h_k$, then the validation benchmark $\lambda_\ell^{\text{bench}}$ inherits submission-rate dependence through $N^\ell/\sum_k h_k^\ell$. A fixed public evaluation set does not by itself establish constant error variance across submissions, but the assumed inverse-duration scaling requires empirical or modeling support and sensitivity assessment. The fixed-grid benchmark check weakens the claim that Figure 6 is solely a mechanical artifact, but it does not establish that the structural estimate of $\lambda$ is robust to a discrete-update observation model without the $1/h_k$ variance restriction.

---

### 10. Figure 1 information set differs from the model

**Status**: [Pending]

**Quote**:
> Note: This figure illustrates the structure of a typical Kaggle contest and the information flow available to contestants. Rectangles and rounded rectangles represent observed data and latent states, respectively. The private leaderboard is visible only to the contest organizer during the contest and is revealed to all participants only after it ends. During the contest period, contestants can observe only the submission times of each player and the public leaderboard.
at its most recent submitted value. The private leaderboard, in particular, evolves according to the real-time effort level and an innovation shock.

**Feedback**:
The stated information structures do not fully align: Figure 1 says contestants observe submission times, while Section 2 defines $I_t=\{Z_s:0\leq s\leq t\}$ and derives filtering and equilibrium without conditioning on counting-process histories. Because Section 3 makes arrivals endogenous through $\tau_i(t)=r m_i(\tilde y_t,t)$, their omission is innocuous only under a conditional-redundancy and common-knowledge assumption; otherwise, arrivals and nonarrivals may affect beliefs and strategies.

---

### 11. Predictive moments are undefined in Section 3.3

**Status**: [Pending]

**Quote**:
> To make the conditioning structure explicit, we evaluate the leaderboard likelihood sequentially. Let $\mathcal{F}_{k-1}$ denote the submission and public-leaderboard history strictly before update time $t_{k}$. Conditional on $\mathcal{F}_{k-1}$ and $\Theta$, the empirical filtering recursion generates the one-step-ahead predictive distribution

$$
\hat{y}_{t_{k}} \mid \mathcal{F}_{k-1}, \Theta \sim \mathcal{N}\left(\mu_{k \mid k-1}, V_{k \mid k-1}\right),
$$

Here, $\mu_{k \mid k-1}$ and $V_{k \mid k-1}$ denote the one-step-ahead predictive mean and variance implied by the empirical state and observation equations over $\left(t_{k-1}, t_{k}\right]$, conditional only on $\mathcal{F}_{k-1}$. These moments are computed before incorporating the leaderboard observation at $t_{k}$. The leaderboard likelihood is therefore

$$
p\left(\left\{\hat{y}_{t_{k}}\right\}_{k=1}^{N} \mid \Theta\right)=\prod_{k=1}^{N} \phi\left(\hat{y}_{t_{k}} \mid \mu_{k \mid k-1}, V_{k \mid k-1}\right) .
$$

**Feedback**:
The leaderboard likelihood is not operationally specified because the recursion for $\mu_{k\mid k-1}$ and $V_{k\mid k-1}$ is absent. Equation (22) constructs a perceived-state path but does not state how latent-state uncertainty is predicted and updated after each irregular, noisy readout. Different plausible treatments of the accumulated Brownian innovation, the interval-specific measurement variance, and the initial covariance produce different factors in equation (24).

---

### 12. Design discussion overstates neutrality to signal precision

**Status**: [Pending]

**Quote**:
> Under the terminal-payoff rule and the steady-state restriction $S_{t}=\bar{S}$, both performance criteria are neutral with respect to signal precision $\lambda$, because $\lambda \bar{S}^{2}=\sigma^{2}$ and $\lambda$ therefore drops out of the equilibrium effort policy. This neutrality is specific to the maintained assumptions and need not hold when filtering uncertainty varies over time or contestants respond to posterior winning probabilities. Nevertheless, $\lambda$ affects belief updating and the noise in the observed public leaderboard, and therefore remains an estimable parameter in the structural model developed below.

**Feedback**:
The claim that both performance criteria are neutral to $\lambda$ is too broad. The identity $\lambda\bar S^2=\sigma^2$ makes the filtered-state diffusion and equilibrium effort rule independent of $\lambda$, but the exact criterion $B(\theta,T)=M(0,0)/2+\mathbb{E}|y_T|/2$ contains the actual terminal gap. Under the maintained posterior $y_T\mid I_T\sim\mathcal N(\tilde y_T,\bar S)$, its conditional absolute moment depends on $\bar S=\sigma/\sqrt{\lambda}$. Neutrality is therefore established for effort and the later leading-order criterion, not for exact expected best output; the simultaneous use of $S_0=\bar S$ and a known $y_0=0$ also requires reconciliation.

---

### 13. Invalid consistency benchmark due to asynchronous updates

**Status**: [Pending]

**Quote**:
> Under the diffusion approximation, the drift-adjusted increment satisfies

$$
\Delta y_{k}-\hat{\mu}_{k} \sim \mathcal{N}\left(0, \sigma^{2} h_{k}\right), \quad k=1, \ldots, n .
$$

Treating these increments as approximately independent, the log-likelihood for $\sigma^{2}$ is, up to constants,

$$
\ell\left(\sigma^{2}\right)=-\frac{n}{2} \log \left(\sigma^{2}\right)-\frac{1}{2} \sum_{k=1}^{n} \log h_{k}-\frac{1}{2 \sigma^{2}} \sum_{k=1}^{n} \frac{\left(\Delta y_{k}-\hat{\mu}_{k}\right)^{2}}{h_{k}} .
$$

**Feedback**:
The increment likelihood in Appendix B.2 does not match the asynchronously updated, carried-forward private gap. A submitting player's displayed-score change generally accumulates since that player's own previous submission, whereas the benchmark subtracts gap drift and divides by the pooled inter-event interval $h_k$. Consequently, a short $h_k$ following an opponent's update can greatly inflate variation accumulated over a much longer player-specific interval. In a constant-rate competing-Poisson special case this mechanism can produce a non-finite first moment, although that conclusion is not established directly for the paper's state-dependent intensities. While the distortion need not always be upward, the Figure 7 relationship is substantially confounded by asynchronous timing. The temporal mismatch alone is sufficient to invalidate the stated Gaussian likelihood as justification for a consistency benchmark.

---

### 14. Signal benchmark omits interval-averaging error

**Status**: [Pending]

**Quote**:
> Let $\ell \in\{1, \ldots, L\}$ index contests. Under the maintained Gaussian discrepancy approximation induced by the discretized signal readout, we construct the following signal-precision benchmark. Let $\mathcal{E}_{\ell}$ denote the ordered set of submission-triggered leaderboard readouts for the focal pair, and let $N^{\ell}=\left|\mathcal{E}_{\ell}\right|$. Let $h_{k}^{\ell}=t_{k}^{\ell}-t_{k-1}^{\ell}$ denote the elapsed time associated with readout $t_{k}^{\ell}$. Under the maintained approximation

$$
\hat{y}_{t_{k}}^{\ell}-y_{t_{k}}^{\ell} \sim \mathcal{N}\left(0, \frac{1}{\lambda_{\ell} h_{k}^{\ell}}\right), \quad t_{k}^{\ell} \in \mathcal{E}_{\ell},
$$

we construct

$$
\lambda_{\ell}^{\text {bench }}=\frac{N^{\ell}}{\sum_{t_{k}^{\ell} \in \mathcal{E}_{\ell}} h_{k}^{\ell}\left(y_{t_{k}}^{\ell}-\hat{y}_{t_{k}}^{\ell}\right)^{2}} .
$$

**Feedback**:
The benchmark for $\lambda$ may conflate signal noise with within-interval state evolution. Averaging the continuous signal over an interval gives noise variance $1/(\lambda h_k)$ around the interval-average state, not the endpoint state $y_{t_k}$. Under Brownian latent dynamics, replacing that average by the endpoint adds variance $\sigma^2h_k/3$ even with zero drift, while nonzero effort drift can add a mean discrepancy. Consequently, irregular interval lengths can affect $\lambda_\ell^{\mathrm{bench}}$ independently of signal precision; the stated approximation and serial-dependence caveat do not establish that this contamination is negligible.

---

### 15. Posterior estimates truncated by prior bounds

**Status**: [Pending]

**Quote**:
> | 3064 | 13 | 5 | 0.20 | 25 | 14.106 | 0.630 | -0.788 | 1.160 | 2.780 | 13.821 | 8157921.303 | 0.263 |
| 3080 | 25 | 27 | 0.09 | 25 | 23.442 | 0.531 | -1.217 | 0.623 | 0.643 | 19.327 | 149115.198 | 0.187 |
| 3288 | 20 | 10 | 1.00 | 0 | 10.864 | 3.729 | -1.169 | 0.822 | 1.750 | 15.448 | 3.518 | 0.616 |

**Feedback**:
Table 9 appears to show several posterior means for $\sigma$ close to its lower support bound of $0.5$, and some cost means close to their upper bound of $5$. Posterior means alone do not establish boundary pile-up or the direction of any resulting regression distortion, but the repeated proximity makes prior-bound sensitivity a potentially substantive concern; without posterior boundary diagnostics or wider-support results, it is uncertain whether these estimates and the validation relationships are support-driven.

---

### 16. Section 5.1 leaves focal-pair selection undefined

**Status**: [Pending]

**Quote**:
> The second criterion identifies two highly active, top-ranked contestants whose participation overlaps substantially during the contest. We select candidate focal contestants beginning from the top of the final private leaderboard. This focuses the two-player model on rivalry among highperforming contestants. Private path values do not enter the estimation likelihood, but final private ranks determine the focal pair; hence, the validation exercises are path-wise holdout diagnostics conditional on ex post pair selection. To ensure overlapping activity and adequate data for estimation, each contestant must submit at least five times, and the pair must satisfy the temporal-overlap criterion used in sample construction.

**Feedback**:
The focal-pair rule is not operationally defined. “Beginning from the top,” “overlaps substantially,” and the referenced temporal-overlap criterion do not specify a unique pair when several contestants qualify. Because pair selection determines every contest-level likelihood and validation path, the exact ranking, overlap, and tie-breaking rules are substantively important.

---

### 17. Pooled recovery conditions on random Poisson counts

**Status**: [Pending]

**Quote**:
> Replications. To begin, we investigate recovery in a fixed-design information-scaling exercise. Starting from the baseline synthetic contest, we replicate the same contest design under identical structural parameters and pool the resulting panels for Bayesian inference. The submission counts are held fixed at the baseline realization in each replicated panel, so the exercise should not be read as an unconditional simulation of the Poisson count distribution. Instead, it asks whether posterior recovery improves when the likelihood is given more observations from the same underlying information environment.

**Feedback**:
The pooled experiment fixes each panel’s Poisson submission counts but does not document whether Table 3 replaces the unconditional point-process likelihood in (23) with a likelihood conditional on those counts. If the panels were generated conditional on fixed counts while estimation retained (23), the factor $e^{-\Lambda_i(\Theta)}\Lambda_i(\Theta)^{N_i}$ would be treated repeatedly as parameter-dependent evidence even though the counts were fixed by design. In that case, the reported concentration could not be attributed cleanly to additional observations generated under the estimation likelihood. This concern would be resolved if a count-conditional likelihood was used, but the implementation as described is ambiguous.

---

### 18. Section 5.1 prize mapping may not match the focal pair

**Status**: [Pending]

**Quote**:
> As a first selection criterion, we focus on monetary prize structure. We restrict attention to Kaggle competitions offering one, two, or three monetary prizes denominated in U.S. dollars. Competitions with a single prize naturally conform to the winner-take-all incentive structure. For competitions with two or three prize tiers, we align the empirical setting with the model by treating the prize difference between the top two contestants as the effective gain from winning. Because non-winning contestants may still receive symbolic recognition such as medals, focusing on the top-two prize gap helps mitigate confounding incentives and brings the empirical setting closer to the model's winner-take-all benchmark.

**Feedback**:
The prize mapping may not correspond to the selected focal rivalry. If activity or overlap screening can select contestants outside final ranks 1 and 2, the top-two prize difference need not equal the monetary payoff difference between their realized rank positions; this issue also applies to the second–third pair robustness exercise. The interpretation is valid only if the top-two spread is intentionally maintained as a contest-level incentive proxy rather than the selected pair’s actual payoff increment.

---

### 19. Section 5.1 does not specify score-domain filtering

**Status**: [Pending]

**Quote**:
> We restrict attention to contests reporting accuracy-based scores, which provides a common bounded score domain and facilitates a consistent transformation across contests. The retained focal paths are restricted to score observations satisfying the domain $p_{t} \in[0.5,1)$ used below. To map raw leaderboard accuracies into the model's state scale, we first transform each individual leaderboard score using the logit (log-odds) function:

$$
x_{t}=a \cdot \log \left(\frac{p_{t}}{1-p_{t}}\right), \quad p_{t} \in[0.5,1)
$$

where $p_{t}$ is an individual leaderboard score, $x_{t}$ is the corresponding transformed score, and $a$ is a scaling parameter in the transformation. ${ }^{9}$

**Feedback**:
The score-domain filter is not operationally specified. It is unclear whether a focal-path value below $0.5$ or equal to $1$ causes removal of that observation, its submission event, the contestant, or the entire contest. These choices affect submission counts, readout intervals, likelihood contributions, and private-path diagnostics.

---

### 20. Algorithm 1 leaves thinning intensity undefined

**Status**: [Pending]

**Quote**:
>     for \(t=0, \Delta, \ldots, T-\Delta\) do
        \(m_{i(j)}\left(\tilde{y}_{t}, t\right) \leftarrow(6), \tau^{i(j)}(t) \leftarrow(18) \quad\left\{\right.\) Use \(\left.\theta, c_{i(j)}, \sigma, r, T\right\}\)
        \(y_{t+\Delta} \leftarrow(1) \quad\left\{\right.\) Use \(\left.m_{i}-m_{j}, \sigma, \Delta, W_{t+\Delta}\right\}\)
        \(\tilde{y}_{t+\Delta} \leftarrow(22) \quad\left\{\right.\) Use \(m_{i}-m_{j}, \lambda, \sigma, \Delta\), and available \(\left.\hat{y}_{t}\right\}\)
        for \(s_{k}^{i(j)} \in[t, t+\Delta)\) do
            if \(u_{k}^{i(j)}<\tau_{i(j)}\left(s_{k}^{i(j)}\right) / \tau_{i(j)}^{\star}\) then

**Feedback**:
Algorithm 1 evaluates the target intensity only at grid points but applies the thinning test using $\tau_{i(j)}(s_k)$ at off-grid candidate times without defining interpolation or state evolution to $s_k$. It therefore does not establish whether submissions are drawn by exact continuous-time thinning or from a piecewise-constant discretization of the equilibrium intensity.

---

### 21. Algorithm 1 omits readout-interval noise scaling

**Status**: [Pending]

**Quote**:
>         \(\tilde{y}_{t+\Delta} \leftarrow(22) \quad\left\{\right.\) Use \(m_{i}-m_{j}, \lambda, \sigma, \Delta\), and available \(\left.\hat{y}_{t}\right\}\)
        for \(s_{k}^{i(j)} \in[t, t+\Delta)\) do
            if \(u_{k}^{i(j)}<\tau_{i(j)}\left(s_{k}^{i(j)}\right) / \tau_{i(j)}^{\star}\) then
                \(\hat{t}_{\ell}^{i(j)} \leftarrow s_{k}^{i(j)} ; \ell^{i(j)} \leftarrow \ell^{i(j)}+1\) {Accept the submission event}
                \(\hat{y}_{t+\Delta} \leftarrow\) discretized readout of (21) \(\quad\left\{\right.\) Use \(\sigma, \lambda, B_{t+\Delta}\); stored for subsequent filtering}
            end if

**Feedback**:
Algorithm 1 does not specify whether each public readout uses the irregular elapsed interval $h_k$ and a corresponding Brownian increment, as required for noise variance $1/(\lambda h_k)$ in the likelihood, or instead uses grid-scaled noise based on $\Delta$. The synthetic recovery evidence for $\lambda$ depends on these observation models being aligned; the displayed algorithm does not establish that alignment, particularly when multiple submissions occur in one grid cell.

---

### 22. Signal validation reuses its public-data component

**Status**: [Pending]

**Quote**:
> The signal precision parameter $\lambda$ is contest-specific and captures the informativeness of public leaderboard feedback relative to latent performance. We compare the structural estimate obtained from submissions and public leaderboard paths with an approximate benchmark constructed from public-private leaderboard discrepancies. This benchmark is informative but not fully independent of the maintained structural setup, because it still relies on the transformed leaderboard scale used in the model.

Let $\ell \in\{1, \ldots, L\}$ index contests. Under the maintained Gaussian discrepancy approximation induced by the discretized signal readout, we construct the following signal-precision benchmark.

**Feedback**:
Figure 6 is not an independent holdout comparison: $\hat{\lambda}_\ell$ and $\lambda_\ell^{\mathrm{bench}}$ share the realized public gaps, update times, and interval lengths, while the benchmark adds the excluded private path. This reuse can correlate the two quantities’ estimation errors, and common transformed-score scaling can move both in the same direction. The positive slope therefore provides directional consistency rather than independent validation; the later time-split diagnostic partly addresses common-path dependence but is itself imprecise.

---

### 23. Prize-gap changes are treated as spending cuts

**Status**: [Pending]

**Quote**:
> For each contest, let $\left(\theta^{\text{obs }}, T^{\text{obs }}\right)$ denote the observed prize and duration, and let $B_{0}=B\left(\theta^{\text{obs }}, T^{\text{obs }}\right)$ denote predicted expected best output under the observed design. For a duration multiplier $a \geq 1$, let $T_{a}=a T^{o b s}$ and define

$$
\theta_{a}^{\mathrm{BO}}=\min \left\{\theta \in\left[\theta_{\min }, \theta^{o b s}\right]: B\left(\theta, T_{a}\right) \geq B_{0}\right\},
$$

where $\theta_{\text{min }}=0.01$ thousand USD is the lower prize bound used in the numerical search. For each duration multiplier considered below, direct numerical evaluation confirms that $\theta=\theta^{\text{obs }}$ satisfies the best-output constraint, so the feasible set is nonempty. The required prize change is $100\left(\theta_{a}^{\mathrm{BO}}-\right.$ $\left.\theta^{\text{obs }}\right) / \theta^{\text{obs }}$, so negative values indicate prize reductions that keep predicted expected best output at least at the observed-design baseline. This exercise is not a claim that longer contests are costless or welfare improving. Rather, it reports the iso-best-output prize-duration frontier: how much prize spending can be substituted by a small, externally specified increase in duration.

**Feedback**:
The spending interpretation of Table 8 does not follow for multi-tier contests. In those contests, $\theta$ is defined empirically as the first-versus-second prize gap, so the reported percentage change measures a reduction in that incentive gap, not a uniquely determined percentage reduction in total prize spending. The numerical frontier remains interpretable in terms of the effective top-two winner premium.

---

### 24. Contradiction between data normalization and prior

**Status**: [Pending]

**Quote**:
> More generally, the mean would include the latent initial gap $y_{0}$; in the empirical implementation we normalize $y_{0}=0$ by measuring subsequent gaps relative to the initial latent-gap origin. This normalization is distinct from the empirical initialization $\mu_{0}$, which relaxes the theoretical restriction $\tilde{y}_{0}=0$.

**Feedback**:
The initialization convention is underspecified. It is unclear whether the observed public-gap series and $\hat y_0$ in the prior for $\mu_0$ are expressed relative to the same origin used to normalize $y_0=0$, and whether the first predictive mean begins from the latent origin or from $\mu_0$. The distinction between a normalized latent state and a nonzero initial belief is coherent in principle, but without these coordinate and initialization details the early likelihood and effort path cannot be reconstructed unambiguously.

---

### 25. Endogenous effort complicates the Gaussian increment law

**Status**: [Pending]

**Quote**:
> Let $\left(t_{k}^{\ell}, y_{k}^{\ell}\right)_{k=0}^{n_{\ell}}$ denote the private-leaderboard gap evaluated at the ordered update times used for this benchmark, including the normalized initial observation. Under the diffusion specification in (1), the private-leaderboard increment satisfies $y_{k}^{\ell}-y_{k-1}^{\ell} \mid\left\{m_{i}^{\ell}, m_{j}^{\ell}\right\} \sim \mathcal{N}\left(\mu_{k}^{\ell}, \sigma^{2}\left(t_{k}^{\ell}-t_{k-1}^{\ell}\right)\right)$ for $k=1, \ldots, n_{\ell}$. Here, $\mu_{k}^{\ell}=\int_{t_{k-1}^{\ell}}^{t_{k}^{\ell}}\left[m_{i}^{\ell}(s)-m_{j}^{\ell}(s)\right] d s$. Motivated by the Gaussian increment formula, we compute the following contest-specific consistency benchmark; the corresponding diffusion calculation is provided in Appendix B.2:

**Feedback**:
The stated conditional Gaussian law is too strong. Equilibrium efforts are feedback functions of the filtered state and can therefore be informative about the Brownian innovations; conditional on the entire realized effort paths, the residual innovation need not retain its original $\mathcal{N}(0,\sigma^2\Delta t)$ distribution. The integrated SDE provides a pathwise residual identity that can motivate an approximate diagnostic, but it does not by itself establish the displayed conditional distribution.

---

### 26. Section 6.2 does not test the advertised assumption

**Status**: [Pending]

**Quote**:
> ### 6.2. Normality Caveat and Interpretation

The multiplicative parameter $a$ in (26) rescales transformed scores and can affect scale-dependent likelihood and validation quantities, but it cannot alter scale-invariant normality diagnostics. Auxiliary checks across alternative values of $a$ should therefore be interpreted as scale-sensitivity checks, not as evidence for the Gaussian approximation. Additional scale-sensitivity results are available from the authors.

We do not conduct a fully non-Gaussian robustness analysis here, because both the benchmark constructions and the Bayesian estimation procedure rely on the maintained Gaussian structure. Extending the analysis beyond this setting would require re-specifying the likelihood and re-deriving the benchmark estimators under alternative distributional assumptions.

**Feedback**:
Section 6 states that both focal-pair selection and the Gaussian assumption are examined through re-estimation, but subsection 6.2 explicitly reports no non-Gaussian analysis. Varying the multiplicative scale $a$ cannot assess distributional-shape robustness, and the scale-sensitivity results are not reported. Thus, the Gaussian assumption remains a maintained limitation rather than a tested robustness dimension.

---

### 27. Alternative-pair eligibility is unclear in Section 6.1

**Status**: [Pending]

**Quote**:
> The baseline empirical analysis pairs two highly active top-ranked contestants in each contest. To assess whether the validation results are sensitive to this focal-pair choice, we re-estimate the model under an alternative pairing rule.

As a robustness check, we estimate the model on the second- and third-ranked contestants in each contest, based on final private leaderboard ranks. This specification shifts the focal rivalry away from the winner and tests whether the validation pattern is driven by one specific top-pair definition.

**Feedback**:
Subsection 6.1 does not establish whether every second–third-ranked pair satisfies the baseline requirements of at least five submissions per contestant and temporal overlap. Because the robustness regressions still use 73 contests, it is uncertain whether only rank position changes or whether eligibility and data-support rules also change; this affects how cleanly the comparison isolates focal-pair choice.

---

### 28. Mischaracterization of the amplitude term's role

**Status**: [Pending]

**Quote**:
> The equilibrium effort $m_{i(j)}$ can be decomposed into two components: a normal density term $\phi\left(\tilde{y}_{t} ; 0, \sigma^{2}(T-t)\right)$ and an amplitude term $K_{i(j)}\left(z ; \theta, \sigma, c_{i}, c_{j}\right):=\frac{\sigma^{2}}{2}\left[\gamma\left(\rho_{i}\right)+\gamma\left(\rho_{j}\right)\right]\left[1-\rho(z)^{2}\right][1 \pm \rho(z)]$. The density term governs how competitive intensity is distributed over the state space, whereas the amplitude term governs the asymmetry of effort across players.

**Feedback**:
The decomposition overstates the separation between the two terms. Although the common normal-density factor is central to state-space concentration and $[1\pm\rho(z)]$ determines the player-specific effort ratio, $K_{i(j)}$ is itself state dependent through $\rho(z)$. In particular, the common factor $[1-\rho(z)^2]$ also shapes total effort across states, so $\theta$, $c_i$, and $c_j$ can affect the state-space profile as well as effort levels and asymmetry. The equilibrium formula itself is unaffected.

---

### 29. Conclusion understates the fixed-variance restriction

**Status**: [Pending]

**Quote**:
> Several limitations deserve mention. First, the empirical setting differs from the baseline model in several respects: many contests use multi-tier or team-based prizes, actual contests involve more than two strategic contestants, leaderboard metrics vary across contests, and the focal pair is selected using final private ranks. Second, the empirical filter assumes that inference variance quickly converges and approximates asynchronous submission-triggered updates with a continuousupdate equation. A hybrid filter with discrete measurement updates would represent irregular observation times more faithfully but would substantially reduce tractability. Third, despite the common logit transformation, cross-contest comparisons of innovation risk and private-path volatility may retain a shared transformed-score scale component. These limitations suggest natural extensions involving richer prize structures, multi-agent dynamics, and more flexible learning mechanisms.

**Feedback**:
The conclusion’s statement that inference variance “quickly converges” may misleadingly suggest that the implemented filter includes a transient variance process. The model instead imposes $S_0=\bar S$ and therefore $S_t\equiv\bar S$ from the outset, while the empirical recursion has no variance state. This fixed-variance restriction is consequential: it removes $\lambda$ from equilibrium effort and excludes learning through changes in posterior uncertainty.

---

### 30. Appendix C: zero public-data share needs reconciliation

**Status**: [Pending]

**Quote**:
> | 3080 | 25 | 27 | 0.09 | 25 | 23.442 | 0.531 | -1.217 | 0.623 | 0.643 | 19.327 | 149115.198 | 0.187 |
| 3288 | 20 | 10 | 1.00 | 0 | 10.864 | 3.729 | -1.169 | 0.822 | 1.750 | 15.448 | 3.518 | 0.616 |
| 3338 | 31 | 29 | 1.50 | 30 | 24.827 | 0.541 | 0.203 | 3.508 | 3.345 | 7.687 | 9639.132 | 0.244 |

**Feedback**:
Contest 3288 appears to have “Public Data %” equal to zero even though public leaderboard observations must have been available to estimate $\lambda$ and construct $\lambda^{\mathrm{bench}}$. This may reflect rounding, missing metadata, or a definition distinct from public-score availability, but the column’s current meaning leaves the contest’s eligibility and data construction uncertain.

---
