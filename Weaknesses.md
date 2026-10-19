### Weaknesses

- Technical limitations or concerns
    - Reliance on a leading-order (small-incentive) approximation raises scope questions: how large can θ/c_k be in practice before the approximation meaningfully biases inference or design counterfactuals?
    - The proportional mapping from effort to submission intensity via a Poisson process is ad hoc and assumes independence across players; strategic withholding and bunching are not modeled.
    - The continuous-update approximation that carries forward the last observed public gap may overstate learning relative to true discrete readouts; identification of λ may be sensitive to this.
    - Local identification requires c_i ≠ c_j; in practice, top rivals may be near-symmetric, potentially weakening identification of σ, λ, and r.
    - The two-player approximation excludes strategic externalities from the broader field and assumes the top rivalry is effectively isolated.
- Experimental gaps or methodological issues
    - The selection of contests and focal pairs (using final private ranks) can introduce selection-on-outcome concerns; while time-split helps, further checks on selection bias would strengthen claims.
    - Limited reporting on posterior predictive checks, goodness-of-fit diagnostics, and comparisons to reduced-form or nonstrategic baselines (e.g., a random-walk gap with Kalman observation) to demonstrate value added by structure.
    - Sensitivity of results to alternative intensity specifications (e.g., overdispersion, self-excitation) or to heavy-tailed observation noise is not reported.
    - The treatment of θ as observed “effective prize” is convenient but may embed unmodeled heterogeneity (e.g., reputation or nonpecuniary incentives).
- Clarity or presentation issues
    - The conditions and intuition behind Theorem 1’s existence and the uniform remainder order could be further summarized in the main text to aid readers who cannot access long appendices.
    - The numerical design regions (Figure 3) are compelling; however, the algorithmic details, convergence checks, and sensitivity to u(·), D(·), and S0 could be elaborated.
    - More detail on the transformation of bounded leaderboard scores into the model’s real-valued state would help interpret Gaussian assumptions and scaling.
- Missing related work or comparisons
    - The literature on leaderboard information design for adaptive evaluation (e.g., the Ladder mechanism: Blum & Hardt/Moritz) provides complementary perspectives on precision and adaptivity that merit explicit discussion.
    - Point-process modeling with stochastic intensities (e.g., DSPP/affine-intensity methods) could inform alternative submission-intensity specifications or robustness checks.
    - The continuous-time information-design work that trades off estimation error and information cost in Kalman–Bucy settings (e.g., recent optimal gain-control formulations) may provide additional theoretical context.

### Detailed Comments

- Technical soundness evaluation
    - The Kalman–Bucy filtering equations are consistent with the linear-Gaussian setting: Ṡ_t = σ^2 − λ S_t^2 yields the standard steady state S̄ = σ/√λ for a driftless state observed through a Gaussian channel. The perceived-state recursion dỹ_t reflects effort-driven drift plus innovation in the innovations process.
    - The HJB system and symmetry/boundary conditions are appropriate for a deadline contest with normal terminal uncertainty. The leading-order expansion produces a plausible, interpretable effort kernel that peaks at ỹ_t = 0 and spreads with Q_t.
    - The design proposition is interesting and nontrivial: showing strict dominance of finite precision without a direct precision cost is conceptually strong. The stated assumptions (concave, sufficiently curved u, convex D, large S0) appear crafted to sustain interior designs and highlight when uncertainty preserves incentives; the paper would benefit from sketching the key comparative statics that drive the strict inequality.
    - Several approximations are made explicit and handled carefully: (i) small-incentive effort approximation; (ii) mapping effort to Poisson submissions; (iii) continuous-time belief updating with carry-forward readouts; and (iv) Gaussian observation errors with variance 1/(λ h_k). These are reasonable first-order choices, but the paper should present robustness to these assumptions given their central role in identifying λ and σ.
- Experimental evaluation assessment
    - The joint likelihood that conditions on realized readout times helps address endogeneity from submission-triggered observations. The decomposition into a submission-count likelihood and a public-gap likelihood, both driven by the same evolving perceived state, is coherent.
    - The path-wise holdout of private trajectories is a strong design choice that goes beyond in-sample fit. The reported associations—innovation risk with private volatility and precision with public–private disagreement—match model predictions and are reinforced by time-split analyses, which is commendable.
    - To more fully establish claims, I encourage adding: (a) posterior predictive checks comparing simulated vs. observed submission counts and public-gap distributions; (b) model-comparison to simpler nonstrategic or constant-intensity alternatives; and (c) sensitivity analyses for the discretization Δ_num, the intensity mapping (e.g., allowing overdispersion), and heavy-tailed public-signal noise.
    - Reporting effect sizes, confidence bands, and contest-level heterogeneity in the validation regressions would help calibrate practical significance. Clarify whether validation regressions control for competition-level features (duration, prize, category) to address confounding.
- Comparison with related work
    - The paper clearly situates itself relative to dynamic contests with continuous effort (Harris–Vickers; Moscarini–Smith; Ryvkin) and to structural work on Kaggle (Lemus–Marshall). Emphasizing the distinct informational channel—signal precision shapes belief filtering rather than inducing variance-seeking submissions—usefully contrasts with recent findings on training/test splits in competitions.
    - Additional connections to leaderboard-information mechanisms (e.g., Ladder) would help bridge the proposed “finite precision” prescription with established practices for adaptive robustness, and distinguish the goals (sustaining incentives vs. preventing overfitting) and their interaction.
    - Point-process and filtering literatures (e.g., affine-intensity DSPPs; ensemble/Kalman–Bucy filtering variants) could be referenced to motivate alternative intensity/noise models and future robustness checks.
- Discussion of broader impact and significance
    - The finding that finite precision can strictly dominate full disclosure in joint design has clear managerial relevance for competition platforms and sponsors. It provides a structured way to think about information policies beyond coarse rules of thumb.
    - The estimation framework enables contest organizers to infer latent primitives from readily available public data, opening the door to ex ante diagnostics and midstream policy adjustments. At the same time, recommending less informative leaderboards has welfare and fairness trade-offs: it may sustain competition but reduce transparency or foster mis-calibration.
    - The work complements strands aiming to improve statistical reliability of leaderboards; the interplay between incentive-sustaining uncertainty and robustness to adaptive overfitting is a promising avenue for future empirical tests.

### Questions for Authors

1. Scope of the leading-order approximation: What ranges of θ/c_k observed in your data imply that w_i + w_j is “small enough”? Can you report posterior mass over regimes where higher-order terms might be material, or show robustness with a second-order correction?
2. Submission intensity mapping: How sensitive are estimates of λ and σ to the assumption τ_i(t) ∝ m_i(ỹ_t,t)? Have you tested alternatives with overdispersion or self-exciting effects, or a flexible link (e.g., log-linear) between effort and intensity?
3. Continuous update vs. discrete readouts: Since carrying forward the last public gap may overstate learning, can you quantify the bias by simulating from a hybrid filter that updates only at submissions and comparing estimated λ under both approaches?
4. Identification near symmetry: In contests where cost asymmetry is small ex post, how stable are posterior inferences for (σ, λ, r)? Do you impose informative priors or regularization in such cases, and can you flag potentially weakly identified contests?
5. Contest and pair selection: Because focal pairs are chosen using final private ranks, can you quantify selection-on-outcome effects (e.g., via placebo re-pairing within contests or using early public ranks for selection) on your validation relationships?
6. Score transformation and Gaussianity: What transformation is applied to map bounded leaderboard scores to ℝ? Do posterior predictive checks indicate approximate normality of gap innovations and measurement errors across contests?
7. Design counterfactuals: For a subset of contests, can you show counterfactual gains from optimizing λ at observed θ and T using your estimated primitives, and compare them with platform heuristics (e.g., coarser public splits or Ladder-like rules)?
8. Priors, computation, and diagnostics: Please report priors and numerical bounds for all parameters across exercises, effective sample sizes, R̂ statistics, and any divergences in NUTS, as well as sensitivity to Δ_num.