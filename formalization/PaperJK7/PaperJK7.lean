import Mathlib.Tactic
import Mathlib.Analysis.Calculus.Deriv.Basic

/-!
Independently checked algebraic components of PaperJK7, not a formalization of
its stochastic model or PDE existence theory. All analytic/probabilistic inputs
are explicit hypotheses. No manuscript theorem is assumed as an axiom.
-/
namespace PaperJK7

/-- HJB completion of the square, before imposing nonnegative effort. -/
theorem complete_square (c p q : ℝ) (hc : c ≠ 0) :
    p * q - c * q ^ 2 / 2 =
      p ^ 2 / (2 * c) - c / 2 * (q - p / c) ^ 2 := by
  field_simp [hc]
  ring

/-- With a positive marginal value, the unconstrained maximizer is feasible. -/
theorem effort_maximizer (c p q : ℝ) (hc : 0 < c) (hp : 0 < p) :
    0 < p / c ∧ p * q - c * q ^ 2 / 2 ≤ p ^ 2 / (2 * c) := by
  constructor
  · exact div_pos hp hc
  · rw [complete_square c p q (ne_of_gt hc)]
    have h := mul_nonneg (le_of_lt (div_pos hc (by norm_num : (0:ℝ) < 2)))
      (sq_nonneg (q - p / c))
    linarith

/-- The two displayed forms of the HJB driver agree, including cross signs. -/
theorem driver_decomposition (ai aj zi zj : ℝ) :
    zi * (ai * zi + aj * zj) - ai * zi ^ 2 / 2 =
      ai * zi ^ 2 / 2 + aj * zi * zj := by ring

/-- The negative-coordinate bound used for Jackson's boundedness condition. -/
theorem negative_coordinate_bound (L a z : ℝ) (hL : 0 ≤ L) (ha : a ≤ L) :
    L * a / 2 * z ^ 2 ≤ L ^ 2 / 2 * z ^ 2 := by
  nlinarith [mul_nonneg hL (sub_nonneg.mpr ha), sq_nonneg z,
    mul_nonneg (mul_nonneg hL (sub_nonneg.mpr ha)) (sq_nonneg z)]

/-- Riccati dynamics imply Q' = -lambda*S^2. -/
theorem remaining_variance_derivative (S : ℝ → ℝ) (sigma lam T t : ℝ)
    (hS : HasDerivAt S (sigma ^ 2 - lam * (S t) ^ 2) t) :
    HasDerivAt (fun u => S u + sigma ^ 2 * (T - u))
      (-lam * (S t) ^ 2) t := by
  convert hS.add (((hasDerivAt_const t T).sub (hasDerivAt_id t)).const_mul
    (sigma ^ 2)) using 1
  ring

/-- A quadratic polynomial cannot vanish at three distinct real points. -/
theorem quadratic_three_roots (a b c x y z : ℝ)
    (hxy : x ≠ y) (hxz : x ≠ z) (hyz : y ≠ z)
    (hx : a*x^2+b*x+c=0) (hy : a*y^2+b*y+c=0)
    (hz : a*z^2+b*z+c=0) : a=0 ∧ b=0 ∧ c=0 := by
  have h1 : (x-y)*(a*(x+y)+b)=0 := by nlinarith
  have h2 : (x-z)*(a*(x+z)+b)=0 := by nlinarith
  have h1' := (mul_eq_zero.mp h1).resolve_left (sub_ne_zero.mpr hxy)
  have h2' := (mul_eq_zero.mp h2).resolve_left (sub_ne_zero.mpr hxz)
  have h3 : a*(y-z)=0 := by nlinarith [h1', h2']
  have ha := (mul_eq_zero.mp h3).resolve_right (sub_ne_zero.mpr hyz)
  have hb : b=0 := by simpa [ha] using h1'
  have hc : c=0 := by simpa [ha, hb] using hx
  exact ⟨ha, hb, hc⟩

/-- Three first-readout variances identify (S0, sigma^2, 1/lambda).
The Gaussian law and event-conditioning argument are not asserted here. -/
theorem variance_identification (S a b S' a' b' x y z : ℝ)
    (hx : 0<x) (hy : 0<y) (hz : 0<z)
    (hxy : x ≠ y) (hxz : x ≠ z) (hyz : y ≠ z)
    (ex : S+a*x+b/x=S'+a'*x+b'/x)
    (ey : S+a*y+b/y=S'+a'*y+b'/y)
    (ez : S+a*z+b/z=S'+a'*z+b'/z) :
    S=S' ∧ a=a' ∧ b=b' := by
  have px : (a-a')*x^2+(S-S')*x+(b-b')=0 := by
    field_simp at ex
    nlinarith [ex]
  have py : (a-a')*y^2+(S-S')*y+(b-b')=0 := by
    field_simp at ey
    nlinarith [ey]
  have pz : (a-a')*z^2+(S-S')*z+(b-b')=0 := by
    field_simp at ez
    nlinarith [ez]
  obtain ⟨ha, hS, hb⟩ := quadratic_three_roots (a-a') (S-S') (b-b')
    x y z hxy hxz hyz px py pz
  exact ⟨sub_eq_zero.mp hS, sub_eq_zero.mp ha, sub_eq_zero.mp hb⟩

/-- Two unequal cumulative intensity differences identify r and mu. -/
theorem mean_identification (mu mu' r r' I J : ℝ)
    (hr : r ≠ 0) (hr' : r' ≠ 0) (hIJ : I ≠ J)
    (hI : mu+I/r=mu'+I/r') (hJ : mu+J/r=mu'+J/r') :
    r=r' ∧ mu=mu' := by
  have h : (I-J)*(r'-r)=0 := by
    field_simp at hI hJ
    nlinarith [hI, hJ]
  have heq : r=r' := by
    have := (mul_eq_zero.mp h).resolve_left (sub_ne_zero.mpr hIJ)
    linarith
  constructor
  · exact heq
  · rw [heq] at hI
    linarith

/-- Positive common kernel and asymmetric costs give unequal intensities. -/
theorem asymmetric_intensities (K ci cj : ℝ) (hK : K ≠ 0)
    (hi : ci ≠ 0) (hj : cj ≠ 0) (hij : ci ≠ cj) : K/ci ≠ K/cj := by
  intro h
  apply hij
  field_simp at h
  exact ((h.resolve_right hK)).symm

/-- Given r and a positive common kernel, intensity identifies cost. -/
theorem cost_identification (r K c c' : ℝ) (hr : r ≠ 0) (hK : K ≠ 0)
    (hc : c ≠ 0) (hc' : c' ≠ 0) (h : r*K/c=r*K/c') : c=c' := by
  field_simp at h
  exact h.symm

/-- The simultaneous scaling used in the symmetric nonidentification example. -/
theorem common_scale_invariance (r c K scale : ℝ) (hs : scale ≠ 0) :
    (scale*r)*K/(scale*c)=r*K/c := by
  by_cases hc : c = 0
  · simp [hc]
  · field_simp [hs, hc]
    ring

/-- Endpoint factorization in the radial kernel positivity argument. -/
theorem kernel_endpoint_factorization (x : ℝ) :
    1+2*x^3-3*x^4=(1-x)*(3*x^3+x^2+x+1) := by ring

/-- Positivity holds over the entire required domain, not only a grid. -/
theorem kernel_numerator_positive (x z : ℝ) (hx : 0 ≤ x) (hx1 : x < 1)
    (hz : 0 ≤ z) (hzx : z ≤ x) :
    0 < 1+x^3+x*z^2-3*x^2*z^2 := by
  have hz2 : z^2 ≤ x^2 := by nlinarith
  have hx3 : 0 ≤ x^3 := pow_nonneg hx 3
  have hxpos : 0 < 1-x := by linarith
  have hend : 0 < (1-x)*(3*x^3+x^2+x+1) := by positivity
  by_cases h : 0 ≤ x-3*x^2
  · nlinarith [mul_nonneg h (sq_nonneg z)]
  · have hm := mul_nonneg_of_nonpos_of_nonpos (le_of_not_ge h) (sub_nonpos.mpr hz2)
    rw [← kernel_endpoint_factorization x] at hend
    nlinarith [hm]

/-- The low-diffusion barrier's residual has the claimed sign. -/
theorem barrier_residual (alpha d tau Q : ℝ) (_ha : 0 ≤ alpha)
    (hd : 0 ≤ d) (ht : 0 ≤ tau) (hQ : 0 < Q) :
    (486/1000:ℝ)*alpha^2*d*tau/Q+2*alpha*d-
      2*alpha*d*(1+(486/1000:ℝ)*alpha*tau/Q) ≤ 0 := by
  have hp : 0 ≤ (486/1000:ℝ)*alpha^2*d*tau/Q := by positivity
  calc
    _ = -((486/1000:ℝ)*alpha^2*d*tau/Q) := by ring
    _ ≤ 0 := neg_nonpos.mpr hp

/-- Exact polynomial margin used in the full-disclosure payoff screen. -/
theorem payoff_screen_identity (x : ℝ) :
    (1/2-(1-x)/40) -
      x*((1+x)/4+29*(1-x)*(1+x)^2/160) =
      (1-x)^2*(29*x^2+87*x+76)/160 := by ring

/-- This part of the payoff screen holds on a continuum of durations. -/
theorem payoff_screen_bound (x : ℝ) (hx : 0 ≤ x) :
    x*((1+x)/4+29*(1-x)*(1+x)^2/160) ≤ 1/2-(1-x)/40 := by
  have hp : 0 ≤ (1-x)^2*(29*x^2+87*x+76)/160 := by positivity
  rw [← payoff_screen_identity x] at hp
  linarith

/-- Upper bound obtained by completing the square against delay costs. -/
theorem quadratic_delay_bound (a d T : ℝ) (hd : 0 < d) :
    a*T-d*T^2/2 ≤ a^2/(2*d) := by
  rw [complete_square d a T (ne_of_gt hd)]
  have h := mul_nonneg (le_of_lt (div_pos hd (by norm_num : (0:ℝ)<2)))
    (sq_nonneg (T-a/d))
  linarith

/-- Counterfactual coefficient: asymmetry enters only through (a-b)*H. -/
theorem first_order_bracket (T a b H : ℝ) :
    T/2*(a+b)+(a-b)*H = a*(T/2+H)+b*(T/2-H) := by ring

theorem first_order_coefficient_positive (T a b H G : ℝ)
    (ha : 0<a) (hb : 0<b) (hG : 0<G)
    (hlo : -T/2 < H) (hhi : H < T/2) :
    0 < G*(T/2*(a+b)+(a-b)*H) := by
  rw [first_order_bracket]
  exact mul_pos hG (add_pos (mul_pos ha (by linarith)) (mul_pos hb (by linarith)))

theorem symmetric_precision_cancellation (T a H : ℝ) :
    T/2*(a+a)+(a-a)*H=T*a := by ring

/-- Weighted quadratic flux bound at kappa = 7/4. -/
theorem weighted_flux_bound (X Y : ℝ) (hX : 0 ≤ X) (hY : 0 ≤ Y) :
    X*Y/4+(7/4:ℝ)/8*(3*Y^2+X^2) ≤
      (7/4:ℝ)/8*(X+(7/4:ℝ)*Y)^2 := by
  nlinarith [mul_nonneg hX hY, sq_nonneg Y]

/-- First radius bound for the high-diffusion contraction. -/
theorem high_diffusion_radius_one (w : ℝ) (hw : 0 ≤ w) (hb : w ≤ 63/250) :
    (2+6*w^2)/(4*(72/100:ℝ)) < 1 := by
  have hw2 : w^2 ≤ (63/250:ℝ)^2 := by nlinarith
  rw [div_lt_one (by norm_num)]
  nlinarith

/-- Second radius bound for the high-diffusion contraction. -/
theorem high_diffusion_radius_two (w : ℝ) (hw : 0 ≤ w) (hb : w ≤ 63/250) :
    (1+w^2*(51+36*w^2)/4)/(3*(72/100:ℝ)) < 1 := by
  have hw2 : w^2 ≤ (63/250:ℝ)^2 := by nlinarith
  have hw4 : (w^2)^2 ≤ ((63/250:ℝ)^2)^2 := by nlinarith [sq_nonneg (w^2)]
  rw [div_lt_one (by norm_num)]
  nlinarith

/-- First Lipschitz row bound for the high-diffusion contraction. -/
theorem high_diffusion_lipschitz_one (w : ℝ) (hw : 0 ≤ w) (hb : w ≤ 63/250) :
    w/(72/100:ℝ)*((2+3*w+6*w^2)/2) < 1 := by
  calc
    _ ≤ (63/250:ℝ)/(72/100)*((2+3*(63/250)+6*(63/250)^2)/2) := by gcongr
    _ < 1 := by norm_num

/-- Second Lipschitz row bound for the high-diffusion contraction. -/
theorem high_diffusion_lipschitz_two (w : ℝ) (hw : 0 ≤ w) (hb : w ≤ 63/250) :
    w/(72/100:ℝ)*((4+18*w+12*w^2)/4) < 1 := by
  calc
    _ ≤ (63/250:ℝ)/(72/100)*((4+18*(63/250)+12*(63/250)^2)/4) := by gcongr
    _ < 1 := by norm_num

end PaperJK7


#print axioms PaperJK7.complete_square
#print axioms PaperJK7.effort_maximizer
#print axioms PaperJK7.driver_decomposition
#print axioms PaperJK7.negative_coordinate_bound
#print axioms PaperJK7.remaining_variance_derivative
#print axioms PaperJK7.quadratic_three_roots
#print axioms PaperJK7.variance_identification
#print axioms PaperJK7.mean_identification
#print axioms PaperJK7.asymmetric_intensities
#print axioms PaperJK7.cost_identification
#print axioms PaperJK7.common_scale_invariance
#print axioms PaperJK7.kernel_endpoint_factorization
#print axioms PaperJK7.kernel_numerator_positive
#print axioms PaperJK7.barrier_residual
#print axioms PaperJK7.payoff_screen_identity
#print axioms PaperJK7.payoff_screen_bound
#print axioms PaperJK7.quadratic_delay_bound
#print axioms PaperJK7.first_order_bracket
#print axioms PaperJK7.first_order_coefficient_positive
#print axioms PaperJK7.symmetric_precision_cancellation
#print axioms PaperJK7.weighted_flux_bound
#print axioms PaperJK7.high_diffusion_radius_one
#print axioms PaperJK7.high_diffusion_radius_two
#print axioms PaperJK7.high_diffusion_lipschitz_one
#print axioms PaperJK7.high_diffusion_lipschitz_two
