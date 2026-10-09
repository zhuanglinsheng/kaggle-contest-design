# Exact-HJB joint design without a precision cost

This experiment searches for finite precision using the exact nonlinear HJB
equilibrium and the organizer's exact expected best output. It does not use
the second-order output approximation and does not modify the manuscript.

## Exact output calculation

For each candidate design, the program solves the two nonlinear player HJBs.
It then solves the organizer's linear backward equation under those equilibrium
policies,

```text
J_t + (m_i-m_j) J_y + lambda*S_t^2*J_yy/2 + (m_i+m_j)/2 = 0,
```

with terminal value equal to the conditional expected best final score. Thus
`B=J(mu0,0)` includes all equilibrium orders, not only the leading or
second-order terms.

An adaptive time grid resolves equal increments of posterior-mean quadratic
variation in addition to calendar time. This is necessary at high precision,
where initial uncertainty is revealed in a short boundary layer. The numerical
identity that `B` is independent of precision at zero prize is satisfied to
about `5e-5` or better on the verification grids.

## Broad search

The normalization is `sigma=c=S0=1`. The grid covers:

- initial perceived gaps `mu0` in `{0, 0.5, 1, 2}`;
- prizes from 0 to 2;
- durations from 0.25 to 4;
- precision from `1e-3` to `1e4`.

The organizer criterion is `A*log(B)-theta-D(T)`. Both linear and quadratic
delay costs were examined over 240 preference/initial-gap cases. Finite
precision strictly beats the design reoptimized at `lambda=1e4` in 139 cases.
Many of these coarse optima lie on a prize or duration bound, so the result
below focuses on a fully interior case.

## Refined interior case

Take

```text
u(B)=6.2 log(B),  D(T)=0.35 T^2/2,  mu0=2,
sigma=c=S0=1.
```

Continuous differential-evolution search over
`theta in [0.5,2.5]`, `T in [0.5,4]`, and `lambda in [0.03,30]` gives

```text
theta = 2.0133, T = 2.8602, lambda = 0.4655.
```

The high-precision comparison fixes `lambda=1e4` and reoptimizes both other
instruments, giving `theta=1.4082`, `T=2.6431`. On successively finer grids:

| Design | dx / time steps | Exact B | Organizer objective |
| --- | --- | ---: | ---: |
| Finite precision | 0.050 / 350 | 1.855449 | 0.387433 |
| Finite precision | 0.035 / 600 | 1.855325 | 0.387020 |
| Finite precision | 0.025 / 1000 | 1.855289 | 0.386899 |
| High precision | 0.050 / 350 | 1.600622 | 0.285696 |
| High precision | 0.035 / 600 | 1.600377 | 0.284747 |
| High precision | 0.025 / 1000 | 1.600287 | 0.284399 |

The finest-grid objective gap is `0.102500`. Both finite instruments are well
inside their search bounds, and the finite precision is also interior. The
profiled broad grid checks the intervening precision range through `1e4`, so
the comparison is not merely against the upper endpoint of the local search.
At the high-precision design, extending the tail to `lambda=1e5` and `1e6`
reduces the fine-grid objective further, from `0.2842` to `0.2813` and
`0.2800`. Two additional global-search seeds converge to
`(theta,T,lambda)=(2.0126,2.8565,0.4663)` to four displayed digits.

The no-effort best-output components are `1.1582` at the finite design and
`1.1451` at the high-precision design. The equilibrium increments above those
benchmarks are `0.6971` and `0.4541`, respectively. The finite design's
advantage is therefore primarily an equilibrium-effort effect rather than a
mechanical difference in the sampling value of the two durations.

## Interpretation and limits

The exact equilibrium supplies a higher-order discouragement effect absent
from the second-order ranking. When feedback is very precise, contestants
learn the gap early and equilibrium effort falls. Moderate noise preserves
competition long enough to raise expected best output, even after the prize
and duration are reoptimized.

This is a constructive numerical example, not a theorem. It uses symmetric
costs, one organizer utility/delay specification, and an interior-equilibrium
solver. The current reliable broad-search range is approximately
`theta/(sigma^2 c) <= 2`; substantially larger incentives at very low
precision require a monotone solver for the nearly first-order HJB.

## Reproduction

```text
.venv/bin/python -m archive.experiments.exact_joint_design --quick
.venv/bin/python -m archive.experiments.exact_joint_refine
```
