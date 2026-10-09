# Finite precision without a precision cost

This numerical experiment evaluates the second-order coefficient
`R(mu,T,lambda;S0)` used by Proposition 1 after removing the precision cost.
It does not modify the paper or establish an analytical monotonicity result.

## Normalization and method

The coefficient scales by `sqrt(T)/sigma^3`, so its shape depends on three
dimensionless quantities:

- `s0 = S0/(sigma^2 T)`;
- `z = abs(mu)/sqrt(S0+sigma^2 T)`;
- `ell = lambda sigma^2 T^2`.

The calculation integrates out the Gaussian state analytically and evaluates
the two remaining time integrals by Gauss--Legendre quadrature. The change of
variables `t=1-a^2`, `s=1-a^2 b^2` regularizes the complete-information
endpoint. This avoids Monte Carlo noise.

The primary scan uses 64 quadrature nodes per integration dimension:

- 17 values of `s0` from `1e-4` to `1e4`;
- 41 values of `z` from 0 to 5;
- 73 finite values of `ell` from `1e-8` to `1e10`;
- a separately evaluated complete-information limit, `ell=infinity`.

This gives 697 parameter curves and 50,184 adjacent finite-precision
comparisons. A second scan uses 128 quadrature nodes on an independent grid.

## Results

- No finite precision in either scan exceeds the complete-information limit.
- Of the 697 primary curves, 615 are increasing, 81 first decrease and then
  increase, and one is numerically flat at the stated tolerance.
- No curve has a finite local maximum on the grid. Thus the observed
  nonmonotonicity is U-shaped, not hump-shaped.
- The closest finite point occurs at the extreme grid boundary
  `(s0,z,ell)=(1e4,5,1e10)` and remains `1.49e-11` below its
  complete-information limit. The 128-node calculation gives the same gap to
  the displayed precision.

The numerical evidence therefore points to `lambda -> infinity` when
precision is costless. Some parameter values make an initial increase in
precision harmful, but sufficiently high precision eventually dominates that
local effect. This rules out neither an off-grid counterexample nor the need
for an analytical proof.

## Reproduction

Run:

```text
.venv/bin/python -m archive.experiments.second_order_precision_no_cost
```

Machine-readable results are in `no_cost_precision_scan.json`; the associated
figure is `no_cost_precision_scan.png`.
