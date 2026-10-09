# Optimal-policy hourly data generation

Theoretical nonlinear HJB policy m replaces the first-order policy. No Bayesian estimation run.
Both steady (S0=2) and nonsteady (S0=4) samples retain the original hourly grid, held public readouts,
readout-noise intervals, thinning, RNG seeds, initial draws, and long-horizon r values.
The hourly belief recursion remains an empirical approximation to the theoretical continuous-signal filter.

| Experiment | Steady i / j | S0=4 i / j |
|---|---:|---:|
| baseline | 30 / 19 | 28 / 18 |
| pool_contest01 | 8 / 6 | 16 / 14 |
| pool_contest02 | 19 / 7 | 18 / 6 |
| pool_contest03 | 14 / 12 | 14 / 12 |
| pool_contest04 | 21 / 18 | 20 / 16 |
| pool_contest05 | 31 / 28 | 32 / 28 |
| pool_contest06 | 15 / 14 | 15 / 14 |
| pool_contest07 | 40 / 35 | 40 / 35 |
| pool_contest08 | 25 / 23 | 25 / 23 |
| pool_contest09 | 20 / 27 | 21 / 27 |
| pool_contest10 | 27 / 24 | 26 / 24 |
| pool_contest11 | 22 / 27 | 21 / 27 |
| pool_contest12 | 23 / 22 | 20 / 19 |
| pool_contest13 | 26 / 18 | 26 / 18 |
| pool_contest14 | 25 / 28 | 25 / 28 |
| pool_contest15 | 30 / 34 | 30 / 34 |
| pool_contest16 | 19 / 17 | 19 / 17 |
| pool_contest17 | 23 / 9 | 22 / 9 |
| pool_contest18 | 19 / 11 | 19 / 11 |
| pool_contest19 | 28 / 24 | 28 / 23 |
| pool_contest20 | 13 / 13 | 13 / 13 |
| days180 | 61 / 54 | 59 / 55 |
| days360 | 132 / 99 | 135 / 100 |
