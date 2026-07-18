---
date: 2026-07-17
bump: minor
---
Microlensed likelihood fast path (Builds 3/3b): coarse full-cluster
kernel-node spline grid (`_DEFAULT_KERNEL_NODES = 100`), numba-njit
dd/1F1 ladder + operator contraction (refusal contract untouched),
njit nearest-caustic search; certified by the new
`cogwheel/tests/test_lensing_fast_path.py` (numba-vs-mpmath
preservation, null-safe production-grid interpolation gate,
RB-vs-brute on every lens regime, single-thread timing guards).
