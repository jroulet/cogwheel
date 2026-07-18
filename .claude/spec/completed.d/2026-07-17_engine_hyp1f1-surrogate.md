---
date: 2026-07-17
section: engine
---
# Fast hypergeometric evaluation — resolved by numba + coarse kernel grid

The few-ms directive's two levers landed as: (1) numba-njit compilation
of the existing DD shared-numerator 1F1 ladder and operator contraction
(the 2D tabulation/surrogate was REJECTED by Professor + Simplifier as
research-grade certification risk — 85 derivatives spanning ~100 orders
of magnitude — and remains the deferred escalation path); (2) the
`h_L = F*h_UL` factorization exploited via a coarse deterministic
kernel-node grid (base 100, full-cluster transition placement, cubic
spline to bin sub-samples), replacing the 506-point dense inheritance
of the waveform bin grid. Measured outcome: ~15 s -> ~0.3 s/eval warm
single-thread (~50x over brute force) at unchanged accuracy gates.
The remaining gap to few-ms is the real FLOP floor of the order-40
85x85 operator contraction (~2.3 ms/point) — closing it requires the
2D surrogate table (owner decision pending).
