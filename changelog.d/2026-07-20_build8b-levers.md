---
date: 2026-07-20
---
### Serving-path levers: Newton caustic search + fused operator contraction

The per-proposal serving path sheds its two dominant non-spline costs.
`geometry.nearest_caustic_point` now runs an analytic Newton iteration
on the stationarity condition (wedge-clamped per lobe/branch, bounded
Brent fallback at cusps): 1.23 -> 0.095 ms on positive parity, 4.54 ->
0.99 ms on saddles, with the caustic distance certified value-preserving
to 9.3e-12 relative over 5677 both-parity configurations and the arc
parameter gated in arc-length currency against an independent dense
oracle (it is MORE accurate than the previous Brent at shallow minima).
The batched operator's weight-vector build and grid contraction are
fused into a single njit pass, certified 0-bit different across the
certified sweep with exact refusal parity. Public behavior unchanged.
