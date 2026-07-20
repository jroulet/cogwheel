---
bump: minor
---
### Build 8b-levers — serving-path levers (Newton caustic search + fused contraction)

Engine-row addendum: `nearest_caustic_point` is now an analytic Newton
iteration (distance value-preserving at 9.3e-12 worst rel; theta ruled
gauge and gated in arc length vs an independent oracle — F017; 13x /
4.6x faster), and the batched operator's weight-vector build + grid
contraction are fused into one njit pass, certified 0-bit different
with exact refusal parity (F010 falsifications re-homed through
`_fused_contraction.py_func`).
