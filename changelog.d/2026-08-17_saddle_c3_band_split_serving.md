---
date: 2026-08-17
---

Lensing: the macro-saddle exterior tier-1 analytic intercept (`saddle_c3`)
now serves via a per-draw **band-split** instead of a whole-band gate --
above the closed-form split point `w_split` it serves the certified
zero-envelope analytic form, and the exact engine is stitched in below
`w_split`. Previously the certificate could only admit a draw as a
whole band, so most of the prior fell through to the exact engine even
though the draw's high-`w` nodes were individually well inside the
certified regime. Combined with the sibling above-ceiling ppGO rung
becoming per-node in the same build, the corrected 10k-draw demand
census shows: `saddle_c3` 0.32% -> 14.09% of the prior, `ppgo_above_ceiling`
0.00% -> 15.87%, `wave_refused` (deterministic refusals) 12.03% -> 2.13%,
and total exact-engine demand (`engine_residual`) 72.25% -> 53.30%.
