---
date: 2026-08-14
bump: minor
---

Tier-1 macro-saddle far-field rung re-gated: the scalar
`_SADDLE_FARFIELD_RHO_FLOOR = 2.0` and the `delta_tau > 0` resolution leg
are replaced by c3-led certificate admission
(`_SADDLE_FARFIELD_SAFETY * ppgo_error_estimate <= _SADDLE_FARFIELD_CERT_BAR`
at the band floor) with None-refusal as the coalescence discriminator and a
separation backstop; symmetry-tied mirror pairs now serve. Census mirror
moved in the same build. Rho-floor-era suites
(`test_lensing_saddle_gauge.py`, `test_lensing_saddle_tier1_accuracy.py`,
`test_lensing_saddle_tier1_refusal.py`) retired; successor
`test_lensing_saddle_serve_gate.py`. Build `symmetry_tie_c3_admission`.
