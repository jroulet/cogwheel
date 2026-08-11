---
date: 2026-08-11
section: Backlog
---

# Schwinger mpmath band (60 < w <= 150): fixed-panel rule — COMPLETED

The production fix (postponed by the test-level fix of
`2026-08-11_mpmath_hang_fast_tier`) landed in
`cogwheel/lensing/chang_refsdal/_schwinger.py`:

- `_MP_PANEL_ORDER = 32` and `_mp_gl_rule(order, dps)` (lru-cached mpmath
  Gauss-Legendre nodes/weights) replace the adaptive per-panel
  `mp.quad(..., maxdegree=5)` in `_f_schwinger_mpmath`'s
  `_raw_integral_mp` — now a fixed-order composite GL panel loop, the same
  structure the DD path uses, at mpmath precision so the `e^{πw/4}`
  cancellation stays certified above `w = 60`.
- The N/2N paired-rule certification is computed on the RECONSTRUCTED F in
  mpmath (`_mp_reconstruct`) and converted to float only at the return
  (`return complex(f_2n_mp)`).
- Order-32 (not 24): order-24 under-resolves the upper band (measured N/2N
  disagreement ~3e-4 > `_CERTIFICATION_TOL = 3e-10` at w=100 across tested
  `(y, gamma')`), which would regress serving coverage in the cusp-exterior
  windows where the exact engine is the last rung (astroid cusp-exterior
  source `(1.5, 0.05)` gamma=0.5 at w=100 is REFUSED by order-24 but SERVED
  by order-32/48, `|F| = 0.8268`). Order-32 certifies across the band and is
  cheap on this fallback-only path (the surrogate covers the bulk:
  astroid <= 480 / saddle <= 148 per `_POSITIVE_W_CEILING` /
  `_SADDLE_W_CEILING`).

ACCEPTANCE (met): `f_schwinger` for `w in (60, 150]` in the served box
completes in O(seconds) — measured w=80: ~160 s -> ~14 s — and the band no
longer diverges. Certification at `_CERTIFICATION_TOL = 3e-10` preserved.
All 68 tests in `cogwheel/tests/test_lensing_schwinger.py` pass (~4-5 min),
and `test_refusal_precedes_coherent_score` (the `CANCELLATION_LENS`
hard-core nodes in the band) no longer hangs — resolving the last open guard
in `lensing_serving_ladder_guards_are_red`, whose STILL RED section now
holds only `test_thresholds_have_one_home` (tracked separately in
`lensing_one_home_routing_disagreement`).
