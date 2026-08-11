---
date: 2026-08-11
---
### Schwinger QD band (60 < w <= 150) now bounded: fixed-order composite Gauss-Legendre rule

The mpmath arbitrary-precision path in
`cogwheel/lensing/chang_refsdal/_schwinger.py` (`_f_schwinger_mpmath`)
replaces the adaptive per-panel `mp.quad(..., maxdegree=5)` (tanh-sinh) with
a fixed-order composite Gauss-Legendre rule: `_mp_gl_rule(order, dps)`
(lru-cached mpmath nodes/weights) at `_MP_PANEL_ORDER = 32` per panel, the
same composite structure the DD path uses. The N/2N paired-rule
certification is now computed on the RECONSTRUCTED F in mpmath
(`_mp_reconstruct`) and converted to float only at the return.

The band is bounded and deterministic — previously unbounded/divergent (the
adaptive refinement never converges at some `(w, y)`; a call could hang for
hours), it now completes in O(seconds) (measured w=80: ~160 s -> ~14 s) and
no longer diverges. Order-32 rather than 24 preserves serving coverage at
the certification bar: order-24 under-resolves the upper band (measured N/2N
disagreement ~3e-4 > `_CERTIFICATION_TOL = 3e-10` at w=100), which would
regress serving in the cusp-exterior windows where the exact engine is the
last rung; order-32 certifies across the band.

Because the band is bounded, the `CANCELLATION_LENS` hard-core nodes in
`(60, 150]` complete fast — `test_refusal_precedes_coherent_score` no longer
hangs (the last open serving-ladder guard, see
`lensing_serving_ladder_guards_are_red`). All 68 tests in
`cogwheel/tests/test_lensing_schwinger.py` pass (~4-5 min).
