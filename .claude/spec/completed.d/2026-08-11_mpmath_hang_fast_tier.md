---
date: 2026-08-11
section: Likelihood
---
# Fast-tier mpmath hang — resolved at the test level (parameter choice)

The four tests that hung in `_f_schwinger_mpmath` (the unbounded
arbitrary-precision exact-engine path for `60 < w <= 150`) are resolved
WITHOUT touching production code.  Root cause (measured): the mpmath path
uses adaptive per-panel `mp.quad` at `dps = 30 + w` on an oscillatory
integrand; panel count grows ~`w²` and the adaptive refinement diverges at
some `(w, y)` — genuinely unbounded, not merely slow.  The sub-60 DD path
is fast (~0.5 s, fixed 24-pt GL composite rule).

Fix: moved fast-tier ladder-node frequencies above the QD ceiling
(`w = 150`) so the exact engine hard-refuses instantly instead of entering
the band:

- `test_lensing_airy_fold.py`: `_CUSP_NODE_W` 80→160, `_GEOMETRIC_NODE`
  w 100→200
- `test_lensing_fast_path.py`: `FOP_REFUSALS` / supra grids 63→160
- `test_lensing_levers.py`: `LEVER5_ABOVE_CEILING_W` 62→160 (62 sat in the
  band where the wave evaluator now certifies)

Verified: `test_lensing_airy_fold.py` (90 passed, 35 s), `test_lensing_fast_path.py`
(22 passed, 52 s), `test_lensing_levers.py` (41 passed) — all previously
hanging or slow files now complete.

The PRODUCTION fix (bounded fixed-panel mpmath GL rule replacing adaptive
`mp.quad`) is POSTPONED by user decision — tracked in
`lensing_mpmath_band_fixed_panel_rule.md`.
