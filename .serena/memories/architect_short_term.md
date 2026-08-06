Last session: 2026-08-04 production batch. Clean.

2026-08-06 brief_mpmath_band_tests:
- Four fast-tier tests wander into f_schwinger mpmath band w in (60,150]
  (~85-120s/eval, F061). NO production change (f_schwinger + both ceilings
  frozen). ALL fixes are in TEST FILES (fixtures/geometry/cost comments)
  + one new guard test in test_lensing_schwinger.py -> is_test_only=true,
  zero Coder WPs, per-suite Test Dev shards.
- DDWCeilingTestCase: change GEOMETRY so DD cap (58/(r_max*reach_max))
  lands <60; formula assertions are geometry-independent (its own docstring
  says "verify the FORMULA not the success rate").
- Existing slow-tier mechanism = _MPMATH_TIER_SKIP (skipUnless
  COGWHEEL_TRAIN_TIER) in test_lensing_schwinger.py:1989. Use it if a
  test genuinely needs w>60.
- M_LENS_MSUN=90 shared (marginalized imports into saddle); lens w scales
  with m_lens -> mass lever. Don't prescribe numbers; Test Dev iterates.

2026-08-06 brief_wire_interior_wedge_chart:
- from_wedge_engine + _from_wedge_fixed + wedge NPZ round-trip ALREADY
  complete in surrogate.py (build 56a223a). Brief is STALE claiming it's
  missing. Only surrogate_training.py wiring is genuinely absent.
- _interior_admission MUST BE KEPT: live exterior-tiler dependency
  (surrogate_training.py:3949) + 5 test suites. Brief's "interior-only,
  move/delete" premise is WRONG.
- _farfield_interior_tiles genuinely dead after swap -> DELETE (but ported
  by 2 test suites: exterior_windows, ppgo_bandsplit).
- Professor rulings: 1 angular column [0,pi/2] (carrier smooth through
  pi/4, empirically confirmed by test_lensing_wedge_dd_arclength), uniform
  n_per_side radial rows, r_min>0, r_extent capped below 1 by tube shell
  (leave Airy edge to tube); in-build eps gate = ABSOLUTE floor <5e-2 +
  chart-count (ffin relative baseline is driver post-build since ffin is
  deleted).
- Simplifier trims: no 2D tiler, inline/minimal radial split; no verify-
  only WP.
- Single Coder WP (all edits in surrogate_training.py; multiple WPs on
  _train_band_charts would conflict). _heldout_eps annotation add =
  biggest risk (G3).
- Gated/flip wedge tiles -> ladder-served gap (mirror LOBE, NOT ffin
  subdivision).
