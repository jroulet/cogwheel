# Inspector Short-Term — 2026-08-20 diffractive_certificate_fit_fenced (pass 3, final)

Scope: re-review after Coder resolved INS-2-001/002/003. Ran both changed fast-tier
suites (test_lensing_part0_mechanical.py 61 pass; test_lensing_diffractive.py 34 pass /
3 skip, all 3 skips COGWHEEL_DIFFRACTIVE_FULL_BAKE-gated). Verified fence math + the
interior over-serve with engine-free and engine probes.

## RESOLVED (confirmed)
- INS-2-002: test_removing_derate_trips_overserve docstring now ~1.18x / 0.844967. FIXED.
- INS-2-003: TestWLlowFitDeepInteriorCeiling deleted (branch removed => "delete the class"
  path taken); replaced by TestWLlowFitDeepInteriorServedByFit. FIXED.
- The hard-coded `if rho < RHO_LO: return CEILING` branch is GONE (w_low_fit ~line 532).
  Consumer passthrough correct: _diffractive_bottom_ceiling returns w_low_fit transparently,
  single except DiffractiveDomainError; census mirror guards `w_low is not None and
  float(w_low) > w_lo`. Fence None falls through byte-identically. _caustic_rho DRY-single-
  sourced (identical to old _fit_features caustic_feature arithmetic).

## NOT RESOLVED — INS-2-001's over-serve persists via a NEW mechanism (INS-3-001, blocker)
Removing the interior ceiling branch did NOT fix the interior over-serve — it moved it into
the fit+clip. The provisional smoke fit is calibrated on ONE interior cell (gamma=0.5, r=0.3)
and over-runs everywhere else in the deep interior, clipping to _DIFFRACTIVE_FIT_CEILING=60.
Measured (w_low_fit vs _measure_w_low_true, n_w=16):
  gamma=0.2 rho=0.3 (r~0.11): 60 vs 33.97 (1.77x)
  gamma=0.3 rho=0.3 (r~0.09): 60 vs 20.50 (2.93x)
  gamma=0.3 rho=0.5 (r~0.16): 60 vs 22.04 (2.72x)
  gamma=0.5 rho=0.3 theta=pi/4 (r~0.17): 15.58 vs 6.14 (2.54x)
STRUCTURAL: these cells live at r<0.3, below BOTH the smoke grid (radii 0.5/0.9 + r=0.3
cell) and the full grid (_unfenced_grid_points radii linspace(0.3,1.3,5) + rand 0.3..1.3).
The deep interior at low gamma (gamma<=0.3) needs r<0.22 but the grid r-floor is 0.3, so
NO bake (smoke OR full) samples it. The docstring claim "the driver's full bake calibrates
those cells" is FALSE. The gated FullGridCertificateOracleTestCase sweeps the same
r in [0.3,1.3] grid, so it would ALSO miss this. w_low_fit's own docstring ("the fit is
calibrated on the interior cell, so it serves it conservatively -- NOT at the ceiling") is
contradicted: at gamma=0.2/0.3 the interior IS served at the ceiling (60).

## TRIVIAL/DESIGN
- INS-3-002: TestWLlowFitDeepInteriorServedByFit.test_deep_interior_served_not_declined
  asserts only `not None` for gamma=0.2/0.3 (value=60) — it certifies "served" but not
  "served conservatively", and its docstring admits "the provisional fit over-runs and is
  clipped to the ceiling". Green-test-advertises-overserve pattern (A GREEN test can be the
  finding). Fix with INS-3-001: either decline the deep interior too, or extend the grid
  below r=0.3 AND add an engine-backed conservative assertion.

## NEW BUG PATTERN
- FENCE MOVES THE OVER-SERVE, DOESN'T KILL IT: removing a hard-coded "serve at ceiling"
  branch and letting a fit serve a region only fixes the over-serve if the fit is actually
  CALIBRATED+CONSERVATIVE across the whole served region. Here the fit over-runs and the
  min(·, ceiling) clip re-serves 60. Rule: when a "serve-at-ceiling" branch is removed in
  favour of "the fit serves it", verify the fit's raw value at the previously-flagged point
  is BELOW the honest ceiling — not just that the branch is gone. Also: a "full bake will
  fix it" mitigation is FALSE unless the full grid's coordinate range actually covers the
  over-serve region — check the grid's r/gamma floors against the over-serve witness's
  coordinates, don't trust the prose.
