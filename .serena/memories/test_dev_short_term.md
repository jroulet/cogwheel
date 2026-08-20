# Test Dev Short-Term Observations

## 2026-08-20 (grid-change prose + deep-interior gamma extension — MY work, test_lensing_part0_mechanical.py)

- WP-1 deep-interior re-bake landed: production `_diffractive.py` now has derate
  0.85 (clamped), caustic coeff -0.82368, and `_unfenced_grid_points` full branch
  radii `linspace(0.3,1.3,5)` -> `linspace(0.1,1.3,7)`.  MEASURED live: the deep
  interior is now served by the FIT (values ~2-34, all < ceiling 60) -- NOT the
  clip -- but the clamped 0.85 de-rate STILL over-serves the honest ceiling at
  the cusp direction: w_fit/w_true = 1.1244 at (gamma=0.5, theta=pi/4, rho=0.2),
  1.0233 at rho=0.3 (n_w=16 oracle).  So the zero-over-serve pin is RED BY DESIGN,
  but the mechanism changed from "clip to 60 (7x)" to "de-rate clamp (1.12x)".
- PROSE/SKIP-REASON correction (no assertion-value changes): rewrote the stale
  "UNCALIBRATED low-gamma interior -> DD-ceiling clip, ~7x over-serve" story in
  module docstring, `_DEEP_INTERIOR_GAMMAS` comment, `_BRUTE_ACCURACY_REASON`
  (+its comment), `TestWLlowFitDerateTeeth` docstring ("then clipped to CEILING"
  -> "the de-rate is the sole margin; the clip is the hard oracle-domain cap"),
  `TestWLlowFitDeepInteriorServedByFit` + `TestWLlowFitDeepInteriorHonestServe`
  docstrings.  New story: provisional bake calibrates the interior (served by the
  fit) but the clamped de-rate over-serves the cusp direction ~1.12x; flips green
  only after the DRIVER's full interior-inclusive bake.  `_CALIBRATION_S_MAX`
  comment `linspace(0.3,1.3,5)` -> `linspace(0.1,1.3,7)` (value 1.3**2 unchanged).
- EXTENDED fast-tier `test_deep_interior_served_below_ceiling_at_calibrated_cell`
  from gamma={0.5} to {0.2,0.3,0.5} (all `_DEEP_INTERIOR_GAMMAS`); removed the
  now-obsolete `_CALIBRATED_GAMMA` constant and the `_fixtures(gammas=...)`
  parameter (dead code).  Green 60 passed / 3 skipped (~17s).
- VERIFIED the gated `TestWLlowFitDeepInteriorHonestServe` is correct as written:
  ran it with COGWHEEL_BRUTE_ACCURACY=1 -> 1 failed (value pin, exactly the
  gamma=0.5/theta=0.785/rho=0.2 7.534 > 6.700 over-serve, matching my probe),
  2 passed (clip-not-mechanism + diagnostic).  No assertion edits; only prose.
- FLAGGED OUT-OF-SCOPE: the Architect's "zero-over-serve class docstring
  '[0.3,1.3] ... 252 on-grid' row-count claim" lives in
  `test_lensing_diffractive.py` (FullGridCertificateOracleTestCase docstring
  ~line 751/778, still stale "252 on-grid + 240 off-grid") -- ANOTHER run's file,
  NOT edited per scope discipline.

## 2026-08-20 (corner re-pin r=1.05->1.1 after radii change — MY work, test_lensing_diffractive.py)

- WP1 re-baked the smoke and pasted NEW provisional coefficients: derate=0.85,
  caustic coeff=-0.8236838383495544 (was -0.72669/derate 0.503444).  The full-branch
  radii changed linspace(0.3,1.3,5) -> linspace(0.1,1.3,7) = [0.1,0.3,0.5,0.7,0.9,
  1.1,1.3], DROPPING r=1.05, so CornerRawOverPredictionTestCase's premise
  (witness is an off-grid row) failed.
- Re-pinned CORNER_R 1.05->1.1 (rho~2.19, nearest surviving radius to the original
  rho~2.09; r=0.9 also survives at rho~1.79).  Both (0.5, 0.0, 0.0, r, theta=25pi/32)
  rows CONFIRMED present in live _off_grid_points('full',42) (250 rows).  Kept
  CORNER_THETA=3pi/4+pi/32; re-derived the premise from live off-grid output (added
  the radii-change note to the CORNER_R comment, did NOT delete the membership check).
- MEASURED at the new provisional coeffs: corner ratio raw_fit/w_low_true = 1.0062
  (raw 1.2302 / w_low_true 1.2226) at r=1.1 (was ~1.19x at old smoke); self-fals
  inflation (caustic coeff->0.0) = 1.9059x at r=1.1 vs 1.6155x at r=0.9 (r=1.1 gives
  stronger teeth).  Docstring "measured ~1.19x" -> "~1.01x at the current provisional
  re-baked smoke coefficients".
- VERIFIED: gated corner test GREEN with COGWHEEL_DIFFRACTIVE_FULL_BAKE=1 (both tests
  pass, ratio 1.0062 < 1.5); fast tier 34 passed / 3 skipped (~87s).
- Cross-suite requirements (extend TestWLlowFitDeepInteriorServedByFit gamma set;
  verify TestWLlowFitDeepInteriorHonestServe; clip-vs-derate prose in TestWLlowFit
  DerateTeeth) ALL target test_lensing_part0_mechanical.py -- ANOTHER run's file; NOT
  edited per scope discipline.  part0_mechanical green (60 passed, 3 skipped) as a
  no-regression check.
- Also fixed stale de-rate literal in test_removing_derate_trips_overserve docstring:
  "the fenced smoke de-rate being 0.844967" -> "the shipped de-rate being 0.85"
  (kept "~1.18x", internally consistent with 1/0.85).

## 2026-08-20 (fenced-domain grid oracle + consumer fall-through — MY work, test_lensing_part0_mechanical.py)

- WP-1 fence (RHO_LO=0.6, DELTA=0.4, shell->None DECLINE) landed; this shard
  added the two REMAINING specs to part0_mechanical.py.  Sole file, 51->62
  tests, 62 passed ~17s.
- Spec "grid generators fenced" -> TestFenceGridGenerators (3 tests): pins
  `_grid_points('full',42)` / `_off_grid_points('full',42)` return ONLY
  non-shell rows (rho outside [RHO_LO,RHO_HI], both interior+exterior
  present), that the fence drops exactly the shell rows (63/259 dropped,
  anti-vacuity), and that `_fence_excluded` single-sources `_caustic_rho` +
  `_DIFFRACTIVE_FIT_FENCE_*` (AST Name-node check, no re-typed 0.6/1.4).
  This is the UPSTREAM fact that makes the diffractive suite's
  `_full_grid_sweep` fenced; that sweep's engine-oracle re-scope is the
  DIFFRACTIVE run's (out of my scope).
- Spec "fence fall-through byte-identity" -> TestFenceFallThroughByteIdentity
  (4 tests, RUNTIME): binds REAL `_diffractive_bottom_ceiling` to an
  UNINITIALIZED `LensedRelativeBinningLikelihood` shell
  (`object.__new__`, the census's own idiom; likelihood import ~3.5s lazy in
  setUpClass), asserts a fenced lens -> None and that fence-None and
  wall-None are byte-identical DOWN TO `_band_split_mask` (same split flag,
  `np.array_equal` below-mask).  AST teeth: wrapper's try body returns
  `w_low_fit(...)` transparently, sole except is DiffractiveDomainError (no
  new exception class).  NOTE the degenerate-sqrt_mu None path is UNREACHABLE
  on positive parity (det_a=0 => gamma'=1 already refused at the wall; y=0
  raises ValueError in `_born_factors` math.log(0) BEFORE the s<=0 branch) --
  documented, pinned the two reachable None paths instead.
- Spec "census mirror fenced draw" -> TestCensusMirrorFencedDrawRouting
  (4 tests, AST+runtime): every `_result('diffractive_analytic', ...)` in
  `classify_draw` is inside an `If` guarded by `w_low is not None`
  (helper `_test_requires_w_low_not_none`/`_guarded_bodies`), so a fenced
  draw (w_low=None) can never be labelled analytic; self-falsification
  test proves the guard detector flags an unguarded synthetic route; runtime
  tie-in `w_low_fit` returns None for the mid-shell draw.
- GOTCHAS: (1) method name `_ceiling` COLLIDES with WLlowFitBaseTestCase's
  `cls._ceiling` (=DD ceiling 60.0, set in setUpClass) -- the base setUpClass
  OVERWRITES a subclass method of the same name; rename to `_wrapped_ceiling`.
  (2) `_band_split_mask` stored as a class attr is a plain FUNCTION -> binds
  `self` via descriptor protocol (the known footgun) -- call via
  `type(self)._band_split_mask`.
- OUT-OF-SCOPE (flagged, diffractive run owns): test_lensing_diffractive.py
  has 2 failures from the fence -- TruncationCertifiedBandTestCase::
  test_truncation_within_bar_over_band and CornerRawOverPredictionTestCase::
  test_dropping_caustic_feature_inflates_over_prediction (corner witness
  gamma=0.41 r=0.55 now rho=1.341 in-shell -> w_low_fit None ->
  assertIsNotNone fails).  Both are the OTHER run's re-scope (the
  FullGridCertificateOracleTestCase / corner re-point specs).

## 2026-08-20 (just-outside-shell conservative pin — MY work, test_lensing_part0_mechanical.py)

- Spec 1 (Just outside the shell is conservative) IMPLEMENTED as new class
  TestWLlowFitJustOutsideShellConservative (3 tests: conservative pin + derate
  teeth + diagnostic plot).  Engine-oracle pin: sources DERIVED at the fold
  (cusp) directions theta=7pi/32 & 25pi/32 (off-grid theta midpoints) of
  gamma=0.3, r = rho*|y_c(theta)| for rho in (1.42,1.5,1.6,1.7) just above
  RHO_HI=1.4; asserts w_low_fit <= _measure_w_low_true(n_w=16) and w_low_fit>0.
  8 engine probes (~1.2s each) paid ONCE in setUpClass, shared by all 3 tests
  (~16s total).  Measured wf/wt ~0.87-0.95 (conservative); raw(derate=1.0)/wt
  ~1.03-1.13 (teeth: raw over-serves).  File 48->51 tests, 51 passed ~17s.
- FINDING (flagged, NOT in my pin): at gamma=0.3 the SMOKE fit OVER-SERVES just
  outside the shell at the pi/2 (max-|y_c|) direction -- wf/wt = 1.08 at
  rho=1.6, 1.29 at rho=1.8 (w_fit 6.41/6.47 > w_true 5.92/5.00).  That is a
  smoke-grid COVERAGE GAP: gamma=0.3 r=0.5/0.9 at pi/2 are fenced (rho<1.4), so
  the fit extrapolates there; the FULL bake's r=1.05 row covers it.  Spec 1 is
  therefore correctly scoped to the fold direction (its example).
- Spec 3 (monotonicity re-scope) ALREADY DONE in a prior shard: Gamma/S
  monotonicity filter _fence_rho > RHO_HI (exterior only); deep-interior
  ceiling-serve asserted by TestWLlowFitDeepInteriorCeiling.  Verified green.
- Spec 2 (re-scope CornerRawOverPredictionTestCase) OUT-OF-SCOPE: it names
  test_lensing_diffractive.py's CornerRawOverPredictionTestCase (another run's
  file; scope discipline forbids editing it).  CONFIRMED the fence broke it:
  test_dropping_caustic_feature_inflates_over_prediction now FAILS in the fast
  tier (w_low_fit returns None at the corner, rho=1.34 in-shell) --
  assertIsNotNone(raw_with_caustic).  The diffractive run must re-point it
  (test_raw_fit_over_prediction_within_twofold_bar is gated and dead too).

## 2026-08-20 (near-fold fence: part0_mechanical fence pins + backward-compat repair — MY work)

- WP-1 two-sided near-fold fence landed in `w_low_fit` (RHO_LO=0.6, DELTA=0.4,
  RHO_HI=1.4; `_caustic_rho(gamma_prime, s, theta) = sqrt(s)/|caustic_point|`;
  interior->ceiling, shell->None DECLINE, exterior->fit).  It BROKE 8 pre-existing
  tests in my sole suite test_lensing_part0_mechanical.py by returning None for
  fixtures/sweeps that now fall in the shell (monotonicity sweeps at fixed y cross
  exterior->shell->interior; DerateTeeth/SelfFals/D2 fixture (0.05,0.9,g=0.45) has
  rho=1.29 in shell; wall-collapse fixture y=(0.5,0.3) becomes interior (rho<0.6)
  as gamma->wall since |yc| blows up).  Fixed all 8 + added the 2 new fence pins.
- NEW pins (Architect specs): TestWLlowFitNearFoldFence (3 tests) sweeps (theta,r)
  across the directional caustic at gamma=0.41 beta=kappa=0, derives rho live via
  `_caustic_rho`, asserts None EXACTLY where rho in [RHO_LO, RHO_HI] with served
  values on both sides (non-vacuity) + diagnostic rho-vs-return scatter PNG + teeth
  (patch DELTA=-0.4 collapses shell -> mid-shell point served).
  TestWLlowFitDeepInteriorCeiling (3 tests): DERIVED fixtures at fixed witness rho
  {0.3,0.5} (r = rho*|yc(theta)| live), asserts == _DIFFRACTIVE_FIT_CEILING (or w_hi
  cap), never None; teeth patch RHO_LO=0 -> deep interior now declined (assertNotEqual
  raw vs ceiling proves interior branch load-bearing).  D2 verify: fixture 3 moved
  (0.05,0.9,g=0.45)->(0.8,0.6,g=0.35,beta=kappa=0) (at g=0.45 no r~0.8-1.1 source is
  rho>1.4 for BOTH the base and its pi/2 image -> fence would None the self-fals
  image); added test_fixtures_are_outside_the_shell premise.
- Backward-compat repairs: GammaMonotonicity re-scoped to served exterior (filter
  `_fence_rho > RHO_HI`, monotone 0 viols); SMonotonicity re-scoped to exterior
  falling branch (exterior start -> live argmin) -- the PROVISIONAL fence-smoke
  coeffs (poly log(s)^2 = +0.1517) reintroduce a ~1-4% large-s up-turn past the
  minimum, so the "no up-turn to s_max" claim is DROPPED (documented; full bake
  expected to remove it; conservativeness is separately certified by the diffractive
  full-grid oracle).  never_exceeds_ceiling now `value is None or <= ceiling`.
  wall_collapse_monotone_and_finite RETIRED -> test_wall_serves_ceiling_or_declines
  (fence makes the fit's log(1-gamma') collapse unreachable: fixed source goes deep
  interior as |yc| blows up).  SelfFals pi/4 + DerateTeeth fixture 3 moved out of shell.
- Shared helper: WLlowFitBaseTestCase.setUpClass now also loads `_module` + `_rho_lo`/
  `_rho_hi`; added `_fence_rho(y,gamma,beta,kappa)` (mirrors w_low_fit's discriminator).
  48 passed ~4.7s.

## 2026-08-20 (INS-2-001 corner-pin re-point to resonance-limited 2.0x bar — test_lensing_diffractive.py)

- INS-2-001 resolved (corner pin + teeth alignment).  The corner raw-over-prediction
  pin now asserts the RESONANCE-LIMITED twofold bar: assertLess(ratio, 2.0) with a
  message citing INS-1-001 (de-rate ~0.5), replacing the abandoned 1.5x/1.43x/0.70
  target.  MEASURED ratio at the CURRENT smoke coefficients: 1.9863 (raw_fit 6.8656
  vs w_low_true 3.4564, un-clipped vs ceiling 60) -- ALREADY inside the 2.0 bar, so
  the pin is green-with-gate-lifted even at smoke and flips/keeps green post-bake.
- Renamed test_raw_fit_over_prediction_below_150_percent ->
  test_raw_fit_over_prediction_within_twofold_bar (old name contradicted the 2.0 bar).
  Own skip reason constant _CORNER_RAW_OVER_PREDICTION_REASON (NOT the shared
  _COGWHEEL_DIFFRACTIVE_FULL_BAKE_REASON, whose 'red in-build BY DESIGN' claim is
  full-grid-sweep-specific and would have mislabelled the pin as a red-until-bake
  gate).  Class docstring rewritten so docstring/skip-reason/assertion agree the pin
  WILL satisfy < 2.0 post-bake (0.70/1.43 abandoned per Professor-backed resonance
  ruling; a future dense-w-scan measurement-robustness fix is the real path to a
  tighter cert).
- Teeth alignment (INS-2-003 interaction): the teeth test test_dropping_caustic_
  feature_inflates_over_prediction ALREADY used the monotone comparison
  raw_nocaustic > raw_with_caustic * 1.05 (not a fixed '> 1.5' bar) -- satisfies the
  Inspector's suggested alternative; measured margin 1.238x vs the 1.05 floor.
  Updated its docstring parenthetical: with the twofold bar, the with-caustic ratio
  1.986 is just UNDER the pin, so a bare 'above-the-pin' comparison WOULD be
  satisfiable by a no-op surface -- the monotone form is load-bearing.  Teeth probe:
  dropping the caustic coeff gives ratio 2.459 > 2.0 -> pin goes RED when the feature
  is inert (assertion load-bearing, verified live).
- All premise checks kept intact (off-grid midpoint witness via script._off_grid_
  points('full', 42), w_low_true not None and > 0, raw_fit < ceiling, n_compared += 1).
  Backward-compat grep: no other test file pins the retired 1.5/1.43/0.70 target
  (all 0.70/1.5x hits elsewhere are unrelated gamma/threshold literals).
- VERIFIED: default fast tier = 33 passed / 3 skipped / 0 failed (3 distinct skip
  reasons, all accurate); with COGWHEEL_DIFFRACTIVE_FULL_BAKE=1 the corner-pin class
  is 2 passed.  NOTE: serena insert_at_line can land mid-block when the target line
  index is the LAST line of a preceding multi-line tuple -- verify surrounding
  structure after insert (caught a displaced closing line and repaired).

## 2026-08-20 (D2 symmetry re-scope — test_lensing_part0_mechanical.py)

- D2 symmetry re-scope (WP-1 even-harmonic basis landed): renamed
  `TestWLlowFitFourFoldSymmetry` -> `TestWLlowFitD2Symmetry` and
  `_FOURFOLD_TOL` -> `_D2_TOL`.  Replaced the retired `test_pi2_rotation_
  invariance` (pinned the WRONG cos(4k theta) 4-fold symmetry) with THREE
  tests using EIGENFRAME transforms via new `_eig_z`/`_from_eig` helpers
  (complex `z_eig = exp(-i beta) y'`; theta+pi = -z, reflection = z.conjugate(),
  pi/2 = 1j*z): test_period_pi_invariance + test_reflection_invariance (both
  to ~1e-15, pass) and test_pi2_rotation_changes_value (self-falsification,
  diff ~0.075-0.42).  41 passed.
- EMPIRICAL CORRECTION TO THE OTHER RUN'S MEMORY CLAIM: `|y_c(theta)| =
  |geometry.caustic_point(gamma_prime, theta)|` is period-PI and
  reflection-symmetric (D2), NOT period-pi/2.  The astroid caustic SET is
  4-fold symmetric, but the critical-ANGLE parametrisation is only 2-fold:
  under theta->theta+pi/2 the `gamma_prime cos(2 theta)` term in effective_u
  flips sign, so |y_c| changes (measured diff ~0.02-0.47 at gp in 0.15-0.45).
  So the pi/2 non-symmetry comes from BOTH the odd harmonics AND the caustic
  feature, not just the odd harmonics.  Teeth check: zeroing the odd harmonics
  (a_1,a_3,a_5,a_7) still leaves pi/2 diff ~0.42-0.65 (caustic feature alone
  breaks pi/2).  Docstrings written to the accurate D2 story.
- Corner raw-over-prediction pin NOT duplicated: it already lives in
  test_lensing_diffractive.py (CornerRawOverPredictionTestCase, other run's
  suite) and is RED BY DESIGN (1.986x > 1.5, de-rate 0.5034 = 1/1.986) --
  flips green when the driver's FULL bake (de-rate >= 0.70) lands.  part0_
  mechanical stays engine-free.
- Monotonicity re-verification (cross-suite): GammaMonotonicity / SMonotonicity
  / CeilingCapAndWallCollapse / DerateTeeth all PASS with provisional
  coefficients (caustic coeff -0.72669 < 0 reinforces wall-collapse + falling
  s-dependence) -- no re-derivation needed; the peak bounds are already
  LIVE-derived.  No other test file pins the retired 4-fold w_low_fit symmetry
  (the "4-fold" hits in ppgo_map / caustic_cusps / interior_wedge_chart are
  about the astroid caustic's geometric symmetry, unrelated to the fit basis).

## 2026-08-20 (WP-1 even-harmonic + caustic corner pin + off-grid oracle gating — test_lensing_diffractive.py)

- Sole owned suite test_lensing_diffractive.py (36 tests). WP-1 (even-harmonic
  basis cos(2k theta) k=1..7 + parametric-caustic feature) landed with
  PROVISIONAL smoke coefficients: de-rate 0.503444, caustic coeff -0.72669,
  provenance SHA 3827f48 (corner raw over-prediction 1.986x at gamma=0.41
  r=0.55 theta=3pi/4+pi/32=2.454369).
- Off-grid midpoint zero-over-serve oracle (spec 1): extended
  `_full_grid_sweep` to iterate BOTH `_grid_points('full',42)` (252 on-grid)
  AND `_off_grid_points('full',42)` (240 theta-MIDPOINT probes), tagging rows
  with an `off_grid` bool (9-tuple).  The on-grid-only sweep was BLIND to the
  sub-grid caustic dip (passed GREEN while the smoke surface over-served
  off-grid).  Gated the ENTIRE zero-over-serve sweep + diagnostic plot behind
  COGWHEEL_DIFFRACTIVE_FULL_BAKE=1 via `@unittest.skipUnless` with a LOUD
  reason (can only pass with FINAL driver-baked coefficients; smoke coeffs are
  de-rated over the smoke grid only).  Added `import unittest` (file used
  `from unittest import TestCase, main, mock`).  MEASURED the extended off-grid
  sweep: 240 off-grid rows, 2 marginal over-serves (worst rel 1.00000005e-4 vs
  bar 1e-4) at (0.41, 0.55, theta~5.596) -- red-with-provisional, green once
  the full bake lands.  The derate falsification (test_removing_derate_trips_
  overserve) stays UNGATED in the fast tier (~39s measured, independent of
  final coefficients).
- Corner raw-over-prediction pin (cross-suite, NEW permanent): added
  CornerRawOverPredictionTestCase (2 tests).  Measures w_low_true at
  (gamma=0.41,kappa=0,beta=0,r=0.55,theta=3pi/4+pi/32) via script
  _measure_w_low_true(n_w=16) and raw_fit via w_low_fit with
  _DIFFRACTIVE_FIT_DERATE patched to 1.0; asserts raw_fit/w_low_true < 1.5
  (driver target <=1.43 = de-rate>=0.70).  MEASURED 1.986x -> RED BY DESIGN
  at authoring time (the smoke de-rate 0.5034 = 1/1.986 does NOT meet the
  driver's 0.70 target); flips green with zero edits when the driver's FULL
  bake lands.  Kept honest (no expectedFailure).  Teeth test patches
  _DIFFRACTIVE_FIT_CAUSTIC_COEFF to 0.0 -> ratio 2.459 > 1.5 (green now),
  proving the negative caustic coeff is load-bearing.  Premise assertions:
  corner witness IS in the script's off-grid set; raw_fit < ceiling (not
  clipped); caustic coeff < 0.
- CRITICAL _measure_w_low_true n_w SENSITIVITY (non-monotone rel): at the
  corner, rel(w) has a REAL truncation bump (peak 1.22e-4 > bar 1e-4 at
  w~3.48, dips back below until ~6.4, then rises monotonically).  n_w=16
  bisection LANDS on the bump -> w_low_true=3.456 (first breach, CORRECT
  conservative); n_w=48 bisection SKIPS it -> 6.54 (last breach, WRONG
  over-optimistic).  The bake's default n_w=16 is therefore the CORRECT
  choice here (matches the certificate's prefix-closed semantics); a higher
  n_w silently over-certifies.  Documented in the pin's docstring.
- CROSS-SUITE BACKWARD-COMPAT FINDING (NOT mine, flagged): WP-1's even-harmonic
  basis breaks test_lensing_part0_mechanical.py::TestWLlowFitFourFoldSymmetry
  ::test_pi2_rotation_invariance (1 failed: pi/2 rotation diff 0.424 vs tol
  2.3e-12) -- the OLD cos(4k theta) basis was 4-fold symmetric, the NEW
  cos(2k theta) basis (odd k) is only 2-fold (pi-periodic) BY DESIGN.  The
  caustic feature is 4-fold symmetric (astroid |y_c| period pi/2), so the
  break is purely from odd harmonics.  Owner must REWRITE the 4-fold test to
  2-fold (theta->theta+pi), NOT retire it (physical: F(-y)=F(y) => pi-periodic
  honest ceiling).  I do NOT edit test_lensing_part0_mechanical.py.
- Monotonicity re-verification (cross-suite): TestWLlowFitGammaMonotonicity /
  SMonotonicity / CeilingCapAndWallCollapse / DerateTeeth all PASS (9/9) with
  the provisional coefficients -- no re-derivation needed (caustic coeff
  negative reinforces wall-collapse and the falling s-dependence past the
  small-s peak; gamma-monotonicity holds).
- File final state: 33 passed, 2 skipped (gated sweep+plot, loud reason), 1
  failed (corner pin, RED by design).  My ONLY change is
  test_lensing_diffractive.py; production _diffractive.py / geometry.py /
  fit_diffractive_certificate.py are the WP-1 coder's (pre-existing).

## 2026-08-19 (INS-3-002 diffractive full-grid oracle + SMonotonicity closure — MY work)

- Added `FullGridCertificateOracleTestCase` (3 tests) to test_lensing_diffractive.py:
  re-runs served-vs-engine relerr over the calibration script's OWN grid
  (`importlib.util.spec_from_file_location` on scripts/fit_diffractive_certificate.py —
  single source of truth, grid can't drift; NOT a package, no __init__.py).
  (a) `_calibration_rows`: sub-samples the 8 grid thetas to every-other {0,pi/4,pi/2,3pi/4}
  (fit's only angular model is cos(4k theta), period pi/2 -> 2 distinct harmonic classes;
  2 physical orientations each; keeps all (gamma,r) corners + 12 random rows; 252->132 rows).
  (b) probes at w=w_low (literal boundary) AND 0.9*w_low (interior), both <= CERTIFICATION_BAR;
  FINAL design keeps the FULL 252-row grid (no theta sub-sampling — measured ~35s < 60s
  single-test ceiling, so the finding's fallback is not triggered); `_full_grid_sweep` is
  functools.lru_cache(maxsize=1) so assert+plot tests share ONE engine sweep (~35s paid once);
  falsification uses its OWN loop under mock.patch derate=1.0 (early-exits ~10s).
  (c) rows where diffractive_amplification raises HypergeometricDomainError at w_low are
  counted as domain-refused (certificate OVER-REACH, not over-serve) and must stay a strict
  minority (< len(rows)); premise len(rows)>50 guards vacuity.
- MEASURED (was smoke-baked derate 0.85 + old coeffs): full 252-row grid over-serves
  124/232 measurable rows, worst rel=97.5 (gamma=0.32 r=0.3); 20 rows domain-refuse
  (w_low*r>=60, kernel cap). After full re-bake (derate 0.745168, provenance SHA 7eeedee):
  ALL 252 rows measurable, 0 over, worst rel 9.99998e-5 (worst-margin row gamma=0.5 r=1.3
  w_low=0.66; ULP-scan of every coefficient +-1 ULP does NOT flip it -> knife-edge is stable).
  Cost: full 252-row sweep ~35s (series+engine ~70ms/row); sub-sampled 132-row ~18s.
- Re-scoped TestWLlowFitSMonotonicity (part0_mechanical): re-baked surface is NOT monotone in
  s — RISES for s<~0.1 (small-s peak, degree-2 poly extrapolating below the r>=0.3 grid),
  falls monotonically through s=1.69 (=r_max^2, _CALIBRATION_S_MAX). Old docstring flagged a
  LARGE-s up-turn at s~0.5-0.6 (smoke's POSITIVE log(s)^2 coeff) — the re-bake eliminated it
  (new coeff idx7 log(s)^2 = -24.07). New test: derive peak from LIVE surface, assert
  non-increasing past it + premise ss[peak]<0.4 (former up-turn region now falls). RED at
  smoke (up-turn), GREEN at re-baked.
- Re-scoped TestWLlowFitGammaMonotonicity: re-baked surface rises for gamma in [0.05,~0.062]
  (low-calibration-edge extrapolation bleed), falls past it to the wall. New test derives the
  peak live, asserts non-increasing past it; sweep starts at gamma=0.05 (calib low edge).
- RETIRED test_small_gamma_returns_cap (1 test): re-baked surface maxes at 51.4 < ceiling 60
  — NO fixture returns the cap (the old premise pinned the smoke clip). Cap invariant still
  covered by test_never_exceeds_ceiling; clip is defense-in-depth vs raw fit (raw max ~71).
- VERIFICATION (item 3): at smoke/stale constants -> exactly 2 REDs (grid overserve +
  SMonotonicity up-turn), 71 pass; at re-baked constants -> 73/73 GREEN. Production
  _diffractive.py RESTORED byte-identical to smoke state (coder INS-3-001 owns the re-bake).
  Grid test + SMonotonicity are RED at current working tree BY DESIGN; flip GREEN when the
  re-bake lands. Fit-script emission (from my full-scale run at SHA 7eeedee): derate 0.745168,
  236/236 conservative + tight, worst ratio 1.0000, median 0.7506, p90 0.8852.


## 2026-08-19 (diffractive certificate-fit: part0_mechanical DERATE-TEETH — 4th shard)

- Fourth shard of test_lensing_part0_mechanical.py (sole owned suite): implemented
  the SELF-FALSIFICATION (teeth) spec (perturb the de-rate D to 1.0 / inflate a
  coefficient -> the conservative/tight oracle pin goes RED).  Added
  TestWLlowFitDerateTeeth (4 tests) + `_load_diffractive_module()` helper (importlib
  of _diffractive for mock.patch targets) + `from unittest import mock`.  36->40
  tests, green ~4.6s.
- TEETH MECHANISM: `_DIFFRACTIVE_FIT_DERATE=0.85` is a module GLOBAL read at call
  time by w_low_fit (`w_fit = _DIFFRACTIVE_FIT_DERATE * math.exp(fitted)`), so
  `mock.patch.object(module,'_DIFFRACTIVE_FIT_DERATE',1.0)` reaches the production
  path (function.__globals__ IS the module dict).  Same for the tuple constant
  `_DIFFRACTIVE_FIT_POLY_COEFFS` (zip over poly features).  Probe-measured: served
  == 0.85 * raw EXACTLY at un-clipped points (base=1.5891, noderate=1.8695,
  inflate-const+0.5=2.620 = *exp(0.5)).  If D were shipped as 1.0, THREE tests go
  red: test_derate_is_a_genuine_margin (assertLess(D,1)), test_derate_is_sole_
  source_of_margin (assertGreater(raw,served) fails since raw==served), and
  test_no_derate_inflates_the_ceiling.
- FIXTURE MUST BE UN-CLIPPED or the teeth are masked: w_low_fit clips
  `min(w_fit, _DIFFRACTIVE_FIT_CEILING=60)`; at a clipped point (small gamma/large
  s) D=1.0 and coefficient inflation change nothing and `assertGreater` fails
  vacuously.  Used interior fixtures (y=(0.4,0.7),g=.3 / (1.1,-.2),.15 / (.05,.9),.45)
  where raw<<60; asserted `served<ceiling` and `raw<ceiling` as PREMISES in each
  loop iteration (derived-from-boundary, not pinned literals).
- STILL BROKEN (pre-existing, OWNED BY ANOTHER RUN): test_lensing_diffractive.py
  fails at collection `ImportError: cannot import name 'diffractive_w_low'` — the
  WP2 rename diffractive_w_low->w_low_fit never updated it.  NOT my scope; flagged
  again.

## 2026-08-19 (diffractive certificate-fit: part0_mechanical RUNTIME specs — 3rd shard)

- Third shard of test_lensing_part0_mechanical.py (sole owned suite): added the
  3 RUNTIME specs the Architect assigned (MONOTONICITY IN gamma' AND s, 4-FOLD
  ANGLE SYMMETRY, CEILING CAP AND WALL-AWARE COLLAPSE) as numerical tests, NOT
  AST.  These are pure-function behavior specs of w_low_fit (O(1), engine-free),
  so they belong here despite the file's "no lensing import" character; kept the
  mechanical tests import-free via a LAZY `_load_w_low_fit()` + WLlowFitBase
  TestCase.setUpClass import.  Added numpy top-level import.  27->36 tests (9 new:
  GammaMonotonicity, SMonotonicity, FourFoldSymmetry, CeilingCapAndWallCollapse
  (4 methods), SelfFalsification (2)).  Green 36 passed ~4.3s.
- GOTCHA: a plain FUNCTION stored as a CLASS ATTRIBUTE binds to the instance via
  the descriptor protocol — `self._w_low_fit(y, gamma, beta, kappa)` prepends
  `self` and raises "TypeError: w_low_fit() takes from 2 to 4 positional
  arguments but 5 were given".  Fix: call `type(self)._w_low_fit(...)`.  (Classes,
  floats, tuples stored as class attrs do NOT bind — only functions.)  This is the
  same footgun to watch in any TestCase that caches a module function on the class.
- SPEC DISCREPANCY FLAGGED: "non-increasing in s" is FALSE for the baked fit.
  `_DIFFRACTIVE_FIT_POLY_COEFFS` monomial (0,2,0) = +0.4615 (positive log-s^2
  coefficient) makes w_low_fit U-SHAPED in s: decreases to a minimum at s~0.5-0.7
  (gamma'-dependent), then INCREASES (e.g. gamma=0.3: s=0.75->6.46, s=1.33->8.10,
  s=1.78->10.2).  The derate keeps it conservative on the calibration grid
  (r=sqrt(s) in [0.3,1.3]), so it's a fit-LOOSENESS artifact not an over-serve, but
  the spec's "no local spikes" is contradicted WITHIN the calibration domain.
  Scoped the s-monotonicity test to the small-s decreasing branch (s<=0.4, gamma'
  in [0.1,0.5]) and documented the U-shape; gamma' monotonicity holds exactly (0
  viols over 4 (beta,kappa) configs).  4-fold symmetry holds to ~6e-15 (pi/2
  rotations exact via -y1,y0 swap); pi/4 rotation changes value by ~0.29 (self-fals
  teeth).  Ceiling cap = W_CEILING_SCHWINGER=60; small gamma returns exactly 60;
  wall collapse monotone to ~5e-11 at gamma=0.9949, raises DiffractiveDomainError
  at exactly 1-DELTA_GAMMA_P=0.995.
- STILL BROKEN (pre-existing, OWNED BY ANOTHER RUN): test_lensing_diffractive.py
  fails at collection `ImportError: cannot import name 'diffractive_w_low'` (line
  88) — the WP2 rename diffractive_w_low->w_low_fit never updated it.  Re-confirmed
  this shard; NOT my scope.

## 2026-08-19 (diffractive certificate-fit: part0_mechanical structural purity)

- Second shard of test_lensing_part0_mechanical.py (sole owned suite). The
  prompt's 3 specs (BAND SEMANTICS PRESERVED, REFUSAL BOUNDARY, ENGINE-FREE
  PURITY) are RUNTIME specs naming w_low_fit/_diffractive_bottom_ceiling and
  rewriting classes (BandFloorAndWholeBandAdmissionTestCase,
  NestedNullSplitByteIdentityTestCase, WrapperFidelityBandRoutingTestCase,
  WallRefusalTestCase) that live in test_lensing_diffractive.py -- which is
  OWNED BY A DIFFERENT RUN and is BROKEN at collection
  (ImportError: cannot import name 'diffractive_w_low', line 88).
  Flagged the runtime value/behavior parts out-of-scope (per the SPEC SPLIT
  precedent) rather than duplicating them into a "no lensing import" AST
  file.  Implemented the STRUCTURAL core instead as 2 new AST classes + 3
  self-falsification tests (18->27): TestDiffractiveFitStructuralPurity
  (w_low_fit defined; retired scan symbols diffractive_w_low/_rootfind_w_low/
  _rootfind_w_high/_honest_tail_ratio/_DIFFRACTIVE_CERT_SAFETY/_CERT_REFERENCE_W
  have no Name-node ref in _diffractive.py or likelihood.py; w_low_fit body has
  no loop and no engine/kernel/mpmath call via _called_identifiers/_has_loop)
  and TestDiffractiveBottomCeilingWrapper (calls w_low_fit not the retired scan;
  try/except DiffractiveDomainError -> return None).  Green: 27 passed in ~0.9s.
  NOTE: _reduced_shear raises DiffractiveDomainError at lam<=0 or
  |gamma'|>=1-DELTA_GAMMA_P; the ONLY remaining 'diffractive_w_low' mentions in
  production are w_low_fit's docstring (lines 352/378/402) -- strings, not Name
  nodes, so _all_name_ids correctly skips them.

- diffractive_w_low honest up-bracket (WP1) assumes `_honest_tail_ratio` is
  monotone in w, but it is NOT at gamma=0.1/Y_REF=(0.8,0.4): breaches BAR near
  w~12.4 (1.148e-4), DIPS under ~13.0-13.6 (8.5e-5), breaches again ~13.9.
  Engine oracle agrees to 6 sig figs (real truncation-error dip, not estimator
  artefact). `_rootfind_w_high` returns the LAST crossing (13.9) -> over-
  certifies a band containing the 12.4 breach. Broke TWO pre-existing engine-
  oracle soundness tests (TruncationCertifiedBandTestCase, KappaEngineOracle
  TestCase) -- both RED, escalated, not weakened. New suites in
  test_lensing_diffractive.py: CeilingTightnessTestCase (crossing scoped to
  monotone gammas 0.03/0.2/0.3; witness pins gamma=0.1 dip + over-cert),
  CeilingMonotonicityTestCase (w_new>=candidate, ~485x gain at 0.03),
  BandFloorAndWholeBandAdmissionTestCase (floor-fail->None via CLEAN_GAMMAS,
  NOT OPTIMISTIC -- floor check only fires on the candidate-clears branch).
  `_closed_form_candidate` helper reproduces production candidate bit-exactly
  (feeding w_hi=candidate returns candidate).

## 2026-08-19 (diffractive certificate-fit: w_low_fit replaces diffractive_w_low)

- Part0 mechanical update (test_lensing_part0_mechanical.py, sole owned suite): WP2
  RENAMED `diffractive_w_low` -> `w_low_fit` (pure O(1) fit, baked coeffs) and
  DELETED `_DIFFRACTIVE_CERT_SAFETY` + `_CERT_REFERENCE_W` + `_rootfind_w_low`/
  `_rootfind_w_high`/`_honest_tail_ratio`.  The absorber allowlist had one stale
  entry `('.../_diffractive.py','_DIFFRACTIVE_CERT_SAFETY')` -- removed it.
  New fit constants (_DIFFRACTIVE_FIT_DEGREE/_POLY_COEFFS/_N_HARM/_HARMONIC_COEFFS/
  _DERATE/_LIP/_CEILING) do NOT match `_ABSORBER_PATTERN`
  (_EPS|_MARGIN|_FRAC|_STANDOFF|_SAFETY), so NO new allowlist entry needed despite
  the spec's "add _DIFFRACTIVE_FIT_DERATE/_CEILING to the absorber list" -- the
  spec named those as if absorber-shaped but they aren't.  Suite green (18 passed).
- SPEC SPLIT NOTE: the same test-dev prompt bundled THREE specs, but only the
  MECHANICAL one maps to test_lensing_part0_mechanical.py (pure AST/text, no
  lensing import).  The ORACLE CONSERVATIVE/TIGHT PIN ("rewrite
  TruncationCertifiedBandTestCase, extend to the fit") and END-TO-END WITHIN-BAR
  specs name `_engine_reference`/`_engine_reference_kappa`/`Y_REF`/`Truncation
  CertifiedBandTestCase`, all of which live in test_lensing_diffractive.py (owned
  by a DIFFERENT test-dev run per scope discipline "write ONLY part0_mechanical").
  Flagged out-of-scope rather than duplicating engine-oracle tests into a
  "mechanical" AST-scan file.

## 2026-08-19 (diffractive_w_low honest-ceiling: 0.9 sweep re-target + wrapper fidelity)

- 0.9*w_low CEILING-MARGIN RE-TARGET (spec "ENGINE HONESTY OVER THE SERVED
  BAND"): sweeping the engine-oracle truncation check to [w_lo, w_low] was RED
  at the ceiling (N/2N estimator has ZERO margin at w==w_low, so ~1.0001e-4 >
  1e-4 on float round-off). Re-targeting `_band_worst_relerr` /
  `_band_worst_relerr_kappa` to `linspace(w_lo, 0.9*w_low, N_BAND)` makes every
  monotone draw pass (max ~7.4e-5). BUT it does NOT fix (kappa=0, gamma=0.1,
  beta=0.0): its non-monotone first breach sits at ~0.8995*w_low, INSIDE the
  0.9 cut (1.1435e-4). That draw is the KNOWN non-monotone over-certified
  geometry already pinned by CeilingTightnessTestCase's witness, so it is
  EXCLUDED from the sweep via a module constant `NONMONOTONE_DRAW = (0.1, 0.0)`
  (breach is beta-specific: (0.1, 0.7) and (0.1, -1.1) are monotone and pass).
- WRAPPER FIDELITY pin: `_diffractive_bottom_ceiling(lens, w_lo, w_hi)` ==
  `diffractive_w_low((y1,y2), gamma, beta, kappa, w_lo=w_lo, w_hi=w_hi)`
  exactly (the wrapper adds only DiffractiveDomainError->None at the saddle
  wall). Give the pass-through TEETH by including band combos that CHANGE the
  result (whole-band-clear w_hi=3.0 -> returns w_hi not the unbounded ceiling;
  floor-above-ceiling w_lo=10.0 -> None) — else a wrapper that silently drops
  w_lo/w_hi passes vacuously.
- WP2 (thread band through `_diffractive_bottom_ceiling`, 3-arg call sites
  `(lens, dense.min(), dense.max())`) BREAKS test_lensing_born_certificate.py:
  its `_make_probe`/`_make_floor_probe` stub it with 1-arg lambdas
  (`lambda lens: None` at 3 sites, lines 744/808/1725) -> TypeError "takes 1
  positional argument but 3 were given" at likelihood.py:3080 -> 20 failed +
  1 error. CROSS-SUITE backward-compat break from the PRODUCTION change, NOT my
  test edits; FLAGGED (born_certificate owned by another test-dev run). Fix
  pattern for owner: widen stubs to `lambda lens, w_lo=None, w_hi=None: ...`.
  `test_lensing_born_analytic_reachability.py` binds the REAL method (not a
  lambda) so it stays green (30 passed).
