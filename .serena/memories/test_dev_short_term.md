# Test Dev Short-Term Observations

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
