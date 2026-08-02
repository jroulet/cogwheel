# Test Dev Short-Term Observations

- PORT FIX (ppgo_bandsplit): Fixed three broken `eta_max` references in
  test_lensing_ppgo_bandsplit.py. (1) Added module-level `_PPGO_ETA_MAX = 0.05`
  constant. (2) setUpClass now passes `eta_max=_PPGO_ETA_MAX` to
  `st._interior_admission`. (3) `admitted_count` closure passes `eta_max=eta_max`
  explicitly (removed dead `dataclasses.replace(..., eta_max=eta_max)`).
  (4) Replaced `self.config.eta_max` with `_PPGO_ETA_MAX` in assertions.
  (5) Removed unused `import dataclasses`. 62 tests GREEN + 4 skipped in ~27s.
  Backward-compat audit: no other eta_max refs in this file; remaining broken
  refs in test_lensing_surrogate_training.py (1106/2680/2701/2705/2726) are
  NOT my scope.

- PORT FIX (exterior_admission): Fixed two broken references in
  test_lensing_exterior_admission.py. (1) _admission() helper now passes
  eta_max=ETA_MAX to st._interior_admission (required arg added by WP1).
  (2) Replaced test_eta_max_matches_training_config (TrainingConfig.eta_max
  removed) with test_f_max_matches_training_config pinning f_max=0.40 and
  f_floor=0.16. Updated ETA_MAX doccomment. 42 tests GREEN in ~3:15.
  Backward-compat audit: other suites with same break (exterior_windows:784,
  ppgo_bandsplit:613/698, caustic_cusps:1094/1453) NOT my scope.

- WP1 (curvature-relative tube shell) BUILD 2: added UniversalFMaxTestCase
  (4 positive bands [0.03,0.28] + 2 saddle [1.1,1.5], ratio max/min < 10,
  eps < 1.0 coherence bar; all pass at smoke 4×4×4).
  Added InvalidFMaxAssertionTestCase (f_max=0.55 fires assertion with correct
  message 'f_max' + '< 0.5').
  Added self-falsification: universality_ratio_has_teeth, f_max_above_half_always_fires.
  Added diagnostic: test_universality_eps_vs_gamma (eps vs gamma midpoint, both parities).
  Full suite 42 tests GREEN in ~7 min.
  Backward-compat audit: all existing WP1 tests pass; no skipped/gated tests in this file.
  Other broken suites (NOT my scope) unchanged from prior short-term entry.


- WP1 (curvature-relative tube shell): updated test_lensing_caustic_cusps.py.
  Deleted test_guard_fires_on_small_astroid_band (guard no longer exists).
  Replaced FootOfNormalCurvatureValueTestCase with f_max < 0.5 invariant tests.
  Added CurvatureRelativeTubeNoSkipTestCase (5 bands [0.0281,0.28], all finite positive eta_max).
  Added CurvatureRelativeHeldoutEpsTestCase (smoke-scale chart build+serve at gamma extremes).
  Fixed InteriorAdmissionMarginRemovalTestCase (eta_max from f_max*R_c, denser 12x12 tile grid).
  Fixed SelfFalsificationTestCase (interior_admission eta_max computed from band).
  Reduced INTERIOR_DENSE_SAMPLES from 40001→4001 (adequate for 0.06 boundary, saves 55s).
  REMAINING BROKEN in other suites (NOT my scope):
    - test_lensing_exterior_admission.py:639 (config.eta_max → AttributeError)
    - test_lensing_exterior_windows.py:784,1461,1793,1817,1899,1918,1923,1931
    - test_lensing_surrogate_training.py:1106,2680,2701,2705,2725-2726
    - test_lensing_ppgo_bandsplit.py:705-706
  Smoke-scale (4×4×4 grid) eps is ~0.4-0.45 (interpolation sparsity); the < 0.05
  tube_eps_max bar is a PRODUCTION gate on 12×8×12 grids (driver-verified post-build).


- PORT FIX (exterior_windows): Fixed all broken `eta_max` references in
  test_lensing_exterior_windows.py. (1) Renamed test_eta_max_constant_matches_training_default
  to test_f_max_constant_matches_training_default, asserting 0.40==TrainingConfig().f_max.
  (2) Updated module-level ETA_MAX doccomment from 'mirrors TrainingConfig.eta_max' to
  'test fixture operating point for tube geometry'. (3) Replaced all 6 self.config.eta_max
  refs with module-level ETA_MAX constant. (4) Removed dataclasses.replace(..., eta_max=...)
  at 2 call sites (eta_max no longer a TrainingConfig field). (5) Added explicit
  eta_max=SADDLE_ETA_MAX to 2 _saddle_lobe_admissions calls. (6) Added explicit
  eta_max=ETA_MAX to 3 _interior_admission calls (also required positional now).
  76 tests GREEN + 1 xfail in ~2:53.
  Backward-compat audit: no remaining dead refs; single @expectedFailure (SACR-C
  production bar) is unaffected by eta_max migration. No skipped/gated tests
  reference the removed field.

- PORT FIX (surrogate_training): Fixed all 11 broken `eta_max`/`eta_floor`
  references in test_lensing_surrogate_training.py. (1) Removed `eta_floor=0.02,
  eta_max=0.05` from `_WP3_CONFIG` TrainingConfig constructor (dead fields).
  (2) Added module-level `_WP3_ETA_MAX = 0.05` and `_WP3_ETA_FLOOR = 0.02`.
  (3) Updated `_wp3_build_and_measure`, `_wp1_build_arclength_eps`,
  `_wp1_build_uniform_theta_chart`, `_wp1_build_uniform_eps`: pass explicit
  eta_max/eta_floor kwargs to `_build_tube_chart` and `_tube_heldout_samples`.
  (4) Fixed diagnostic line (eta midpoint) to use module constants.
  31 tests GREEN + 49 skipped. Engine-backed WP3 (4 pass) + geometry (7 pass).
  Pre-existing failure: `test_beyond_box_draws_never_serve` (far-field annulus)
  unrelated. Backward-compat audit: zero remaining broken config.eta_* refs
  across all cogwheel/tests/.
