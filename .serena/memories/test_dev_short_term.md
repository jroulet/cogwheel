# Test Dev Short-Term Observations

## Current build: test_lensing_born_residual_wiring.py — WP1 Born residual chart wiring (EXTENDED)

- Extended suite from 18 to 34 tests (4.7 s total runtime).
- KappaBetaGuardPrecedenceTestCase (6 tests): proves kappa!=0 and beta!=0
  guards at lines ~1554-1575 fire BEFORE the Born residual slot.
  Uses _BornResidualProbe with born_residual_chart attached; config has
  gamma=0.5, |y| giving rho~3.0 (inside chart grid). Control test proves
  same config with kappa=beta=0 reaches the Born path (non-None), while
  kappa=0.1 or beta=0.3 yields None. Sweeps multiple kappa/beta values.
- BornResidualChartCoversTestCase (10 tests): verifies axis-aligned box
  containment on spec-prescribed grids (gamma=[0.3..0.7], rho=[1.5..5.0]).
  Tests: interior point, all 4 corners, all 4 mid-edges (all True);
  gamma_below/above, rho_below/above (all False); machine-epsilon boundary
  (1e-10 outside each edge → False).
- Backward-compat audit: WP1 is purely additive (new module, defaulted kwarg,
  code behind `if not served:`). All existing tests pass born_residual_chart
  default (None). Searched: _surrogate_coefficients (5 files),
  BornResidualChart (0 hits outside new test), LensedRelativeBinningLikelihood(
  (12 files, all use default). No changes needed to existing tests.


- Created `cogwheel/tests/test_lensing_born_residual_wiring.py` (18 tests, ~4.4 s).
- NoChartByteIdentityTestCase (3 tests): verifies _surrogate_coefficients
  returns None when born_residual_chart=None and surrogate declines.
  Uses _BornResidualProbe (lightweight class binding real methods from
  LensedRelativeBinningLikelihood without heavy event/waveform construction).
- MockChartServePathTestCase (4 tests): verifies Born path fires with a
  synthetic BornResidualChart (residual = 0.01*exp(-rho)*(1+0.001j)). Config:
  gamma=0.5, rho~3.0, w in [0.5,20], M_lens=100 Msun, z_lens=0.5.
  Reconstruction identity: k0/k1 from _surrogate_coefficients match
  independently-computed carrier+residual through reconstruct_farfield to
  <1e-13 relative. Diagnostic plot saved.
- OutOfBoxFallthroughTestCase (5 tests): three sub-cases (rho>5.0 above grid,
  1.0<rho<1.5 below grid, rho<1.0 interior) + gamma outside grid. All return
  None. The rho<=1.0 guard fires independently of chart.covers.
- SelfFalsificationTestCase (5 tests): chart.covers rejects/accepts correctly;
  rho guard fires even when chart would cover; mock surrogate declines;
  wrong residual (1000x) produces detectable k0 difference (>1e-3 relative).
- Anti-vacuity: every test class has setUp/tearDown n_checks guard.
- Key design: _BornResidualProbe binds _surrogate_coefficients,
  _reduce_dense_kernels, _image_delays, _lens_params, _ppgo_band_split from
  the real class. _MockSurrogate passes may_serve but refuses serve. No engine
  needed — geometry_partition is analytic.
- Backward-compat audit: WP1 adds born_residual_chart=None (keyword, defaulted)
  to __init__. All existing tests pass the default. The Born path only fires
  when born_residual_chart is non-None AND surrogate declines (served=False).
  No existing test exercises that combination. Searched: _surrogate_coefficients
  (5 files), born_residual_chart (0 hits outside new test), caustic_rho
  (unchanged function), born_carrier_from_partition (unchanged function).
  No changes needed to existing tests.



## Current build: test_lensing_log_reach_gamma.py — WP1 log-reach gamma axis

- Created `cogwheel/tests/test_lensing_log_reach_gamma.py` (23 tests, ~20 s
  with COGWHEEL_TRAIN_TIER=1; 17 pass + 6 skip without it, ~3 s).
- LogReachStructuralTestCase (12 tests): array size, strict ascending,
  endpoint pinning (1e-14), log-reach round-trip (5e-4 tol — the 200-pt
  internal linspace gives ~3e-6 positive, ~1e-4 saddle), node clustering
  direction (last<first near wall for positive; first<last for saddle),
  validation errors (reversed range, <4 nodes).
- LogReachComparativeAccuracyTestCase (4 tests, engine-backed): builds two
  7×4×4 tube charts on (0.90, 0.98) with SHARED spatial grids and w-grid,
  differing ONLY in gamma placement (uniform vs log-reach). Evaluates at
  30 off-grid gamma × on-grid (u, theta) to isolate gamma interpolation.
  Asserts log-reach max eps < 0.7 × uniform max eps AND < 5e-2 absolute.
  KEY DESIGN: with 4 spatial nodes, random-spatial probes have ~0.30-0.45
  eps (dominated by spatial interpolation), masking the gamma benefit.
  Using on-grid spatial coords isolates gamma-only error (~0.003-0.02).
- LogReachRegressionTestCase (2 tests, engine-backed): tube chart on
  (0.35, 0.65) with log-reach nodes; gamma-only held-out eps < 5e-3
  (measured ~1.2e-3). Confirms no degradation on smooth interior bands.
- LogReachSelfFalsificationTestCase (5 tests): proves clustering detector
  fires on uniform grid; proves round-trip detector fires on wrong grid;
  proves endpoint detector fires on perturbed array.
- Anti-vacuity: every test class has setUp/tearDown n_checks guard.
- Backward-compat audit: WP1 changed `from_engine`/`from_lobe_engine` to
  use `_log_reach_gamma_axis` instead of `np.linspace`. No existing test
  asserts gamma-grid uniformity or specific node values from from_engine.
  Tests that build grids manually (via `np.linspace` + `_build_tube_chart`
  or `FarFieldChart.from_values`) are unaffected. No changes needed to
  existing tests.

## Current build: test_lensing_part0_mechanical.py — structural invariant scanner

- Created `cogwheel/tests/test_lensing_part0_mechanical.py` (13 tests, 0.66 s).
- Pure AST/text scanning — no lensing module imports, no numerical computation.
- TestNoPriorBoxConstants: scans for diagonal ≈ 4.2426 and box-named 3.0
  constants. Includes anti-vacuity assertion (>10 files, >20 constants).
- TestNoRetiredConceptNames: loads .claude/hooks/retired_concepts.json,
  checks __all__ exports, top-level symbols, and full source lines against
  word-boundary patterns. Exclusion carveout for commentary lines.
- TestNoNewDiscretizationAbsorbers: regex `^_[A-Z][A-Z0-9_]*(_EPS|_MARGIN|
  _FRAC|_STANDOFF|_SAFETY)$` — currently 5 matches, all allowlisted in
  surrogate_training.py.
- TestSelfFalsification: 5 tests proving each detector fires on synthetic
  violations.
- Backward-compat audit: no WP changes this build (empty list), no
  production changes to audit against.

## C12 build: test_lensing_ghost_gate.py — orthogonality witness

- Added GhostGateOrthogonalityWitnessTestCase (5 tests, ~3.7 s) proving the
  separation gate is NOT subsumed by the decay gate.
- Key finding: at positive parity (gamma<1), Im(tau_c)>=0.4 physically implies
  separation>=1.2 (exhaustive scan gamma 0.10-0.99, all angles/offsets). The
  orthogonality witness requires saddle parity (gamma=5.0, y=(5.2,1.5)) where
  Im(tau_c)=0.502>=0.4 but separation=0.600<0.7. The gate function itself is
  parity-agnostic (no parity check inside farfield_ghost_term).
- Config: ORTH_GAMMA=5.0, ORTH_SOURCE=np.array([5.2, 1.5])
  - Im(tau_c) = 0.5016, margin = 0.1016 above threshold 0.4
  - separation = 0.6001, margin = 0.0999 below threshold 0.7
  - GhostDomainError message correctly says "separation" not "decay"
- test_disabling_separation_gate_admits_the_config: patches only
  _GHOST_SEPARATION_MIN to 0 (decay gate stays live) → config admits,
  proving the gates are independent.
- Diagnostic scatter plot saved: ghost_gate_orthogonality_scatter.png
- Pre-existing failure: test_raising_constant_to_two_refuses_an_admit_config
  (ADMIT_CONFIGS[0] has sep=2.012, so MIN=2.0 doesn't refuse it). NOT caused
  by WP1 changes (confirmed via git stash test on prior commit).
- Backward-compat audit: WP1 is pure documentation (added Part 0 resolution
  comments to _GHOST_SEPARATION_MIN). No API/value/signature changes. All
  existing tests referencing these constants use unchanged values. The
  pre-existing failure is unrelated to WP1.

## C11 build (extension): test_lensing_ghost_decay_gate.py — protective refusal + skew impossibility

- Extended suite from 11 to 18 tests (5.4 s total runtime).
- ProtectiveRefusalTestCase (3 tests): builds full Schwinger partition for
  REFUSE config, computes FARFIELD_KERNEL_SUM envelope, force-computes ghost
  via geometry.ghost_kernel bypassing the gate, proves mean|E - G| > mean|E|
  (ghost worsens residual → refusal is protective, not overprotective).
  Uses _frame_phase to demodulate ghost into same frame as envelope.
- TrainServeSkewImpossibilityTestCase (4 tests): proves gate decision is
  frequency-independent — TRAIN_W (0.5..10) and SERVE_W (2..10) both admit
  for well-decayed config; both refuse for near-axis config; bit-identical
  ghost values on shared w-points. This is the formal skew-impossibility
  proof: Im(tau_c) >= min_sep contains no w.
- Backward-compat audit: confirmed C11's findings still current (4 other
  suites broken, not touched per scope discipline).
- Imports added: ChangRefsdalChannels, FARFIELD_KERNEL_SUM,
  farfield_envelope_from_partition, _frame_phase.

## C11 build: test_lensing_ghost_decay_gate.py — new decay gate certification

- Created `cogwheel/tests/test_lensing_ghost_decay_gate.py` (11 tests, ~3.7 s).
- WP1 added `Im(tau_c) >= min_delay_separation` gate to `farfield_ghost_term`.
- Decay-refused config: gamma=1.6, source near theta=0.02 (Im(tau_c)=0.044,
  min_sep=4.076, separation=1.28 > 0.7 → separation gate alone would admit).
- Admitted config: gamma=1.5, y=(2,2) (Im(tau_c)=0.825 > min_sep=0.249,
  separation=1.57 > 0.7).
- FewImages: real_images=[] and real_images=[single] both refuse (min_sep=0).
- SelfFalsification: mock geometry.delay to bypass/tighten the gate; pass
  t_min and real_images explicitly to avoid StopIteration on side_effect.
- BACKWARD-COMPAT AUDIT — 3 OTHER SUITES BROKEN by the new decay gate:
  * test_lensing_chang_refsdal_ghost_frame.py: 8 failures (probe config
    gamma=0.5, theta=45°, offset=0.6 has Im(tau_c)=0.404 < min_sep=2.916).
  * test_lensing_ghost_gate.py: 7 failures (ADMIT_CONFIGS have Im(tau_c)
    < min_sep under the new gate).
  * test_lensing_exterior_windows.py: 9 failures (fold configs refused).
  * test_lensing_born.py: 2 failures (SaddleGhostRefusedNodeCount).
  These are other runs' suites; reported but not touched per scope discipline.

## C10 build: test_lensing_born.py — fence retirement verification (no changes needed)

- Suite already fully updated in C8 build; all 53 tests pass in ~25 s.
- Backward-compat audit confirmed: no stale references to GAMMA_FENCE,
  annulus_rho, saddle_fence, or saddle_caustic_max_y in any test file.
- All spec requirements already implemented: ExteriorFenceTestCase/
  SaddleExteriorFenceTestCase deleted, fence constants removed,
  BornCensusReachableRedTestCase uses caustic_rho mock, C8FenceRetirementTestCase
  and CausticRelativeClassificationTestCase present and passing.
- Pyright type warnings are all duck-typing false-positives (SimpleNamespace
  as surrogate, **kwargs dict unpacking) — runtime correct.


## C9 build: test_lensing_ppgo_map.py — caustic_rho rename verification

- Suite already fully ported to `caustic_rho` (was `CausticRhoByteEquivalenceTestCase`).
- 37 tests pass in ~10 s; no modifications needed — the rename was purely cosmetic.
- Backward-compatibility audit: `annulus_rho` absent from entire `cogwheel/tests/`
  except the descriptive class name `OuterAnnulusRhoCapTestCase` in
  `test_lensing_ppgo_bandsplit.py` (doesn't call the old function, still green).
- No fence/classify_fallthrough references in this file (those are other suites' domain).


## C8 build: test_lensing_born.py fence retirement

- Deleted: ExteriorFenceTestCase, SaddleExteriorFenceTestCase, _plot_astroid,
  _plot_saddle_fence, _saddle_off_on_axis, all SADDLE_FENCE_* constants,
  FENCE_SERVE/REFUSE/ABSY/ASTROID_TOL constants.
- Updated CENSUS_NONANNULUS_Y1_EIG from 2.0 to 0.5 (caustic reach ~ 1.214,
  needs rho < 1 for interior); SADDLE_CENSUS_NONANNULUS_Y1_EIG from 2.0 to 1.0.
- BornCensusReachableRedTestCase: mock.patch on GAMMA_FENCE -> caustic_rho mock.
- SaddleCensusReachableRedTestCase/SelfFalsification: saddle_caustic_max_y mock
  -> caustic_rho mock returning 0.0.
- SelfFalsificationTestCase: test_refuse_gamma_actually_raises (gamma=0.80) ->
  test_parity_wall_margin_actually_raises (gamma=0.998).
- SaddleSelfFalsificationTestCase: removed fence test, added wall-margin test.
- New C8FenceRetirementTestCase: gammas 0.80, 0.90, 1.04 now pass born_gate.
- New CausticRelativeClassificationTestCase: classify_fallthrough uses rho > 1
  on both parities (positive gamma=0.5, saddle gamma=1.3).
- Key insight: the old annulus inner radius (3.0) is no longer the boundary;
  caustic_rho = |y| / caustic_reach > 1 is the new one-for-both-parities gate.
