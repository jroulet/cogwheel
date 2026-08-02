# Test Dev Short-Term Observations

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
