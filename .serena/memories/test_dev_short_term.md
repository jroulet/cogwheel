# Test Dev Short-Term Observations

## 2026-08-17 (WP2 c3 in-band accuracy + WP3 above-ceiling partition; ppgo_above_ceiling stale-mock break)
- Extended test_lensing_saddle_serve_gate.py with 8 tests / 2 classes, 0 retired:
  * SaddleC3InBandAccuracyTestCase (3, Spec1/WP2): reconstruct analytic
    zero-envelope saddle serve above w_split, compare F vs exact f_schwinger
    over [w_split*1.001, 55] (w<=60 stays on exact DD path). Oracle
    INDEPENDENCE: f_schwinger (DD) vs reconstruct_farfield (switched-analytic)
    are distinct derivations. Frame lift is LOAD-BEARING: absolute F =
    total*exp(+1j*w*t_min); test_alignment_is_load_bearing pins aligned<=1e-3
    AND unaligned>100*bar (raw unaligned ~0.9).
  * LEAKY-GATE WITNESS pinned + ESCALATED: config (gamma=2, y=(1.1,0)) has the
    lowest w_split (~9.6, closest to caustic); the c3 certificate ADMITS it
    (serve premise True) yet |F_ana-F_eng|.max() ~ 3.1e-3 > bar=1e-3. Test
    asserts CERT_BAR < worst < 1e-2 to PIN the measured optimism. Per spec
    rationale this is a calibration miss where the gate admits -> STOP/escalate,
    NOT a plumbing bug. Certified clean domain (3 configs) green alongside.
  * PpgoAboveCeilingPartitionTestCase (5, Spec2/WP3): ENGINE-FREE structural
    proof. _CeilingProbe binds unbound _ppgo_above_ceiling + spies
    _engine_envelope_below_split (MagicMock side_effect returns
    sentinel*below_mask). Patch _likelihood_module.reconstruct_farfield to
    CAPTURE the stitched envelope arg. Resolved (gamma=1.5, y=(0.6,0.9)):
    envelope[below]==sentinel byte-exact, envelope[above]==independent fold
    rebuild byte-exact, above sentinel-INVARIANT (A vs B), spy sees only the 8
    below-150 nodes. Near-caustic (gamma=1.5, y=(1.5,0)): None, spy
    assert_not_called. Production gate is CEILING-keyed:
    W_CEILING_SCHWINGER_QD(150)*min_delta_tau < RHO_END(4) -> None (NOT w_lo).
- BACKWARD-COMPAT AUDIT (step 7, reading + confirmatory runs):
  * Spec3 Born null-split byte-identity: canonical pin lives in
    test_lensing_born_analytic_reachability.py (147 passed) — GREEN after WP1
    factored the inline mask into shared _band_split_mask. No parallel added.
  * GREEN neighbors: ppgo_bandsplit, born, ppgo_map, born_residual_wiring (85+
    passed). My own suite 62 passed.
  * *** RED / FLAGGED (OUT OF MY EDIT SCOPE) ***
    test_lensing_ppgo_above_ceiling.py: 7 failed + 6 errors. ROOT CAUSE: WP3
    added self._engine_envelope_below_split and _ppgo_above_ceiling now calls
    it; that suite's MagicMock stubs (_build_stub + inline test_b2/test_d
    stubs) never set it, so an auto-child MagicMock flows into
    reconstruct_farfield -> "ValueError: operands could not be broadcast
    together with shapes (0,) (N,)" in channels.py:1254, then anti-vacuity
    tearDown errors ("zero comparisons ran"). This is the "MagicMock hides a
    new attribute read" pattern (memory, 2026-08-13/14). FIX RECIPE for the
    owning run: set stub._engine_envelope_below_split = MagicMock(
    side_effect=lambda lens, dw, below_mask: np.zeros(np.shape(dw),
    dtype=complex)) on every stub, OR use a real spy like this suite does.
    NOT edited here (scope discipline: other suites owned by other runs).

## 2026-08-17 (band_split_mask / saddle c3 split-point / null-split identity)
- Extended test_lensing_saddle_serve_gate.py (re-pointed the existing serve-gate
  pin, no parallel file) with 3 spec classes / 21 tests, 0 retired:
  * BandSplitMaskTestCase (10): `_band_split_mask(dense_w, split)` unit —
    band_split True iff split not None AND w_lo<split<w_hi (STRICT interior);
    False -> below_mask all-True (null-split identity precondition); True ->
    below_mask == dense_w<=split exactly (np.array_equal); node-coincident
    split inclusive; monotone below-count; boolean diag table.
  * SaddleC3SplitPointTestCase (7): gamma=2 source=[1,0] resolved 2-image
    fixture; w_split inverts cert exactly (S*est(w_split)/bar==1 to 1e-9);
    reference-frequency-independent; PASS at w_split*(1+1e-6) / FAIL at
    *(1-1e-6); est strictly-decreasing exact w**-3 (log-log slope pin lives
    in sibling CertificateMonotoneDecayTestCase, not duplicated).
  * SaddleFarfieldNullSplitIdentityTestCase (4): admit-band -> zero-envelope
    k0/k1 byte-identical to independent reconstruct_farfield+_reduce rebuild,
    engine spy assert_not_called; refuse-band -> None no engine;
    self-falsification proving the zero-envelope identity has teeth.
- SPEC-2 None-branch DISCREPANCY (documented in class docstring, not a test
  bug): `_saddle_c3_split_point` hardcodes w_ref=1.0 so the w_min<=0 None
  trigger is unreachable; an image ON the critical curve (gamma=2, img=[1,0])
  makes magnification do 1/0 -> ZeroDivisionError (RAISES, not None), near-
  critical gives finite-huge mu (not None). Used the contract-guaranteed
  EMPTY-images trigger (np.zeros((0,2)) -> len==0 -> None) instead.
- `_ppgo_band_split` (LensedRelativeBinningLikelihood method, ppGO cell-ceiling
  splitter) is DISTINCT from and unaffected by WP1's new module-level
  `_band_split_mask`; a tests-wide grep for the new symbol found no stale
  pins. Regression-clean: saddle_serve_gate 54, born_analytic_reachability 30,
  born 53.
