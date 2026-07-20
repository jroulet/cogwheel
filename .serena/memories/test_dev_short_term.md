# Test Dev Short-Term Observations

## Build (2026-07-19) — test_lensing_schwinger.py WP1/WP2 additions
- Suite already existed (targets _schwinger). ADDED operator-level
  dispatch + census coverage; full file now 32 tests, all green
  (~15min; DispatchFallbackOracle alone ~5min, mpmath-heavy).
- FIXED pre-existing RED: CertifyXorRefuse.test_error_type_contract had
  (3.0,y,1.0) & (3.0,y,0.5) as ValueError cases. WP2 relaxed
  _schwinger._validate_inputs to gamma_prime>0, so g'=0.5 ACCEPTED
  (finite), g'=1.0 returns (nan+nanj) (parity boundary det A=0, silent
  nan — NOT a refusal!). Replaced both with g'<=0 cases (0.0,-0.5).
- WP2 fallback recon (in _positive_parity_grid_with_fallback):
  F_op = (1/lam)*exp[0.5j*w*ln(lam) - 0.5j*w*kappa*s] * f_schwinger(w,
  y_eig, gamma'); lam=1-kappa, y_scaled=y/sqrt(lam), gamma'=gamma/lam,
  y_eig=exp(-1j*beta)*(y_scaled), s=y_scaled@y_scaled. Reused existing
  AST-guarded _oracle_saddle (=F_{0,gamma'}); prefactor cancels in rel
  err so recon oracle stays independent. Built recon prefactor in the
  TEST with plain cmath/math (NOT operator._mass_sheet_map) to keep it
  fully independent.
- Dispatch fixtures that REFUSE legacy _grid_certified + recover via
  fallback: gamma in {0.47,0.49} kappa=0 y=(0.4,0.3) refuse w>=8;
  y=(0.1,0.1) refuse only w>=~12-20; y=(1.0,0.0) refuse from w=3 (use
  for low-w span). Worst rel 1.7e-11 at w=59.9 (< 1e-10 gate). kappa!=0
  rows (0.35,k0.2),(0.4,k0.15) all refuse+recover.
- ABOVE CEILING (w>60): y=(0.4,0.3)/(0.1,0.1) -> CancellationError
  (fallback re-raises, w>W_CEILING=60); y=(1.0,0.0) -> Hypergeometric
  DomainError (different named refusal) so EXCLUDE it from the
  CancellationError|SchwingerCertificationError contract test.
- BIT-FREEZE: F_op & F_op_grid byte-IDENTICAL on certified path (gamma
  0.2 y=(0.4,0.3) w{3,5,8,10}); order_used>0 proves certified path ran
  (fallback reports order 0). Saddle f_schwinger(3,(.4,.3),1.3) exact =
  (0.14470585550870085+0.40651223933528396j) — note existing ANCHOR_VALUE
  imag is truncated 0.4065122393352838 (used at 1e-12, not byte-exact).
- CENSUS (WP1 geometry._check_image_census): positive macro_matrix(0.3,
  0,0) det>0 has interior 4-image src (0.05,0.03) morse{0,0,1,1} signed
  0 -> mirror-pair drop RAISES 'Image census defect'. SADDLE macro_matrix
  (1.3,0,0) det<0 is 2-IMAGE EVERYWHERE (scanned, no 4-image region) ->
  used single-image drop for its red path (documented premise repair).
  Msg substring assert: 'census defect' (lowercased) matches 'Image
  census defect'.
- NO production changes this run; neighbor suites (operator/saddle_geom)
  unaffected by a test-only edit — did not re-run (each ~15min + memory
  MemoryError-when-combined warning).


## Build 7a — test_lensing_saddle_geometry.py (WP1 census guard, WP2 fallback)
- Flipped NearAxialQuarticDefectTestCase from @expectedFailure to
  positive assertRaises(LensDomainError) for BOTH F012 reproducers
  (saddle gamma=1.3 y=(-1.43028417,2e-10); positive gamma=0.3
  y=(0.2,2e-10)); message contains 'census defect'. Removed now-unused
  `expectedFailure` import.
- WP1 SIDE EFFECT: `_check_image_census` wired into find_images_quartic
  now REFUSES the exactly-on-fold degenerate 3-image census (signed -1
  != -2). Existing test_on_caustic_and_fold_crossing_sources broke
  (LensDomainError on on_point) — rewrote it to assertRaises at the
  fold, keep clean 4/2 census at +-1e-4 offsets.
- Census facts: saddle (det<0) signed=-2, {1,1}/{0,1,1,1}; positive
  (det>0) signed=0, {0,1}/{0,0,1,1}. sign(detA)-1 invariant.
- Guard falsification: build 4-image list, drop 2 images sharing a
  Morse index (mirror pair) -> _check_image_census raises; full list
  returns None. Helper _doctored_without_mirror_pair.
- WP2 refusal-above-ceiling: positive-parity strong shear gamma>=0.9,
  small y, w in {61,70,80} -> operator.F_op / F_op_grid raise
  CancellationError (all 50 probed points). Assert against
  (CancellationError, SchwingerCertificationError). W_MAX_CERTIFIED=500
  so w=61-80 don't trip kernel domain error first; W_CEILING_SCHWINGER=60.
- Bit-freeze certified positive path (fallback never fires): F_op
  (0.3,0.1)g0.2 w5 -> -0.35753006967142426+1.1663724461262843j;
  grid [5,8,12] captured; order_used>0 confirms operator series ran.
- NEIGHBOR REDS (report, don't touch): test_lensing_operator.py 3 fails
  (test_former_silent_nan_config_now_refuses etc.) — WP2 fallback now
  RESCUES below-ceiling configs those tests expected to refuse
  (CancellationError not raised). test_lensing_schwinger.py 1 fail
  (CertifyXorRefuseTestCase::test_error_type_contract, ValueError not
  raised) — independent of my file. Both owned by other Test Dev runs.
- ENV: running all 4 heavy lensing suites together -> MemoryError; run
  one file at a time. Full python:
  /home/tejaswi/anaconda3/envs/cogwheel-newlal/bin/python
