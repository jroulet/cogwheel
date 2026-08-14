# Test Dev Short-Term Observations

(empty — last consolidated by Dreamer on 2026-08-13)

## 2026-08-13 fold-refusal + ghost-rung exterior suite (WP-1 len!=4 guard, WP-2 _ghost_ppgo_amplification)
- New suite test_lensing_fold_ghost_exterior.py (17 tests, 3.88s, all green):
  Q4a FoldExteriorRefusalTestCase (2-image gamma=0.5 rho=1.40 w=70:
  fold_amplification->None; fold_ppgo_correction==geometric_amplification
  EXACT; born_carrier==no-fold reference via np.array_equal) +
  FoldInteriorByteIdentityTestCase (4-image rho=0.90: fold ACTIVE, all three
  differ from raw). Q4b GhostRungGateTestCase (serve/caveat/interior premises
  asserted via independent _ghost_im_and_sep; boundary-flip teeth by
  monkeypatching geometry._GHOST_DECAY_IM_THRESHOLD / _GHOST_SEPARATION_MIN
  individually to flip serve<->decline, isolating each gate). Q4c
  GhostSignConventionTestCase (served == geometric + ghost.kernel*exp(1j*w*
  ghost.delay) to 1e-12; negated + conjugated variants must fail).
  FoldGuardSelfFalsificationTestCase (teeth).
- KEY GOTCHAS: (1) ChangRefsdalChannels REQUIRES a >=2-point strictly-
  increasing freq grid — build np.array([w, 2*w]) not [w]; per-node fold
  decisions are w-independent so the comparison stays valid. (2) The naive
  4-image-SPOOF shim (2 real + 2 decoy images) to prove the guard is
  load-bearing DIES on geometric_amplification's Morse census guard
  (operator.py:1517: signed-magnification-sum != sign(detA)-1 ->
  LensDomainError). REPLACED with _merging_fold_pair returning non-None on
  the genuine 2-image exterior census (guard load-bearing, no census fight)
  + a 1e-9-perturbation teeth test on the exact-equality assertion.
  (3) Ghost sign teeth (minus vs plus) only resolvable at LOW w: ghost
  decays as exp(-w*Im tau_c), so at serve-band w~90 the term is ~1e-23 and
  minus teeth are INERT — run Q4c at w=12 (term ~4e-4). Conjugate teeth
  resolve at any w (conjugation flips decay->growth).
- ORACLE INDEPENDENCE: no-fold reference via monkeypatch _airy_fold.
  _merging_fold_pair = lambda: None (restore in finally) forces raw ppGO
  through the SAME shipping code — git-independent HEAD surrogate. Q4c is an
  algebraic identity between the rung and ghost_kernel (no external oracle).
- BACKWARD-COMPAT AUDIT (WP-1 len!=4 guard is monotone-toward-refusal):
  existing "fires" tests use 4-image configs (guard is a no-op there);
  existing "refuses" tests already had _merging_fold_pair=None (guard cannot
  flip None->non-None). No existing test breaks. Confirmed empirically:
  test_lensing_fold_ppgo_correction.py + test_lensing_fold_ppgo_handoff.py
  -> 37 passed, 3 skipped, no regression.

## 2026-08-13 ppGO certificate consistency shard (WP1 ppgo_error_estimate + WP2 leg-2 drop)
- Extended test_lensing_ppgo_certificate.py (now 16 tests, all green in
  6.8s): added InteriorCertificateConservativeTestCase (Spec B: 5 strict-
  interior 4-image configs x w in {20,40,60}, true_err<=cert ratio<=1.02;
  measured worst ratio 0.885 << 1.0) and NearCausticCertificateSelfRefusal
  TestCase (Spec A: 3 configs at rho 0.90-0.95 x 3 w; EVERY point self-
  refuses cert*2 > CERTIFICATION_BAR=1e-4 — cert 10..2.8e6, all admit=False;
  no point both admits AND is optimistic). Plus 2 self-falsification methods
  in the existing teeth class (shrunk-cert caught on interior; near-caustic
  true_err dwarfs BAR so any admitting cert is optimistic).
- ORACLE (INDEPENDENT): kappa=0,beta=0 positive parity collapses the mass-
  sheet+eigenframe reconstruction to lam=1, y_eig=y, gamma'=gamma,
  mass_sheet_phase=1 -> oracle = _schwinger.f_schwinger(w, source, gamma)
  DIRECTLY (no operator call needed). served = sum_a image_kernel(w,im,mat)
  *exp(1j*w*delay(im,source,mat)); BOTH demod by exp(-1j*w*t_min),
  t_min=min real-image geometry.delay. true_err=|served-oracle| after demod;
  the t_min demod cancels in the magnitude (kept for spec fidelity).
- KEY MEASUREMENTS (scratch-verified before authoring): general interior
  ratios in [0.008, 0.885] all < 1.0 (confirms Fact 3 conservativeness);
  near-caustic cert explodes as rho->1 (rho=0.99 gives cert~5.7e6) so the
  gate refuses long before c3 goes optimistic — this is why dropping leg 2
  is sound. All near-caustic fixtures stay 4-image (nimg=4) and f_schwinger
  serves (w<=60 DD path, no SchwingerCertificationError).
- FIXTURE DERIVATION: source via _source_on_ray = rho*r_caustic(gamma,theta)
  *[cos,sin]; rho<1 interior (4 img), rho in [0.9,0.99] near-caustic merging.
  Keep w<=60 so oracle stays on the exact DD path (>60 is mpmath, >150
  hard-refuse). ~24 oracle evals total, 6.8s — comfortably fast tier.
- BACKWARD-COMPAT AUDIT: grep of cogwheel/tests for ppgo_error_estimate /
  _c3_coefficient / _series_coefficients -> ONLY test_lensing_ppgo_
  certificate.py; WP1/WP2 fully additive, no existing test broken. Neighbor
  test_lensing_geometry green (28 tests, 14.8s).

## 2026-08-13 ppGO interior certificate (WP1 c3 + ppgo_error_estimate, WP2 4-image re-gate)
- New suite test_lensing_ppgo_certificate.py (12 tests, all green): Leg1
  interior/exterior predicate (rho=0.5->4 img, rho=1.5->2 img across 4
  rays; caustic-flip bisection offset ~1e-13 << 1e-3 bar), C3Coefficient
  (c1 vs 1j*saddle_coefficients()[0], c2 vs [1]; rtol 1e-12, measured
  ~1e-15/1e-14; c3 purely imaginary, real part exactly 0 or ~2e-14),
  PpgoErrorEstimate (w^-3 ratio == (w2/w1)**3 rtol 1e-12 measured ~1e-16;
  None for w<=0 and nan-image), self-falsification (_expect_checks=False).
- KEY FIXTURE GOTCHAS: (1) ChangRefsdalChannels(w).reset() is IN-PLACE and
  returns None — never chain .reset() (memory-confirmed, hit it again).
  (2) ppgo_error_estimate/_series_coefficients require GENUINELY interior
  sources (4 images) — place via rho*r_caustic(gamma,theta,kappa=...) at
  rho=0.5; a hand-picked (gamma,y) like (0.3,(0.4,0.2)) is EXTERIOR (2 img)
  and images[real_mask] mismatches shape (4,) vs (2,2). Derive interior
  fixtures from the LIVE closed-form reach, don't pin y literals.
  (3) None-branch for non-finite magnification: use a nan-coordinate image
  np.array([[np.nan,0.5]]) (drives RuntimeWarning + None cleanly); the
  critical_point image gives det(H)=0.0 exactly -> ZeroDivisionError, NOT
  the None path.
- INDEPENDENT ORACLE: saddle_coefficients (closed-form _c1/_c2_polynomial)
  vs _series_coefficients (Gaussian moment-table algebra) are two shipped
  derivations of C1/C2 — a valid cross-check gate; c3 purely-imaginary is
  an analytic property (no external oracle).
- BACKWARD-COMPAT AUDIT (WP1/WP2 additive): _series_coefficients/
  ppgo_error_estimate/_c3_coefficient/_PPGO_MAXEPS are NEW symbols, no
  existing test references them; saddle_coefficients still returns a
  2-tuple (unchanged); real_mask.sum() semantics unchanged. WP2's interior
  fold-ppGO re-gate lives in likelihood.py; test_lensing_fold_ppgo_handoff
  .py tests the SEPARATE _airy_fold saddle-merging path (14 pass/3 skip),
  not the new certificate. Neighbors green: geometry 28, channels 16.
