Build: exterior_followup WP4 ppGO above-ceiling (brief_exterior_followup_four_items, 2026-08-08)
Working tree: cogwheel/lensing/likelihood.py (+87 lines), test_lensing_ppgo_above_ceiling.py (new, untracked, 15/15 PASS)

Verified:
- Production: _ppgo_above_ceiling gate order correct (w_max>150, >=2 real, min_dt>0, w_lo*min_dt>=RHO_END)
- Reconstruction pattern matches existing fold-ppGO interior in _surrogate_coefficients: fold_ppgo_correction → f_minrel → ppgo_sum subtraction → envelope → reconstruct_farfield(FARFIELD_KERNEL_SUM)
- Phase convention: fold_ppgo_correction returns absolute-frame F, demodulates by exp(-1j*w*t_min), envelope re-modulates by exp(+1j*w*t_min), reconstruct_farfield de-tilts by exp(-1j*_frame_phase(w,t_min)) — round-trip correct
- NaN guard: np.where(isfinite, f_total, 0.0) — belt-and-braces, more conservative than existing fold-ppGO interior (no NaN guard)
- Lazy import of fold_ppgo_correction from _airy_fold — correct pattern (circular import avoidance)
- Intercept placement in _amplification_coefficients: after surrogate + lens/dense_w, before engine seed eval — correct
- Import of W_CEILING_SCHWINGER_QD (150.0) and RHO_END (4.0) correct
- Tests: 15/15 pass, 7 test classes, comprehensive coverage (boundary continuity, error decay, gate borders, self-falsification, fallthrough, no-surrogate, telescoping identity)
- Pre-existing test failure: test_fop_refuses_uncertifiable_contractions (confirmed by stash+run — NOT caused by this change)
- No regressions: test_lensing_likelihood.py (17P/12S/1X), test_lensing_surrogate.py (66P)
- All callers of _amplification_coefficients forward-compatible (return shape unchanged)

No bugs or design issues found. Verdict: PASS.
