# Inspector Short-Term Observations

## Review 2026-07-17 (2nd pass, full access) — Lensing Build 2b crown gate

Worktree: /Users/tejaswi/Work/cogwheel-claude-dev (branch claude-dev).
First pass this date was blocked (Bash false-denial + serena timeout). After
the owner-confirmed one-retry on the bare denial signature, `git diff
--no-index` and serena read_file/write recovered, so the crown-gate files WERE
fully read this pass. pytest could NOT be run (execute_shell_command denied
twice) — numerical green is UNVERIFIED, but full static review found no
correctness defects.

### Verdict: ISSUES (one TRIVIAL finding only). 3 prior findings RESOLVED.

### RESOLVED
- INS-2-001 → GeometricOpticsSlopeTestCase SLOPE_W capped linspace(12,27,84),
  L=0.9w <= 24.3, no longer errors in the refusing band.
- INS-2-002 → ContractionCertificationTestCase asserts certify-XOR-refuse over
  L in [24,48] vs independent mpmath oracle; SelfFalsification
  test_certification_band_gate_can_go_red (1% perturb breaches RTOL_GATE).
- INS-2-003 / INS-3-001 → test_lensing_likelihood.py now EXISTS and is a
  thorough, correct crown gate (528 lines). Fully reviewed this pass.

### likelihood.py (813 lines) — full read, algebra verified correct
- Near-cusp fix (F006): `_amplification_coefficients` evaluates
  ChangRefsdalChannels on a dense per-bin grid (kernel_subsamples=8 interior
  midpoints, symmetric about f_center → LS intercept=mean=value@center,
  slope=<offset,K>/sum(offset^2)). reshape(n_bins,n_sub,n_ch) matches the
  C-order flatten of dense_f. Only kernel k0/k1 VALUES change; contraction
  structure unchanged. Correct.
- `_norm_term` verified by hand: m_s = sum_p mu_p B^(p+s), final =
  sum_{ac} phase * sum_s nu_s m_s = sum_{ac} phase sum_{p,s} mu_p nu_s B^(p+s),
  truncated consistently at p+s<=3 (B has moments 0..3). mu = mode-ratio poly
  (deg2), nu = image poly incl. delay-phase linear expansion (deg3). The
  dropped p+s>=4 terms are the documented cubic in-bin truncation, gated by
  the bin-density criterion + RB-vs-brute tolerance. Correct.
- `_data_term` coeff_k0/k1 with -2πi*tau in-bin delay expansion: consistent.
- Refusal symmetry: both `_amplification_coefficients` (RB) and
  `lnlike_bruteforce` (via LensedWaveformGenerator.amplification) let
  geometry.LensDomainError / operator.CancellationError propagate unswallowed.
  Note: refusal is grid-point-wise identical only if the same w is sampled;
  RB samples dense sub-grid, brute samples full FFT grid — configs are chosen
  deep in the wave branch to avoid the refusal band, tested by
  MacroSaddleRejectionTestCase (both paths raise LensDomainError).

### test_lensing_likelihood.py — crown gate, well-designed
- BruteForceAgreement over 2/4-image, near-cusp, kappa, rotated-shear,
  waveform-offset, near-fold. NearCuspRegressionPin + kernel_subsamples=2
  edge-secant canary (SECANT_ALIAS_MIN=1e3) proves the fix load-bearing.
- Determinism: explicit seed=SEED (gaussian_noise uses default_rng(seed)),
  bit-identical strain + assertEqual repeatability.
- F→1: loose noisy factor gate (0.1) + TIGHT zero-noise floor (1e-2) that
  zeroes strain then injects, removing the noise tail. Addresses the brittle
  0.1-floor nondeterminism from the failing report.
- Timing reframed per brief: (a) RB faster than brute (SPEEDUP_MIN=3),
  (b) contraction subdominant to `_amplification_coefficients` (NOT the coarse
  waveform call — a shared co-cost). Justified in docstring + FINDINGS F007.

### FINDINGS F007 — present, spec↔code consistency verified
Documents (1) timing-gate mis-spec, (2) F→1 floor = template-construction
asymmetry (~1e-3), and corrects F006's overstated "exact to 1e-8". Verified vs
code: `_set_summary` forces precession on + `_stall_ringdown` for `_h0_edges`;
`_candidate_bin_ratios` does neither → the ratio != 1 in ringdown. Accurate.

### OPEN findings this pass
- INS-3-002 (TRIVIAL): `_amplification_at_bins` in likelihood.py is dead code
  — not called in the hot path, lnlike_bruteforce, or the test suite. The
  module docstring + FINDINGS F006 claim it's "retained (the ratio path and the
  timing test still use them)", but the ratio path uses
  `_edge_linear_coefficients` and the timing test uses
  `_amplification_coefficients`. Remove the method or correct the claim.

### Carried forward (unchanged, not this build's scope)
- mpmath undeclared test dependency (F003).
- pytest not runnable in Inspector sessions (shell denial) → numerical green of
  both suites remains UNVERIFIED; re-run in a shell-capable session:
  `python -m pytest cogwheel/tests/test_lensing_likelihood.py
  cogwheel/tests/test_lensing_operator.py`.
