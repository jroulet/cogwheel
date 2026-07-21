# Inspector Short-Term Observations

## 2026-07-21 — Build 8e review (uniform-asymptotic fold/cusp arms)

Scope: uncommitted tree, worktree /home/tejaswi/Work/cogwheel-claude-dev.
Full python: /home/tejaswi/anaconda3/envs/cogwheel-newlal/bin/python.
Changed code: chang_refsdal/{operator.py,__init__.py,_airy_fold.py(new),
_pearcey_cusp.py(new)}, surrogate.py, scripts/census_homogenization_corners.py,
tests/{test_lensing_airy_fold.py(new),test_lensing_marginalized_likelihood.py}.

### VERDICT: ISSUES

### Verified CORRECT
- Pearcey primitive `_pearcey_cusp.pearcey`: rotated steepest-descent contour
  (central + two 9pi/8-reflected tails), paired N/2N cert at 3e-10 BEFORE
  prefactor. P(0,0) matches closed form (1/2)Gamma(1/4)e^{ipi/8} to ~13 digits.
  Left-tail Jacobian sign (+_VALLEY_DIR) is correct & documented (wrong sign
  cancels both tails for even integrands). Asymptotic P/P_asymp -> 1 with R.
- Cusp arm reconstruction is SOUND in structure: uniform = cluster_sum*(P/P_asymp)
  + far_sum — preserves true image magnitudes, -> geometric sum at large R.
  Calibration certificate (real stationary values match distinct scaled delays).
- geometry API used by both arms all resolve (NearestCausticPoint/CriticalPoint
  fields image/source/hard_axis/soft_axis/hard_eigenvalue/theta; morse_index,
  delay, magnification, saddle_coefficients, macro_matrix, critical_point,
  find_images, nearest_caustic_point). imports clean, no circular-import.
- operator ladder intercept fires ONLY on w>W_CEILING_SCHWINGER previously-
  refusing nodes (byte-identity of resolved & w<=ceiling confirmed by passing
  CertifiedPathByteIdentityTestCase). census is measurement-only (no engine/
  threshold edits). marg test gating (5 heavy classes behind
  COGWHEEL_BRUTE_ACCURACY) correct; RefusalContract/BinGuard kept fast.
- test_lensing_airy_fold.py: 48 passed / 7 skipped / 1 xfail.
- surrogate.py change (`_CUSP_ARM_COVERAGE=0.0`, residual=max(0,dtheta-0.0)==
  dtheta) is provably a no-op. (Suite too slow to finish <500s; not a defect.)

### FINDINGS (introduced by 8e)
- INS-2-001 (implementation/BUG): FOLD arm serves leading-order-INACCURATE
  amplitudes into production. `_airy_fold._fold_amplitudes` sets q=0 and builds p
  from curvatures (2^-1/6 |lam_h|^-1/2 |b3|^-1/3). For generic ASYMMETRIC folds
  (mag ratio 1.22 in fixture — the common case) the max-normalized envelope error
  vs the exact geometric two-image sum PLATEAUS at ~0.095-0.10 and does NOT fall
  with xi (leading-order error), > the 0.05 crown bar. The only accuracy test
  (AiryFoldFarFieldEnvelopeTestCase) certifies a DIFFERENT amplitude set
  (test-only `_farfield_amplitudes`: q=difference, p=sum) which converges to
  <1e-3 — so the GREEN suite masks the production inaccuracy; NO test certifies
  production `_fold_amplitudes`. The self-cert `c_A xi^-3/2 < envelope_bar` bounds
  only the higher-order uniform term, NOT the q=0/curv-p model error, so it PASSES
  wrong values -> violates the "never serve where wrong" refusal-conservative
  contract. Demonstrated: fold arm serves ALL 43 wave nodes of CANCELLATION_LENS
  in the marginalized production path (cusp arm never reached — fold tried first).
  Fix: reconstruct like the cusp arm (geometric image sum x Ai-uniform-ratio, keeps
  asymmetric mags & nonzero-q), OR derive p,q from real sqrt|mu_+/-| (sum/diff) +
  add a certificate bounding amplitude error vs geometric sum; until certified the
  fold arm must REFUSE (fall through), not serve.
- INS-2-002 (implementation/REGRESSION): tests/test_lensing_marginalized_
  likelihood.py::RefusalContractTestCase::test_refusal_precedes_coherent_score is
  RED with 8e, GREEN at HEAD (verified via git stash). The fold arm now SERVES the
  CANCELLATION_LENS wave nodes HEAD refused, so the config no longer raises the
  wave-branch refusal (CancellationError, SchwingerCertificationError); it surfaces
  LensedBinningError later (candidate delay 0.0355s > delta_t_max 0.02s). Non-gated
  load-bearing falsification -> fast tier is RED, violates acceptance. Coupled to
  INS-2-001. Resolve by re-baselining the expected refusal / choosing a fixture that
  still refuses at the wave branch AND/OR fixing INS-2-001.
- INS-2-003 (design/SPEC divergence, flag to Librarian): SPEC.md line 54 still says
  positive-parity gamma'>0 refuses "UNCONDITIONALLY every w>60 ... refuses by name
  until the Build-8e uniform-asymptotics build serves it — accepted interim state,
  sampling parked." Post-8e the w>60 corner IS served by the uniform arms -> the
  "unconditionally refuses" claim is false. Plan expected SPEC.md to change; it did
  NOT (SPEC untouched). I own accuracy, Librarian owns the edit. Interp: spec needs
  updating — but the update must reflect that serving is currently INACCURATE
  (INS-2-001), so arguably the corner is not yet correctly served.
- INS-2-004 (trivial): WP4 cusp-window shrink is INERT by default — _CUSP_ARM_
  COVERAGE=0.0 makes _tube_serves residual==delta_theta, so mission hole class 1
  (8c cusp exclusion windows) is NOT shrunk; cusp arm only serves the operator-
  ladder w>60 corner. Documented as pending census pin; conservative default, not a
  blocker, but the "shrink cusp exclusion windows" deliverable is deferred.

### Resolved from prior review
- INS-1-003 (8d census undercount): FIXED. census positive-parity geometric gate is
  now `resolved & (l_arr > L_MAX)` (dropped the `high &` mask); diff comment cites
  INS-1-003 explicitly.

### Carry-forward (NOT re-checked this round — out of 8e scope)
- Prior INS-1-001 (8d): several lensing suites (schwinger/operator/fast_path/
  ratio/batched) were RED pending re-baseline. Not re-run this review; check if
  still open.
- Prior INS-1-002 (8d): SPEC line-54 "bit-frozen"/"byte-identical" positive-parity
  claims — superseded by/entangled with INS-2-003; needs Librarian pass.
