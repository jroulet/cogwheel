# Inspector Short-Term Observations

## Review 2026-07-17 — Lensing Build 2c (switch-neighbourhood fix) — RE-REVIEW after test amendments

Worktree: /Users/tejaswi/Work/cogwheel-claude-dev (branch claude-dev).
Reviewed uncommitted working tree. Several git reads hit the transient
false-denial signature; per owner confirmation re-issued the WP3 diff ONCE and
it succeeded, so WP3 is now directly verified. pytest NOT run this session
(runtime green still UNVERIFIED), but every blocking finding is statically
resolved.

### Verdict: PASS. INS-4-001 and INS-4-002 both RESOLVED.

### WP1 (channels.py) — CORRECT
- `_channel_switch` neighbour set = `np.delete(np.arange(_N_CHANNELS), channel)`
  (min over ALL cluster labels incl. parked virtual). Docstring rewritten to
  Eq.(delay-separation). No-op when all 4 labels real (old/new neighbour sets
  identical) -> 4-image regions bit-unchanged.
- `_min_delay_separation` kept REAL-only with justified docstring (branch gate
  runs stationary phase over real saddles only). Correct.

### WP2 (likelihood.py) — CORRECT
- `_DEFAULT_KERNEL_SUBSAMPLES = 2`; all 4 docstrings reframed (secant accurate
  now kernels bounded; sub-samples = robustness margin). Verified.

### INS-4-001 RESOLVED — crown suite amended (test_lensing_likelihood.py)
- Canary re-based: `test_edge_secant_canary_reproduces_aliasing_pathology` ->
  `test_real_only_switch_variant_blows_up_kernels`. New module-level helper
  `_real_only_channel_switch(w,delays,real_mask)` independently re-implements
  the BUGGY real-only rule (NOT imported from module-under-test -> non-circular,
  satisfies F002). Test monkeypatches `channels._channel_switch` via
  `mock.patch.object` and asserts on kernel magnitudes from
  `_amplification_coefficients` (returns delays,k0,k1,partition — verified):
    max_k_bug >= 1e3*|F|; max_k_prod < 1e3*|F|; max_k_bug >= 1e3*max_k_prod.
  |F| = max|partition.exact_total| (switch-INDEPENDENT, computed separately at
  channels.py:661 vs switch at :658). Brief table (buggy 5.22e5, prod 0.975,
  |F|~3): all three asserts pass. Non-vacuous, pins WP1 load-bearing.
  SECANT_ALIAS_MIN -> SWITCH_PATHOLOGY_FACTOR=1e3. `cls.like_secant` REMOVED.
- Monkeypatch validity: engine `evaluate` calls bare global `_channel_switch`
  (channels.py:658) -> patching module attr takes effect. Helper uses
  channels.smootherstep/RHO_START/RHO_END/_N_CHANNELS — all module-level
  (imported l.65-67, _N_CHANNELS=4 l.73).
- Zero-noise NaN fix (NormalizationFloorZeroNoiseTestCase): construction wrapped
  in warnings/errstate-ignore, then `cls.zero_like.asd_drift = np.ones(n_det)`.
  VERIFIED sound: relative_binning.py:631-632 — asd_drift NOT baked into
  summaries; applied at eval (l.500, weight identities /asd_drift^2). Plain
  attribute. Reassignment overrides NaN for lnlike_fft/lnlike/lnlike_bruteforce
  consistently. ZERO_NOISE_TOL=1e-2 unchanged.

### INS-4-002 RESOLVED — stale docstrings fixed
- NearCuspRegressionPin class docstring no longer says "default 8"; rewritten to
  switch mechanism (F008 supersedes F006). test_production_lnlike... docstring
  "(kernel_subsamples=8)" -> "at the near-cusp config". Serena scan: no residual
  subsamples=8/default 8/SECANT_ALIAS/like_secant/kernel_subsamples=2 refs. Only
  2 F006 mentions remain — both intentional "F008 supersedes F006" notes.

### WP3 (spec/FINDINGS/changelog) — DIRECTLY VERIFIED this session. F006 header
marked SUPERSEDED-by-F008, history kept + accurate right/wrong account
(dense-sample null result correct; edge-secant slope-squaring sign-disproven:
squaring real slope only ADDS to (h|h), cannot give the negative-huge
excursion). F008 added: real cause (real-only _channel_switch vs Eq.
delay-separation), measured table matches brief+code, _min_delay_separation
sibling audit (correctly NOT fixed, justified), F005/F007 cross-refs unaffected.
SPEC.md 0.2.1 -> 0.2.2; SPEC_CHANGELOG + CHANGELOG rendered/accurate. No
spec-code divergence. CORRECT. Minor non-blocking (pre-existing render_fragments
convention, NOT a finding): SPEC.md last_updated stays 2026-06-05, SPEC_CHANGELOG
entry empty () date — same shape as the 0.2.1 entry.

### Not deep-reviewed (Test Dev domain, out of WP scope)
- test_lensing_waveform.py amendments #4 (macro-saddle certified-band control +
  CancellationError companion) and #6 (small-mass floor restricted to w>=1e-3,
  ticket ref). Prior memory notes these handled by test_dev; not re-audited.

### Carried forward (non-blocking)
- pytest ungated re-run STILL OWED from WORKTREE root:
  `python -m pytest cogwheel/tests/test_lensing_likelihood.py
  cogwheel/tests/test_lensing_waveform.py cogwheel/tests/test_lensing_operator.py`
  Confirm crown suite green at RB_ATOL=1.5 (no tolerance widening). Static
  analysis strongly predicts green.
- mpmath undeclared test dependency (F003).
