---
section: Backlog
---

- **THE FAST FULL-SUITE GATE CANNOT COMPLETE — tests fall into the mpmath
  branch** `[housekeeping]` — diagnosed 2026-08-06 by py-spy on a gate run
  frozen at 99% for SIX HOURS with four workers spinning at ~120% CPU each.
  ONE of the two offenders is FIXED; the other is not yet identified.

  `.claude/sdk/run_full_suite.sh /tmp/gate_at_head_171802.log` started
  2026-08-05 17:18, last wrote its log at 19:59, and was still "running" at
  01:58 the next morning. Log growth zero for 6h while CPU stayed pegged —
  the signature CLAUDE.md's detached-run health rule calls a kill. Killed
  2026-08-06 02:05.

  ## What py-spy found

  Both stuck workers were inside `mpmath.quad` summation, reached through
  `_raw_integral_mp` -> `_f_schwinger_mpmath` -> `f_schwinger`
  (`cogwheel/lensing/chang_refsdal/_schwinger.py:845/866/940`):

  - worker `gw2`: `test_lensing_surrogate_census.py::LnlTierTestCase::
    test_real_likelihood_tiers_within_bars` -> `_dense_farfield_source` ->
    `_pos_farfield_dense` -> `FarFieldChart.from_engine` -> `_exact_total` ->
    `_positive_parity_grid`;
  - the other worker: the same `_f_schwinger_mpmath` leaf reached through
    `Posterior.lnposterior` -> `lensing/posterior.py:81` ->
    `marginalized_extrinsic.lnlike_and_metadata` -> `_get_dh_hh_timeshift`
    -> `_amplification_coefficients` -> `_evaluate_envelope` ->
    `_exact_total` -> `_saddle_grid` (`operator.py:915`). The py-spy dump was
    truncated above the pytest frames, so the TEST is unidentified.

  ## DONE

  `LnlTierTestCase.test_real_likelihood_tiers_within_bars` now carries
  `@_TRAIN_TIER_SKIP` (the file's existing `COGWHEEL_TRAIN_TIER` gate, line
  535). It was the only unguarded method in that file touching
  `_dense_farfield_source`, and every other engine-backed class there
  (`EndToEndPartitionTestCase`, `HeldoutEnvelopeEpsTestCase`,
  `TubeBeatsRawTestCase`, `FoldApproachRayTestCase`,
  `MutationFalsificationTestCase`) was already guarded — so this was a plain
  oversight, not a design choice. Applied per-METHOD, not per-class: the two
  siblings (`test_assign_tier_is_theta_independent`,
  `test_tiers_aggregate_with_a_mock_pair`) are cheap and stay in the fast
  tier. VERIFIED: the class now runs 2 passed / 1 skipped in 3.70 s.

  ## NOT DONE — the second offender is still unidentified

  Ruled out by measurement, not inference:
  - `test_posterior.py` — the obvious suspect (it evaluates
    `lnposterior_pardic_and_metadata` over every prior x likelihood pair).
    Runs clean in 52.07 s. NOT the offender.
  The gate log cannot name it: under xdist the start line and the result are
  emitted together, so an in-flight test leaves no trace (all 1457 started
  tests in that log also completed).

  Remaining candidates, all reaching a MARGINALIZED lensed posterior — the
  stack's `marginalized_extrinsic` frame is the discriminator:
  `test_lensing_marginalized_likelihood.py` (builds
  `LensedPosterior(marg_prior, lensed_marg)` at :321; unguarded classes
  `RefusalContractTestCase`, `BinGuardTestCase`,
  `RegistrationPairingSerializationTestCase`), `test_lensing_prior.py`
  (`LensedPosterior` at :303), `test_lensing_saddle_likelihood.py` (all six
  classes unguarded).

  ## Work

  - Install `pytest-timeout` (NOT currently available — `--timeout` is an
    unrecognized argument) and add a per-test timeout to
    `run_full_suite.sh`. This is the highest-value item and must come FIRST:
    with it, the next gate run NAMES the hung test instead of requiring a
    py-spy autopsy, and an unbounded test fails the gate LOUDLY instead of
    pinning cores overnight. The script self-emits `[beat] n/N` on progress,
    which is exactly why six hours of no beats read as "still running".
  - Then re-run the gate on a CLEAN tree to identify the second offender, and
    gate or shrink it the same way.
  - Establish which condition in `f_schwinger` (`_schwinger.py:940`) routes to
    `_f_schwinger_mpmath` rather than the fast branch, and whether these
    fixtures cross it by design or by accident (a dense far-field source at
    high `w` is the suspect — the DD product cap and the mpmath ceiling are
    different thresholds and may disagree). Prefer shrinking a fixture over
    tier-gating it where the test is a genuine fast-tier smoke check: a
    smoke test moved behind an opt-in env var stops guarding anything.

  ## Consequence to remember

  The post-build tally for the Born residual chart build (`849e580`, shipped
  2026-08-04) was never actually verified — the gate meant to verify it is
  the run killed here. A test that never terminates is not a slow test, it is
  an ABSENT gate.

  ACCEPTANCE: `run_full_suite.sh` completes end to end and reports a tally;
  no fast-tier test enters `_f_schwinger_mpmath` (assert it directly — patch
  the symbol and fail if called during the fast tier).
