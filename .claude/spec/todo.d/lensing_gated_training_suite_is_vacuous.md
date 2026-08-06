---
section: Backlog
---

- **THE TIER-GATED TRAINING SUITE IS BROKEN AT HEAD — 16 failed, 16 errors,
  every one vacuous** `[housekeeping]` — measured 2026-08-06, PRE-EXISTING
  (not caused by the interior-wedge build).

  `COGWHEEL_TRAIN_TIER=1 pytest cogwheel/tests/test_lensing_surrogate_training.py
  -k "EpsRegistrationGateTestCase or EpsGateResumeTestCase or
  SelfFalsificationTestCase"` gives **16 failed, 11 passed, 16 errors in 94 s**.

  All 16 failures are the same assertion:

      AssertionError: 0 not greater than 0 :
      anti-vacuity: this test asserted nothing (zero comparisons).

  So `_wp1_gate_fixture()` (~:908) is yielding ZERO charts, and every test
  built on it is a no-op. The fixture's own anti-vacuity guard is the only
  thing reporting it — which is exactly what that guard is for.

  ## Provenance — established by A/B, not by inference

  Run against the pre-build tree (`c08f506`) in an isolated worktree and
  against the post-build tree: **the failing/erroring sets are IDENTICAL**
  (16 vs 16, set equality, zero new, zero fixed). The interior-wedge wiring
  did not cause this and did not fix it.

  This matters because the pre-commit gated-drift guard flagged these three
  classes when `_build_farfield_chart` lost its keyword-only `definition`
  parameter. That flag was CORRECT to fire (skipped tests cannot report their
  own breakage) but the breakage it pointed at is older than the change that
  triggered it. The commit was made with
  `GATED_DRIFT_ACK="EpsRegistrationGateTestCase,EpsGateResumeTestCase,SelfFalsificationTestCase"`
  on the strength of the A/B above.

  ## Why this is worse than an ordinary red test

  These are TIER-GATED, so a default run skips them and the suite reports
  green. They have presumably been vacuous for some time without anyone
  noticing: a test that asserts nothing and a test that passes look the same
  in a skip-heavy summary. Every guarantee these classes were written to
  provide — eps-gate registration, resume determinism, the poisoned-chart
  self-falsification set — is currently UNENFORCED.

  ## Work

  - Find why `_wp1_gate_fixture()` produces no charts. It builds three real
    engine-backed far-field charts via `_build_farfield_chart` with
    `_GATE_GAMMA_BAND` / `_GATE_HALF = (0.25, 0.2)` (~:941); the likely
    suspects are a tile that no longer admits under current thresholds, or a
    `half` that is now out of range for the `(s, d)` bridge.
  - Restore the assertions, then confirm non-vacuity by mutation: poison a
    chart and check the test actually FAILS.
  - Then consider whether these belong in the training tier at all — 94 s for
    the class set is fast enough that the gate may be costing more coverage
    than it saves.

  ACCEPTANCE: the three classes pass under `COGWHEEL_TRAIN_TIER=1` with a
  non-zero comparison count, and a deliberately poisoned chart makes them
  fail.
