# Build 1e-farfield-port — finish the `(s, d)` test port

## Mission

Finish the existing port of far-field construction and selection sites to the
implemented caustic-relative `(s, d)` API. This is a compatibility port, not a
physics redesign: preserve each test's oracle and tolerance. It is a test-only
build: set `is_test_only: true`, emit zero Coder WPs, and partition the listed
existing test files into explicit disjoint Test Developer descriptions. Complete
the Inspector and Professor review plus required fast verification, then leave
the tree ready for the driver post-build full fast-suite gate.

## Current facts

- Worktree source and test changes already equal
  `refs/sdk/farfield_port_live` (`f587dc9`). Do **not** restore any older ref
  and do not discard staged work.
- That snapshot contains the restored implementation and the port of
  `test_lensing_exterior_admission.py`, `test_lensing_exterior_windows.py`,
  `test_lensing_farfield_envelope.py`, and
  `test_lensing_surrogate_census.py`.
- A later porter investigated `test_lensing_surrogate.py` but made no source
  or test edit. `scratch_ff_probe*.py` are diagnostic artifacts only; do not
  use them as test oracles and do not delete unrelated worktree files.
- The prior build's lobe and training porters never ran: its Claude CLI message
  reader failed, followed by an unreviewed Inspector loop. This is an SDK
  interruption, not a code finding.
- The changed API is real: `FarFieldChart.from_values` no longer accepts
  `rho_grid`; `LensAmplificationSurrogate.from_engine` no longer accepts
  `rho_range`; `select_chart` no longer accepts `rho`; the old top-level
  `rho_grid` and `theta_c_grid` accessors are gone; and `_evaluate_chart` /
  `_farfield_serves` have the new coordinate arguments.
- Current gated-drift output is authoritative. It names skipped references in
  the four already-touched files and in
  `test_lensing_surrogate_training.py` and
  `test_lensing_ppgo_bandsplit.py`; a passing ordinary fast tier cannot clear
  those. Run `python .claude/hooks/check_gated_test_drift.py` against the
  staged port and resolve every named stale reference.
- The last full collection baseline was 1171. Any numerical value mismatch is
  a finding: stop and report it rather than loosening a tolerance or changing
  an oracle.

## Scope

IN: all remaining construction/selection call sites for the new far-field API,
including `test_lensing_surrogate.py`,
`test_lensing_surrogate_lobe.py`,
`test_lensing_surrogate_training.py`,
`test_lensing_ppgo_bandsplit.py`, and the specific skipped sites reported by
the gated-drift check in already-touched suites. Keep the original test claim;
only translate how its fixture constructs, projects, or selects a chart.

OUT: new acceptance tests; any change to the `(s, d)` implementation except a
demonstrated port blocker; coordinate-policy changes; chart retraining; engine
or physics campaigns; slow accuracy/timing sweeps; unrelated scratch cleanup.

## Acceptance

1. Targeted changed-suite fast tests pass and no anti-vacuity guard remains.
2. The gated-drift check is clean, or every still-named skipped test was run in
   its own tier and acknowledged specifically—never blanket-bypassed.
3. Existing numerical claims, reference oracles, and tolerances are unchanged.
4. The `DATA_CONTRACTS.yaml` change has its required changelog fragment and
   rendered generated outputs are current.
5. The build review reaches a parseable clean verdict. The driver will run the
   full 1171-collection fast gate afterward; slow tiers remain post-build only.

## Constraints

- Branch `claude-dev`; preserve the current staged WIP and other users' files.
- Keep the plan to at most three focused work packages. This is a port.
- Test only files changed by a role, with small/synthetic fixtures. Do not run
  slow sweeps or a bulk campaign inside the build.
- No `git show HEAD` test oracle. A historical ref may establish restoration
  provenance but must not participate in a committed test.
