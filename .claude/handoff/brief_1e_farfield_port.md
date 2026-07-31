# Build 1e-farfield-port — restore the (s,d) coordinate and make the suite green

## Mission

The far-field `(s, d)` coordinate is already IMPLEMENTED. It is preserved at
the git ref `refs/sdk/coder_checkpoint` (commit `df176fdf`), written by a build
that died before any test was authored. Restore it and port the construction
sites the API change broke, until the full suite is green.

This build writes NO new acceptance tests. Those are the NEXT build
(1e-farfield-accept). Splitting them is deliberate: the previous attempt
bundled the port with 11 acceptance specs, and its Test Developer died twice at
`error_max_turns` with zero output.

## Restore first — HALF THE PORT IS ALREADY DONE

A previous attempt restored the implementation and ported three of the six test
files before the watchdog killed it mid-escalation (F059, now fixed). That work
is preserved. Start from it:

    git checkout refs/sdk/farfield_port_wip -- \
        cogwheel/lensing/ cogwheel/tests/ .claude/spec/DATA_CONTRACTS.yaml

ALREADY PORTED — do not redo, but DO re-verify they still pass:
    test_lensing_farfield_envelope.py, test_lensing_surrogate_census.py,
    test_lensing_exterior_windows.py

REMAINING:
    test_lensing_exterior_admission.py
    test_lensing_surrogate.py
    test_lensing_surrogate_lobe.py
    test_lensing_surrogate_training.py  <- SEE BELOW, this one is not optional

## test_lensing_surrogate_training.py is IN scope (the last plan got this wrong)

The previous plan scoped it out because the gate showed 0 failures there. Its
tests are SKIPPED (gated tier), so they CANNOT fail — that is the whole reason
the drift hook exists. Five classes reference the changed API and are silently
broken:

    ServeFractionTestCase             select_chart, from_values
    EpsRegistrationGateTestCase       select_chart
    SelfFalsificationTestCase         select_chart, from_values
    ArcLengthBoundShiftMarginTestCase from_values
    ArcLengthNodeEfficiencyTestCase   from_values

Run `python .claude/hooks/check_gated_test_drift.py` with the port staged to
get the current authoritative list; port every entry it names.

Read the restored code before changing it. It is a real implementation, not a
sketch: `_caustic_arclength_map` (gamma-RESOLVED, a 2-D `s(theta, gamma)` table
on the spline's own gamma nodes), `_to_farfield_smooth` / `_from_farfield_smooth`,
cusp-span and near-tied-foot rejection guards, and `_evaluate_chart` dispatching
the far-field branch through `_to_farfield_smooth` at the query's OWN gamma.

## Measured facts — the exact failure surface (2026-07-30, full gate)

Collection is CLEAN at 1171. The code imports and the suite collects; it is the
old construction sites that break.

    59 failed, and every one is a construction-site failure:
      68  TypeError    — FarFieldChart.from_values() / select_chart() signatures
      58  AssertionError — anti-vacuity guards ("the test made zero
                           comparisons"), i.e. a fixture could not BUILD, so
                           the test body asserted nothing and the guard
                           correctly refused to let it pass silently
       0  numerical mismatches — nothing disagrees on a VALUE, anywhere

    failures by file:
      29  cogwheel/tests/test_lensing_surrogate.py
       9  cogwheel/tests/test_lensing_surrogate_census.py
       8  cogwheel/tests/test_lensing_exterior_windows.py
       6  cogwheel/tests/test_lensing_surrogate_lobe.py
       4  cogwheel/tests/test_lensing_farfield_envelope.py
       3  cogwheel/tests/test_lensing_exterior_admission.py

Zero numerical failures is the load-bearing fact: this is a PORT, not a
physics repair. If a ported test fails on a VALUE, that is a real finding —
stop and report it rather than adjusting the expectation.

## Scope

IN — port the ~60 construction/selection sites to the `(s, d)` API across those
six files; keep each test's ORIGINAL claim intact (same oracle, same tolerance,
same thing asserted); the `contracts_changelog.d/` fragment for the
`DATA_CONTRACTS.yaml` change the restored code carries.

OUT — the 11 acceptance tests (next build); any change to the restored
coordinate implementation beyond what porting demands; tube charts; lobe
charts; the `w` and `gamma` node measures; any training or engine sweep.

If porting a site is impossible without changing what it asserts, that site is
telling you something. Report it; do not weaken the assertion.

## Acceptance

1. Full suite GREEN, 1171 collected, driver-verified post-build.
2. No test's claim was weakened to achieve it. For every ported test, the
   oracle and tolerance are unchanged; only the construction call changed.
3. Any test that CANNOT be ported without changing its claim is reported
   explicitly, with the reason, rather than edited into passing.
4. `contracts_changelog.d/` fragment present with `bump:`;
   `python scripts/render_fragments.py` run.
5. No anti-vacuity guard is left tripped anywhere.

## Constraints

- Branch `claude-dev`.
- **Every domain-test description MUST name its target suite file**
  (`test_<x>.py`) in its text. A description that names no file is routed to
  cross-suite and, with several suites in play, is appended to every agent's
  prompt without being counted by the shard cap or the `60 + 20*n` budget
  (F057 — that is precisely how the previous attempt died).
- Slow tiers stay empty in-build; fast tests only. Agents verify ONLY the
  tests they changed; the driver runs the full tally once.
- Assert VALUES, not code paths. No `git show HEAD` oracle (pre-commit
  enforced) — note this build legitimately reads a git REF to restore source,
  which is not an oracle; no TEST may read a ref.
- Keep the WP count at or below 3. This build is a port; it does not need a
  wide decomposition, and an over-wide plan will be rejected at the gate.
