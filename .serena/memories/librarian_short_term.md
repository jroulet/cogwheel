# Librarian Short-Term Observations

## 2026-08-07 post-commit sync (commits d711934, 67338d6)

**Scope**: spec: record driver probe findings (exterior coordinate disease confirmed,
wedge probe invalid) + scripts: probe_wedge_v3.py update.

**Changed files in scope**:
- `.claude/spec/completed.d/2026-08-07_driver_probes_exterior_wedge.md` (new)
- `.claude/spec/COMPLETED.md` (regenerated)
- `scripts/probe_wedge_v3.py` (scripts-only, no-op per SCRIPTS/ REWRITE NO-OP RULE)
- Memory files (no doc surfaces)

**Issue found and fixed**: `lensing_exterior_recursion_never_measured.md` was not deleted
by the commit that added the completion fragment covering its work. The three measurements
it required (pass rate, depth histogram, depth-3 cap hits) are all answered in
`2026-08-07_driver_probes_exterior_wedge.md`. Deleted the stale TODO fragment.

**Cascade fix**: `lensing_exterior_should_chart_in_polar_not_sd.md` had
`depends_on: [..., lensing_exterior_recursion_never_measured]` — a dangling reference
after deletion. Updated to `2026-08-07_driver_probes_exterior_wedge` (the completed.d
entry covering that work). Render confirmed clean (no warnings).

**Files changed**:
- `.claude/spec/todo.d/lensing_exterior_recursion_never_measured.md` — deleted
- `.claude/spec/todo.d/lensing_exterior_should_chart_in_polar_not_sd.md` — depends_on updated
- `.claude/spec/TODO.md` — regenerated

**sync_derived_docs.py**: same four test-only-caller consumer-graph warnings for
`lens_amplification_surrogate` — pre-existing, already escalated via
`surrogate_contract_test_consumer_warning.md` todo fragment. No new action.

**CHANGELOG**: no `changelog.d/` directory in this repo; internal builds use
only `completed.d`/`todo.d`. No CHANGELOG entry needed for driver probe findings.

**Pattern noted**: when a driver records probe findings in a completion fragment
WITHOUT deleting the corresponding todo.d fragment, the renderer does NOT warn —
the dangling depends_on in downstream fragments only surfaces if render_fragments.py
is run after deletion. The two-step (delete todo + update depends_on) is easy
to miss in the same commit.

**Fragile cross-references to watch**:
- `lensing_exterior_should_chart_in_polar_not_sd.md` now depends on
  `2026-08-07_driver_probes_exterior_wedge` — if that completed.d fragment is
  renamed, the depends_on breaks silently (no dangling-link check for completed.d entries?).
- `lensing_wedge_probe_charts_need_retraining_under_v3.md` references
  `[[2026-08-07_lensing-training-path-per-region]]` — still valid.
