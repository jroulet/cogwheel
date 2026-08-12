# Librarian Short-Term Observations

## Run: 2026-08-12 — post-commit sync for {27e458b}

**Scope**: one commit in `.claude/sync_issues.json`

**Result: NO-OP — no doc surfaces stale**

**Commit triage**:

1. `27e458b` — "docs: record deltoid exterior geometry bugs (lobe-local coords, negative corridor rho, cusp exclusion frame, disconnected exterior topology)"
   - Changed files: `.claude/spec/TODO.md` and `.claude/spec/todo.d/lensing_deltoid_exterior_geometry.md`
   - This is a pure spec/TODO bookkeeping commit: added a new `todo.d` fragment recording four deltoid exterior geometry bugs (origin-based coords, negative corridor rho, wrong cusp exclusion frame, topologically disconnected exterior). Also regenerated `TODO.md`.
   - No `cogwheel/` Python changes, no dependency changes, no API changes, no data contract changes.
   - Triage result: skip per "spec/TODO-only change" rule (analogous to test-only rule).

**sync_derived_docs.py**: ran cleanly (no-op state flush; git diff showed only unrelated agent_state + other agents' memory files). The known `lens_amplification_surrogate` test-only-consumer warnings are the pre-existing ones; escalation TODO fragment exists and is open — no duplicate created.

**What was NOT stale**:
- `overview.rst`, `api.rst`, `crash_course.rst`, `installation.rst`: no relevant changes
- `DATA_CONTRACTS.yaml`: no new disk artifacts
- `SPEC.md`: not touched by this commit
- `FINDINGS.md`: bugs recorded in `todo.d`, not yet in FINDINGS.md (correct: they are open/unresolved bugs, not findings)

**Pattern confirmed**: post-commit hooks fire after spec/TODO-only internal commits too (not just code commits). These are always no-ops for doc surfaces. Just verify `git diff --name-only` is clean after `sync_derived_docs.py` and move on.

**New TODO fragment to watch**: `lensing_deltoid_exterior_geometry.md` tracks four deltoid exterior bugs; when the fix lands, the Librarian will need to propagate any SPEC.md updates and check for DATA_CONTRACTS.yaml changes (if a new lobe-centered exterior chart type ships with a new disk artifact).
