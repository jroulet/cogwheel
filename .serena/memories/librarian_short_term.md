# Librarian Short-Term Observations

## Run: 2026-08-12 — post-commit sync for {897bff8, 26d088a, cfc4377}

**Scope**: three commits in .claude/sync_issues.json

**Result: NO-OP — no doc surfaces stale**

**Commit-by-commit triage**:

1. `897bff8` — "docs: post-commit sync (c8cad0c deltoid exterior cusp gap + mid-w ppGO band)"
   - This IS the previous librarian's own post-commit sync commit. SPEC.md changes
     (ppGO full mid-w band, saddle CUSP-EXCLUSION now ADMITS, corridor parity guard)
     were already propagated in this commit itself.
   - `overview.rst` search for `cusp|ppGO|ppgo|FARFIELD|KERNEL_SUM|exclusion|high-w|mid-w|saddle corridor|force_minus_ghost` → zero matches. Not stale. Consistent with previous librarian's finding.
   - No new cogwheel modules, no dependency changes, no Sphinx RST edits needed.

2. `26d088a` — test-only change (`cogwheel/tests/test_lensing_airy_fold.py`)
   - Skip per triage rules: test-only → no-op.

3. `cfc4377` — spec housekeeping only (COMPLETED.md, TODO.md, completed.d/todo.d fragments)
   - Closes "consolidate duplicate routing pins" TODO (resolved in 26d088a).
   - No Sphinx RST, no data contracts, no cogwheel code → no-op.

**sync_derived_docs.py**: ran cleanly (via cogwheel-newlal python). "Some issues auto-fixed"
was a no-op state flush — confirmed by `git diff --name-only` showing only agent_state +
memory files, not doc files. (Known pattern: "auto-fixed" with no actual diff = internal flush.)

**Surrogate escalation TODO**: `.claude/spec/todo.d/surrogate_contract_test_consumer_warning.md`
EXISTS and is open — repeated `lens_amplification_surrogate` test-only-consumer warning from
sync_derived_docs.py is covered; no duplicate created, per escalation rule.

**What was NOT stale**:
- `overview.rst`: pitched at architecture level; none of the implementation details from
  these builds (ppGO gates, cusp exclusion d_exclude, FARFIELD_KERNEL_SUM_MINUS_GHOST,
  saddle corridor parity guards) appear there.
- `api.rst`, `crash_course.rst`, `installation.rst`: no relevant changes
- `DATA_CONTRACTS.yaml`: no new disk artifacts
- `SPEC.md`: already correct (897bff8 was the previous librarian's fix commit)

**Pattern confirmed**: the post-commit hook fires again after a librarian's own doc-sync
commit — the next librarian run triggered by that commit is always a no-op (the sync was
already done). This is expected; just verify overview.rst hasn't grown implementation detail
and move on.
