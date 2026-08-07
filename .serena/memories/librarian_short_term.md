# Librarian Short-Term Observations

## 2026-08-07 FarFieldChart deletion post-commit sync

**Scope**: One pending commit from sync_issues.json:
- `0a31fcf` — `feat(lensing): delete FarFieldChart and (s,d) machinery (post-strand cleanup)`
  Changed files: `cogwheel/lensing/surrogate.py`, `cogwheel/tests/test_lensing_farfield_envelope.py`

**Stale surfaces found and fixed**:

1. **SPEC.md lines 64-67** (Key abstractions paragraph): The sentence "The `FarFieldChart` class
   (fold-adapted `(s, d)` FAR-FIELD-SMOOTH coordinates, tag
   `'farfield_arclength_s_perp_d_framewinv'`) is retained for backward
   compatibility with pre-ExteriorPolarChart artifacts." was removed. FarFieldChart is
   now fully deleted from the codebase, so this backward-compat note is dead.

2. **SPEC.md line 53** (NAMING HAZARD in the Microlensing engine table row): Removed
   `` `FarFieldChart` / `` from "NAMING HAZARD: `far-field` / `FarFieldChart` / `farfield_*`".
   The terminology `farfield_*` still applies (many functions in channels.py, surrogate.py,
   surrogate_training.py use the "far-field" naming for "outside the caustic") but the
   class itself is gone.

3. **DATA_CONTRACTS.yaml** (lens_amplification_surrogate description): Removed the sentence
   "The prior FarFieldChart class (fold-adapted (s, d) FAR-FIELD-SMOOTH coordinates, tag
   'farfield_arclength_s_perp_d_framewinv') is retained for backward-compatible loading of
   pre-ExteriorPolarChart artifacts; _farfield_serves declines any saddle-labelled FarFieldChart
   and falls through to the engine as a safe compatibility response."

**Created changelog fragments**:
- `.claude/spec/spec_changelog.d/2026-08-07_farfield-chart-deleted.md` (bump: patch)
- `.claude/spec/contracts_changelog.d/2026-08-07_farfield-chart-deleted.md` (bump: patch)

**Skipped / not stale**:
- `docs/source/`: No mentions of FarFieldChart anywhere — confirmed clean.
- `reconstruct_farfield` / `FARFIELD_KERNEL_SUM` in SPEC.md: These functions still exist in
  `channels.py` and are still used throughout. The SPEC.md references to them are still correct.
- `_validate_farfield_axis_schema` in surrogate.py: Still present in code at line 3758 but
  references a schema that no longer applies to any live chart. This is a code-level dead-code
  issue (Inspector territory), not a doc-sync issue.
- Docstring at surrogate.py line 3614: "An 8a single-box artifact loads as a one-chart
  `FarFieldChart` for backward compatibility." — stale code comment, cannot touch (code-only).

**Pattern from this run**: The previous librarian run (ExteriorPolarChart introduction) correctly
anticipated the FarFieldChart deletion scenario and flagged exactly the sentences that went stale.
The prediction-then-fix pattern worked: the short-term memory from a prior run predicted exactly
which two passages would go stale when FarFieldChart was deleted. Read short-term memory carefully
before any post-commit that involves class deletions.

**Side effect note**: render_fragments.py modified `.claude/tidy_advisory.json` as a side effect —
reverted with `git checkout --` before committing (known behavior, documented in librarian_knowledge).
