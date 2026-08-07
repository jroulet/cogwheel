# Librarian Short-Term Observations

## 2026-08-07 _validate_farfield_axis_schema deletion post-commit sync

**Scope**: Single pending commit from sync_issues.json:
- `c8fd88c` — `fix(lensing): restore _validate_exterior_polar_axis_schema (accidentally deleted)`

**What the commit did**:
1. Restored `_validate_exterior_polar_axis_schema` function in `cogwheel/lensing/surrogate.py`
   (it had been accidentally deleted in a prior commit).
2. Deleted the orphaned `_validate_farfield_axis_schema` function as dead code — this function
   was a thin wrapper over `_validate_axis_schema` binding `_KNOWN_FARFIELD_AXIS_SCHEMAS`, which
   no longer had any callers since `FarFieldChart` was deleted in `0a31fcf`.
3. Updated the docstring of `_validate_axis_schema` to reference `_KNOWN_EXTERIOR_POLAR_AXIS_SCHEMAS`
   instead of `_KNOWN_FARFIELD_AXIS_SCHEMAS`.

**What needed updating**:
- `.claude/spec/todo.d/lensing_farfield_name_spans_three_regimes.md`: "Remaining scope" list
  still cited `_validate_farfield_axis_schema` as a name that needed renaming. Since the function
  was deleted (not renamed), it should be removed from the list. Fixed.

**Confirmed no-ops**:
- SPEC.md: no reference to either function name
- DATA_CONTRACTS.yaml: no reference
- FINDINGS.md: no reference
- docs/source/: no reference
- `.claude/handoff/brief_saddle_lobe_serve.md`: references `_validate_farfield_axis_schema` but
  is historical operational context, not a doc surface — left as-is.
- Test files referencing deleted function: code files, not librarian scope (Inspector/Coder territory).

**Files changed**:
- `.claude/spec/todo.d/lensing_farfield_name_spans_three_regimes.md` — removed `_validate_farfield_axis_schema` from Remaining scope list
- `.claude/spec/TODO.md` — regenerated from fragments

**Pattern to watch**:
TODO fragments that list "remaining scope" of a rename task go stale silently when one of the
listed symbols is DELETED rather than renamed. The fragment only tracks what still exists to
rename — deletion resolves the item without renaming it.

**Known stray side effects from render_fragments.py**:
- `.claude/tidy_advisory.json` and `.claude/agent_state/librarian.json` always get a stray diff
  from `render_fragments.py`. Reverted both before committing (git checkout --).
