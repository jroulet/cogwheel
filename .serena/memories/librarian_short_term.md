# Librarian Short-Term Observations

## Run: 2026-08-12 — post-commit sync for 288f37c

**Scope**: commit `288f37c` — "fix(lensing): saddle corridor origin-rho misclassification in ppGO/Born serving"

**Partially pre-done**: Two other commits preceded this librarian run:
- `288f37c` already updated SPEC.md certified-by list (added `test_lensing_saddle_rho_guards.py`) and the spec_changelog fragment (`2026-08-12_saddle_origin_rho_guards.md`).
- `ba7a2e9` (a separate docs commit between the fix and this librarian run) already retired the `lensing_saddle_origin_rho_assumption.md` TODO fragment to `completed.d/2026-08-12_saddle_origin_rho_assumption.md` and regenerated TODO.md/COMPLETED.md.

So when this librarian run started, MOST of the sync was already done by the builder and the separate retire commit.

**What was stale and why**:
- SPEC.md Born rung section, lines 147-149: described `classify_fallthrough` as attributing 'born' by checking `rho > 1` only. The fix commit (`288f37c`) added a second criterion at lines 290-291 of `surrogate_census.py`:
  ```python
  if gamma > 1.0 and image_count == 2:
      return 'born'
  ```
  This marks saddle corridor sources (deltoid corridor where `rho < 1` but `image_count == 2`) as 'born'. The SPEC sentence was left with only `rho > 1` — the builder updated the certified-by list but missed this prose sentence.
- Fixed to: "by `rho > 1` (exterior-to-caustic) OR, for the saddle (`gamma > 1`), `image_count == 2` (corridor source -- the deltoid caustic does not enclose the origin, so `rho < 1` does not imply interior on the saddle)"

**Fragment created**: `.claude/spec/spec_changelog.d/2026-08-12_saddle_born_census_criterion.md` (patch bump)

**render_fragments.py output**: SPEC_CHANGELOG.md updated; spec_version bumped (now includes new patch fragment).

**DATA_CONTRACTS.yaml**: No census born-category description there; no changes needed.

**Skipped**: All 4 test-only files (born, ppgo_map, saddle_rho_guards, surrogate_census tests — triage table: notebook/test-only = skip entirely).

**sync_derived_docs.py**: Only the pre-existing `lens_amplification_surrogate` test-consumer warning (same as last run, escalated via open TODO `surrogate_contract_test_consumer_warning.md`). No new issues.

**Fragile cross-refs created**: The born-attribution sentence now cites both `gamma > 1` and `image_count == 2`. If a future build changes either condition in `classify_fallthrough`, the sentence goes stale. The `classify_fallthrough` docstring (line 244 in surrogate_census.py) still says "rho = |y| / caustic_reach > 1" only — this is a code docstring (read-only for Librarian), so it was left for Inspector/Coder to update.

**Pattern noted**: The builder who committed 288f37c updated the SPEC certified-by list but left the 'born' prose sentence stale. This is a recurring pattern: SPEC prose sentences that describe runtime behavior (not just module locations or test coverage) are missed when a quick "SPEC bump patch" adds only to the certified-by list. Watch for: any commit that says "SPEC bump patch" + adds a test + changes runtime behavior — the prose description of that behavior may also need updating.

**Pre-existing wt dirty files NOT touched**: `.claude/agent_state/*.json`, `.claude/tidy_advisory.json`, other agents' `.serena/memories/*.md` — these are from other agents' uncommitted state, correctly excluded from this commit.
