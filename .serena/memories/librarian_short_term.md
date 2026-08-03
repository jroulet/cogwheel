# Librarian Short-Term Observations

## Run: 2026-08-03 — post-commit audit after "docs+tidy: post-commit sync (DD ceiling + arc-length axis)" (d17f42e)

### Scope

sync_issues.json covered commit d17f42e: a "docs+tidy" commit that contained:
- `cogwheel/lensing/surrogate.py`: 3-line tidy fix (inline `_ARC_MAP_NODES = 2001` deduplicated
  to module-level `_FARFIELD_ARC_MAP_SIZE`). No public API change, no new behavior.
- `cogwheel/tests/test_lensing_wedge_dd_arclength.py`: 12-line tidy fix (inline import hoisting).
  Test-only → skipped per triage rule.
- `.claude/spec/contracts_changelog.d/2026-08-03_interior_wedge_dd_ceiling_arclength.md`: NEW fragment
  added (bump: patch). This was NOT accompanied by a re-render of `DATA_CONTRACTS_CHANGELOG.md`
  in that commit.
- `CHANGELOG.md`: already rendered in the commit itself.

### What was stale

**DATA_CONTRACTS_CHANGELOG.md** — the new contracts_changelog fragment
`2026-08-03_interior_wedge_dd_ceiling_arclength.md` was not rendered into
`DATA_CONTRACTS_CHANGELOG.md`. Ran `render_fragments.py` to add it as version `0.2.2`
(alphabetically between `interior_wedge_chart` and `wedge_dd_arclength_contracts`).
The existing `0.2.2` became `0.2.3` (re-numbering is the known rendering quirk,
not a bug to fix).

### What was NOT stale (verified)

- `docs/source/overview.rst` / `api.rst` / `crash_course.rst`: no public API change from tidy fix.
- SPEC.md "Certified by" line: already updated in prior post-commit run (283d435).
- `sync_derived_docs.py` consumer-graph warnings: pre-existing test-file-only callers,
  excluded by convention (stay off in DATA_CONTRACTS.yaml consumer lists).
- `tidy_advisory.json`: `sync_derived_docs.py` produces a spurious diff here; reverted with
  `git checkout --`.

### Files changed

- `.claude/spec/DATA_CONTRACTS_CHANGELOG.md` — rendered from new fragment

### Pattern identified

A "docs+tidy" commit that adds a `contracts_changelog.d/` fragment may NOT include the
rendered `DATA_CONTRACTS_CHANGELOG.md` if rendering was missed. Always check whether
`render_fragments.py` output matches the fragment count in `contracts_changelog.d/`.
The quick check: `grep -c "^- \`0\." DATA_CONTRACTS_CHANGELOG.md` should equal the number
of fragments in `contracts_changelog.d/`.

### Fragile cross-references (continued from prior run)

- `_DD_PRODUCT_MARGIN = 58.0` duplicated in surrogate.py and surrogate_training.py —
  a value change needs both files AND the w-ceiling description in SPEC.md and DATA_CONTRACTS.
- `_FARFIELD_ARC_MAP_SIZE` now used as arc-length map size for wedge tiles too
  (the tidy fix that motivated this run) — if renamed, the wedge arc-map resolution changes silently.
- `lensing_remaining_coverage_gaps.md` items 1 and 2 remain open `[→ spec]` —
  watch commits touching ppGO paths or `from_wedge_engine` w-ceiling serving logic.
