# Librarian Short-Term Observations

## Run: 2026-08-03 — post-commit audit after InteriorWedgeChart DD w-ceiling + arc-length axis (56a223a)

### Scope

sync_issues.json covered commit 56a223a: "feat(lensing): InteriorWedgeChart DD w-ceiling + arc-length axis".
Changed code: `cogwheel/lensing/surrogate.py` (43 lines, WP1). New test: `cogwheel/tests/test_lensing_wedge_dd_arclength.py` (666 lines, test-only → skipped per triage rule).

### What was stale

**SPEC.md InteriorWedgeChart "Certified by" line** — the new test file `test_lensing_wedge_dd_arclength.py` directly tests two new `from_wedge_engine` capabilities (DD-product w-ceiling + theta_to_s construction) but was absent from the certification list. Added it alongside `test_lensing_interior_wedge_chart.py` with a parenthetical description.

### What was NOT stale (verified)

- "Optional `theta_to_s` (shape `(2, N_map)`) reparametrises the `theta_wedge` axis..." — still accurate; class accepts `None` for backward compat, even though `from_wedge_engine` now always builds it.
- "eliminating the DD cap bottleneck for high-w draws at small `|y|`" — design-rationale sentence; still true (more so now that the training cap is enforced).
- DATA_CONTRACTS "optional theta_to_s shape (2, N_map)" — still accurate.
- `lensing_remaining_coverage_gaps.md` TODO fragment — NOT completed by this commit. The fragment's items are: (1) d-axis normalization [→ spec], (2) ppGO handoff above chart w-ceiling [→ spec], (3) ppGO interior certification fix [research], (4) sidecar silent death [housekeeping], (5) xdist gate infra [housekeeping]. None are done; the commit only implements the training-side w-ceiling, not the serving-side handoff (item 2).

### Files changed

- `.claude/spec/SPEC.md` — "Certified by" line extended
- `.claude/spec/SPEC_CHANGELOG.md` — regenerated
- `.claude/spec/spec_changelog.d/2026-08-03_wedge_dd_arclength_tests.md` — new fragment (bump: patch)
- `.serena/memories/librarian_short_term.md` — this file

### Pattern identified

When a commit adds a new test file covering existing-class features without adding a new class, the ONLY stale surface is the "CERTIFIED BY" citation in the SPEC.md row for that class. The rest of the architecture description stays accurate. Triage shortcut: if changed_files contains a new `test_*.py` and an existing production module (no new class), check only the "Certified by" sentence for that module's class in SPEC.md.

### Fragile cross-references carried forward

- `test_lensing_interior_wedge_chart.py` AND `test_lensing_wedge_dd_arclength.py` are now both cited in SPEC.md for InteriorWedgeChart — renaming either breaks the citation.
- `_WEDGE_AXIS_SCHEMA = 'wedge_caustic_relative_v1'` cited in SPEC.md + DATA_CONTRACTS.yaml — rename in code requires both doc updates.
- `_WedgeCausticMap` cited in SPEC.md — rename/removal goes stale silently.
- `lensing_remaining_coverage_gaps.md` items 1 and 2 are `[→ spec]` — watch commits touching `FarFieldChart`, `InteriorWedgeChart.from_wedge_engine`, or ppGO paths as completion triggers.
- `_DD_PRODUCT_MARGIN = 58.0` now in surrogate.py (duplicated from surrogate_training.py) — a value change needs to be updated in both files AND in the prior description in SPEC.md ("keeping `w*sqrt(s) <= 58` by construction").
