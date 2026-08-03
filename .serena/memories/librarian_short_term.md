# Librarian Short-Term Observations

## Run: 2026-08-03 — post-commit audit after lensing_remaining_coverage_gaps TODO filing

### Scope

sync_issues.json covered commit 7dbae47: "spec: file remaining coverage gap TODOs".
Changed files: `.claude/spec/TODO.md` (regenerated) and
`.claude/spec/todo.d/lensing_remaining_coverage_gaps.md` (new fragment).

### Outcome: no-op — all surfaces clean

The commit added five new TODO items to the backlog:
- Far-field d-axis normalization by curvature radius `[→ spec]`
- ppGO handoff above chart w-ceiling for interior draws `[→ spec]`
- ppGO interior certification fix `[research]`
- Sidecar callback silent death `[housekeeping]`
- xdist tree-gate infra fix `[housekeeping]`

None of these items were **completed** — they were only **added**. The
`[→ spec]` tag means SPEC.md updates are required when these items are
finished, not when they are filed. No cogwheel/*.py code changed; no
SPEC.md changed; no API signatures changed; no disk artifacts added.

Triage conclusion: zero downstream doc surfaces are stale. No doc fixes,
no commit. Trigger file deleted.

### Pattern identified

TODO-only commits (adding fragments, regenerating TODO.md) are the most
common no-op post-commit trigger. The triage shortcut: if changed_files
contains ONLY `.claude/spec/TODO.md` + `.claude/spec/todo.d/*.md` (and no
code, SPEC.md, or `completed.d/` files), the answer is always no-op — the
`[→ spec]`/`[→ docs]` tags only fire on completion, not filing.

### Fragile cross-references carried forward from last run

- `_WEDGE_AXIS_SCHEMA = 'wedge_caustic_relative_v1'` cited in SPEC.md +
  DATA_CONTRACTS.yaml — rename in code requires both doc updates.
- `_WedgeCausticMap` cited in SPEC.md — rename/removal goes stale silently.
- lensing_coverage_map.md region 1 remains OPEN (high-gamma crown band
  measurement not yet done).
- `test_lensing_interior_wedge_chart.py` cited in SPEC.md — rename breaks
  the CERTIFIED BY citation.
- New fragile ref: once the `[→ spec]` items above are completed, SPEC.md
  will need d-axis normalization and ppGO-handoff architecture described.
  Watch for commits touching `FarFieldChart`, `InteriorWedgeChart`, or
  `fold_ppgo_correction` as triggers.
