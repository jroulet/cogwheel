# Librarian Short-Term Observations

## Run: 2026-08-03 — post-commit audit after InteriorWedgeChart

### Scope

sync_issues.json covered commit ff06b8a: `InteriorWedgeChart` class (812
insertions into surrogate.py), new test file (skipped), binary npz update,
agent-state and memory files (agent-only, skipped).

### What went stale and why

**SPEC.md row 55 (surrogate chart collection)** — the surrogate section listed
three chart types: TubeChart, FarFieldChart, LobeInteriorChart. A fourth type,
`InteriorWedgeChart` (positive-parity astroid interior), was added in this
commit but not mentioned. Pattern: a new class added to surrogate.py that
covers a previously un-charted region (astroid interior, gamma < 1) creates a
new SPEC row entry. The flag is "new public class that's not in SPEC's chart
collection list" — scan symbols_overview for new Classes whenever surrogate.py
has large insertions.

**DATA_CONTRACTS.yaml lens_amplification_surrogate** — the description
ended with LobeInteriorChart axis-schema tags but did not mention
`InteriorWedgeChart`'s `axis_schema='wedge_caustic_relative_v1'`
(`_WEDGE_AXIS_SCHEMA`). Pattern: whenever a new chart class is added with a new
axis_schema constant, the DATA_CONTRACTS.yaml description needs a matching
entry for the record format, fields, and tag semantics.

### Surfaces checked and found clean

- overview.rst: no chart implementation names cited; no update needed.
- api.rst: no new top-level cogwheel module/subpackage; `:recursive:` autosummary
  covers lensing subpackage without manual entry.
- DATA_CONTRACTS.yaml `certified_ppgo_map`: already has a complete entry (binary
  npz changed but schema/description unchanged — just retraining result).
- todo.d fragments: lensing_coverage_map.md region 1 still correctly OPEN (the
  chart type exists but training quality at high-gamma crown band is unmeasured).
  No todo fragment to close.
- sync_derived_docs.py: "5 checks run, some issues auto-fixed" with no actual
  git diff → no-op per institutional memory rule. Consumer-graph warnings about
  test-file callers of lens_amplification_surrogate: skipped per convention
  (test-only callers stay off the consumer list).
- `.claude/tidy_advisory.json` untracked: stray artifact from render_fragments.py,
  NOT committed per institutional memory.

### Fragile cross-references to watch next run

- `_WEDGE_AXIS_SCHEMA = 'wedge_caustic_relative_v1'` is now cited in both
  SPEC.md and DATA_CONTRACTS.yaml. If this constant is renamed in code, both
  doc surfaces need simultaneous updating.
- `_WedgeCausticMap` is cited in SPEC.md. If renamed or removed, SPEC.md goes
  stale silently.
- lensing_coverage_map.md region 1 stays "OPEN at the high-gamma CROWN band"
  until a training run actually measures eps at that band.
- The InteriorWedgeChart certification test is `test_lensing_interior_wedge_chart.py`
  (1400 lines). If the test file is renamed, the SPEC.md CERTIFIED BY citation
  goes stale.
