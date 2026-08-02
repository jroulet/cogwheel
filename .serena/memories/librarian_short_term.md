# Librarian Short-Term Observations

## Run: 2026-08-01, Build C11 — BornResidualChart wiring

### Scope
Single code change: new `cogwheel/lensing/born_residual_chart.py`
(frozen `BornResidualChart` dataclass) + wiring into
`likelihood._surrogate_coefficients` fact-4 slot.

### What went stale and why
- **SPEC.md line ~131**: The "STILL NOT wired into the serve path" sentence
  was stale because the coder wired the serve path (conditionally on chart
  attachment) — a classic case where a status-description sentence that
  says "X is not yet done" goes stale silently the moment X gets done.
  Fixed by replacing the 11-line "STILL NOT wired" block with a current
  description of the conditional wiring.

### Fragments created
- `.claude/spec/spec_changelog.d/2026-08-01_born_residual_wiring.md` — patch bump
- `.claude/spec/completed.d/2026-08-01_born_residual_chart_wiring.md` — C11 completion
- `lensing_born_b1_derivation.md` and `lensing_saddle_born.md` todo fragments
  updated to record C11 wiring landed; remaining work is TRAIN_TIER only.

### What was NOT stale
- `docs/source/api.rst` — `:recursive:` autosummary covers new module automatically
- `docs/source/overview.rst` — BornResidualChart is internal/TRAIN_TIER, not public API
- `DATA_CONTRACTS.yaml` — BornResidualChart has no `save`/`load` methods; pure in-memory
- `docs/source/crash_course.rst` — no import examples changed
- `cogwheel/lensing/__init__.py` — BornResidualChart is not a public export (correct)

### Fragile cross-references to watch
- The SPEC.md now says "the fact-4 slot... is now wired: when a
  `BornResidualChart`... is attached" — this will go stale if the
  dataclass is renamed or the conditional wiring is restructured.
- `lensing_born_b1_derivation.md` now says "What remains — TRAIN_TIER only"
  — once the chart is actually trained, this fragment should be closed
  (moved to completed.d).

### Surprises
- Inspector finding INS-11-002 correctly called out the stale SPEC sentence —
  sync was straightforward once the stale block was identified.
- `BornResidualChart` is a frozen dataclass with no serialization — confirmed
  no DATA_CONTRACTS entry needed.
- `sync_derived_docs.py` ran clean (only the pre-existing test consumer gap
  in `lens_amplification_surrogate`).
