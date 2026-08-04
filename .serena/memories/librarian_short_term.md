# Librarian Short-Term Observations

## Run: 2026-08-04 — Retire stale 'annulus' terminology (commit abd4ce40)

**Scope:** Post-commit sync for commit `abd4ce40` (chore: retire stale 'annulus' terminology
across codebase — 97 code references updated: 'far annulus' → 'Born exterior' / 'exterior region',
'annulus rung' → 'exterior rung', 'Born-annulus' → 'Born residual', 'annulus coordinate' → 'rho coordinate').

**Changed files of doc relevance (from commit):**
- `cogwheel/lensing/` Python files (born_residual_chart.py, _born.py, channels.py, likelihood.py,
  ppgo_map.py, surrogate.py, surrogate_census.py, surrogate_training.py)
- `.claude/spec/COVERAGE_DESIGN.md` — already updated by the commit itself (8 changes)
- `.claude/hooks/retired_concepts.json` — ANNULUS_INNER_RADIUS, GAMMA_FENCE, _SADDLE_GAMMA_FENCE added

**Stale surface found and fixed:**
- `SPEC.md` line 138 (Key abstractions → BornResidualChart bullet): "draws in the annulus fall
  through to the exact engine" → "exterior draws fall through to the exact engine"
  — was using the retired 'annulus' term to describe current behavior of the exterior region.

**Surfaces confirmed NOT stale:**
- `SPEC.md` line 101: "The prior-box annulus `3.0 < |y| <= 4.2426`... were retired in C8 (F036)"
  — historical record of the retired concept; appropriate to name the old term.
- `SPEC.md` line 55 (FAR-FIELD TILING): "the ~98% of the annulus that is genuinely exterior"
  — geometric/colloquial noun describing a ring-shaped region, not a code identifier; left as-is.
- `COVERAGE_DESIGN.md` lines 192, 194, 235: all in historical audit section describing what
  existed before C8 retired the annulus concept ("retires the annulus concept") — appropriate.
- `FINDINGS.md`: all 10+ references are historical findings with time-stamped context — left as-is.
- `docs/source/**/*.rst`: zero annulus references — confirmed clean.

**sync_derived_docs.py output:** "5 checks run, some issues auto-fixed" — confirmed no-op
(only tidy_advisory.json dirtied as stray side-effect; reverted before commit).

**Stale pattern this commit reveals:**
- TERMINOLOGY RETIREMENTS PROPAGATE INCOMPLETELY: a commit that sweeps code identifiers often
  misses isolated doc-surface uses of the same term in current-behavior sentences. Pattern:
  search SPEC.md for the retired term AFTER any terminology-retirement commit; the code sweep
  won't touch .claude/spec/SPEC.md.
- Three-way distinction matters: (1) current-behavior uses of retired term → must update;
  (2) historical-context sentences naming the retired concept → correct to keep old name;
  (3) geometric/mathematical use as plain noun → acceptable, judgement call.

**Fragile cross-references to watch:**
- SPEC.md "born exterior" / "exterior region" language — next rename commit that touches
  these must also sweep SPEC.md.
- `.claude/hooks/retired_concepts.json` is now the canonical list of retired names; if
  SPEC.md grows new references to ANNULUS_INNER_RADIUS or GAMMA_FENCE, those are violations.
