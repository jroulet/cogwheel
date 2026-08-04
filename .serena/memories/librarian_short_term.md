# Librarian Short-Term Observations

## Run: 2026-08-04 — Cusp arm coverage enabled (commit ddd8980)

**Scope:** Post-commit sync for commit `ddd8980` (feat(lensing): enable cusp arm
coverage — `_CUSP_ARM_COVERAGE = 0.07 rad`).

**Changed files of doc relevance (from commit):**
- `cogwheel/lensing/surrogate.py` — `_CUSP_ARM_COVERAGE` changed from 0.0 to 0.07;
  `_tube_serves` comment updated

**Stale surfaces found and fixed:**

1. **SPEC.md** — Chart selection description contained "cusp neighborhoods are EXCLUDED
   (2/3-power singularity; served exact until the cusp fast-serving build)". This was a
   forward-looking status sentence. Updated to describe current behavior: draws within
   `_CUSP_ARM_COVERAGE = 0.07 rad` of the cusp vertex are now served by the Pearcey arm;
   residual window beyond the arm's certified reach falls through to the exact engine.
   Created `spec_changelog.d/2026-08-04_cusp-arm-coverage.md` (bump: patch).

2. **FINDINGS.md F040** — The finding said "`_CUSP_ARM_COVERAGE` was never going to be
   pinned by a census." Added addendum noting it was pinned at 0.07 rad by direct boundary
   sweep (not census, not analytic derivation). The core finding (w-dependent delta_theta)
   remains open.

3. **`todo.d/likelihood_cusp-fast-serving.md`** — Deleted; this build discharged the
   cusp fast-serving TODO. Created `completed.d/2026-08-04_cusp-arm-coverage.md`.

4. **`todo.d/lensing_coverage_map.md`** — Row 4 (cusp neighbourhoods) updated from
   "OPEN, both parities" to "PARTIALLY CLOSED (ddd8980, 2026-08-04)". Section B item 3
   (Cusp fast-serving) marked DONE with commit reference.

**Surfaces confirmed NOT stale:**
- `docs/source/` — no RST pages cover `_CUSP_ARM_COVERAGE` or cusp arm serving detail
- `DATA_CONTRACTS.yaml` — no new disk artifact (the constant is applied at query time,
  not stored; `pearcey_table.npz` contract already existed from c715bcd)
- `COVERAGE_DESIGN.md` — had no cusp arm status sentences to update
- `CHANGELOG.md` — no `changelog.d/` directory in this repo's `.claude/spec/`

**Stale pattern this commit reveals:**
- STATUS SENTENCES IN SPEC that say "until the X build" go stale the moment X ships.
  Pattern: after any build that closes an explicitly named "pending build" reference in SPEC,
  grep SPEC.md for that build name and update the sentence.
- FINDINGS that pre-diagnose why a measurement "can't happen" go partially stale when the
  measurement happens via a different route. Add addendum rather than removing — the core
  finding may still be valid (as here: F040's w-scaling thesis is still open).

**Fragile cross-references to watch:**
- SPEC.md now cites `_CUSP_ARM_COVERAGE = 0.07 rad` — if this constant changes, SPEC.md
  must be updated.
- `lensing_coverage_map.md` row 4 status "PARTIALLY CLOSED" — closes fully once a census
  confirms near-zero cusp-window fall-through.
- F040 addendum references commit ddd8980 and the measurement script — stable references.
