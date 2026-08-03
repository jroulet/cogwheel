# Librarian Short-Term Observations

## Run: 2026-08-03 — WP1 min_gamma_band threshold reduction (0.02 → 0.005)

**Scope:** Build reduced `TrainingConfig.min_gamma_band` default and
`stable_gamma_bands` default `min_width` from 0.02 to 0.005, added 9-test
suite in `test_lensing_min_gamma_band.py`, fixed two comments in
`surrogate_training.py`.

**What was stale and why:**
Three doc surfaces cited `min_gamma_band = 0.02` as the *current default*:
1. `.claude/spec/todo.d/lensing_dropped_gamma_slivers.md` — two occurrences
   of "default `0.02`" and a "Do NOT close by lowering" note that needed
   updating since the threshold was lowered (but the TODO is still OPEN:
   mass measurement and treatment decision remain owed).
2. `.claude/spec/todo.d/lensing_coverage_map.md` — row 10 cited
   `min_gamma_band = 0.02`.
3. `.claude/spec/COVERAGE_DESIGN.md` — audit table entry for `min_gamma_band`
   listed 0.02.

SPEC.md itself does NOT pin the specific value — it names the concept and
references the mechanism without a literal constant, so it was not stale
(confirmed by inspector memory's note).

FINDINGS.md and historical handoff briefs cite 0.02 as empirical measurement
values taken at the old threshold — those are historical facts, not stale.

**Pattern confirmed:**
- SCRIPTS/ REWRITE NO-OP RULE: `scripts/measure_dropped_slivers.py` change
  was internal to scripts/ with no new disk artifacts → librarian no-op.
- Test-only file (`test_lensing_min_gamma_band.py`) → no doc surface updates.
- The "Do NOT close by lowering" advisory in a TODO fragment needs to be
  updated when a build does lower it, even if the TODO stays open.

**Fragile cross-references to watch:**
- `lensing_dropped_gamma_slivers.md` — TODO is still OPEN. Row 10 in
  `lensing_coverage_map.md` tracks this too. When sliver mass is finally
  measured and treatment decided, both fragments need DONE markers.
- `COVERAGE_DESIGN.md` C9 section still references the OLD `min_width = 0.02`
  measurement narrative ("at min_width = 0.02 the saddle drops 40.6%") as a
  historical measurement fact — this is intentionally left as-is (it records
  a past measurement, not the current default).

**Surprises:**
- The `lensing_dropped_gamma_slivers.md` fragment explicitly said "Do NOT
  close this by lowering `min_gamma_band`" — so the build did something the
  fragment advised against. The TODO is correctly kept open with updated wording.
- COVERAGE_DESIGN.md's C9 section cites "measured 2026-07-28 — at min_width
  = 0.02 the saddle drops 40.6%" as a historical measurement; that measurement
  text is accurate and should NOT be updated (it describes a past empirical
  run, not the current config).
