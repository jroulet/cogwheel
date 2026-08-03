# Librarian Short-Term Observations

## Run: 2026-08-03 — min_gamma_band=1e-6 fix (commit 70affbb)

**Scope:** Build set `TrainingConfig.min_gamma_band` and `stable_gamma_bands(min_width=...)`
default from `0.005` to `1e-6`. With the log-reach gamma axis, any band is
well-spaced, making the old 0.005 floor redundant. Total dropped prior mass at
the new threshold: ~1.5e-6 (~1e-6 fraction) — negligible. Comment in `train()`
also updated to describe correct behavior.

**What went stale and why:**

1. `todo.d/lensing_dropped_gamma_slivers.md` — all three owed items (measure, fix
   comment, decide treatment) are now done: mass measured (~1e-6 fraction),
   comment fixed in the build commit, treatment not needed at this residual mass.
   TODO closed → moved to `completed.d/`.

2. `todo.d/lensing_coverage_map.md` — row 10 had "OPEN. `min_gamma_band = 0.005`
   (lowered from 0.02 in WP1); total prior mass at new threshold NEVER MEASURED".
   Updated to CLOSED status. Section B item 5 also needed to be marked CLOSED.

3. SPEC.md — training paragraph described "metamorphosis slivers below
   `min_gamma_band` are DROPPED refusal-conservatively and recorded" and the
   test cert said "at reduced `min_gamma_band = 0.005` threshold". Both needed
   updating to describe the new behavior.

**sync_derived_docs.py behavior — important pattern:**

The script ran as Step 0 and made the correct structural changes
(deleted `lensing_dropped_gamma_slivers.md`, created
`completed.d/2026-08-03_min_gamma_band_zero.md`,
`spec_changelog.d/2026-08-03_min_gamma_band_zero.md`, updated SPEC.md and
`lensing_coverage_map.md`). BUT it used stale/wrong values throughout:
- **Wrong value**: wrote `0.0` everywhere instead of `1e-6`
- **Wrong commit**: wrote `2e01ae9` everywhere (not in git log) instead of `70affbb`

I had to manually fix all four locations:
- `SPEC.md` (2 occurrences)
- `completed.d/2026-08-03_min_gamma_band_zero.md`
- `spec_changelog.d/2026-08-03_min_gamma_band_zero.md`
- `todo.d/lensing_coverage_map.md` (row 10 and Section B item 5)

**Root cause of sync_derived_docs.py errors:** The script appears to use
internal state/heuristics to determine what changed rather than reading the
actual committed values. For constant-change builds, ALWAYS verify the values
sync_derived_docs.py writes against the actual code (grep for the constant in
the source file), especially:
1. The numeric value of constants
2. The commit hash cited in coverage-map rows and completed entries

**Fixes applied:**
- Corrected `0.0` → `1e-6` and `2e01ae9` → `70affbb` in all four files
- Section B item 5 in coverage_map: marked CLOSED with correct value/hash
- Ran `render_fragments.py` → regenerated COMPLETED.md, TODO.md, SPEC_CHANGELOG.md

**Files committed by caller:**
- `.claude/spec/SPEC.md` (modified)
- `.claude/spec/COMPLETED.md` (modified)
- `.claude/spec/TODO.md` (modified)
- `.claude/spec/SPEC_CHANGELOG.md` (modified)
- `.claude/spec/todo.d/lensing_coverage_map.md` (modified)
- `.claude/spec/todo.d/lensing_dropped_gamma_slivers.md` (deleted)
- `.claude/spec/completed.d/2026-08-03_min_gamma_band_zero.md` (new, untracked)
- `.claude/spec/spec_changelog.d/2026-08-03_min_gamma_band_zero.md` (new, untracked)
- `.serena/memories/librarian_short_term.md` (this file)

**Docs/source NOT touched:** No Sphinx RST changes needed — this is a training
constant change only, no public API change.

**Fragile cross-references to watch:**
- `min_gamma_band = 1e-6` is now cited in SPEC.md (2 places) and
  coverage_map row 10. If the constant changes again, all three need updating.
- `_XI_FOLD_THRESHOLD = 4.0` and `CERTIFICATION_BAR` from prior run still cited
  in SPEC.md — watch for renames.
- `lensing_remaining_coverage_gaps.md` still has two OPEN items: "ppGO interior
  certification fix" (research) and infrastructure items.

**Surprises:**
- `sync_derived_docs.py` using phantom commit hash `2e01ae9` (not in git log)
  suggests it may have cached state from a prior aborted run or derives the
  hash incorrectly. Never trust its cited commit hashes without cross-checking
  `git log --oneline`.
- `tidy_advisory.json` appeared in the initial git status as M (modified), then
  sync_derived_docs.py updated it to point to current HEAD commit, making it
  clean again. The `M` was because the working tree still pointed to the
  previous commit's hash pre-session.
- FINDINGS.md has `min_gamma_band = 0.02` reference at line ~2071 — this is a
  historical measurement record (correct to preserve as-is).
