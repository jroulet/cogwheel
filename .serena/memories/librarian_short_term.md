# Librarian Short-Term Observations

## Run: 2026-08-03 — post-commit sync (step 1c DONE markers in source fragments)

**Scope:** Single commit `bd1f99d` changed only `.claude/spec/TODO.md`.

**What was stale and why:**
Commit `bd1f99d` edited TODO.md *directly* to add a "DONE" marker to step 1c
("The serving path, plus third order", completed 2026-07-30 in commit `b9c3ed6`).
But TODO.md is generated from `todo.d/` fragments — the direct edit would be
reverted on the next `render_fragments.py` run. Two source fragments needed
updating (same two-fragment pattern as 1b, documented in knowledge memory):
1. `lensing_caustic_relative_coordinates.md` — step 1c ordered entry (line 101)
2. `lensing_analytic_derivatives.md` — item 1 (`_pearcey_cusp._cusp_vertex`
   inventory cross-reference for build 1c)

**Pattern confirmed:** Two-fragment DONE-marking rule from knowledge memory
applied correctly again. The cross-reference `[[fragment_name]]` backlink
in `lensing_caustic_relative_coordinates.md` to `lensing_analytic_derivatives`
means every step-DONE mark requires both fragments to be updated.

**Fragile cross-references to watch:**
- `lensing_caustic_relative_coordinates.md` ↔ `lensing_analytic_derivatives.md`:
  Items 2 and 3 in the derivatives fragment (`_branch_speed_profile`,
  `_find_cusps`) are still OPEN — they survived build 1b partially. Item 2's
  `np.gradient` was removed but the function remains; item 3's `_find_cusps`
  still uses an inlined `0.2` literal (carried forward by design per the
  "Carried forward from build 1b" subsection). These should be marked DONE
  only when the cusp-window schema build (F040) lands.

**Surprises:**
- None. Clean sync. `sync_derived_docs.py` showed "some issues auto-fixed"
  but the git diff confirms no additional file changes beyond my 3 edits —
  consistent with the known internal-state-flush false positive (knowledge memory).
- The 4 test-consumer warnings for `lens_amplification_surrogate` are
  pre-existing and test-only; per convention they stay off the DATA_CONTRACTS
  consumer list.
