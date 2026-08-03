# Librarian Short-Term Observations

## Run: 2026-08-03 — Fold-ppGO interior handoff (Build ppgo_interior_handoff)

**Scope:** Build added a new serve path in `likelihood._surrogate_coefficients`
for positive-parity interior draws above the `InteriorWedgeChart` w-ceiling,
mirrored gate logic in `surrogate_census.characterize_sample` (category
`ppgo_fold`), and added 17 tests in `test_lensing_fold_ppgo_handoff.py`.

**What was stale and why:**
1. SPEC.md "Microlensed waveform & likelihood" row — missing description of the
   fold-ppGO interior handoff path. The row described the fact-4/Born slot but
   did not mention that `_surrogate_coefficients` now has a separate interior
   (rho <= 1.0) serve path for draws above the `InteriorWedgeChart` w-ceiling.
   Inspector flagged this as INS-1-001 (trivial).
2. SPEC.md census description said "6-way MECE fall-through breakdown" — now 7-way
   with `ppgo_fold` as a new category (served, not a fall-through, but tracked
   distinctly by `characterize_sample`).
3. `lensing_remaining_coverage_gaps.md` TODO fragment — had an OPEN `[→ spec]`
   item for the ppGO handoff; the build implemented option (b) (xi threshold
   gate), so the item needed to be marked DONE.
4. `doc_debt.json` — recorded an owed entry about `test_lensing_min_gamma_band.py`
   not being cited in SPEC.md. Added it to the training "Certified by" section.

**Fixes applied:**
- SPEC.md: Added "FOLD-PPGO INTERIOR HANDOFF" paragraph after the Born fact-4
  slot sentence in the likelihood row (via Python str.replace — pipe-escape rule).
- SPEC.md: Updated census "6-way" → "7-way fall-through breakdown" with `ppgo_fold`.
- SPEC.md: Added `test_lensing_fold_ppgo_handoff.py` cert to the new FOLD-PPGO
  paragraph.
- SPEC.md: Added `test_lensing_min_gamma_band.py` cert to the training section
  (after the `test_lensing_caustic_cusps.py` reference).
- `todo.d/lensing_remaining_coverage_gaps.md`: Marked ppGO handoff item DONE.
- Created `completed.d/2026-08-03_fold_ppgo_interior_handoff.md`.
- Created `spec_changelog.d/2026-08-03_fold_ppgo_interior_handoff.md` (bump: patch).
- Deleted `.claude/doc_debt.json` (item addressed).
- Deleted `.claude/sync_issues.json` (prior scripts-only no-op commit).

**Docs/source NOT touched:** No Sphinx RST changes needed — changes are
SPEC.md architecture table only, not public Python API or install instructions.

**Pattern confirmed:**
- The TODO fragment's `[→ spec]` tag is the trigger for a spec_changelog.d
  fragment; verify it's there before closing the TODO.
- When SPEC.md describes "N-way MECE breakdown", a new census category in
  `characterize_sample` always stalens that count — grep for the category name
  in SPEC.md immediately.
- `doc_debt.json` can carry owed work from prior builds; read it at the start
  of each doc sync (it was overlooked in prior runs — now tracked).
- The SPEC.md "fact-4 slot" description appears in TWO places: (1) the main
  table row for "Microlensed waveform & likelihood" (updated here) and (2) the
  design-note bullet under "Born rung" (lines ~134–145). The note describes the
  Born-specific path only; it does NOT need to describe fold-ppGO.

**Fragile cross-references to watch:**
- `ppgo_fold` category is now in SPEC.md census description, code constants,
  and TODO fragment. If the category string changes in code, update both SPEC.md
  occurrences and any test that pins it.
- The `_XI_FOLD_THRESHOLD = 4.0` and `CERTIFICATION_BAR` constants are now
  cited by name in SPEC.md — if they're renamed, update SPEC.md.
- `lensing_remaining_coverage_gaps.md` still has two OPEN items:
  "ppGO interior certification fix" (research) and infrastructure items.

**Surprises:**
- The `doc_debt.json` file was untracked (not in git) but had actionable content
  about a previously missed SPEC.md test-file citation. Deleting it after
  addressing the item is the correct outcome.
- The census description "MECE" was not quite accurate after adding `ppgo_fold`
  (which is a SERVED category, not a fall-through). Updated description to
  "7-way fall-through breakdown" with a clarifying parenthetical that
  `ppgo_fold` is served but tracked distinctly.
