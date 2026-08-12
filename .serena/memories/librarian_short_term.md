# Librarian Short-Term Observations

## Run: 2026-08-12 — post-commit sync for c8cad0c

**Scope**: commit `c8cad0c` — "feat(lensing): close the deltoid exterior cusp gap + astroid mid-w ppGO band"

**What was actually stale and why**:

1. **SPEC.md CUSP-EXCLUSION FILTER paragraph**: the builder's SPEC.md edit (in c8cad0c) only updated the certified-by list; the CUSP-EXCLUSION FILTER prose still said "excluded from exterior training on BOTH parities" — stale because WP-2 changed the saddle (parity -1) from excluding near-cusp tiles (d_exclude=0.35) to ADMITTING them with `force_minus_ghost=True` and `FARFIELD_KERNEL_SUM_MINUS_GHOST` label. Also the certified-by test description "exclusion boundary: tiles just beyond the cut are admitted, tiles within are excluded" was technically only accurate for astroid now; prefixed with "astroid cusp".
   - Fixed via `replace_content` literal mode

2. **Three spec_changelog.d fragments created**: my own (`2026-08-12_cusp_exclusion_saddle_admitted.md`) plus two the builder left uncommitted (`2026-08-12_ppgo_full_midw_band_minus_ghost.md` and `2026-08-12_saddle_corridor_ppgo_refusal.md`). The builder's fragments described the ppGO rung mid-w band fix and saddle corridor ppGO refusal. render_fragments.py bumped spec_version to 0.37.13 (from 0.37.10; builder had manually set 0.37.12 in violation of the convention — render_fragments.py overwrote it).

3. **Builder left substantial uncommitted state**: the builder created the completed.d fragments, deleted todo.d fragments, modified SPEC.md further, and created spec_changelog.d fragments — all uncommitted. I committed all of it as part of this sync.

4. **changelog.d fragments (4) + CHANGELOG.md update**: four untracked changelog.d fragments for Aug-12 completed builds were untracked; render_fragments.py updated CHANGELOG.md (+103 lines). Committed together.

5. **TODO closures**: `lensing_deltoid_exterior_cusp_gap.md` and `lensing_saddle_interior_cusp_serving.md` both deleted (by builder). Corresponding completed.d fragments created and committed.

**What was NOT stale**:
- `overview.rst`: no mention of `_CUSP_EXCLUSION_DISTANCE`, ppGO rung, or FARFIELD_KERNEL_SUM_MINUS_GHOST (pitched at architecture level)
- `api.rst`, `crash_course.rst`, `installation.rst`: no relevant changes
- `DATA_CONTRACTS.yaml`: no new disk artifacts
- The census "cusp-window" category description: still valid (it's just a fall-through category; fewer saddle draws fall into it now)

**Fragile cross-references created/noted**:
- SPEC.md CUSP-EXCLUSION FILTER now explicitly says "for positive parity (astroid)" — if astroid behavior also changes to admit near-cusp tiles, this sentence goes stale
- The `FARFIELD_KERNEL_SUM_MINUS_GHOST` label is now mentioned in both the CUSP-EXCLUSION FILTER section and the certified-by sentence for `test_lensing_ppgo_midw_and_minus_ghost.py` — if this label is renamed, both spots need updating
- `_R_PPGO_ERROR_CONST = 0.10` is cited in the certified-by description (via test file name suffix "midw") but NOT in SPEC prose (SPEC doesn't cite the numeric value) — pattern consistent with prior builds

**Pattern noted**: builder frequently commits feature work with only the certified-by list updated in SPEC.md, leaving prose paragraphs that describe the mechanism stale. This is the second time (after 288f37c) the CUSP-EXCLUSION / behavior paragraphs needed updating separately. Watch for: commits that add tests to certified-by but also change constants or behavioral flags (d_exclude, force_minus_ghost) — those always need prose updates too.

**Builder state left uncommitted (all now committed)**:
- Modified SPEC.md (ppGO "high-w" → full mid-w band, saddle corridor ppGO paragraph, surrogate section + MINUS_GHOST window)
- Two spec_changelog.d fragments
- Two completed.d fragments
- Two todo.d deletions
- Four changelog.d root fragments
- CHANGELOG.md update
