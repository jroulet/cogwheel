# Librarian Short-Term Observations

## 2026-08-10 — Post-commit sync: exterior cusp-exclusion + ppGO above-ceiling

### Scope
Commits 7a4a8ce through 5ac7d99 (2026-08-07 to 2026-08-10). Key code changes:
`cogwheel/lensing/surrogate_training.py` (exterior cusp-exclusion cut, both parities),
`cogwheel/lensing/likelihood.py` (_ppgo_above_ceiling), `cogwheel/lensing/surrogate.py`
(lobe cusp-adapted coordinate, _LOBE_AXIS_SCHEMA_NEW, _EXTERIOR_POLAR_AXIS_SCHEMA rename).

### What went stale and why

1. **SPEC.md — likelihood row**: `_ppgo_above_ceiling` in `likelihood.py` (commit 609d8d3)
   was never documented in SPEC row 55. The likelihood row described Born rung, fold-ppGO
   interior handoff, but omitted the above-ceiling ppGO intercept entirely. Fixed: added
   "ABOVE-CEILING ppGO INTERCEPT" paragraph with gate description and
   `test_lensing_ppgo_above_ceiling.py` certification.

2. **SPEC.md — surrogate row**: The FAR-FIELD TILING section never mentioned the
   `_exclude_near_cusp` filter or `_CUSP_EXCLUSION_DISTANCE`. Commit d685ebe extended
   the filter to BOTH parities and bumped the distance 0.2 → 0.35 — an architecture-level
   change (now covers saddle exterior too). Fixed: added "CUSP-EXCLUSION FILTER" paragraph
   with `_CUSP_EXCLUSION_DISTANCE = 0.35`, `_deltoid_cusp_source_angles`, and
   `test_lensing_exterior_admission.py` certification.

3. **todo.d/lensing_exterior_cusp_exclusion_cut.md**: Was open, implementation completed
   in d685ebe. Moved to completed.d/2026-08-09_exterior_cusp_exclusion_cut.md.

4. **todo.d/lensing_exterior_followup_four_items.md**: Items 2 (cusp exclusion) and 4
   (ppGO fallback) completed; marked DONE in-place. Fragment stays open (items 1 and 3
   remain).

5. **todo.d/lensing_exterior_w_axis_powerlaw_conditioning.md**: Had dangling
   `depends_on: [lensing_exterior_cusp_exclusion_cut]` after the todo was moved to
   completed.d; repointed to `2026-08-09_exterior_cusp_exclusion_cut`.

### What was already up to date

- SPEC.md schema tags (`exterior_polar_rho_u_v1`, `lobe_caustic_relative_v1`,
  `_LOBE_AXIS_SCHEMA_NEW`, `theta_to_u` for ExteriorPolarChart) — all already
  documented from the 98c4e7f librarian run (8a2e654).
- DATA_CONTRACTS.yaml — unchanged by these builds.
- docs/source/ — no user-facing API or narrative changes needed.
- Lobe cusp-adapted coordinate (b18e6a8/98c4e7f) — already synced in prior run.
- surrogate_contract_test_consumer_warning TODO fragment — already exists, not re-created.

### Fragile cross-references to watch

- SPEC row 55 now cites `test_lensing_ppgo_above_ceiling.py` — if this test is
  renamed, SPEC goes stale silently.
- SPEC row 56 now cites `_CUSP_EXCLUSION_DISTANCE = 0.35` — if the constant changes
  again, SPEC goes stale silently. Same family as the schema-constant naming hazard.
- SPEC row 56 cites `_deltoid_cusp_source_angles` — if renamed, stale.

### Surprises

- `sync_derived_docs.py` modified `tidy_advisory.json` as a side-effect of its internal
  state flush ("some issues auto-fixed" with no actual git diff). Reverted.
- The `lensing_exterior_followup_four_items.md` TODO (items 1 and 3 still open) was
  NOT closed — confirming "multi-part TODO stays open until every part finishes".
- The ppGO above-ceiling test (test_lensing_ppgo_above_ceiling.py) was NOT in SPEC.md
  at all despite being certified in commit 609d8d3 on 2026-08-08. A full two days passed
  before this sync. Pattern: ppGO intercepts added to likelihood tend to be omitted from
  SPEC when the commit message focuses on the build work-package label rather than the
  architecture change.
