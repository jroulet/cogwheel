# Librarian Short-Term Observations

## 2026-08-10 — Post-commit sync: exterior rho-axis log(rho-1) coordinate (f6b8b05)

### Scope
Commit f6b8b05 (2026-08-10). Key code changes:
- `ExteriorPolarChart` gains `rho_log_axis: bool` flag (serialized per-chart)
- Schema bumped: `exterior_polar_carrier_demod_v2` → `exterior_polar_rho_log_v3`
- Constant renamed: `_EXTERIOR_POLAR_AXIS_SCHEMA_CARRIER` → `_EXTERIOR_POLAR_AXIS_SCHEMA_V3`
- 7 test references to old constant fixed in test_lensing_farfield_envelope.py +
  test_lensing_surrogate_lobe.py

### Stale surfaces fixed
1. **SPEC.md** (Key abstractions section, line 62-64):
   - Schema tag updated to `exterior_polar_rho_log_v3` / `_EXTERIOR_POLAR_AXIS_SCHEMA_V3`
   - Retired-schemas list extended to include `exterior_polar_carrier_demod_v2`
   - `rho_log_axis` field description inserted after `carrier_rate` sentence
   - SPEC.md spec_version bumped to 0.36.4 by render_fragments.py
2. **DATA_CONTRACTS.yaml** (lens_amplification_surrogate description, line 198):
   - Same schema tag + constant rename
   - `exterior_polar_carrier_demod_v2` added to retired-tags list with updated rationale
   - `rho_log_axis` field description added after `carrier_rate` sentence
   - schema_version bumped to 3.0.2 by render_fragments.py

### Fragments created
- `spec_changelog.d/2026-08-10_exterior_rho_log_v3.md` (patch bump)
- `contracts_changelog.d/2026-08-10_exterior_rho_log_v3.md` (patch bump)
- `completed.d/2026-08-10_exterior_rho_axis_conditioning.md`
- `completed.d/2026-08-10_exterior_w_axis_powerlaw_conditioning.md` (this TODO was held
  open pending spatial-axis fix; both w-axis and rho-axis done now)

### TODO fragments deleted
- `todo.d/lensing_exterior_rho_axis_conditioning.md` (build complete)
- `todo.d/lensing_exterior_w_axis_powerlaw_conditioning.md` (both parts now done)

### Surfaces confirmed clean (no changes needed)
- SPEC.md pipeline table row: does NOT mention the schema tag at all (confirmed via
  search for `axis_schema='exterior_polar_carrier_demod_v2'` returning {} in SPEC.md)
- docs/source/ (overview.rst, api.rst, crash_course.rst): no new modules, no API signature
  changes, no import path changes
- sync_derived_docs.py: ran clean with only the known test-consumer warning for
  `lens_amplification_surrogate` (escalation fragment confirmed still open — no duplicate)

### Fragile cross-references to watch
- `_EXTERIOR_POLAR_AXIS_SCHEMA_V3` is now cited in BOTH SPEC.md and DATA_CONTRACTS.yaml
  (same fragile constant-name cross-reference family as `_LOBE_AXIS_SCHEMA*`/
  `_EXTERIOR_POLAR_AXIS_SCHEMA*` — a future rename breaks both simultaneously)
- If `carrier_rate` or `rho_log_axis` field names are renamed in code, BOTH doc surfaces
  need updating (same pattern as before)
- The `tidy_advisory.json` was pre-modified before this session; left as-is (not a
  render_fragments.py stray)

### Surprises
- The SPEC.md pipeline table row does NOT contain the schema tag verbatim (uses high-level
  description). Only the Key abstractions section needs the tag updated. Search for the
  bare `axis_schema='...'` format returned {} for SPEC.md — confirmed architecture-level
  vs. detail-level split between the two sections.
- Both the w-axis conditioning TODO and the rho-axis conditioning TODO were open
  simultaneously because the w-axis fragment was intentionally held open for the
  "REMAINING — SPATIAL AXES" part. Both closed together in this sync.
