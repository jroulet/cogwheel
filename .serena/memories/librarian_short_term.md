# Librarian Short-Term Observations

## 2026-08-10 — Post-commit sync: exterior carrier demodulation (w-axis conditioning)

### Scope
Commit f4652e7 (2026-08-10). Key code change:
`cogwheel/lensing/surrogate.py` — `ExteriorPolarChart` gains `carrier_rate`
field; axis schema bumped from `exterior_polar_rho_u_v1` to
`exterior_polar_carrier_demod_v2`; constant renamed from `_EXTERIOR_POLAR_AXIS_SCHEMA`
to `_EXTERIOR_POLAR_AXIS_SCHEMA_CARRIER`.

### What went stale and why

1. **SPEC.md Key abstractions section (lines 63-65)**: Referenced the old
   axis-schema tag `'exterior_polar_rho_u_v1'` and did not mention `carrier_rate`.
   The commit bumped the schema but the Key abstractions paragraph was not updated
   by the build commit. Fixed: updated tag to `exterior_polar_carrier_demod_v2`,
   added both retired tags, added `carrier_rate` field description.

2. **DATA_CONTRACTS.yaml lens_amplification_surrogate description**: Same stale
   tag and missing `carrier_rate`. Fixed: updated tag and constant name, added
   carrier_rate field description.

3. **todo.d/lensing_exterior_w_axis_powerlaw_conditioning.md**: W-axis part is
   done (w-axis eps ~1e-4), but acceptance bar NOT cleared because spatial axes
   (rho ~3-decade growth, theta) still have eps ~0.04. TODO stays open. Updated
   fragment to show w-axis DONE with reference to commit f4652e7, and remaining
   spatial-axis work.

4. **New fragments created**:
   - `spec_changelog.d/2026-08-10_exterior_carrier_demod_v2.md` (patch bump →
     SPEC.md rendered at 0.36.3)
   - `contracts_changelog.d/2026-08-10_exterior_carrier_demod_v2.md` (patch bump →
     DATA_CONTRACTS.yaml rendered at 3.0.1)

### What was already up to date

- docs/source/ — no user-facing API or narrative changes needed
- Four-items TODO fragment (items 1 and 3 still open)
- surrogate_contract_test_consumer_warning TODO — still exists, not re-created
- tidy_advisory.json — sync_derived_docs.py side-effect reverted

### Fragile cross-references to watch

- SPEC.md Key abstractions now cites `_EXTERIOR_POLAR_AXIS_SCHEMA_CARRIER`
  (constant name) — if renamed again in code, SPEC and DATA_CONTRACTS both
  go stale simultaneously.
- SPEC.md Key abstractions cites `|E(w)| ~ w^(-0.60)` power-law exponent —
  if the empirical exponent changes in a later build, this sentence goes stale
  silently.
- TODO fragment notes "spatial-axis follow-on not yet in a TODO fragment" —
  when a spatial-axis conditioning build brief is written, a TODO fragment
  should be added.

### Surprises

- The build commit (f4652e7) DID include SPEC.md in changed_files but only
  bumped spec_version (via a pre-existing fragment render). The Key abstractions
  section was NOT updated by the build itself — the schema tag and carrier_rate
  were left stale. Pattern: schema-tag bumps tend to be described in the
  surrogate module itself but are not propagated to SPEC.md Key abstractions
  by the coder.
- The SPEC.md surrogate row (long paragraph in the module table) did NOT contain
  `exterior_polar_rho_u_v1` — confirmed by grep. Only the Key abstractions
  section had the stale tag. This means the module table row is maintained
  separately from the Key abstractions summary paragraph.
- DATA_CONTRACTS.yaml description is a single-line YAML string — the Edit tool
  was not available (file flagged sensitive); used mcp__serena__replace_content
  with literal mode successfully.
