# Librarian Short-Term Observations

## 2026-08-10 — Post-commit sync: ghost-exclusion filter + fold-carrier TODO (cf81d66, 8a3232f)

### Scope
Two commits:
- cf81d66: feat(lensing): ghost-transition-zone tile exclusion in exterior tiler
  - `cogwheel/lensing/surrogate_training.py` +100 lines: `_exclude_ghost_dominated`, wired into `_farfield_exterior_tiles`
  - `cogwheel/tests/test_lensing_exterior_admission.py`: new 759-line test file (19 ghost tests + admission tests)
  - `cogwheel/tests/test_lensing_surrogate_training.py`: new 212-line test file
  - agent state + memory files (not doc surfaces)
- 8a3232f: docs: todo + brief for fold-carrier phase demodulation build
  - new TODO fragment `lensing_exterior_fold_carrier_demodulation.md`
  - brief `brief_exterior_fold_carrier_demodulation.md` (agent-internal)
  - TODO.md (generated)

### What was stale and why
SPEC.md FAR-FIELD TILING section mentioned CUSP-EXCLUSION FILTER but had no GHOST-EXCLUSION FILTER section. `_exclude_ghost_dominated` was a newly-shipped mechanism in surrogate_training.py with no SPEC description.

### Fix applied
Added GHOST-EXCLUSION FILTER paragraph to SPEC.md (FAR-FIELD TILING row) describing:
- `_exclude_ghost_dominated`, positive-parity scope
- ghost-transition zone definition (Im(tau_c) < _GHOST_DECAY_IM_THRESHOLD)
- GhostDomainError → retain; success+below-threshold → exclude
- gamma_band probing mirrors _exclude_near_cusp
- ghost_excluded_tiles counter
- certification by test_lensing_exterior_admission.py (19 ghost tests itemized)

Spec_changelog fragment `2026-08-10_ghost_exclusion_filter.md` was ALREADY PRESENT (created by another agent) and correctly described the change. No duplicate created.

### Pre-existing uncommitted artifacts included in commit
- `completed.d/2026-08-10_exterior_rho_phase_carrier.md` — completion record for ghost-exclusion build
- `todo.d/lensing_exterior_rho_phase_carrier.md` — deleted (moved to completed.d/)
- `todo.d/lensing_exterior_fold_carrier_demodulation.md` — `depends_on` already correctly repointed to `[2026-08-10_exterior_rho_phase_carrier]`
These were done by the coder agent; render_fragments.py incorporated them into TODO.md/COMPLETED.md.

### sync_derived_docs.py result
Ran clean. Same KNOWN recurring test-consumer warnings for `lens_amplification_surrogate` (4 test-only callers in test_lensing_surrogate.py). Escalation fragment `todo.d/surrogate_contract_test_consumer_warning.md` confirmed still open. Stray `tidy_advisory.json` diff reverted per institutional memory.

### Fragile cross-references to watch
- `_GHOST_DECAY_IM_THRESHOLD` cited in SPEC.md FAR-FIELD TILING (ghost-exclusion) AND in the engine's channels.py row — if renamed, both spots need updating
- `_exclude_ghost_dominated` named in SPEC.md — if the function is renamed, SPEC.md goes stale
- `ghost_excluded_tiles` field name in region report — same family
- `_EXTERIOR_POLAR_AXIS_SCHEMA_V3` cited in BOTH SPEC.md and DATA_CONTRACTS.yaml (prior session)
