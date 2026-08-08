# Inspector Short-Term — lobe_cusp_coordinate salvage audit (2026-08-08)

## Scope
Salvage audit of commit b18e6a8 — build died at inspector-17 (quota exhaustion) after coder-16 fixed revision-2 findings. Verifying the unverified salvage commit.

## Files changed (confirmed complete)
- surrogate.py: _lobe_cusp_axis_map, schema/field rename (theta_to_s -> theta_to_u), from_lobe_engine
- surrogate_training.py: _lobe_nearest_cusp, _lobe_child_boxes u-midpoint split, _build_lobe_chart signature, _LOBE_CUSP_EXCLUSION_DISTANCE deletion
- test_lensing_surrogate_lobe.py: migrated tests + new coverage
- test_lensing_lobe_subdivision.py: carve-out retirement + cusp_axis_map + round-trip + schema + open-cusp-edge tests
- test_lensing_wedge_dd_arclength.py: field exposure test renamed/updated

## Test results (all green)
- test_lensing_lobe_subdivision.py: 49/49 PASSED
- test_lensing_surrogate_lobe.py: 63 passed, 10 skipped (pre-existing golden/slow), 0 failed
- test_lensing_wedge_dd_arclength.py: 37 passed, 6 skipped (pre-existing slow), 0 failed
- Total: 149 passed across 3 files, 0 failures

## Findings resolved from revision 2/2
- INS-4-001 (trivial): _validate_theta_to_u docstring now says "wedge-interior and lobe-interior charts" ✅
- INS-4-002 (trivial): DATA_CONTRACTS.yaml stale — deferred to Librarian (F050), not a code issue
- INS-4-003 (impl): Missing `return theta_fine, u_fine` on _wedge_cusp_axis_map — reinserted by foreman-lite-15 and coder-16, verified present at line 702 ✅
- INS-4-004 (impl): test_lobe_still_exposes_theta_to_s renamed to test_lobe_exposes_theta_to_u, now correctly asserts theta_to_u present and theta_to_s absent ✅

## Open issues
- DATA_CONTRACTS.yaml still describes old lobe axis schemas (INS-4-002, Librarian scope)
- 10 skipped tests in test_lensing_surrogate_lobe.py are pre-existing golden-file regeneration skips (D₂ fold), not related to this build
- 6 skipped tests in test_lensing_wedge_dd_arclength.py are pre-existing slow-tier skips

## Verified invariants
- _LOBE_AXIS_SCHEMA_NEW = 'lobe_caustic_relative_v1', _KNOWN_LOBE_AXIS_SCHEMAS = frozenset({only this})
- No old schema constants (_LOBE_AXIS_SCHEMA_V1, _LOBE_AXIS_SCHEMA) remain anywhere
- No theta_to_s references in ANY lobe code path (all remaining theta_to_s refs are TubeChart/FarField)
- _LOBE_ARC_MAP_SIZE deleted (unused)
- _LOBE_CUSP_EXCLUSION_DISTANCE deleted, docstring updated
- _lobe_cusp_axis_map: both sides produce u_fine[0]≈0, monotonic, endpoint-exact, validated
- lobe_cusps threaded through all tiers: _train_band_charts → tile dict → _subdivide_tile children → _lobe_nearest_cusp → _lobe_child_boxes → _build_lobe_chart → from_lobe_engine
- from_lobe_engine: cusp-adapted path works, raw-theta fallback works
