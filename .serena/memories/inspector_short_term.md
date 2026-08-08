# Inspector Short-Term — lobe_cusp_coordinate follow-up review (2026-08-08)

## Scope
Review of all uncommitted changes. Same build (lobe cusp-adapted coordinate).

## Files changed
- surrogate.py: _lobe_cusp_axis_map, schema/field rename (theta_to_s -> theta_to_u), from_lobe_engine
- surrogate_training.py: _lobe_nearest_cusp, _lobe_child_boxes u-midpoint split, _build_lobe_chart signature, _LOBE_CUSP_EXCLUSION_DISTANCE deletion
- test_lensing_surrogate_lobe.py: renamed tests + new coverage
- test_lensing_lobe_subdivision.py: carve-out retirement + cusp_axis_map + round-trip + schema tests

## Test results
- test_lensing_lobe_subdivision.py: 49/49 PASSED
- test_lensing_surrogate_lobe.py: 63 passed, 10 skipped, 0 failed
- test_lensing_wedge_dd_arclength.py: 3 FAILED, 15 passed, 6 skipped, 21 errors — 2 findings caused by this build

## Findings from this review

### BUG: _wedge_cusp_axis_map missing return statement (INS-4-003)
The `return theta_fine, u_fine` line at the end of `_wedge_cusp_axis_map` (line ~610) was accidentally deleted when `_lobe_cusp_axis_map` was inserted immediately after it. Verified: `_wedge_cusp_axis_map(0.2, 1.2, 'low')` returns `None`. This cascades to 3 FAIL + 21 ERROR in test_lensing_wedge_dd_arclength.py (wedgie-cusp_axis_map crashes on every call). Also breaks wedge chart training silently (from_wedge_engine line 3258 unpacks None).

Suggested fix: reinsert `    return theta_fine, u_fine` before the blank line that precedes `def _lobe_cusp_axis_map`.

### BUG: Stale field-name assertion in test_lensing_wedge_dd_arclength.py (INS-4-004)
`FieldExposureTestCase.test_lobe_still_exposes_theta_to_s` (line 532) asserts `'theta_to_s' in LobeInteriorChart fields` and `'theta_to_u' not in fields`. LobeInteriorChart now has `theta_to_u` (not `theta_to_s`). The test should be renamed (e.g. test_lobe_exposes_theta_to_u) and assert *theta_to_u present, theta_to_s absent*. TubeChart still correctly has theta_to_s. Also the module-level docstring lines 30-31 and the class docstring at line 505 are stale ("Lobe still exposes theta_to_s").

### Not resolved (carried forward from previous review)
- **INS-4-001**: `_validate_theta_to_u` docstring still says "Used by the wedge-interior chart" — now also called by LobeInteriorChart.from_lobe_values.
- **INS-4-002**: DATA_CONTRACTS.yaml still describes old lobe axis schemas with theta_to_s. Librarian scope.
