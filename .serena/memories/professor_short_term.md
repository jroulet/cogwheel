# Professor Short-Term Observations — lobe cusp-adapted coordinate review, 2026-08-08

## Context
Post-build domain review of commit b18e6a8: lobe cusp-adapted u = d**(2/3)
coordinate. All 149 fast-tier tests passed.

## Domain verdict: PASS with 1 concern

### Confirmed correct
1. **2/3 exponent universal for A3**: the deltoid lobe cusps are A3 fold-cusp
   singularities (same catastrophe class as astroid cusps). The 2/3 exponent in
   u = d**(2/3) is the exact, gamma-universal caustic-reach scaling. The old
   sqrt-edge coordinate (exponent 1/2) was wrong for cusps — designed for A2
   fold edges, not A3.

2. **Smoothing verified**: rho_lobe normalizes by deltoid radius (smooth at
   cusps, rho ~ 1 + O(dtheta)). The u-coordinate absorbs the d**(2/3) term in
   caustic reach, removing the d**(-1/3) derivative singularity that a spline
   in raw theta would see at the cusp vertex. Combined (rho_lobe, u) is smooth
   everywhere in the lobe interior.

3. **_lobe_cusp_axis_map mirrors _wedge_cusp_axis_map**: uniform-in-u grid
   (np.linspace), node-exact endpoints explicitly pinned, offset so u(θ_lo)=0.
   Both 'left' and 'right' sides correctly implement the monotone-increasing
   d→u→θ inverse mapping. The np.clip guard on the 'right' side protects
   against FP roundoff near the cusp.

4. **u-midpoint subdivision correct**: _lobe_child_boxes computes u_mid from
   the parent's cusp-adapted map, splits children with equal u-range via
   np.interp inverse mapping. Angular children have unequal θ-widths (near-cusp
   child narrower) — correct for a cubic spline in u.

5. **Schema hard-refuse**: both old schemas removed from _KNOWN_LOBE_AXIS_SCHEMAS;
   _validate_lobe_axis_schema rejects them at load. Tests verify this for both
   old tags, None, and unknown tags. No silent degradation possible.

6. **Carve-out retirement correct**: _LOBE_CUSP_EXCLUSION_DISTANCE removed
   because the cusp-adapted coordinate now handles near-cusp tiles. The
   caustic-cloud nearest-distance test in _SaddleLobeAdmission.admits already
   excludes tiles too close to the caustic — a separate lobe-specific
   carve-out was always redundant.

### Concern (not a blocker)
- **_chart_from_npz / _chart_to_npz asymmetry**: _chart_from_npz unconditionally
  accesses data['theta_to_u'] for lobe charts (will KeyError if absent), but
  _chart_to_npz only writes theta_to_u when it is not None. The raw-theta
  fallback path in from_lobe_engine (cusp_angle=None) produces charts with
  theta_to_u=None that CAN be built but CANNOT survive an NPZ round-trip.
  Not triggered in the current training pipeline (all tiles carry cusp angles),
  but is a latent trap. Mitigation: either have _chart_from_npz tolerate
  missing theta_to_u, or have _chart_to_npz raise a clear error when saving
  a theta_to_u=None chart.

## Test coverage
- 149 passed, 0 failed, 16 skipped in 3:48
- Schema hard-refuse: 6 tests (old tags, None, unknown, cross-contamination
  with far-field set, self-falsification)
- U-axis node-exact: B-spline reproduction at stored u-nodes to 1e-7 (both
  test files)
- Subdivision u-midpoint tests
- Round-trip save/load with new schema
