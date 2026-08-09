# Coder Short-Term Observations

(exterior_cusp_exclusion build, 2026-08-09):
- Added _deltoid_cusp_source_angles(gamma, n) -> list[float] for saddle/deltoid
  cusp angles D₂-folded into [0, π/2], mirroring _lobe_cusp_source_angles but
  measuring from origin (not lobe centroid). Sweeps both branches around both
  lens-plane centers 0 and π.
- Modified _exclude_near_cusp: added optional gamma_band parameter. When
  provided, checks at gamma_lo, gamma_mid, gamma_hi. Backward-compatible
  (gamma_band=None falls back to single-gamma check).
- Modified _farfield_tiles: added keyword-only args cusp_angles, gamma,
  gamma_band. When provided, calls _exclude_near_cusp and continues (skips
  tile). Backward-compatible.
- Wired saddle exterior: _train_band_charts now computes deltoid cusp angles
  via _deltoid_cusp_source_angles in the parity != 1 branch and passes them
  to _farfield_tiles with gamma=gamma_mid, gamma_band=band.
- Bumped _CUSP_EXCLUSION_DISTANCE from 0.2 to 0.35. Updated docstring to
  cover both astroid and deltoid cusps. Removed "astroid only" claim.
- Wrote scripts/measure_cusp_exclusion.py probe to calibrate the constant.
- Lobe interior unchanged: _SaddleLobeAdmission.admits comment confirms
  _LOBE_CUSP_EXCLUSION_DISTANCE is retired.
- Orphaned comment line "#: Deltoid-lobe interior near-cusp carve-out
  distance..." on former _LOBE_CUSP_EXCLUSION_DISTANCE line left as-is
  (cosmetic only, not in scope).
- (empty — last consolidated by Dreamer on 2026-08-09)
