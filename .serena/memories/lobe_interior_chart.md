# LobeInteriorChart — cusp-adapted coordinate (u = d**(2/3))

Consolidated durable knowledge for the macro-saddle lobe-interior chart
(`surrogate.py` / `surrogate_training.py`). Build `lobe_cusp_coordinate`,
2026-08-08, commits 98c4e7f (+ docs f55454c); completed fragment
`completed.d/2026-08-08_lobe-cusp-adapted-coordinate.md`. 149 fast tests
pass, 16 skipped (pre-existing golden/slow).

## Coordinate design
- Axis: cusp-adapted `u = d**(2/3)`, `d` = angular distance to the nearest
  deltoid cusp vertex. The 2/3 exponent is the UNIVERSAL A3 fold-cusp
  caustic-reach scaling (`r_deltoid ~ const - c*d**(2/3)`) — exact for the
  deltoid lobes (same catastrophe class as astroid cusps), gamma-universal.
- Why: rho_lobe already normalizes by the deltoid radius, so it is smooth at
  cusps (rho ~ 1 + O(dtheta)); the u-axis absorbs the d**(2/3) reach term,
  removing the d**(-1/3) derivative singularity a spline in raw theta would
  see at the cusp vertex. Combined (rho_lobe, u) is smooth everywhere in the
  lobe interior.
- OLD sqrt-edge coordinate (`s = sqrt(span) - sqrt(theta_max - theta)`,
  exponent 1/2, `theta_to_s`) was designed for A2 fold edges and was wrong
  for A3 cusps — RETIRED.
- `_lobe_cusp_axis_map` mirrors `_wedge_cusp_axis_map`: uniform-in-u grid
  (np.linspace, shape (2, 2001), same node count as `_FARFIELD_ARC_MAP_SIZE`),
  node-exact endpoints explicitly pinned, offset so u(theta_lo)=0; both
  'left'/'right' sides implement the monotone d->u->theta inverse; the
  np.clip guard on the 'right' side protects against FP round-off near the
  cusp. Verified: both sides u_fine[0]~0, monotonic, endpoint-exact.

## Schema contract (single tag, hard-refuse)
- `_LOBE_AXIS_SCHEMA_NEW = 'lobe_caustic_relative_v1'` is the ONLY known lobe
  tag (`_KNOWN_LOBE_AXIS_SCHEMAS` = frozenset of just it). Both old tags (V1
  raw-theta, sqrt-edge) are GONE from the known set and hard-refuse at load;
  None and unknown tags refuse too. No silent degradation possible.
- `theta_to_u` is REQUIRED on load: `_chart_from_npz` reads it
  unconditionally, so an absent map hard-refuses (KeyError). No identity/None
  fallback — artifacts must be rebuilt with a real map.
- Field/params renamed theta_to_s -> theta_to_u / s_grid -> u_grid on
  `LobeInteriorChart` ONLY; Tube/FarField keep genuine arc-length
  theta_to_s (deliberately untouched). `_validate_theta_to_u` docstring
  covers both wedge-interior and lobe-interior callers.
- `_LOBE_ARC_MAP_SIZE` deleted (unused).

## Subdivision rule (u-midpoint, not theta-midpoint)
- `_lobe_child_boxes` splits children at the U-MIDPOINT of the parent's
  cusp-adapted map, mapped back to `theta_local` via np.interp inverse — NOT
  the raw theta midpoint. Angular children have UNEQUAL theta-widths
  (near-cusp child narrower) — correct for a cubic spline in u.
- `_lobe_nearest_cusp` is the SINGLE authoritative nearest-cusp derivation,
  shared by `_lobe_child_boxes` AND the production build sites (build_lobe
  closure, `_subdivide_lobe_tile` build_child). INS-3-001 lesson: the
  cusp-adapted design was BUILT but NOT ACTIVATED in production (callers
  passed no cusp_angle/cusp_side, so the raw-theta fallback silently won);
  fix = derive nearest-cusp+side via the SHARED helper at both build sites.
  When cusp_angle is None, from_lobe_engine falls back to a raw-theta
  uniform grid (keeps legacy callers backward-compatible).

## Carve-out retired
- `_LOBE_CUSP_EXCLUSION_DISTANCE` DELETED (retired, not just dead): the
  eta_max tube-shell nearest-caustic-distance test in
  `_SaddleLobeAdmission.admits` alone excludes near-cusp tiles (cusp vertices
  are in caustic_cloud); a tile centred at a cusp vertex clears the eps bar
  without subdivision. A lobe-specific carve-out was always redundant.

## Professor note — latent NPZ round-trip trap (non-blocking)
- ASYMMETRY: `_chart_from_npz` UNCONDITIONALLY accesses data['theta_to_u']
  for lobe charts (KeyError if absent), but `_chart_to_npz` only WRITES
  theta_to_u when it is not None. The raw-theta fallback path
  (cusp_angle=None) produces charts with theta_to_u=None that CAN be built
  but CANNOT survive an NPZ round-trip. Not triggered in the current training
  pipeline (all tiles carry cusp angles) — LATENT TRAP for future external
  callers. Mitigation: tolerate missing theta_to_u in `_chart_from_npz`, or
  raise a clear error in `_chart_to_npz` when saving a theta_to_u=None chart.

## Build history / workflow lesson
- Quota-death at Inspector: the build died at inspector-17 (quota
  exhaustion) AFTER coder-16's revision-2 fixes were committed; the code was
  salvaged as b18e6a8, then a FRESH Inspector audit ran the full changed-file
  suites (149 pass) + re-derived the invariants independently -> PASS, Tidier
  clean (6 unused imports removed), Professor PASS. Pattern: a quota-killed
  build at a late stage must be re-audited FROM SCRATCH on the salvage
  commit, never trusted from the partial pre-death pass.
- Audit-verified invariants (b18e6a8): no old schema constants remain
  anywhere; zero theta_to_s in any lobe code path (remaining refs are
  Tube/FarField); lobe_cusps threaded through all tiers
  (_train_band_charts -> tile dict -> _subdivide_tile children ->
  _lobe_nearest_cusp -> _lobe_child_boxes -> _build_lobe_chart ->
  from_lobe_engine); from_lobe_engine cusp-adapted path AND raw-theta
  fallback both work; _lobe_cusp_axis_map both sides produce u_fine[0]~0.
