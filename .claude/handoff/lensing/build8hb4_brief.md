# Build 8h-b4 — Exterior admission repair, saddle axis, gamma=1 guards, port finish

## Mission

Repair the defects that block exterior coverage and close out the
caustic-fixed migration. The Professor ruling (2026-07-26) settled the
coordinate question: the exterior positive-parity axis is ALREADY
Einstein-scale (additive, `rho = 1 + |y| - r_caustic(gamma, theta_c)`,
verified `drho/d|y| = 1.0000`) and measured BETTER than a plain Einstein
axis at every radius. Fitting axes for exterior(+1), interior and tube
charts are therefore CORRECT AS SHIPPED and must not change. The real
defects are elsewhere. Three work packages:

1. **Exterior admission (the coverage defect).** `_farfield_tiles`
   excludes on a single SCALAR `exclusion_rho` built from
   `reach_max + eta_max`. The astroid is a SPIKE, not a disk: at
   gamma=0.9 `r_caustic` is ~0.9-1.5 over most angles but 5.69 at the
   90-degree cusp, and `_caustic_reach` returns that spike. Measured
   consequence: exterior coverage of the true `{eta >= eta_max}` region
   falls 0.891 (gamma 0.10-0.40) -> 0.496 (0.50-0.70) -> 0.094
   (0.70-0.80) -> 0.000 (0.80-0.90), because for gamma >= ~0.85 the
   exclusion circle (4.69-5.99) exceeds the whole prior source box
   (|y|max = 4.24) and NO exterior chart is built at all. Replace the
   scalar test with the direct per-`theta_c`-column test the INTERIOR
   tiler already implements (`_InteriorAdmission.admits`): a tile is
   admitted iff, for every gamma in the band and every probe point on
   the tile's INNER rho edge across its `theta_c` span,
   `geometry.nearest_caustic_point(gamma, 0, y).distance >= eta_max`
   AND the point is outside the caustic. Per-column `rho_inner`, not
   one scalar. Do NOT substitute a radial proxy: the radial offset
   overstates true caustic distance by up to 10x (ratio falls to 0.105
   at gamma=0.95) because a ray near 80 degrees passes laterally close
   to the long-axis cusp. Recovers 0.97-0.98 coverage per band at a
   measured ~2-3% cost vs exact per-gamma admission.
2. **Saddle (parity -1) exterior axis + the two gamma=1 guards.**
   (a) The parity -1 arm of `_to_caustic_fixed`/`_from_caustic_fixed`
   still uses the MULTIPLICATIVE `rho = |y| / _caustic_reach(gamma)`.
   This is the one place the reach-stretch is real (M1/M3 apply here).
   Switch it to the same additive form the positive-parity arm uses
   (~2 lines each). gamma in [0, 1.6] is in the prior, so these charts
   ship. Nothing is trained yet, so there is no retraining cost.
   (b) `_box_region_labels` (~L1344) calls `_from_caustic_fixed` on the
   box CENTER outside any guard, so a box whose centre gamma is exactly
   1.0 raises `LensDomainError` and crashes chart construction. Catch
   it and mark `image_count`/`parity` unknown (`None` is already the
   declared type; the guard stack handles unknown conservatively). The
   sibling node-loop site is ALREADY FIXED in-tree (commit c28408b) —
   do not redo it, verify it.
3. **Finish the test-fixture port (test-only).** Complete the migration
   of the old suites to the caustic-fixed API and get the far-field
   regression battery green. Same rules as before: no weakening of any
   assertion or tolerance; express the SAME physical configuration in
   the new coordinates via the shared `_caustic_reach`; migrate a
   genuinely retired premise's INTENT rather than deleting (any
   deletion needs a one-line justification).

## Measured facts (pre-answered — do not re-derive)

- `drho/d|y| = 1.0000` on the exterior positive-parity arm (driver-
  verified). Corrected label-spread across a gamma band (reach varying
  4.9x), `max|E_ff|/max|F|`, shipped-additive vs plain-Einstein:
  |y|=1.5 -> 2.89 vs 3.96; 2.5 -> 1.58 vs 2.63; 4.0 -> 1.07 vs 1.16.
  Additive wins everywhere; true caustic distance varies only 10%
  across the band at fixed additive rho vs 27% at fixed Einstein |y|.
- Exterior coverage collapse table and the gamma>=0.85 box-swallow are
  driver-verified (`_caustic_reach(0.85)=4.39`, `(0.9)=5.69`, prior box
  corner 4.24).
- Professor verdict (2026-07-26) on the saddle exterior LABEL: (A)
  benign near-caustic physics, NOT a conditioning defect. The large
  `|E_ff|/|F|` lives entirely below `w_floor`; inside the label's valid
  window it is 0.01-1.5. NO SACR-C treatment for the exterior.
- Interior axes: KEEP the multiplicative `rho = |y|/r_caustic` in
  [0, 1]. The argument is DOMAIN not smoothness — the interior shrinks
  to zero with gamma, so only a multiplicative normalisation gives a
  gamma-independent tiling domain with the caustic at rho=1. The
  measured interior eps failure is a fixed-gamma CONDITIONING error,
  not gamma-interpolation. Do not touch.
- Tube charts: unaffected, confirmed correct (`u = sqrt(eta)` is the
  same caustic-anchored Einstein-scale principle; `rho - 1` is its
  outward continuation).
- Ghost/annulus: the additive coordinate makes the `w * Im tau_c` gate
  boundary a near-coordinate surface — reinforces the ruling. CAVEAT:
  all driver probes were taken at theta=45deg where `Im tau_c = 0`
  identically (a symmetry diagonal), so they say NOTHING about the
  ghost regime; do not extrapolate them to generic angles.
- `ppgo_exclusion_rho` stays scalar-reach based — the ppGO map is
  scalar-reach by contract. Do not change it.

## Out of scope — hard fences

- NO change to exterior(+1), interior, or tube FITTING AXES.
- NO saddle exterior label / SACR-C work. NO Pearcey model.
- NO quad-double, NO Born rung, NO campaign, NO tolerance changes.
- NO chart-schema, serve-path, or retraining-format change from WP1
  (tiles stay rectangles in `(rho, theta_c)`; only each column's
  `rho_inner` differs). WP2(a) DOES change stored saddle axes — that
  is expected and confined to the parity -1 arm.

## Acceptance (two-tier)

1. In-build (FAST, synthetic scale):
   (a) admission: exterior coverage of the true `{eta >= eta_max}`
   region is >= 0.95 in every gamma band tested INCLUDING 0.80-0.90
   (currently 0.000), measured against exact `nearest_caustic_point`
   distance; no admitted tile contains a point with true caustic
   distance < eta_max (reachable-red: restore the scalar test and the
   0.80-0.90 band must collapse to zero tiles);
   (b) saddle axis: the parity -1 arm satisfies `drho/d|y| = 1` to
   1e-12; a saddle exterior chart round-trips
   `_to_caustic_fixed`/`_from_caustic_fixed` to 1e-12;
   (c) gamma=1: a chart whose box CENTRE gamma is exactly 1.0 builds
   without raising, recording unknown image_count/parity; a grid with a
   gamma=1 NODE records it refused (verify the in-tree fix);
   (d) the far-field regression battery is GREEN (test_lensing_surrogate,
   test_lensing_ppgo_bandsplit, test_lensing_surrogate_census,
   test_lensing_exterior_windows); tube byte-identity holds;
   (e) fast tier green (tree gate).
2. POST-BUILD (driver): serving census — the climb from 2.2% must begin,
   and the gamma>=0.8 bands must stop contributing zero exterior
   coverage; then the calibration re-pilot, Born rung, qd, ONE campaign.
