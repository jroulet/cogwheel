# Build: saddle admission predicates — serve the connecting region and transverse cone

## Mission

Two saddle-parity (|gamma| > 1) regions where raw ppGO is measurably
certifiable reach live quadrature today because the admission predicates ask
the wrong geometric questions (audit 2026-08-13, FINDINGS F077 context):

1. `_saddle_farfield_analytic`'s floor `rho >= _SADDLE_FARFIELD_RHO_FLOOR
   (= 2.0)` is a SCALAR-reach test applied to DISCONNECTED deltoids: the
   transverse (hard-axis) cone at rho 0.6-1.9 is far from both lobes
   (directional eta up to 2.5) yet refused.
2. The connecting region (origin -> deltoid, rho < 0.5) can never satisfy a
   `min delta_tau` resolution test on the soft axis: the mirror image pair
   has delta_tau == 0 EXACTLY, so no w resolves it — yet ppGO there is
   1e-5..1e-6 for w >= 25.

## Measured facts (SHA d3dc109; saddle oracle = `_saddle_mass_sheet_map` +
## `f_schwinger` reconstruction, validated at 0.0e0 vs F_op incl. kappa=0.2)

1. Directional distance separates cleanly: every audited point with
   `eta >= 0.5` (via `geometry.nearest_caustic_point`) had ppGO error
   <= 1e-4 at w >= 30; every point with `eta <= 0.05` was over the bar by
   >= 2 orders. 108+ point audit, gamma {1.2, 1.5, 2.0}.
2. Transverse cone, rho=1.5, gamma=1.5: ppGO err 1.4e-5 (w=30), 2.2e-6
   (w=60) — 100x inside the bar, refused today.
3. Connecting region: ppGO clears 1e-4 at w = 12-25 for rho <= 0.4 (all
   three gammas), and the SHIPPED `certified_ppgo_map.npz` saddle rows
   already record w_cert 16-28 for rho < 0.5 — the map and the fresh
   measurement agree; the serve path just never consults the map (F077).
4. Deltoid geometry: lobes on the soft axis, wedge |sin 2theta| <=
   (1-kappa)/|gamma|; 4 real images strictly inside a lobe, 2 everywhere
   else (origin, connecting, transverse, outward). Image count is the exact
   lobe-interior discriminator (parity-blind, per SPEC).

## Scope

IN:
- Re-key the saddle far-field/ppGO admission on a DIRECTIONAL predicate:
  eta from `nearest_caustic_point` (with a derived floor defended by fact 1
  — do not pin 0.5 blind; re-measure the boundary on a denser eta grid and
  state the margin), composed with the existing w-resolution leg where a
  REAL non-mirror pair exists, and with `w_cert` from the certified map
  where a map is installed (wire `set_certified_ppgo_map` at construction
  when the shipped artifact exists — this un-deadens one F077 artifact).
- The mirror-pair delta_tau == 0 case: treat symmetry-tied pairs as
  NON-resolving rather than vacuously resolved (same tie discipline as
  F072's `_CUSP_TIE_EPS`).
- Fast tests, derived fixtures, saddle oracle per the recipe above.

OUT: any positive-parity gate (F076's distance gate belongs to the
fold_exterior_ghost build — coordinate, don't duplicate); charts/training;
`_ppgo_above_ceiling`'s own admission (same fold build); slow tiers.

## Acceptance

- Transverse-cone and connecting-region witnesses (derived, per-gamma) serve
  analytically with measured error vs the saddle oracle under the 1e-4 bar
  at w <= 60, REPORTED numbers.
- Deltoid-lobe interior and near-lobe annulus still refuse to the engine
  (they genuinely need tables — do NOT widen onto them; fact 4's image
  count is the fence).
- Existing saddle suites green (test_lensing_saddle_*, ppgo_bandsplit,
  saddle_rho_guards).

## Constraints

Branch claude-dev; spec/TODO fragments; assert values not paths; the
pairing-gate discipline before any oracle claim (memory:
pair-frames-before-scoring).
