# Build Brief: Exterior follow-up — ghost label, cusp exclusion, A/B test, ppGO fallback

## Mission

Four coupled items from `lensing_exterior_followup_four_items`, post-polar-rechart:

1. **Ghost-subtraction fix**: the `_GHOST_DECAY_IM_THRESHOLD=0.4` gate admits ghost subtraction only where it's pointless (far from fold) and refuses where it helps (near fold). Replace with a uniform (Chester-Friedman-Ursell/Airy) ghost representation valid near the fold, so the label can be used everywhere outside. Retire the decay gate.

2. **Cusp carve-out in the tiler**: the exterior tiler has no cusp-ball exclusion, so cusp-adjacent tiles fail eps by construction. Add explicit carve-out (~0.2 y-units, wider than Pearcey arm's 0.07 rad). Confirm the ladder covers the carved region.

3. **Polar-vs-(s,d) A/B node-budget test**: same tile geometry, same node counts, compare eps. Count how many charts each needs to clear 1e-3. Baseline: 57 charts/39.4 min per band (84% subdivision children). With polar+fold should drop well below 15.

4. **ppGO fallback for beyond-engine-reach**: serve ppGO above engine ceilings as a named rung. Error must DECREASE with w.

## Measured facts (SHA 7a4a8ce)
- polar re-chart deleted ~1500 lines of (s,d) machinery
- ExteriorPolarChart in (rho, theta_c) is single-valued, analytic
- Exterior recursion: 13 depth-3 cap-fails with eps up to 3.6
- Tube charts still use (s,d), untouched by this change

## Constraints
- Fast tests. Follow AGENTS.md.
- Depends on: polar re-chart (DONE at 7a4a8ce), d2 fold (composable, can run in parallel).
