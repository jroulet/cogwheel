# Build Brief: ppGO handoff above chart w-ceiling for interior draws

## Mission

The InteriorWedgeChart has a finite w-ceiling (DD product cap:
`w_max = _DD_PRODUCT_MARGIN / (r_max * reach_max)`). Above that ceiling,
interior draws (rho < 1, w > w_ceiling) have NO serve path — they fall
through to exact quadrature.

The fold-corrected ppGO (`fold_ppgo_correction`) IS physically correct at
high w (the Airy regime where xi >> 1 makes the fold approximation exact),
but the ppGO certification map doesn't certify positive-parity interior
cells because the angular sweep's worst case (axis angles ±π/2 where
Δτ → 0) dominates.

## The physics argument

Above the chart ceiling, `w * |y|` is large (that's WHY the chart can't
train there — the engine refuses). But `w * Δτ` for the fold pair may
still be O(1) at axis angles. The key insight: for draws where
`ξ = (3wΔτ/4)^{2/3} > ξ_threshold` (say ξ > 5), the fold approximation
IS the exact answer to better than 1% — the Airy function's asymptotic
regime. This is a GEOMETRIC criterion, not a certification sweep.

## Implementation

1. In `select_chart` (or `_surrogate_coefficients`), add a serve path for
   interior draws above the wedge chart's w-ceiling:
   - Compute `Δτ` for the nearest fold pair (from the geometry)
   - Compute `ξ = (3wΔτ/4)^{2/3}`
   - If `ξ > XI_FOLD_THRESHOLD` (to be determined, likely 3-5):
     return `fold_ppgo_correction` as the serve value
   - Otherwise: fall through to exact engine

2. Determine `XI_FOLD_THRESHOLD` by measurement:
   - At representative interior (r, theta, gamma) points above the DD cap
   - Sweep w and compare `fold_ppgo_correction` vs exact engine
   - Find the ξ where relative error drops below 1% permanently

3. Wire into the census so these draws register as 'ppgo_interior' not
   'out-of-box'.

## Acceptance

- Interior draws with ξ > threshold are served (not quadrature).
- Measured error < 1% for all served draws (by construction of threshold).
- Census shows nonzero served fraction for interior at high w.
- No regression on existing serve paths.

## Constraints

- Fast tests only.
- The threshold is DERIVED from measurement, not tuned.
- Follow AGENTS.md and the spec/TODO workflow.
