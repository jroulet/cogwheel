# Build Brief: D2 reflection fold — chart one quadrant, serve four

## Mission

The macro matrix is diagonal (`diag(1-gamma, 1+gamma)`), so the lens is invariant under `y1→-y1` AND `y2→-y2`. Every field is determined in ONE quadrant. Currently the exterior tiler (both parities) and the saddle interior tiler chart the FULL circle, wasting a factor of 4 on the dominant training cost (exterior: 57 charts/band at 39.4 min). Fold them to one quadrant.

## Measured facts (SHA 7a4a8ce)

- Verified at gamma=1.3: four sign combinations of y=(±0.6, ±0.25) give identical image delays to 8 decimals.
- Exterior dominates training: 39.4 min/band vs interior's 1.8 min. Folding is a straight 4x on the 57-chart baseline.
- Compose with polar re-chart: the two changes together should bring exterior from 57 charts/band to well under 15.

## Work

1. Fold the positive-parity exterior tiler to one quadrant. The wedge already does this with `(|y1|, |y2|)` in `_to_wedge_fixed`. Mirror queries serve identical values via reflection.
2. Fold the saddle exterior tiler to one quadrant.
3. Fold the saddle interior: chart ONE lobe, and within it ONE half. The other lobe follows by the reflection that swaps them.
4. Pin each fold with bitwise-identical mirror-query tests (same pattern as the wedge build).
5. Watch the seam: tile edges ON the symmetry axes (where image pairs are delay-degenerate). This is benign for the fold itself but must not create coordinate singularities.

## Acceptance
- Exterior charts per band fall by ~4x from the fold alone.
- Mirror-image queries serve bitwise identical values.
- Saddle interior charts one half-lobe.
- No regression on existing chart types.

## Constraints
- Fast tests only. Follow AGENTS.md.
- Depends on: polar re-chart (DONE at 7a4a8ce).
