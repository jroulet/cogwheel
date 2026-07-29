---
date: 2026-07-29
bump: minor
---

### Serving ladder: one authoritative geometric-vs-wave gate; arms and geometric branch are NOT certified

Both operator grids now route the geometric-vs-wave decision through
`operator.select_branch`, giving the predicate one home. Previously three sites
disagreed.

Corrected two false claims in the Build 8e serving-ladder description: the
uniform arms were described as "(certified)" and as firing "ONLY at the
previously-refusing sites". F028 measured the fold arm at 60%–267% relative
error, and the arms no longer fire at every previously-refusing site — resolved
above-ceiling positive-parity nodes are served by geometric optics instead.

Added F029's finding that the geometric branch is not certified either: a
residual ~1% O(1) tail (p99 7.1e-1, max 74) controlled by distance to the
caustic, which `_certify_geometric_census` does not catch.
