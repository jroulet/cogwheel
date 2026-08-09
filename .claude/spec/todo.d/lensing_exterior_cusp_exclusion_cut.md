---
section: Backlog
depends_on: [2026-08-08_exterior_polar_cusp_coordinate]
---

- **Exterior cusp exclusion radius needs correct cut; must cover saddle cusps too**
  `[→ spec]` — identified 2026-08-08 by driver probe analysis.

  The exterior tiler admits tiles whose nearest corner sits just beyond
  `_CUSP_EXCLUSION_DISTANCE = 0.2` source-plane units from an astroid cusp
  vertex, yet the FARFIELD_KERNEL_SUM envelope has a near-cusp zero
  cancellation band extending past that distance (measured: a tile with
  nearest corner at 0.206 is admitted and fails the 1e-3 bar with
  eps ~ 0.076; failing π/2-cusp tiles have nearest-corner distances as low
  as 0.132). The probe showed 123/179 near-π/2-cusp tiles fail (median eps
  0.0039, max 274), driving the ~500-chart tile count instead of the ~70
  target. The u-coordinate fix was correct but addresses a different
  failure mode; the real issue is tile admission into the Pearcey-served
  cusp window.

  **Fix**: measure and set the correct exterior cusp-exclusion cut so tiles
  never straddle the near-cusp cancellation band (serving that window by
  the Pearcey arm / exact engine as designed). Must ALSO cover the macro-
  saddle deltoid cusps (currently `_exclude_near_cusp` checks astroid only)
  and the deltoid-lobe interior cusp handling.

  ACCEPTANCE: a 4×4×4 exterior probe produces ~70 charts (not 500+), no
  tile straddles a cusp window, and the cut is calibrated from measured
  envelope turn-on distances on BOTH parities.
