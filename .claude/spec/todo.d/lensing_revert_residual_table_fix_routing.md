---
section: Backlog
depends_on: [lensing_zero_quadrature_pearcey]
---

- **Revert residual-table reformulation; fix cusp-arm routing instead**
  `[→ spec]` — identified 2026-08-11 by Professor (verdict after driver
  measurement).

  The zero-quadrature Pearcey build's core reformulation — tabulating the
  residual R(x,y) = P(x,y) - P_asymp(x,y) instead of demodulated P — was
  OVER-ENGINEERING. Professor verdict (measured, 568 served configs,
  gamma=0.5 both parities, rho [1.1,5.0], w [10,200]):

  - The x=-71 "structural barrier" (subdominant-saddle spline error
    25000x) exists ONLY in the expanded box's corners (R ~ 115), which no
    served source reaches — the 0.07 rad cusp window forces small
    delta_parallel -> |x| <= 7.95, well inside the demodulated table box
    (x +-27.6).
  - Every served EXTERIOR cusp config has R >= 71.6 > r_ppgo_min (71.1),
    so the ppGO rung preempts the table — the table is never consulted for
    exterior cusp sources.
  - The residual format was ALSO numerically unstable near the fold caustic
    (|P_asymp| ~ 1e9) and was already reverted by the Inspector (INS-1).
  - The table's REAL role is INTERIOR cusp sources (inside the caustic, 3
    comparable images), where R < 71 fails the ppGO gate. Those refuse due
    to a `_cusp_vertex` ROUTING BUG: it selects the cusp via
    `nearest_caustic_point` image-theta seeding, which can snap to the
    wrong cusp or a fold segment, giving wrong (x,y) controls and R too
    small -> the arm refuses -> exact engine.

  **Fix**: (1) keep the demodulated PearceyTable (revert any residual-format
  residue; ensure schema stays the demodulated format), (2) fix `_cusp_vertex`
  routing to select the SOURCE-PLANE-NEAREST cusp vertex (probe the four
  astroid / six deltoid cusps, pick min source-plane distance), (3) regenerate
  + ship the deleted pearcey_table.npz artifact, (4) verify interior cusp
  sources then pass the radius gate and serve via the demodulated table, (5)
  verify exterior cusp sources continue to serve via ppGO.

  ACCEPTANCE: interior cusp sources (3 comparable images) serve via the
  demodulated table (not exact engine); exterior cusp sources serve via ppGO;
  no live quadrature in the hot path; the table artifact is shipped.
