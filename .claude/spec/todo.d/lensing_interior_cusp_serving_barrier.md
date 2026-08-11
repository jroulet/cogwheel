---
section: Backlog
depends_on: [lensing_revert_residual_table_fix_routing]
---

- **Interior cusp sources still refuse — investigate the certification barrier**
  `[→ spec]` — identified 2026-08-11.

  The `_cusp_vertex` routing fix (source-plane-nearest cusp) is necessary
  but NOT sufficient: interior cusp sources (inside the caustic near a
  cusp, 3 comparable images) STILL refuse at all w (measured w up to 300,
  R~11, even with relaxed envelope_bar=0.5). The refusal is in the arm's
  CERTIFICATION (calibration certificate / normal-form stationary-phase
  check), not the routing or radius gate. ppGO returns a finite value at
  those sources, so the physics is evaluable — the Pearcey path just
  refuses to certify.

  The user suggests possibly resurrecting the F - F_ppgo residual idea
  with correct limits (the earlier residual attempt blew up near the fold
  caustic |P_asymp|~1e9 — but bounded limits may work), or a different
  treatment.

  **Investigate + fix**: why does the Pearcey arm refuse interior cusp
  sources, and what serves them? Candidates: (a) relax/fix the calibration
  certificate for interior configs, (b) a bounded F - F_ppgo residual
  table limited to the interior regime, (c) the wedge/tube interior charts
  with the cusp-adapted u should handle the interior cusp (verify), (d)
  exact engine for the interior cusp neighborhood. The interior cusp is
  the diffraction regime (R < 71) where Pearcey is the correct physics —
  it should serve, not refuse.

  ACCEPTANCE: interior cusp sources (3 comparable images) are served by a
  fast path (table or ppGO or wedge/tube), not the exact engine; the
  refusal barrier is understood and fixed; no live quadrature in the hot
  path; exterior cusp sources continue via ppGO.
