---
section: Backlog
depends_on: [lensing_saddle_forensics]
---

- **Lobe interior needs cusp-adapted angular coordinate** `[→ spec]` —
  identified by Professor in saddle forensics.

  `LobeInteriorChart` uses ``(rho_lobe, theta_local)`` where ``rho_lobe =
  |y - centroid| / r_deltoid``. At a cusp vertex ``r_deltoid`` scales as
  ``|dtheta|^(1/3)`` (same exponent as the astroid), so ``rho_lobe`` is
  singular — a cusp-adapted coordinate is required for the angular axis.

  **Fix**: Map ``theta_local → u = d^(2/3)`` where ``d`` is the angular
  distance to the nearest deltoid cusp vertex. Same pattern as
  `InteriorWedgeChart`'s ``u = d^(2/3)`` coordinate (wedge v3). Eliminates
  the ``|dtheta|^(1/3)`` singularity in ``r_deltoid`` and makes the lobe
  interior envelope smooth at cusp vertices.

  **Scope**:
  - Replace ``theta_local`` with ``u`` in `LobeInteriorChart` axis schema
  - Add ``_lobe_cusp_angles(gamma)`` → ``theta_to_u`` map per gamma
  - Update training, serving, and serialization for the new coordinate
  - Remove the cusp carve-out (`_LOBE_CUSP_EXCLUSION_DISTANCE`) — no longer
    needed with a smooth coordinate

  ACCEPTANCE: lobe interior chart trains and serves with ``(rho_lobe, u)``
  coordinates; a tile centered at ``u=0`` (cusp vertex) clears the 1e-3 bar
  without subdivision; the carve-out guard can retire.
