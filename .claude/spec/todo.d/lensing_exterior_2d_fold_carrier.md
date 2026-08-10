---
section: Backlog
depends_on: [2026-08-10_exterior_fold_carrier_demodulation]
---

- **Exterior 2D (rho, u) fold-carrier (extend the 1D rho-carrier)**
  `[→ spec]` — identified 2026-08-10 after probe (killed at 14 charts, all fail).

  The 1D rho-carrier (b061103) fixed the rho-axis phase winding but left
  the u-axis winding: measured 11.66 rad phase span in u (and in theta_c —
  span is coordinate-independent), max dphase/du = 48 (the cusp-adapted u
  reduces the gradient from 82 but not the total winding). The probe failed
  with off-grid theta (u) eps ~0.52.

  Fix (validated): the fold-carrier must be a 2D array Re(tau_c(rho, u))
  on the spline's ACTUAL axes (rho, u), NOT (rho, theta_c). Measured:
  - Re(tau_c) is LINEAR in rho (slope 2.4-2.7, varying with the angular
    coordinate — the theta_c-dependence is really u-dependence).
  - Re(tau_c) is LINEAR in u (dRe/du ~ -1.45, nearly constant) — the ideal
    carrier form.
  - The exterior rho is the ADDITIVE form (rho = 1 + |y| - r_caustic), so
    the linear-in-rho carrier is consistent with the caustic scaling.
  - A 2D (rho, u) fold-carrier flattens the per-rho phase span in u from
    11.66 -> <= 1.63 rad, splineable at 4 nodes/axis.
  - Serve re-modulation MUST happen at the interpolated u (after the
    theta_c -> u map), never at raw theta_c.

  ACCEPTANCE: the exterior probe produces ~70 charts with all held-out eps
  under the 1e-3 bar; the u-axis off-grid eps drops below the bar; round-trip
  to machine precision; re-modulation at interpolated u verified.
