---
section: Backlog
depends_on: [lensing_exterior_w_axis_powerlaw_conditioning]
---

- **Exterior envelope rho-axis conditioning (log/rho-1 coordinate)**
  `[→ spec]` — identified 2026-08-10 after the carrier-demodulation build.

  The carrier demodulation fixed the w-axis (node eps 1e-17, off-grid ~1e-4),
  but the SPATIAL axes now dominate the residual: off-grid rho eps ~0.04,
  off-grid theta ~0.009, against the 1e-3 bar. The envelope magnitude grows
  ~4.5 decades toward rho=1: |E| = 0.95 at rho=1.02 down to 2.6e-5 at
  rho=2.5 (gamma=0.5, theta=0.4, w=12). It is power-law-like in (rho-1)
  (exponent ~-1.7 to -3.2, steepening near rho=1; R²≈0.75-0.78 — not a
  clean single power law). 4 linear-rho nodes cannot track a 4.5-decade
  growth toward the caustic.

  theta_c varies ~20× across the quadrant (0.0037 → 0.107, smooth U-shape
  peaking at the cusps) — likely already handled by the cusp-adapted u
  coordinate. gamma varies only ~6× — probably fine.

  **Fix**: reparameterize the rho axis so the 4.5-decade growth toward
  rho=1 is splineable at low node counts — e.g. `u_r = log(rho-1)` (or a
  tuned power `(rho-1)^q`), consistent with the wedge/lobe cusp-adapted
  coordinate precedent and the carrier-demodulation approach. Must handle
  the rho=1 boundary (log singular), the complex envelope (phase), and
  keep the serve/reconstruction path consistent (round-trip to machine
  precision). The user's note: "you can't just fix the ordinate" — the
  spatial transform must be coherent with the w-axis carrier demodulation.

  ACCEPTANCE: the exterior surrogate clears the 1e-3 eps bar at the
  probe's node count with BOTH the w-carrier and rho-coordinate fixes;
  round-trip to machine precision; serve path consistent.
