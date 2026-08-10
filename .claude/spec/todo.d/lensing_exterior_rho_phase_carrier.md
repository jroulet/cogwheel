---
section: Backlog
depends_on: [lensing_exterior_rho_axis_conditioning]
---

- **Exterior envelope rho-phase carrier demodulation (+ correct log-rho coordinate)**
  `[→ spec]` — identified 2026-08-10 after probe 3 (killed at 56 charts, 30/55 fail).

  Probe 3 (all three prior fixes in HEAD: cusp exclusion, w-carrier
  demodulation, log(rho-1) rho-axis) still fails: 30/55 charts over the
  1e-3 bar, subdivision grinding to the depth-3 cap. At nodes eps ~1e-4
  (fixes work at nodes) but off-grid rho midpoint eps ~0.38 (catastrophic).

  Root cause (measured): the envelope PHASE rotates ~2π every 0.3 in rho
  across the tile (-1.87 -> +2.84 -> -1.73 -> +1.43 rad over rho in
  [1.3, 2.0]). This is a rho-PHASE CARRIER, exactly analogous to the
  w-phase carrier the w-carrier-demodulation build removed. A magnitude
  coordinate (log(rho-1)) cannot fix a phase rotation — the real/imag
  parts oscillate in rho at the phase rate.

  Also corrected: the magnitude is |E| ~ rho^(-p), NOT (rho-1)^p —
  log|E| is linear in log(rho) (R²=0.999) vs log(rho-1) (R²=0.986). The
  rho-log build chose the slightly-wrong coordinate.

  **Fix**: (1) demodulate the envelope by the rho-phase carrier before
  fitting (measure per-node rho-phase slope, median -> k_rho_chart,
  E *= exp(-1j * k_rho * (rho-1)) or similar), re-modulate at serve —
  the rho analog of the w-carrier. (2) Correct the rho magnitude
  coordinate from log(rho-1) to log(rho) (or verify whether the phase
  demodulation alone suffices, in which case the log coordinate may be
  unnecessary). (3) Ensure coherence with the w-carrier and serve
  round-trip.

  ACCEPTANCE: exterior probe produces ~70 charts with all held-out eps
  under the 1e-3 bar at the 4x4x4 node count; round-trip to machine
  precision; serve path consistent.
