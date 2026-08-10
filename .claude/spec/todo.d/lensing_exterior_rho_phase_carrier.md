---
section: Backlog
depends_on: [lensing_exterior_rho_axis_conditioning]
---

- **Exterior ghost-region tile exclusion (fix the unsmoothable-region admission)**
  `[→ spec]` — identified 2026-08-10 after probe 3 (killed at 56 charts, 30/55 fail).

  Probe 3 (all three prior fixes in HEAD: cusp exclusion, w-carrier
  demodulation, log(rho-1) rho-axis) still fails: 30/55 charts over the
  1e-3 bar, subdivision grinding to the depth-3 cap. At nodes eps ~1e-4
  (fixes work at nodes) but off-grid rho midpoint eps ~0.38 (catastrophic).

  Root cause (measured, decisive): the KERNEL_SUM residual is DOMINATED
  by the unsubtracted ghost (|G| ~ 3.2-3.4 x |E_ks| everywhere computable;
  image count stays 2). The ghost gate (F027) REFUSES in the failing band
  [1.1, 1.9] (Im tau_c < 0.4), so MINUS_GHOST is unavailable there, and
  the chart uses KERNEL_SUM (Window iii) which leaves the ghost in. The
  rho-phase winding IS the ghost's phase. No coordinate transform
  (carrier, log-rho) can smooth a dominant oscillatory ghost — that is
  fighting physics.

  **Fix**: EXCLUDE tiles in the ghost-dominated regime from the exterior
  tiler (mirroring the cusp-exclusion precedent), serving those draws by
  the exact engine / Airy-Pearcey arms. Optionally use the
  FARFIELD_KERNEL_SUM_MINUS_GHOST label where the gate permits. This
  collapses the tile count toward ~70.

  ACCEPTANCE: exterior probe produces ~70 charts with all held-out eps
  under the 1e-3 bar at the 4x4x4 node count; excluded regions fall to
  the exact engine (census fall-through); no tile straddles the
  ghost-transition zone.
