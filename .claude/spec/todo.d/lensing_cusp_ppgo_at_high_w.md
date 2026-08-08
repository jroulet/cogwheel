---
section: Backlog
depends_on: [2026-08-08_lobe-cusp-adapted-coordinate]
---

- **High-w cusp serving falls through to live Pearcey quadrature instead of
  geometric optics + ghost** `[→ spec]` — identified 2026-08-08.

  At high ``w`` the Pearcey controls ``(x, y)`` are large and the uniform
  Pearcey integral asymptotes to the geometric image sum: the fold-corrected
  ppGO (`fold_ppgo_correction` including the Airy ghost at merging fold
  pairs) plus non-merging image kernels is accurate and ~10^3× faster than
  live certified quadrature. The cusp arm should certify and serve ppGO
  above a cross-over ``w``.

  WORK: add a ppGO rung in the cusp arm (or a bypass in the ladder) above
  a measured cross-over ``w`` where the ppGO error drops below the bar.
  Retire the live-quadrature fallback for the high-w regime.
