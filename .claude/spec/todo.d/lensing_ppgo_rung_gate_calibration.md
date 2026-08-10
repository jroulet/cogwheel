---
section: Backlog
depends_on: [2026-08-10_exterior_2d_fold_carrier]
---

- **ppGO rung gate calibration — serve the excised regions without quadrature**
  `[→ spec]` — identified 2026-08-10.

  The `cusp_ppgo_high_w` build added a ppGO fast rung inside
  `cusp_amplification` (serves via `fold_ppgo_correction`, ~10^3x faster
  than live Pearcey quadrature), but gated it at `_W_PPGO_FLOOR = 50` and
  `_R_PPGO_ERROR_CONST = 50.0`. The excised exterior regions — the astroid
  cusp windows AND the saddle deltoid-excised exterior — have w-bands
  around [0.88, 19.3] (below the w=50 floor), so the ppGO rung never fires
  there; those draws fall to the Pearcey table / live quadrature instead of
  the fast ppGO path. The whole point of excising is to avoid BOTH the
  surrogate AND the quadrature.

  **Fix**: calibrate the ppGO rung gates (measure the ppGO error vs the
  exact Pearcey/engine across w for the excised regions; lower
  `_W_PPGO_FLOOR` and/or `_R_PPGO_ERROR_CONST` so ppGO certifies in the
  excised exterior w-band) so the excised regions are served by ppGO
  without quadrature. Must remain refusal-conservative (never serve a
  wrong ppGO value). Applies to BOTH parities.

  ACCEPTANCE: the excised cusp/ghost-region draws are served by the ppGO
  rung (not live quadrature) across their w-band; the ppGO error vs the
  exact engine stays under the certification bar; the serving ladder is
  refusal-conservative.
