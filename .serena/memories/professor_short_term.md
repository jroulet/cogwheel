# Professor session: ppGO fast-rung gate calibration (cusp_amplification)

2026-08-10. Reviewed `_pearcey_cusp.py` and `_airy_fold.py` to assess the
physics of lowering _W_PPGO_FLOOR and _R_PPGO_ERROR_CONST to serve the
cusp-window region (source within 0.07 rad of cusp vertex, w-band [0.88, 19.3])
via ppGO instead of Pearcey quadrature.

Key findings:

1. **ppGO at low w is Airy-corrected, not raw geometric**: `fold_ppgo_correction`
   replaces the merging fold pair with the uniform Airy form. This is fold-sound
   and converges to the Pearcey result at large R (cusp-proximity control radius).
   Airy correction is 4-40% at w=5..15; above w~25 diffractive error < Airy residual.

2. **1/w³ kernel truncation floor**: The image_kernel function includes C1/w + C2/w²
   terms, so O(1/w³) truncation error at w=5 is ~0.008 (> 0.005 bar) but at w=6-7 is
   ~0.0046-0.003 (< bar). Physics permits _W_PPGO_FLOOR as low as 5-6 on pure
   kernel-truncation grounds, but the R-gate is the binding constraint.

3. **Calibration must measure R at threshold, not just w**: The ppGO error depends on
   the signed ratio x/y in Pearcey controls, which varies with source direction
   relative to the cusp axes. The calibration must sweep directions at each w and find
   the FIRST direction to fail as w decreases. Derive _R_PPGO_ERROR_CONST from R_cal
   with a 1.5-2× safety factor.

4. **Parities differ**: Negative-parity deltoid lobe cusps have different image
   topology (saddle,saddle merging vs min,saddle) and smaller angular extent. Calibrate
   separately for each parity; use the more restrictive constants as production values.

5. **Expected outcome**: _R_PPGO_ERROR_CONST ≈ 2-4 (down from 50), _W_PPGO_FLOOR ≈ 7-10
   (down from 50), with binding direction likely the hard-axis (pure y-control, pure-
   delta_perp) source offset on the negative-parity side.

6. **The cusp-window lower edge** (w ≈ 0.88): Even with calibrated gates, w=0.88 is
   almost certainly below any plausible ppGO serve threshold — at that frequency the
   Pearcey quadrature is always needed. The excised region's lower portion must stay on
   the certified path.
