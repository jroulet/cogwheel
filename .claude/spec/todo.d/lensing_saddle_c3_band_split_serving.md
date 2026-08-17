---
section: Backlog
---

- **THE SADDLE c3 RUNG IS DEAD UNDER THE PHYSICAL PRIOR — band-split
  serving is what makes its calibration live** `[→ spec]` — measured
  2026-08-17 (demand-census audit): the c3 certificate admits at band
  floors w_lo >= ~28 (rho 0.3) / ~20 (rho 1.5) / ~8.7 (rho 2.5), but
  the physical 20 Hz prior's band floors are w_lo = 2.476e-3 * M <=
  8.67 (M <= 3500) — the calibrated admitting regime is UNREACHABLE as
  a whole-band intercept, and the census's saddle_c3 route at 0.3%
  faithfully mirrors production. The 672-point calibration is not
  wasted: the rung becomes live via BAND-SPLIT serving — serve the
  analytic channel sum above the certificate's admitting floor and the
  engine/chart below, exactly the w_trust-split architecture the Born
  intercept already implements against the certified map. Design: the
  certificate itself yields the per-draw split point (the smallest w
  where S * ppgo_error_estimate(w) <= bar — closed form to invert or
  bisect, cheap); the serve returns split coefficients like the Born
  band-split with its byte-exact null-split identity pattern. Sizing
  input: the corrected demand census shows the residual concentrated at
  w <= 60; a c3 band-split converts every saddle-exterior draw's
  above-floor band from chart demand to analytic serving, shrinking
  the saddle-side table need to [w_lo, min(60, split point)]. Sequence
  with the demand-sized tiling design (the split changes the tiling's
  w-bands); before 7b.
