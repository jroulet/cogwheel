---
section: Backlog
depends_on: [lensing_cusp_ppgo_at_high_w]
---

- **`train()` COVERS ONE 0.04-WIDE GAMMA BAND PER PARITY — about 4% of the
  prior box** `[→ spec]` — measured 2026-08-06.

  `PriorBox.from_prior_classes(f_lo_hz=0.16, f_hi_hz=0.40).gamma_range` is
  `(0.0, 1.6)`. `_gamma_band(box, parity, halfwidth)` returns a band CENTRED
  in the parity's sub-range and `halfwidth` wide — its docstring says "A
  narrow gamma band", so this is deliberate, not a bug in the function:

      parity +1: sub-range (0.00, 0.99), centre 0.495 -> (0.475, 0.515)
      parity -1: sub-range (1.01, 1.60), centre 1.305 -> (1.285, 1.325)

  With the production `gamma_band_halfwidth=0.02` that is **0.04 of 0.99
  (4%)** for positive parity and **0.04 of 0.59 (7%)** for the macro saddle.
  `train()` loops `for parity in (1, -1)` and calls `_gamma_band` ONCE per
  parity, so there is no outer sweep: everything outside those two slivers is
  simply never trained.

  ## Why this went unnoticed

  `scripts/train_surrogate_production.py` is named "production" and prints
  "PRODUCTION SURROGATE TRAINING", so its artifact reads as a full-domain
  surrogate. The narrowness is visible only by evaluating `_gamma_band`
  against the box. The 2026-08-05 run that died after 2h24m was not 60%
  through the domain — it was ~60% through ONE BAND of a 4% slice.

  ## Cost of actual coverage (measured, not estimated)

  Per band on the wedge path: ~42 min (tube 1.2 + exterior 39.4 + interior
  1.8; the interior figure is measured, the exterior from the dead run's
  timestamps). Tiling the parity sub-ranges at 0.04:

      positive parity  0.99 / 0.04 ~ 25 bands
      macro saddle     0.59 / 0.04 ~ 15 bands
      TOTAL            ~40 bands x 42 min ~ 28 h

  `stable_gamma_bands` may bisect further near the parity wall
  (`gamma -> 1`), so treat 28 h as a floor. This is a WEEKEND run, not an
  overnight one, and it wants a resumable driver and a box that will not be
  OOM-killed by another user (the 2026-08-05 death).

  ## Work

  - Decide the intended contract: is `train()` meant to be called REPEATEDLY
    (once per band, with the caller sweeping gamma) and the artifacts merged,
    or should it sweep internally? The `_load_or_build` resume path and the
    `label=f'{label}_b{i_band}'` naming suggest the former was intended, but
    nothing in `scripts/train_surrogate_production.py` does the sweep.
  - Whichever it is, make the narrowness IMPOSSIBLE TO MISS: the training
    report should record the gamma coverage actually trained as a FRACTION of
    the box, and `serve` should refuse (not silently extrapolate) outside it.
  - Only then run the full sweep, with a cost quoted from the measured 42
    min/band.

  ACCEPTANCE: the training report states trained-gamma coverage as a fraction
  of `box.gamma_range`; a serve query at a gamma outside every trained band
  refuses with a named error rather than returning a value; and the pilot
  artifact is labelled as a pilot.
