# Build: fix LowWDiffractiveChart residual representation — Airy F_ref anchor

## Mission

The smoke-scale chart bake (driver step) exposed a REPRESENTATION FAILURE
in the shipped `LowWDiffractiveChart` residual. The current anchor
`r_pure = f_schwinger / (sqrt(mu_pure) * prefactor_c(w))` is WRONG for the
near-fold shell and wall band: neither `sqrt(mu_pure)` (w-independent, strips
only a constant) nor `prefactor_c` (strips only the point-mass `w*ln w`
phase) is an oscillatory carrier, so the fold's Airy oscillation stays in the
residual. Measured: |r| crosses Airy zeros (0.997 at w=0.1 -> 0.023 at w=30
-> 0.10 at w=60) with +/-pi phase jumps at each zero. The smoke chart's cubic
interpolation of this collapsed to de-rate 0.055 and 0/240 grid + 0/112
off-grid points within 1e-4 served error (worst 1.58). This is a
representation bug, NOT grid coarseness.

## Binding fix (Professor ruling, 2026-08-20)

1. REPRESENTATION (primary): anchor on the BEAT-FREE AIRY-UNIFORM TWO-CARRIER
   reference -- the tube chart's `F_ref = airy_fold_value(w, tau_bar, xi, p,
   p, sigma)` with q=p, DRY-imported from
   `cogwheel.lensing.chang_refsdal._airy_fold` (reuse `_merging_fold_pair` +
   `airy_fold_value`; NEVER re-derive xi). Its `|F_ref|^2 ~ w^{1/3} Ai^2 +
   w^{-1/3} Ai'^2` is non-vanishing by the Airy Wronskian, so `r = F/F_ref`
   is always finite. The residual becomes the smooth CFU amplitudes p(w),
   q(w) + the resolved-image part.
2. AXIS (secondary): spline/grid the residual in `xi = (3w*DT/4)^(2/3)` (or
   `w^{2/3}`), NOT log w -- the CFU amplitudes vary on the xi scale. Do NOT
   remap before stripping the fold carrier (the Airy phase (2/3)xi^{3/2} =
   w*DT/2 is linear in w, so a monotone remap cannot remove the oscillation).
3. ONE representation for BOTH the near-fold shell and the wall band (both
   are fold-dominated). "Convergence-collapse" of the order-16 series is an
   ORACLE signal (use f_schwinger as the trainer oracle), NOT a
   representation signal -- do not treat it as a reason to split the chart.
4. CAVEAT TO PIN: confirm the wall-band points are fold-dominated (a real
   merging pair exists) via `_merging_fold_pair` BEFORE committing the chart.
   Deep-interior-at-high-gamma points (4 resolved images, no merging pair)
   would need the multi-image geometric sum instead. Pin the real-image
   count / merging-pair presence per cell.

## Scope

IN:
- `scripts/train_low_w_diffractive_chart.py`: replace the residual target
  `r_pure = f_pure * sqrt(1-gamma'^2) / prefactor_c(w)` with the Airy F_ref
  residual `r = f_pure / F_ref` (F_ref via `_merging_fold_pair` +
  `airy_fold_value`), and grid the residual in `xi` (or w^{2/3}) rather than
  log w. Keep f_schwinger as the offline oracle ONLY. Per-cell merging-pair
  pin (which cells are fold-dominated vs multi-image).
- `cogwheel/lensing/low_w_diffractive_chart.py`: the serve re-modulation
  `F_serve = mass_sheet_phase * F_ref * (derate * r_fit)` (multiply by the
  SAME F_ref, matching the tube r*F_ref serve). The chart class's axis is xi
  (or w^{2/3}); the anchor F_ref is evaluated at serve from the merging pair
  (DRY, same builder as the trainer).
- The decline mask: cells without a real merging pair (multi-image /
  deep-interior) are declined (or served by a second representation) per the
  per-cell pin.
- Re-run the smoke bake: the residual must be smooth (no Airy-zero
  crossings), the de-rate must be far above 0.055, and the served error must
  approach 1e-4 on the smoke grid. Report the smoke margins.
- Tests: re-pin the residual-representation expectations (the smooth residual
  is now the CFU amplitudes, not the raw F/sqrt(mu)/C(w)); the w->0 anchor
  and serve-vs-engine pins stay (they assert served values, not the internal
  residual).

OUT (do not touch):
- The order-16 series, `w_low_fit`, the near-fold fence, the census route
  wiring (WP4 is done and correct).
- Rung S / macro-saddle.
- The tube chart's `_airy_fold` internals (import, don't modify).

## Acceptance

- Smoke bake: served error <= 1e-4 on the smoke grid + off-grid midpoints,
  de-rate >> 0.055 (report the number), residual smooth in the xi axis (no
  Airy-zero crossings).
- Serve re-modulation is `F_serve = mass_sheet_phase * F_ref * (derate *
  r_fit)` (the tube r*F_ref pattern, F_ref the same object the trainer used).
- The per-cell merging-pair pin is shipped: fold-dominated cells use the
  Airy representation; multi-image cells are declined or handled per the
  ruling.
- Full bake + shipped `low_w_diffractive_chart.npz` = DRIVER step.

## Constraints

- Branch `claude-dev`. Slow tiers stay gated.
- DRY: import `_merging_fold_pair` / `airy_fold_value` from `_airy_fold`,
  never a re-derivation of xi or the fold form.
- Spec/TODO workflow: `[→ spec]` + completion record; the tracked todo
  `lensing_low_w_near_fold_serve` is binding.
- The full bake + held-out validation is a DRIVER step.
