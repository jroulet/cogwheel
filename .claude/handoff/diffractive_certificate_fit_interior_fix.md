# Build: complete INS-3-001 — calibrate the deep interior so it is served conservatively, not clipped to 60

## Mission

The fenced-fit build stranded at revision (credits died). The two-sided fence
is in place, but the INS-3-001 fix was NOT applied: the deep interior is
"served by the same fit as the smooth exterior" while the calibration grid
starts at r=0.3, so NO bake samples the deep interior (gamma<=0.3 needs
r<0.22). The un-calibrated fit over-serves there (measured: gamma=0.2/rho=0.3
-> fit 60 vs true 34, 1.77x; gamma=0.3/rho=0.3 -> 60 vs 20, 2.93x;
gamma=0.3/rho=0.5 -> 60 vs 22, 2.72x; gamma=0.5/rho=0.3 @ theta=pi/4 ->
15.6 vs 6.1, 2.54x) and `min(w_fit, _DIFFRACTIVE_FIT_CEILING)` clips to 60,
quietly re-serving the interior where the series is NOT honest. This build
completes the fix.

## Owner ruling (binding, from the INS-3-001 escalation)

Do NOT decline the deep interior (it is 39% of residual demand — gutting the
rung to the engine is wrong). CALIBRATE it honestly:

1. Extend the calibration grid into the deep interior: radii down to
   r ~ 0.1 (e.g. `linspace(0.1, 1.3, ...)` in `_unfenced_grid_points`,
   currently line ~197 `linspace(0.3, 1.3, 5)`), so the fit is trained where
   gamma'*s*w/2 is genuinely small. The honest ceiling there is a SMOOTH
   monotone function of rho that the log-log poly + caustic feature should
   capture once sampled — the over-serve is a calibration gap, not a
   representation wall.
2. Remove `min(., CEILING)` as the interior's conservativeness mechanism:
   the fit must be conservative there ON ITS OWN (de-rated), never by
   clipping to 60. The ceiling cap is a hard physical bound for w_hi, NOT a
   substitute for a conservative fit.
3. Re-examine the lower fence side (RHO_LO): if the fit is genuinely
   calibrated+conservative in the deep interior, the interior no longer
   needs special-casing — simplify to the smooth-exterior fit covering the
   interior too, or keep a minimal fence only if measurement shows the
   interior is still hard. Do NOT keep a fence branch that exists only to
   paper a calibration gap.
4. Extend the engine-backed validation to the deep interior: add fixtures at
   gamma in {0.2, 0.3} x rho in {0.2, 0.3, 0.5} asserting
   `w_low_fit <= w_low_true * (1 + 1e-5)` (exact zero-over-serve tolerance,
   engine-measured via the bake script's `_measure_w_low_true`). These are
   the cells INS-3-001 measured over-serving 1.77x-2.93x; they must be green
   after the re-bake.

## Scope

IN:
- `_diffractive.py`: the interior serve path — extend calibration coverage,
  remove the clip-as-conservativeness, keep the near-fold fence and the wall
  refusal. The even-harmonic cos(2k theta) basis + caustic feature + degree-2
  log-log poly stay (the representation is sound in the smooth region).
- `scripts/fit_diffractive_certificate.py`: extend `_unfenced_grid_points`
  radii into the deep interior (r >= 0.1); the fenced grid and off-grid
  midpoints follow automatically. Re-bake smoke in-build (PROVISIONAL),
  report the interior-cell margin numbers (the INS-3-001 fixtures).
- Tests: restore real value assertions at gamma 0.2/0.3/0.5 in
  test_lensing_part0_mechanical.py (currently `assertIsNotNone` is vacuous —
  it certifies 'served' at clipped-60 where the interior over-serves).
  Add the engine-backed zero-over-serve fixtures above.
- The census mirror (serve_route_census.py) mirrors `w_low_fit` — verify
  interior draws are classified per the corrected serve (no signature change;
  the mirror binds the SAME predicate object).

OUT (do not touch):
- The served series `diffractive_amplification` (exact order-16).
- The near-fold fence and the tracked low-w near-fold serve todo
  (`todo.d/lensing_low_w_near_fold_serve`) — that is the NEXT build, not this
  one.
- Any surrogate-chart or campaign work.

## Acceptance

- INS-3-001 fixtures green: at gamma in {0.2, 0.3} x rho in {0.2, 0.3, 0.5},
  `w_low_fit <= w_low_true * (1 + 1e-5)` (engine-measured). No interior cell
  is served at the clipped ceiling unless the engine truth there is genuinely
  the ceiling.
- The interior is served by a CONSERVATIVE calibrated fit, not by
  `min(., CEILING)`. Remove the clip-as-conservativeness; the de-rate is the
  sole margin.
- Calibration grid covers the deep interior (radii down to ~0.1); the fit's
  interior margin numbers are quoted in the completion record.
- The near-fold fence still holds (shell declines, wall raises, smooth
  exterior conservative); the fenced-domain de-rate target >= 0.70 (or the
  measured best, quoted).
- In-build: smoke bake only (< ~10 min, PROVISIONAL). Full bake + paste +
  on/off-grid validation is the DRIVER step.

## Constraints

- Branch `claude-dev`. Slow tiers stay gated. In-build smoke bake < ~10 min.
- Spec/TODO: `[→ spec]`.
- Mirror-fidelity: production `w_low_fit` and the bake script share the basis
  + grid builders; never two copies.
- The full bake is a DRIVER step (the SDK 1200s ceiling kills in-build full
  bakes).
- The low-w near-fold serve todo (`lensing_low_w_near_fold_serve`) is the
  immediate next build after this one; do NOT scope-creep into it here.
