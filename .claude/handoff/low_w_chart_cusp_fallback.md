# Build: chart reference — Pearcey (cusp) fallback where the fold degenerates

## Mission

The low-w chart's F_ref builder (Airy fold reference) refuses ~56 cells where
`b3 -> 0` (the fold's cubic coefficient vanishes), declaring them
"unbuildable." This is WRONG (owner ruling 2026-08-20): there is no
degenerate fold — a vanishing `b3` is the **fold -> cusp transition**, and the
correct reference there is the **Pearcey (cusp) uniform form**, which is
FINITE (the codebase's `_pearcey_cusp.pearcey` is certified). Both a fold and
a cusp have no NaN in the right coordinates. The serving ladder already
encodes this: `_uniform_arm_value` tries fold (Airy) -> ppGO+ghost -> cusp
(Pearcey). The chart's reference builder MUST mirror that hierarchy.

## Root cause (measured)

At gamma'=0.8, rho=2.0: `_soft_axis_cubic` gives `b3 ~ 1.4e-15` for theta in
[0.0, 0.5] and [1.4, 1.57], and `_fold_amplitudes` returns None when
`|b3| <= _B3_MIN = 1e-6`. Those cells were marked unbuildable (declined),
but they are cusp-adjacent: the fold reference is the wrong normal form
there, and the correct finite reference is the Pearcey function. The chart
should SERVE them with the Pearcey-anchored residual, not decline them.

## Fix (owner ruling)

The F_ref builder (`airy_fold_reference` in
`cogwheel/lensing/low_w_diffractive_chart.py`) gains a PEARCEY FALLBACK:
where `_fold_amplitudes` refuses (b3 -> 0, the cusp transition), build the
reference from the uniform Pearcey form instead of declaring the cell
unbuildable. Mirror the serving ladder exactly:

- FOLD cells (b3 well away from 0): Airy F_ref as now.
- CUSP cells (b3 -> 0, `_fold_amplitudes` None): Pearcey reference
  `F_cusp = A w^{1/2} exp(i w tau_c + i sigma_c) P(x, y)` with the certified
  controls `x = c_x w^{1/2} delta_parallel`, `y = c_y w^{3/4} delta_perp`
  from `_pearcey_cusp` (import, do NOT re-derive the controls). The
  reference is the cusp's uniform form WITHOUT the q=0 / table certificate
  over-restriction -- the same "reference is a demodulation carrier, not a
  served value" logic as the Airy F_ref (F075 does not apply).

The residual anchors consistently: `r = f_pure * sqrt(1-gamma'^2) / F_ref`
where F_ref is whichever normal form applies (Airy or Pearcey) per cell.
`declined_mask` then covers ONLY genuinely-unservable cells (e.g. a
degenerate geometry the Pearcey form also cannot build), NOT the cusp
transition.

## Scope

IN:
- `cogwheel/lensing/low_w_diffractive_chart.py`: `airy_fold_reference`
  (rename to `fold_cusp_reference` or add a Pearcey fallback path) — try the
  Airy fold reference; on `_fold_amplitudes` None (b3 -> 0) build the
  Pearcey reference from `_pearcey_cusp`. Keep the absolute-frame, no-t_min
  convention. Return the reference or None only if BOTH forms fail.
- `scripts/train_low_w_diffractive_chart.py`: the residual and the
  unbuildable/decline classification use the new reference; cusp cells are
  now BUILDABLE (not declined). Update the provenance populations
  (n_unbuildable_cells should drop to ~0 if all cells are fold or cusp).
- Tests: re-pin the F_ref builder contract — a cusp-adjacent cell (b3 ~ 0)
  returns a finite Pearcey-based F_ref, not None; a genuinely-degenerate cell
  (both forms fail) still declines. The serve/trainer consistency pin and
  the node-exact re-modulation pin stay (they assert served values).

OUT (do not touch):
- `_airy_fold` and `_pearcey_cusp` internals (import, don't modify).
- The served series, `w_low_fit`, the fence, Rung S.
- Any surrogate-chart or campaign work.

## Acceptance

- Cusp-adjacent cells (b3 ~ 0) are served via the Pearcey-anchored residual,
  NOT declined: the smoke bake's unbuildable-cell count drops to ~0 (or the
  genuinely-degenerate remnant only), and the served error for those cells
  approaches 1e-4 (report the cusp-cell margins separately).
- The residual is smooth (no NaN, no Airy/Pearcey zero crossings) in the
  w^{2/3} axis for BOTH fold and cusp cells.
- The reference is the fold (Airy) or cusp (Pearcey) uniform form per cell,
  DRY-imported from `_airy_fold` / `_pearcey_cusp`; never a re-derived
  control or normal form.
- Smoke bake: de-rate far above 0.137, served error approaching 1e-4 on the
  smoke grid + off-grid midpoints. Full bake + shipped npz = DRIVER step.

## Constraints

- Branch `claude-dev`. Slow tiers stay gated.
- Spec/TODO workflow: `[→ spec]` + completion record; `lensing_low_w_near_
  fold_serve` binding.
- The full bake + held-out validation is a DRIVER step.
