---
date: 2026-07-29
section: lensing
---

# Analytic caustic derivative cascade (build 1a)

Step 1a of `todo.d/lensing_caustic_relative_coordinates.md`'s "THE ANALYTIC
SWEEP" (itself step 1 of the caustic-relative-coordinates program, ordered
first per `todo.d/lensing_analytic_derivatives.md`). Adds four public
functions to `cogwheel/lensing/chang_refsdal/geometry.py`, beside
`r_caustic`: `caustic_derivatives(gamma, theta, *, kappa=0.0, branch=1) ->
(y', y'')`, `caustic_speed` (`|y'|`), `caustic_curvature_radius`
(`|y'|**3 / |y1'y2'' - y2'y1''|`), and `fold_opening_direction` (unit
`D2y[e,e]`). All four differentiate the exact closed-form caustic curve
directly — no finite difference, no `np.gradient`, no sampled-arc stencil.

Radial weight is `p_i = M_ii - lam*u`, not `M_ii - u`; the two forms coincide
only at `kappa = 0` and differ by 0.19-0.39 in source-plane position at
`kappa = 0.3`. Caught by cross-checking `critical_point`'s own
`source = macro_matrix @ image - image/radius**2` during review.

Verified by a two-stage oracle: stage 1 validates the transcribed curve
against `critical_point`'s shipping output (5.14e-15 over 110 configs), stage
2 differentiates that validated curve at 40 dps. F038 flagged that a
single-stage oracle (re-transcribing the curve and differentiating its own
transcription) cannot catch a transcription error — exactly how the `lam*u`
error above survived a full round before this fix. Measured: `y'` worst
relative 4.39e-13, `y''` worst relative 2.56e-14, 0 failures at
atol=5e-13 + rtol=1e-11 over 110 configs both parities/branches, including
`kappa != 0`, near-axial `theta`, and near the parity wall; fold direction
16/16 to the correct side, unit to 1e-12; `|y'|` at the astroid cusp = 1.3e-16
(cusps are now exact roots).

Domain contract inherited from `critical_point`: positive parity ignores
`branch`; the saddle wedge edge refuses by name rather than dividing by a
clamped-zero discriminant; refusal is whole-call, never per-element `nan`.

Replaces nothing yet. The numerical estimators these will retire
(`_min_curvature_radius`, `_branch_speed_profile`, `_find_cusps`,
`_probe_arc_side`, `_cusp_vertex`) remain in place — that is steps 1b/1c,
still open in `todo.d/lensing_caustic_relative_coordinates.md` and
`todo.d/lensing_analytic_derivatives.md` (which also still owes the
`y'''` cascade extension for the cusp-exclusion half-width, F040).

Full suite green: 905 passed, 126 skipped, 5 xfailed. Inspector PASS,
Professor PASS, zero implementation or design findings.
