# Build brief — analytic cusp vertex on the serving path + `y'''` (step 1c)

## Mission

Two things, both closing the analytic-geometry sweep:
1. Replace `_pearcey_cusp._cusp_vertex`'s finite-difference cusp finder with
   the analytic root. It is on the SERVING path, so the bar is served values,
   not just a geometry tolerance.
2. Extend `geometry.caustic_derivatives` to THIRD order (`y'''`), verified the
   same two-stage way as `y'`/`y''`.

This is build 1c of step 1 of
`.claude/spec/todo.d/lensing_caustic_relative_coordinates.md` (detail:
`todo.d/lensing_analytic_derivatives.md`). Read the master fragment's Part 0
principle and standing rules first. 1a (the `y'`/`y''` cascade) and 1b (the
training-path consumers) have shipped. Do NOT touch 1d (`_WEDGE_EPS`) or 1e
(the interpolation coordinate); those are later.

## Part A — `_cusp_vertex` (serving path)

`_pearcey_cusp._cusp_vertex` (cogwheel/lensing/chang_refsdal/_pearcey_cusp.py,
~line 463) finds the caustic cusp nearest a source by computing caustic speed
`|d source / d theta|` with a hardcoded central difference `delta = 1e-4`,
scanning 129 thetas over a `pi` window for the speed minimum, then
golden-section refining. A cusp is exactly `|y'| = 0` — a ROOT, not a sampled
minimum. It is called once, at ~line 660, and the returned `CriticalPoint`
feeds the Pearcey control arguments, so a wrong or imprecise vertex moves a
served value.

FIX: the analytic root. `geometry.caustic_speed(gamma, theta, *, kappa,
branch)` is exact (1a); the cusp is its zero. Use the same idiom 1b used in
`surrogate_training._refine_cusp_angle`: `g(theta) = y'.y'' =
(1/2) d|y'|^2/dtheta` is real-analytic through the cusp and crosses zero with
`g' > 0` at the speed minimum, so `brentq` on `g` pins the angle to ~1e-10.
Bracket around `seed_theta`, apply a twin gate (sign change AND
`caustic_speed(theta_cusp)` below a small fraction of the local scale), and
return `critical_point` there. No 129-point scan, no golden section, no finite
difference.

CRITICAL FRAME SUBTLETY (measure, do not assume): the 1a cascade works in the
shear-aligned frame — `caustic_derivatives`/`caustic_speed` take NO `beta` and
are rotation-invariant, with the caller passing `theta` relative. But
`_cusp_vertex` handles `beta != 0` and passes `beta` to `critical_point`
(which internally uses `phase = theta - beta`). So the analytic root must be
found in the `phase` frame and mapped back to `theta` before calling
`critical_point`. The single caller passes `beta` from the live config; the
served-values acceptance below is what proves you got the rotation right.

## Part B — extend the cascade to `y'''`

Add third-order derivatives to `geometry.caustic_derivatives` (or a sibling;
your API choice — keep it consistent with the existing `(y', y'')` return).
Differentiate the SAME closed-form cascade one order further:
`u -> u' -> u'' -> u'''`, `r -> ... -> r'''`, then `y_i''' `by the product
rule on `p_i * r * T_i`. Plain numpy, vectorised over `theta`, no new
dependency, NO finite difference.

Do NOT hand-transcribe the `y'''` formula from this brief as truth — an
earlier brief inlined a wrong `p_i = M_ii - u` (should be `M_ii - lam*u`) and
a co-erroneous oracle hid it for a full round (F038). Derive it, and verify
against a TWO-STAGE oracle:
- STAGE 1: validate the transcribed curve against `critical_point(...).source`
  at float64 (F038 measured 5.14e-15; require <= 1e-13). Without this, stage 2
  proves nothing.
- STAGE 2: `mpmath.diff(curve, theta, 3)` at 40 dps on that validated curve.
Mirror `critical_point`'s two quirks in the oracle or it goes complex:
positive parity IGNORES `branch` (only the `+` root is a positive radius); a
slightly-negative saddle discriminant is CLAMPED to zero. `y'''` amplifies
noise, so the mixed tolerance may need to loosen from the `y''` bar — MEASURE
the achievable envelope and state it; do not assert a tolerance you cannot
meet, and do not loosen it beyond what the oracle's own precision forces.

`y'''` has no consumer in this build — F040's cusp-window width (`w^{-1/4}`
with a `|y'''_perp|` coefficient) is a LATER step. 1c just ships and certifies
the primitive.

## Scope

IN:
- `cogwheel/lensing/chang_refsdal/_pearcey_cusp.py::_cusp_vertex` — analytic
  root, frame-correct.
- `cogwheel/lensing/chang_refsdal/geometry.py` — `y'''` on the cascade.
- Tests in `cogwheel/tests/`.

OUT — do not touch:
- `_WEDGE_EPS` (1d), the interpolation coordinate / node placement (1e),
  `_build_tube_chart`, any training or chart artifact, any `F042` fixture.
- The `y''`-based cusp WIDTH (F040) — later; `y'''` ships unused here.

## Acceptance

1. `_cusp_vertex` contains no finite difference, no 129-point scan, no
   golden-section; it returns the `critical_point` at the analytic root of
   `caustic_speed = 0`, frame-correct for `beta != 0`.
2. SERVED VALUES UNCHANGED: the Pearcey arm's served amplification (`|F|` and
   phase) matches the pre-1c implementation to the F016 envelope bar, over a
   set of cusp-neighbourhood configs spanning both parities and `beta != 0`.
   This is the load-bearing gate — the cusp vertex only matters through what
   it serves. State the configs and the measured max deviation.
3. The cusp angle agrees with the analytic root to 1e-10, and the vertex is
   found in O(1) geometry calls (not ~258).
4. `y'''` agrees with the two-stage oracle at the stated (measured) tolerance,
   on both parities and branches, including `kappa != 0` and near-axial
   `theta`. Stage 1 (curve vs `critical_point.source`) reported <= 1e-13.
5. Suites you touched run green (`test_lensing_pearcey*`, geometry). Full
   suite is a post-build driver step; do not run it in-build.

## Constraints
- Two-stage oracle is mandatory (F038). An oracle that re-transcribes the
  curve without stage 1 is circular.
- Assert VALUES against a tolerance, not code paths.
- Slow tests never run in-build; `COGWHEEL_BRUTE_ACCURACY` /
  `COGWHEEL_STRICT_TIMING` stay empty.
- `SDK_CONDA_ENV` from `.env` (`cogwheel-newlal`); interpreter
  `$(conda info --base)/envs/$SDK_CONDA_ENV/bin/python`, never `conda run -n`.
- Named refusals stay named: `_cusp_vertex` returns `None` where the geometry
  refuses, exactly as now; `caustic_derivatives` refuses by name off-domain.
- Prose you change must be true when done. `_cusp_vertex` is described in
  SPEC.md's microlensing-engine row (Pearcey arm) — verify it still reads true
  after the swap.
