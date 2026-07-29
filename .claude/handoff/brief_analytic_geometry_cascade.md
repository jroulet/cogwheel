# Build brief — the analytic caustic-derivative cascade (step 1a)

## Mission

Give `geometry` the analytic derivatives of the caustic — `y'(theta)` and
`y''(theta)` — and the three quantities the package currently estimates from
them: curvature radius, caustic speed, and the fold-opening direction.

This is build **1a** of step 1 of
`.claude/spec/todo.d/lensing_caustic_relative_coordinates.md`, whose detail
fragment is `todo.d/lensing_analytic_derivatives.md` (both rendered into
`.claude/spec/TODO.md`). Read the master fragment's Part 0 governing principle
and standing rules before starting. Builds 1b and 1c retire the CONSUMERS of
these estimates; this build only creates the exact replacements and proves
them. Do not start on the consumers.

## Why the derivatives are the deliverable, not the scalars

The caustic is a closed-form parametric curve, so every local property of it is
closed form too. The package instead estimates three of them numerically, in
three different places, each with its own step size or threshold constant:

- `surrogate_training._min_curvature_radius` — three-point circumradius over a
  sampled arc, biased HIGH 4.9-9.6% (F038).
- `surrogate_training._branch_speed_profile` — `np.gradient` over sampled
  caustic points for `|y'|`; feeds `_find_cusps`, whose thresholds and safety
  factors exist only to cope with sampled dips.
- `_pearcey_cusp._cusp_vertex` — central difference at `delta = 1e-4` across a
  129-point scan, ON THE SERVING PATH.
- `surrogate_training._probe_arc_side` — an absolute `0.05` step to decide
  which side of a fold carries the image pair; the answer moves with the step
  (F039).

All four are the SAME two derivatives. So export `y'` and `y''` themselves, not
only the scalars built from them — otherwise 1b and 1c each re-derive the
cascade and the duplication is back.

## Scope

IN — all in `cogwheel/lensing/chang_refsdal/geometry.py`, beside `r_caustic`
(which currently ends before `class GhostDomainError`); vectorise over `theta`:
- `caustic_derivatives(gamma, theta, *, kappa=0.0, branch=1)` returning
  `y'` and `y''`. This is the primitive; the rest are thin.
- `caustic_curvature_radius(...)` = `|y'|^3 / |y1' y2'' - y2' y1''|`.
- `caustic_speed(...)` = `|y'|`.
- `fold_opening_direction(...)` — the unit vector from the caustic toward the
  two-image side, `D2y[e,e]` (see the derivation below).
- Tests for each, in `cogwheel/tests/`.

OUT — do not touch, in this build:
- Every CONSUMER named above. `_min_curvature_radius`, `_branch_speed_profile`,
  `_find_cusps`, `_probe_arc_side`/`_PROBE_ETA`, `_caustic_inradius`,
  `_CLOUD_MARGIN_FRAC` are build 1b. `_cusp_vertex` is build 1c (separate
  because it serves).
- `_DEFAULT_ETA_MAX`, `_DEFAULT_ETA_FLOOR`, the foot-of-normal skip branch at
  `surrogate_training.py:3330` — step 3.
- `ANNULUS_INNER_RADIUS`, `GAMMA_FENCE`, the saddle fence, `ppgo_map` — step 5.
- Any training, any engine sweep, any chart artifact.

## The fold-opening direction, derived

At a critical point the source-map Jacobian is singular with soft eigenvector
`e` (`J e = 0`), so displacing along `e` kills the linear term:

    y(t) = y_c + (1/2) * D2y[e,e] * t**2 + O(t**3)

BOTH signs of `t` map to the same side — that is what makes it a fold — so the
merging pair lives in the direction of `D2y[e,e]`. For `y(x) = A x - x/|x|^2`
only the point-mass term survives, and contracting its second derivative twice
with a unit `e` gives, in closed form:

    D2y[e,e] = (4*(x.e)*e + 2*x - 8*(x.e)^2 * x/r^2) / r^4,   r^2 = |x|^2

`x` and `e` are already returned by `critical_point` as `.image` and
`.soft_axis`. No step, no tolerance, no image count.

## Measured facts (obtained by the driver; you cannot get these in-build)

**The caustic is an exact closed-form parametric curve.** From
`geometry.critical_point` (geometry.py:945), with `lam = 1 - kappa`,
`eg = gamma / lam`, `phase = theta - beta`:

    u(theta) = eg*cos(2*phase) + branch*sqrt(1 - eg^2 * sin(2*phase)^2)
    r        = 1 / sqrt(lam * u)
    x        = r * (cos theta, sin theta)
    y        = macro_matrix @ x - x / r^2

So `R_c = |y'|^3 / |y1' y2'' - y2' y1''|` with derivatives in `theta`. `beta`
is a rigid rotation and curvature is rotation-invariant, so
`R_c(theta; beta) = R_c(theta - beta; 0)` — do not carry `beta` through the
algebra.

**The derivative cascade, derived and VERIFIED by the driver.** Reproduce or
improve on it; the acceptance is the oracle, not this text. With
`s = sin 2th`, `c = cos 2th`, `c4 = cos 4th`, `e = gamma/lam`, `b = branch`,
`D = sqrt(1 - e^2 s^2)`:

    u   = e*c + b*D
    u'  = -2*e*s - b*2*e^2*s*c/D
    u'' = -4*e*c - b*4*e^2*(c4*D^2 + e^2*s^2*c^2)/D^3
    r   = 1/sqrt(lam*u);  r' = -r*u'/(2u);  r'' = r*(3u'^2/(4u^2) - u''/(2u))

then for each component with `p = (lam -+ gamma) - u`, `p' = -u'`,
`p'' = -u''`, and `T` the component's `cos/sin` factor:

    y_i'  = p'*r*T + p*r'*T + p*r*T'
    y_i'' = p''*r*T + 2p'*r'*T + 2p'*r*T' + p*r''*T + 2p*r'*T' + p*r*T''

About 25 lines of plain numpy, vectorised over `theta`, no new dependency.

**Verification envelope already measured (F038).** This cascade agrees with an
independent mpmath 40-dps numerical-differentiation oracle to **4.4e-13 worst
case** over 42 cases: `gamma` in {0.05, 0.3, 0.9, 0.99, 1.02, 1.3}, both
branches, `kappa` in {0, 0.3}, `theta` in {0.02, 0.17, 0.5, 1.0, 1.3, 2.2,
3.9} — i.e. including near-axial angles, the saddle `-1` branch, and
`gamma = 0.99` where `R_c = 1145`. Your gate is 1e-12; if you cannot reach it,
the algebra is wrong, not the tolerance.

**Oracles.** `SDK_CONDA_ENV` (from the repo-root `.env`) has mpmath 1.3.0 and
sympy 1.14.0. mpmath `diff` at 40 dps produced the envelope above and is the
cheap check. sympy works for a symbolic cross-check, but `lambdify` of the
UNSIMPLIFIED second derivative takes minutes — `cse` or simplify first. Do NOT
hard-code an interpreter path; use
`$(conda info --base)/envs/$SDK_CONDA_ENV/bin/python`.

**Second check, scale and sign only.** As `gamma -> 0` with `kappa = 0` the
caustic tends to the astroid `y = (-2g cos^3 th, 2g sin^3 th)`, so
`R_c -> 3*gamma*|sin 2 theta|`. Measured 4.4e-6..1.2e-4 at `gamma = 1e-4`,
degrading as `O(gamma^2)`. Pins scale and sign; NOT a 1e-12 gate.

**You will NOT reproduce `_min_curvature_radius`'s current numbers, and must
not try.** It is biased HIGH by 4.9-9.6% on production bands (F038), because a
three-point stencil's first usable centre is one sample step inside the arc
endpoint while the true minimum sits AT an endpoint. Matching it would enshrine
a discretization artifact. Reconciling that consumer is build 1b's job, not
yours — this build does not touch it.

## Acceptance

1. `caustic_derivatives` agrees with an independent high-precision oracle to
   **1e-12 relative** in BOTH `y'` and `y''`, across the case set above — both
   parities, both branches, `kappa != 0`, near-axial `theta`, and near the
   parity wall.
2. `caustic_curvature_radius` likewise to **1e-12**. The driver measured
   4.4e-13 on this exact cascade, so the bar has margin; if you cannot reach
   it, the algebra is wrong, not the tolerance.
3. `fold_opening_direction` points to the side carrying the extra image pair,
   checked against a direct image count wherever that count is
   well-conditioned. F039 measured 31/32; the single miss was
   `find_images_quartic` failing to separate a merged pair at `eps = 6e-7`, so
   choose check points where the pair is resolvable and say so in the test.
4. The astroid limit holds: `R_c / (3*gamma*|sin 2 theta|) -> 1` as
   `gamma -> 0`, to the measured `O(gamma^2)` accuracy above.
5. No finite difference, no `np.gradient`, and no sampled-arc estimator appears
   anywhere in the new code. The derivatives are analytic or the build fails.
6. `python -m pytest` on the geometry suites you touched is green. Full-suite
   gate is a post-build driver step; do not run it in-build.

## Constraints

- Assert VALUES against the oracle and a tolerance, not code paths. See the
  "Assert VALUES, not code paths" section of AGENTS.md — the measured cost of
  ignoring it is recorded there.
- Slow tests never run inside a build. `COGWHEEL_BRUTE_ACCURACY` and
  `COGWHEEL_STRICT_TIMING` stay empty.
- `critical_point` raises `LensDomainError` outside the saddle critical wedge
  and at `|gamma| == 1 - kappa`. The new functions inherit that contract —
  refuse by name, never return `nan` or `inf` silently. A straight caustic
  point (genuinely infinite curvature radius) is the one legitimate `inf`.
- Prose you change must be true when you are done. This build ADDS to
  `geometry` and changes no consumer, so `SPEC.md` row 55 and
  `COVERAGE_DESIGN.md`'s description of `_min_curvature_radius` stay accurate —
  verify that rather than assuming it. `SPEC.md`'s module row for
  `geometry.py` gains the new public names.
- Behaviour change in `cogwheel/`: follow the Spec/TODO workflow. This is step
  1 of an existing fragment, so do NOT delete that fragment; strike the step-1
  block and its acceptance from it and add the completion record.
