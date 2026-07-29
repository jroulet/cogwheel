# Build brief — pointwise caustic curvature radius

## Mission

Give `geometry` a pointwise caustic curvature radius, and re-express the
band-minimum in `surrogate_training` as a thin wrapper over it.

This is step 1 of `.claude/spec/todo.d/lensing_caustic_relative_coordinates.md`
(rendered into `.claude/spec/TODO.md`). Read that fragment's Part 0 governing
principle and its standing rules before starting. Nothing downstream of step 1
is in scope here.

## Why this function has to exist

Every later step in that plan replaces an absolute length with a
caustic-relative one, and the local scale of the caustic IS its curvature
radius. Today the only curvature the package computes is
`surrogate_training._min_curvature_radius` — a band MINIMUM, used once, to
REFUSE (skip a tube chart). A boundary that scales with curvature needs the
value AT a point, not the worst value over a band.

That band-min estimates curvature with a three-point circumradius over a
sampled arc. It does not need to: the caustic is a closed-form parametric
curve, so its curvature radius is a closed-form function. **Differentiate it,
do not sample it.** The estimator is DELETED, not relocated.

## Scope

IN:
- New public `geometry.caustic_curvature_radius(gamma, theta, *, kappa=0.0,
  branch=1)` in `cogwheel/lensing/chang_refsdal/geometry.py`, computed
  analytically. Place it beside `r_caustic` (currently ends before
  `class GhostDomainError`). Vectorise over `theta`.
- Re-express `surrogate_training._min_curvature_radius` as a minimum over
  exact values from that function, and DELETE the inlined circumradius,
  including its `area2 < 1e-30` collinearity guard (an artifact of the
  three-point stencil, with no analogue in the closed form). Keep the name,
  signature and call site.
- Tests for both, in `cogwheel/tests/`.

OUT — do not touch, in this build:
- `_DEFAULT_ETA_MAX`, `_DEFAULT_ETA_FLOOR`, or the foot-of-normal skip branch
  at `surrogate_training.py:3330`. Making the shell curvature-relative and
  deleting that branch is step 3.
- `ANNULUS_INNER_RADIUS`, `GAMMA_FENCE`, the saddle fence, `ppgo_map`.
- The small-gamma topology slivers (step 3b).
- Any training, any engine sweep, any chart artifact.

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
sympy 1.14.0. mpmath `diff` at 40 dps is the cheap independent check and is
what produced the envelope above. sympy is available for a symbolic
cross-check, but note: `lambdify` of the UNSIMPLIFIED second derivative of
this expression takes minutes — if you use sympy, simplify or `cse` first.
Do NOT hard-code an interpreter path; resolve it as
`$(conda info --base)/envs/$SDK_CONDA_ENV/bin/python`.

**Second check, scale and sign only.** As `gamma -> 0` with `kappa = 0` the
caustic tends to the astroid `y = (-2g cos^3 th, 2g sin^3 th)`, whose curvature
radius is `R_c -> 3*gamma*|sin 2 theta|`. Measured: 4.4e-6..1.2e-4 at
`gamma = 1e-4`, degrading as `O(gamma^2)` to 4.9e-4..1.2e-2 at `1e-2`. Pins
scale and sign; NOT a 1e-12 gate. Do not write it as one.

**The incumbent's numbers are NOT the gate (F038).** The circumradius
estimator is biased HIGH by 4.9-9.6% on production bands, because a three-point
stencil's first usable centre is one sample step inside the arc endpoint and
the true minimum sits AT an endpoint. Do not assert byte-identity with it and
do not assert that margin either — both enshrine a discretization artifact.
Once the band-min uses exact values, the endpoints are evaluable and the bias
is simply gone.

**The consumer decision does not flip.** `r_min` is read at exactly one place,
`surrogate_training.py:3331`, as `config.eta_max > 0.5 * r_min`. With
`eta_max = 0.05` that decision is unchanged on every production band measured
above, and on the small-gamma bands `(0.0281, 0.0462)`, `(0.0644, 0.0825)`,
`(0.0825, 0.1550)`, `(0.1550, 0.3000)`.

## Acceptance

1. `caustic_curvature_radius` agrees with an independent high-precision
   curvature oracle to **1e-12 relative** across the case set above — both
   parities, both branches, `kappa != 0`, near-axial `theta`, and near the
   parity wall. The driver measured 4.4e-13, so 1e-12 has margin.
2. The astroid limit holds: `R_c / (3*gamma*|sin 2 theta|) -> 1` as
   `gamma -> 0`, to the measured `O(gamma^2)` accuracy above.
3. No finite difference and no sampled-arc estimator survives anywhere in the
   curvature path. `_min_curvature_radius` returns a minimum over EXACT values
   and its `area2 < 1e-30` guard is gone.
4. The `eta_max > 0.5 * r_min` decision flips on NO band listed above. Assert
   the decision, once, in the file that owns it. If a flip appears anywhere,
   STOP and report it — it is a finding, never a number to tune.
5. `python -m pytest cogwheel/tests/test_lensing_surrogate_training.py -q` and
   any geometry suite you touched are green. Full-suite gate is a post-build
   driver step; do not run it in-build.

## Constraints

- Assert VALUES against the oracle and a tolerance, not code paths. See the
  "Assert VALUES, not code paths" section of AGENTS.md — the measured cost of
  ignoring it is recorded there.
- Slow tests never run inside a build. `COGWHEEL_BRUTE_ACCURACY` and
  `COGWHEEL_STRICT_TIMING` stay empty.
- `critical_point` raises `LensDomainError` outside the saddle critical wedge
  and at `|gamma| == 1 - kappa`. The new function inherits that contract —
  refuse by name, never return `nan` or `inf` silently. Collinear samples
  (genuinely infinite radius) are the one legitimate `inf`.
- Prose you change must be true when you are done. `_min_curvature_radius` is
  described in `SPEC.md` row 55, `COVERAGE_DESIGN.md` line 166, and the
  `lensing_coverage_map` fragment. Step 1 keeps the name and the consumer, so
  those stay accurate — verify that rather than assuming it.
- Behaviour change in `cogwheel/`: follow the Spec/TODO workflow. This is step
  1 of an existing fragment, so do NOT delete that fragment; strike the step-1
  block and its acceptance from it and add the completion record.
