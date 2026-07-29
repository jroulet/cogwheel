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

The three-point circumradius is currently inlined inside that band-min. It
belongs in `geometry` because the caustic does.

## Scope

IN:
- New public `geometry.caustic_curvature_radius(gamma, theta, *, kappa=0.0,
  branch=1)` in `cogwheel/lensing/chang_refsdal/geometry.py`. Place it beside
  `r_caustic` (currently ends before `class GhostDomainError`).
- Re-express `surrogate_training._min_curvature_radius` as a thin band-min
  wrapper over the new function. Keep its name, signature and call site.
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

So `R_c = |y'|^3 / |y1' y2'' - y2' y1''|` with derivatives in `theta`. Prefer
an analytic derivative over any finite difference; `critical_point` itself is
the reference for the parametrization, but do NOT call it in a loop and
difference the result — that reproduces the very estimator being replaced.

**Oracles.** The project env (`SDK_CONDA_ENV`, read from the repo-root `.env`)
has mpmath 1.3.0 and sympy 1.14.0. Either is a valid independent oracle:
sympy for a symbolic `dy/dtheta` of the closed form above, mpmath `diff` at
40 dps for a numerical one. Prefer sympy for the derivative and mpmath for the
high-precision evaluation. Do NOT hard-code an interpreter path — resolve it
as `$(conda info --base)/envs/$SDK_CONDA_ENV/bin/python`.

**Second, fully analytic check (scale and sign only).** As `gamma -> 0` with
`kappa = 0` the caustic tends to the astroid `y = (-2g cos^3 th, 2g sin^3 th)`,
whose curvature radius is `R_c -> 3*gamma*|sin 2 theta|`. Measured agreement
with the mpmath oracle: 4.4e-5 at `gamma = 1e-3`, 4.9e-4 at `1e-2`, 9.9e-3 at
`0.1` — it degrades as `O(gamma^2)`, so it pins scale and sign but is NOT a
1e-8 gate. Do not write it as one.

**The incumbent is biased HIGH; byte-identity is NOT the gate (F038).** The
three-point stencil's first usable centre is one sample step inside the arc
endpoint, and the true curvature minimum sits AT an endpoint (curvature is
worst toward the trimmed cusp windows). Measured on production bands, positive
parity, `n_caustic_samples = 200`:

| band | incumbent circumradius | exact | excess |
|---|---|---|---|
| (0.25, 0.35) | 0.16136 | 0.14717 | 9.6% |
| (0.45, 0.55) | 0.30895 | 0.28747 | 7.5% |
| (0.65, 0.75) | 0.46892 | 0.44167 | 6.2% |
| (0.85, 0.95) | 0.78344 | 0.74692 | 4.9% |

The convergence is FIRST order in sample spacing (30.2% / 14.9% / 7.4% / 3.7%
at 100 / 200 / 400 / 800 samples over a quarter-arc), which is the signature of
the endpoint exclusion rather than of the circumradius formula.

**The consumer decision does not flip.** `r_min` is read at exactly one place,
`surrogate_training.py:3331`, as `config.eta_max > 0.5 * r_min`. With
`eta_max = 0.05` that decision is unchanged on every production band measured
above, and on the small-gamma bands `(0.0281, 0.0462)`, `(0.0644, 0.0825)`,
`(0.0825, 0.1550)`, `(0.1550, 0.3000)`.

## Acceptance

1. `caustic_curvature_radius` agrees with an independent mpmath high-dps
   curvature oracle to **1e-8 relative**, on both parities, at several
   `(gamma, theta, branch)` including near-cusp angles where `R_c` is small.
2. The astroid limit holds: `R_c / (3*gamma*|sin 2 theta|) -> 1` as
   `gamma -> 0`, to the measured `O(gamma^2)` accuracy above.
3. The rewritten `_min_curvature_radius` is BELOW the incumbent values in the
   table above, by the stated margin. Pin the new values. **Do not assert
   byte-identity with the incumbent** — that would preserve a discretization
   artifact, which the plan's standing rules forbid.
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
