# Build 1e-farfield — the far-field exterior charts' interpolation coordinate

## Mission

The exterior far-field charts interpolate in RAW UNIFORM axes. Change the
interpolation COORDINATE of the spatial axes (`rho`, `theta_c`) to one in which
the demodulated envelope is smooth, per parity — the same move 1e-tube already
made for the tube charts' `theta` axis. This is not node placement. The spline's
independent variable must BE the new coordinate.

This build gates step 4 (the far-zone crossover measurement), because that
measurement sweeps held-out eps and node cost; on a raw uniform grid it would
pin `rho*` to a placement artifact (F042).

## Measured facts (current tree, do not re-derive)

`LensAmplificationSurrogate.from_engine` (`cogwheel/lensing/surrogate.py`,
~line 1900) builds every exterior chart axis:

    log_w_grid   = _log_w_grid(w_range, w_nodes_per_decade)
    gamma_grid   = _uniform_axis(gamma_range, n_gamma, 'gamma')
    rho_grid     = _uniform_axis(rho_range, n_rho, 'rho')
    theta_c_grid = _uniform_axis(theta_c_range, n_theta, 'theta_c')
    theta_c_grid = _union_cusp_nodes(theta_c_grid, theta_c_range)   # positive parity only

`_union_cusp_nodes` (`surrogate.py:643`) unions `_ASTROID_CUSP_ANGLES` into the
axis so "a cubic chart places a node ON each C2 curvature kink rather than
smoothing across it", deduping within `_CUSP_NODE_DEDUP_TOL`. That is NODE
PLACEMENT bolted onto a raw uniform axis. It is the incumbent, it is the
contrast control, and it cannot satisfy this build's acceptance.

1e-tube's shipped precedent (`surrogate_training.py`): the tube chart now grids
`s_grid = np.linspace(0.0, s_total, config.n_theta)` (:2532) and carries its own
axis map, `theta_fine = np.linspace(arc.theta_lo, arc.theta_hi, n_map)` (:2490),
`N_map = 2001`. Measured round-trip error, clean `h^2`, strictly monotone both
parities: 101 -> 2.42e-5 / 8.55e-5, 501 -> 9.68e-7 / 3.42e-6, 2001 -> 6.05e-8 /
2.14e-7 (astroid / saddle). Tolerance is 1e-6, so 2001 is conservative.

## The parities are TOPOLOGICALLY DIFFERENT — do not mirror them

- **Positive parity (astroid).** Encloses the origin. `rho = |y| /
  r_caustic(theta_c)` is directional-MULTIPLICATIVE; every ray hits once.
- **Macro saddle.** TWO deltoids sitting OFF-origin, enclosing nothing. A ray
  can miss both lobes, so there is no directional radius. `_to_caustic_fixed`
  uses a SCALAR ADDITIVE offset, `rho = 1 + |y| - reach`, by design (F036).
  The quadrant fold (`['u1','u2']`) exploits the reflection swapping the two
  lobes, so only ONE deltoid is charted — the fold removes the DOUBLING, not
  the coordinate difference.

Enumerate the saddle exterior DISTINCTLY. An implementation that assumes the
saddle mirrors the astroid is wrong even when its tests pass on the astroid.

## The design question to answer BEFORE touching the schema

1e-tube had to answer the serve-side cost question first; so does this build.
A query arrives in `(rho, theta_c)`. If the new coordinate is closed form both
ways (as `u = sqrt(eta)` is — one `sqrt`, no stored map), NO schema change is
needed. If it requires a quadrature or a root solve (as arc length does), the
chart MUST carry a baked monotone map like 1e-tube's, because a quadrature in
the likelihood's hot path is not acceptable.

State which case applies, with the arithmetic, in the plan. If a map is needed:
it changes the chart SCHEMA, so it needs a `contracts_changelog.d/` fragment
and a `DATA_CONTRACTS.yaml` update. That is harmless TODAY only because nothing
is trained yet, and expensive after step 9.

Use trapezoid for any cumulative map, not Simpson: for a positive integrand
every increment is `(h/2)(f_i + f_{i+1}) > 0`, so monotonicity holds BY
CONSTRUCTION, which the `np.interp` inversion depends on. (`cumulative_simpson`
also needs scipy >= 1.12; this env is 1.11.4.)

## DRY — import the uniformizing coordinates, never re-derive them

The cusp columns' uniformizing coordinate is the Pearcey `(x, y)` control that
`_pearcey_cusp.py` already computes; the fold's is `xi` in `_airy_fold.py`.
IMPORT them. A test must assert the collocation coordinate equals the arm's own
control to machine precision where they overlap. A second copy of `xi` or
`(x, y)` is the DRY violation this whole fragment exists to prevent.

Precedent for the shape of the answer: F044 found the macro-saddle wedge edge is
a REGULAR point — the `dtheta^-1/2` divergence was an artifact of the `theta`
parametrization alone, and the fix was the coordinate `s = sqrt(theta_max -
theta)`, NOT a standoff margin. A cusp is the same kind of object: fix the
coordinate and the kink is gone, at which case `_union_cusp_nodes` becomes
unnecessary rather than merely adequate.

## Acceptance

1. **Held-out eps is INSENSITIVE to a small chart-bound shift**, both parities.
   This is the criterion that separates a coordinate change from node
   placement, and placement alone cannot pass it.
2. **Contrast control, mandatory.** The SAME test, run against the incumbent
   uniform+`_union_cusp_nodes` axes, must FAIL. A passing test with no
   demonstrated failing counterpart measures nothing (F045). 1e-tube shipped
   this; match it.
3. Held-out eps at fixed node COUNT improves, or the node count for fixed eps
   drops. Report BOTH — either alone can be gamed.
4. The saddle exterior is exercised distinctly from the astroid, with its
   scalar-additive `rho` named in the test, not assumed to mirror.
5. If a stored map ships: strict monotonicity asserted, round-trip within 1e-6,
   and the map is serialized, reloaded, and re-checked (schema tests).
6. Full suite green, driver-verified post-build.

## Out of scope — do not touch

- Tube charts (`_build_tube_chart`) — 1e-tube shipped; do not re-open.
- Lobe-interior charts (`_build_lobe_chart`) — that is 1e-lobe, next build.
- The `w` and `gamma` axes — those are 1e-w and 1e-gamma. `log_w_grid` and
  `gamma_grid` stay exactly as they are in this build even though they are
  visibly in the same function.
- `ANNULUS_INNER_RADIUS`, `GAMMA_FENCE`, the annulus retirement — that is
  step 5 (C8).
- ANY training run or engine sweep. Nothing is trained until step 9.

## Constraints

- Branch `claude-dev`. Never `main`/`master`.
- Slow tiers NEVER run in-build: `COGWHEEL_BRUTE_ACCURACY`,
  `COGWHEEL_TRAIN_TIER`, `COGWHEEL_STRICT_TIMING` stay empty. In-build tests
  must be FAST — small/synthetic configs, analytic or few-eval oracles.
- Assert VALUES against an oracle and a tolerance, never which branch produced
  them. One canonical pin per routing decision, in the file that owns it.
- No test may reconstruct pre-change code from `git show HEAD` — it passes only
  until its own change commits, then skips itself forever (F043/F045, enforced
  by a pre-commit gate).
- Spec workflow: this is a behavior change in `cogwheel/`, so it needs a
  `completed.d/` fragment and the `todo.d/` bullets updated; run
  `python scripts/render_fragments.py` after writing fragments.
