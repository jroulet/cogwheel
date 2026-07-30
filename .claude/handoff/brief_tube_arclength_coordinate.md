# Build brief — the tube chart's `theta` axis becomes arc length (step 1e-tube)

## Mission

Make `TubeChart` **interpolate in arc length**, not in raw `theta`. Today the
axis is `theta_grid = linspace(arc.theta_lo, arc.theta_hi, n_theta)` and the
cubic B-spline's coordinate is literally `theta`, so held-out accuracy depends
on where the arc bounds happen to fall rather than on the geometry. The sibling
axis already shows the shape of the fix: `u = sqrt(eta)` is not node placement,
it is a COORDINATE CHANGE, and the chart genuinely splines in `u`.

This is 1e-tube, the first of the three sub-builds in
`.claude/spec/todo.d/lensing_collocation_from_local_scales.md`. Read that
fragment first. It gates step 2 of
`todo.d/lensing_caustic_relative_coordinates.md` — a driver measurement of
held-out eps that would otherwise pin its constants to placement artifacts.

## Measured facts (F042, driver, 2026-07-29)

* On a real saddle tube arc at `gamma = 1.55`, at the SAME `n_theta = 4`:
  arc-length nodes give held-out eps **0.027** vs uniform-theta's **0.059** —
  2.2x better at identical node count, so this is placement, not node budget.
* Uniform-theta is KNIFE-EDGE: nudging the arc bounds by `+-0.01 rad` swings
  eps `0.0592 -> 0.0727 / 0.0575`, about `+-23%`. Arc length tracks the
  geometry and is insensitive to the bound.
* That 2.2x came from arc-length NODE PLACEMENT with the chart still
  interpolating in `theta` — the HALF-fix. Splining in `s` should beat it, and
  more importantly removes the between-node error that placement alone leaves.

## The coordinate

`s(theta) = integral of |y'| dtheta'` from `arc.theta_lo`, with
`|y'| = geometry.caustic_speed(gamma, theta, branch=...)` — exact, shipped in
build 1a, no estimator. `s` is strictly increasing on a fold arc because
`|y'| > 0` away from cusps, and the arc bounds already exclude the cusp
windows, so the map is invertible on the served interval.

## THE SERVE-SIDE PROBLEM — solve this first, it shapes everything else

A query arrives as `theta`. The spline now wants `s`. Recomputing the integral
per evaluation would put a quadrature in the likelihood's hot path, which is
not acceptable.

So **the chart carries its own axis map**: build a monotone `theta -> s` table
(or 1-D spline) once at build time and store it alongside the existing axes.
One authoritative representation of the coordinate (Part 0 / DRY), one extra
1-D evaluation per serve, and a map that is serializable and testable on its
own. Choose the node count for that map by MEASURING the round-trip error, and
state it; do not pick a round number.

Consequences to handle, not to discover later:
* This is a CHART SCHEMA change. `TubeChart` gains a field, `from_values` and
  `_assemble` gain a parameter, and the npz round trip must preserve it. Write
  a `contracts_changelog.d/` fragment and update `DATA_CONTRACTS.yaml`.
  Harmless today ONLY because nothing is trained yet — that window closes at
  step 9.
* `_tube_serves` range-tests `theta` against `chart.theta_grid[0]/[-1]` and
  `_theta_into_frame` unwraps the query into the arc's frame. Membership tests
  and cusp windows STAY in `theta` — they are not spline coordinates. Only the
  interpolation coordinate changes. Keep the frame unwrap: a deltoid arc can
  span negative angles.

## Scope

IN:
* `cogwheel/lensing/surrogate.py` — `TubeChart` (schema, `from_values`,
  `_assemble`, the evaluation path), and the `theta -> s` map.
* `cogwheel/lensing/surrogate_training.py` — `_build_tube_chart`'s
  `theta_grid`, which becomes a uniform grid in `s` sampled at the
  corresponding `theta(s)`.
* `.claude/spec/DATA_CONTRACTS.yaml` + a `contracts_changelog.d/` fragment.
* Tests, per the ownership map below.

OUT — do not touch:
* the far-field exterior charts (1e-farfield) and `_build_lobe_chart`
  (1e-lobe). Both are later sub-builds; the saddle wedge-edge coordinate
  `s = sqrt(theta_max - theta)` (F044) belongs to 1e-lobe, NOT here.
* the `u = sqrt(eta)` axis — already a coordinate change, leave it alone.
* `gamma` and `log w` axes; cusp-window RULE (F040); `_DEFAULT_ETA_MAX`;
  anything under step 3 or 5.
* Do NOT train anything.

## Test-suite ownership — DISJOINT, one author per file

A plan whose test shards overlap on a file is rejected at the gate (this cost
build 1d a replan). Assigned by which suite owns the predicate:

* `cogwheel/tests/test_lensing_surrogate.py` — the CHART: schema round trip
  (the new field survives npz save/load), the `theta -> s` map's accuracy and
  monotonicity, and served-value equivalence.
* `cogwheel/tests/test_lensing_surrogate_training.py` — the BUILD PATH: the
  `theta` nodes are the images of a uniform `s` grid, and the held-out eps
  results below.
* `cogwheel/tests/test_lensing_farfield_envelope.py` — only if a shared
  helper genuinely moves. Prefer not to touch it.

## Acceptance

State the measured number for each.

1. **The knife-edge is gone.** Held-out eps is INSENSITIVE to a `+-0.01 rad`
   arc-bound shift — the incumbent swings `+-23%`. This is the load-bearing
   gate, and it is the one property node placement alone CANNOT deliver, so it
   is how we know a coordinate change actually happened.
2. Held-out eps at fixed `n_theta` improves on the incumbent at the F042
   configuration (`gamma = 1.55` saddle arc, `n_theta = 4`, incumbent 0.059).
   Report both eps at fixed nodes AND nodes needed for fixed eps; either alone
   can be gamed.
3. The chart SPLINES in `s`: a test asserts the interpolation coordinate is
   the arc-length image, not that nodes merely sit at arc-length points.
4. `theta -> s` round trip: `s(theta(s)) == s` to a stated tolerance, and the
   map is strictly monotone across every production arc, both parities.
5. Served values remain correct: the tube arm's `|F|` and phase agree with the
   engine to the F016 envelope bar on a cusp-free sweep. A coordinate change
   must not move a served number beyond fit error.
6. npz round trip preserves the new field bit-identically.
7. Suites you touched run green. Full suite is a post-build driver step.

## Constraints

- Assert VALUES against tolerances, not code paths. ONE canonical pin per
  decision, in the file that owns the predicate.
- **No `git show HEAD:` oracles** — the whole suite was just purged of them
  (F043/F045) and a pre-commit hook now blocks both a fresh one and a call to
  an existing helper. Freeze any incumbent comparison as golden literals.
- Never preserve an incumbent number by construction; matching is a
  coincidence to report, not a target.
- Slow tests never run in-build; `COGWHEEL_BRUTE_ACCURACY` /
  `COGWHEEL_STRICT_TIMING` stay empty.
- `SDK_CONDA_ENV` from `.env`; interpreter
  `$(conda info --base)/envs/$SDK_CONDA_ENV/bin/python`, never `conda run -n`.
- Reuse `geometry.caustic_speed`; do NOT re-derive arc length or introduce any
  finite difference. Step 1 exists to make every such quantity analytic.
- Prose you change must be true when done — `TubeChart`'s docstring describes
  the axes and will be wrong the moment this lands.
