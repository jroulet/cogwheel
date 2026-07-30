# Build brief — delete `_WEDGE_EPS`, make `_tube_normal` analytic (step 1d)

## Mission

Close the analytic sweep. Two changes, both in
`cogwheel/lensing/surrogate_training.py`:

1. **Delete `_WEDGE_EPS = 1e-3`** and sample the macro-saddle critical wedge
   CLOSED (endpoints exactly at the wedge edge). It is an absolute angular
   standoff installed against a singularity that does not exist; it buys no
   safety and it costs coverage.
2. **Retire `_tube_normal`'s finite difference** — the last numerical
   derivative on the inventory. Replace `dth = 1e-6` forward differencing of
   `critical_point` with the exact tangent `y' / |y'|` from
   `geometry.caustic_derivatives`.

This is build 1d of step 1 of
`.claude/spec/todo.d/lensing_caustic_relative_coordinates.md` (inventory:
`todo.d/lensing_analytic_derivatives.md`, target 5). 1a (the `y'`/`y''`
cascade), 1b (training-path consumers) and 1c (serving-path cusp vertex plus
`y'''`) have shipped. When 1d lands, **nothing in this package's caustic
geometry is estimated rather than derived** — that is the claim step 1 exists
to earn, so the acceptance below is about earning it, not about the constant.

## Measured facts (driver, 2026-07-30 — recorded as FINDINGS F044)

Read F044 before starting; these are the load-bearing numbers.

* At the wedge edge `theta_max = (1/2) arcsin(lam / |gamma|)` the derivatives
  diverge in `theta` at exactly `|y'| = A dtheta^(-1/2)` and
  `|y''| = (A/2) dtheta^(-3/2)`, `A` constant to 5 s.f. over four decades
  (`A = 0.85124 / 0.40826 / 0.14434` at `gamma = 1.3 / 2.0 / 5.0`).
* But in `s = sqrt(theta_max - theta)` the curve is REGULAR: `y(s)` tends to a
  finite limit and `|dy/ds| = |y'| * 2s` to a NONZERO one
  (1.70248 / 0.81652 / 0.28868, stable `s = 1e-4 .. 1e-6`); `A = |dy/ds| / 2`.
  The two branches meet in a smooth turnaround. The singularity is in the
  PARAMETER, not the object.
* `critical_point` serves `dtheta = 0` exactly and raises `LensDomainError` at
  `dtheta = -1e-12` (no silent clamp). `caustic_derivatives` refuses at
  `dtheta <= 0`. The named refusals already guard the edge.
* Cost of the standoff: `_lobe_winding_loop`'s closure gap is 0.279 at
  `gamma = 1.05` (9.3% of lobe reach), 0.107 at 1.3, 0.051 at 2.0 — and that
  loop is the saddle interior-admission test (`_winding_number(loop - probe)`,
  `abs(w) < 0.5` rejects). Against a standoff-free reference interior it
  rejects 1/792 interior probes at `gamma = 1.05`, reaching 0.059 source-plane
  units INSIDE the lobe.
* With the standoff at 0, measured at `gamma = 1.05 / 1.3 / 2.0 / 5.0`,
  `n = 401`: closure gap becomes exactly 0.0; cusp count (6), arc count and
  reach are UNCHANGED; total arc span is slightly larger
  (e.g. 2.908035 -> 2.910714 at `gamma = 1.05`). Nothing degrades.

## Part A — `_WEDGE_EPS`

Six sites, all `np.linspace` bounds over a wedge sweep
(`_saddle_arcs`, `_lobe_caustic_points`, `_lobe_winding_loop`,
`_lobe_cusp_source_angles`, and the two arc-bound computations). Delete the
constant; sample `[center - theta_max, center + theta_max]` inclusive.

Two things to get right:

* **The endpoints must actually reach the edge.** `theta_max` is computed in
  float, so an endpoint can land a few ulp OUTSIDE the wedge and be refused.
  That is correct behaviour, not a bug to paper over — every sampler here
  already skips `LensDomainError` per angle. Do NOT reintroduce a margin to
  avoid the skip; verify instead that the endpoint is served or refused as the
  arithmetic dictates and that the loop still closes to 0.0.
* **The wedge-edge WALL exclusion in `_saddle_arcs` stays.** The walls list
  `[(lo_edge, edge_hw)] + cusps + [(hi_edge, edge_hw)]` guards the near-
  singular foot-of-normal map at the turnaround and is a separate concern from
  the standoff. Anchor it at the true edge now, not at the standoff.

Do NOT replace `_WEDGE_EPS` with a derived constant. The earlier plan asked
for `dtheta >= |y''|_max^(-2/3)`; F044 retires that framing. And do NOT
introduce the `s = sqrt(theta_max - theta)` coordinate here — that is step 1e
(`todo.d/lensing_collocation_from_local_scales.md`), which owns interpolation
coordinates across all three chart types.

## Part B — `_tube_normal`

```
caust2 = critical_point(gamma, theta + 1e-6, ...).source
tangent = caust2 - caust                       # forward difference of a
                                               # closed form
```
becomes `tangent = y' / |y'|` from `caustic_derivatives(gamma, theta,
branch=branch)`, normal unchanged as the left perpendicular
`(-t_y, t_x)`.

**The sign is the tripwire.** `_tube_normal`'s normal is what `_tube_source`
displaces along and what `_make_arc` dots against `fold_opening_direction` to
fix each arc's `inward_sign`. Getting the tangent's ORIENTATION wrong flips
served sides silently — that is the F041 failure, in this exact function. The
forward difference points along increasing `theta`, and so does `y'`; that
agreement is the thing to verify, per production arc, not to assume.

`caustic_derivatives` refuses where `critical_point` succeeds (the wedge edge
itself). `_tube_normal`'s callers already handle `LensDomainError`; keep the
refusal named and do not swallow it.

## Part C — prose that is currently false on these surfaces

Each is one line, each is on a surface this build touches, and each says the
opposite of what the code does:

* `geometry.caustic_derivatives` docstring calls the wedge edge "the deltoid
  cusp, where the derivatives genuinely diverge". The derivatives do diverge;
  it is NOT a cusp (F044). The deltoid's cusps are the interior `|y'| = 0`
  roots. `_saddle_arcs` already says it correctly ("walls, but not cusps").
* `_winding_number` docstring: "the disjoint saddle lobes are NOT such a loop,
  so this is never applied to them" — it is applied to exactly them, at the
  `_SaddleLobeAdmission` probe test.
* `_lobe_winding_loop` docstring: "the loop stays approximately closed" — after
  this build it is exactly closed.
* `cogwheel/tests/test_lensing_caustic_cusps.py`: `SERVE_ALIGN_MIN = 0.1`
  documented as "equals `_make_arc`'s own build floor", and the helper at
  ~line 337 replays "the same `|dot| > 0.1` floor". F041 removed that floor;
  production is now `if dot == 0.0: continue`. The test helper no longer
  mirrors the production loop it claims to mirror, so it can pick a different
  serve theta. Fix the helper to match production and the prose with it.

## Scope

IN: `cogwheel/lensing/surrogate_training.py`,
`cogwheel/lensing/chang_refsdal/geometry.py` (docstring only),
`cogwheel/tests/`.

OUT — do not touch:
* the `s = sqrt(theta_max - theta)` coordinate, any grid placement, any
  interpolation coordinate (all step 1e);
* `_SADDLE_CUSP_WIDTH_SAFETY` / `_SADDLE_CUSP_MIN_HALFWIDTH` / the cusp-window
  rule (F040, a later build);
* `_DEFAULT_ETA_MAX`, `ANNULUS_INNER_RADIUS`, `GAMMA_FENCE` (steps 3 and 5);
* any training or chart artifact; do not train anything.

## Test-suite ownership — DISJOINT, one author per file

Three suites, assigned by which already owns the predicate (one canonical pin
each; do not re-assert a claim in a second file). A plan whose test shards
overlap on any file is rejected at the gate.

* `cogwheel/tests/test_lensing_saddle_geometry.py` — the WEDGE-EDGE PREDICATE.
  It already owns it (`test_outside_wedge_and_boundary_are_refused`, and the
  lobe-closure test that calls `critical_point` at `center +- tmax` and asserts
  the two branches meet to 1e-6). Gate 5 lands here: `critical_point` serves
  `dtheta = 0` and refuses at `dtheta = -1e-12`; `caustic_derivatives` refuses
  at `dtheta <= 0`. Note its closure assertion is currently `gap < 1e-2` with
  the comment "scales as the sqrt-resolved step" — if that gap is now exactly
  representable, tighten it to the measured value rather than leaving slack
  that no longer means anything.
* `cogwheel/tests/test_lensing_surrogate_training.py` — the TRAINING-PATH
  CONSEQUENCES. Gates 1, 2, 3: `_WEDGE_EPS` absent, sweeps run edge to edge,
  `_lobe_winding_loop` closure gap exactly 0.0, and no shrink in cusp count /
  arc count / reach / total arc span.
* `cogwheel/tests/test_lensing_caustic_cusps.py` — SERVE CONSISTENCY, which it
  already owns (the `fold_dir . serve_normal` invariant built on
  `_tube_normal`'s normal). Gate 4 lands here: normal unit and perpendicular to
  `y'`, and `inward_sign` unchanged per production arc. The Part C helper fix
  (`SERVE_ALIGN_MIN`, the stale `|dot| > 0.1` replay) is also this file's.

## Acceptance

State the measured number for each; "unchanged" without a number is not a
result.

1. `_WEDGE_EPS` does not exist in `cogwheel/`, by name and with no inlined
   `1e-3` standing in for it. Wedge sweeps run edge to edge.
2. `_lobe_winding_loop`'s closure gap is **exactly 0.0** at every saddle band
   gamma tested (incumbent: 0.279 at `gamma = 1.05`).
3. No coverage shrinks: at `gamma = 1.05 / 1.3 / 2.0`, cusp count, arc count
   and reach are unchanged and total arc span is `>=` incumbent. A drop in any
   of these fails the build — the point of the change is more served geometry,
   not less.
4. `_tube_normal` contains no finite difference; the normal is unit and
   perpendicular to `y'` to 1e-15; and **`inward_sign` is identical to the
   incumbent for every fold arc on every production band, both parities**.
   This is the load-bearing gate.
5. The named refusals still hold as VALUES: `critical_point` serves
   `dtheta = 0` and raises `LensDomainError` at `dtheta = -1e-12`;
   `caustic_derivatives` raises at `dtheta <= 0`. One canonical pin each, in
   the file that owns the predicate.
6. Suites you touched run green (`test_lensing_surrogate_training`,
   `test_lensing_caustic_cusps`, geometry). Full suite is a post-build driver
   step; do not run it in-build.
7. Every prose item in Part C is true when done.

## Constraints

- Assert VALUES against a tolerance, not code paths. ONE canonical pin per
  routing decision, in the file that owns the predicate — do not re-assert it
  in each consumer suite.
- No `git show HEAD:` oracles. A cross-version comparison that must survive
  its own commit is a landmine (F043, now blocked by a pre-commit hook). If
  you need the incumbent's `inward_sign` for gate 4, compute it ONCE and
  freeze it as a golden table of literals in the test.
- Never preserve an incumbent number by construction; matching is a
  coincidence to report, not a target.
- Slow tests never run in-build; `COGWHEEL_BRUTE_ACCURACY` /
  `COGWHEEL_STRICT_TIMING` stay empty.
- `SDK_CONDA_ENV` from `.env`; interpreter
  `$(conda info --base)/envs/$SDK_CONDA_ENV/bin/python`, never `conda run -n`.
- Prose you change must be true when done; a live document describing the old
  behaviour fails acceptance exactly like a red test.
