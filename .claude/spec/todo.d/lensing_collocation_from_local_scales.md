---
section: Backlog
---

- **COLLOCATION FROM LOCAL SCALES — place chart nodes by the function's own
  analytically-computed scale, not uniformly in whatever coordinate the code
  happens to use** `[→ spec]` — the same principle that fixed the guards
  (F041: a dimensionless O(1) constant on a scale-free ratio of local
  quantities), applied to node placement. Owed before step 9 of
  [[lensing_caustic_relative_coordinates]], because training bakes the node
  layout into a shipped artifact.

  ## The current state

  Every tube-chart axis is a UNIFORM grid in a coordinate with no knowledge of
  the local scale, each governed by a bare dimensionless COUNT:

  | axis | today | count |
  |---|---|---|
  | theta | `linspace(arc.theta_lo, arc.theta_hi, n_theta)` | `n_theta` |
  | eta | `linspace(sqrt(eta_floor), sqrt(eta_max), n_u)` | `n_u` |
  | gamma | `_uniform_axis(band, n_gamma)` | `n_gamma` |
  | log w | `_log_w_grid(w_range, w_nodes_per_decade)` | `w_nodes_per_decade` |

  A count is the right KIND of constant — dimensionless. The defect is the
  measure it counts against: a uniform step in `theta` is not a uniform step
  in anything the envelope cares about.

  ## SCOPE: all THREE chart types, both parities — the saddle is not automatic

  The table above is the TUBE chart, but the coordinate work spans three build
  paths, and getting the geometry parity-general (1a-1d) does NOT make the
  chart coordinates parity-general on its own:

  - **Tube charts** (`_build_tube_chart`) — SHARED by the positive-parity
    astroid arcs and the macro-saddle deltoid arcs. Fixing `theta` here fixes
    both parities' tubes at once (F042 was a saddle tube at `gamma = 1.55`).
  - **Far-field exterior charts** (`_build_farfield_chart`, both parities) —
    the `rho` axis IS the caustic-relative coordinate the C8 redesign builds;
    `theta_c` and node placement still need the treatment. The two parities'
    exteriors are TOPOLOGICALLY DIFFERENT and already carry different radial
    coordinates, so C8 and the node work must NOT assume the saddle mirrors the
    astroid: the astroid encloses the origin (directional-MULTIPLICATIVE
    `rho = |y| / r_caustic(theta_c)`, every ray hits once), while the two
    saddle deltoids sit OFF-origin and enclose nothing, so a ray can miss both
    lobes and there is no directional radius — the existing `_to_caustic_fixed`
    uses a SCALAR ADDITIVE offset (`rho = 1 + |y| - reach`) on the saddle
    exterior arm by design (F036). The quadrant fold (`['u1','u2']`) exploits
    the reflection that swaps the two lobes, so only ONE deltoid is charted,
    not two — but the folded lobe is still off-origin, so the fold removes the
    DOUBLING, not the coordinate difference. Enumerate the saddle exterior
    distinctly.
  - **Lobe-interior charts** (`_build_lobe_chart`, MACRO SADDLE ONLY) — a
    SEPARATE build path in lobe-local `(rho_lobe, theta_local)` on an
    axis-aligned box. `rho_lobe = |y - centroid| / r_deltoid` is already
    caustic-relative (good), but its grid placement and interpolation
    coordinate are its own and are NOT touched by the tube-chart work. Left
    alone it keeps uniform grids and re-grows the F042 knife-edge near the
    deltoid cusps. It must be enumerated explicitly or it is silently missed —
    which is exactly the trap this note exists to prevent. Its cusp coordinate
    is the SAME Pearcey `(x, y)` control as everywhere else (DRY, per the reuse
    section below): the deltoid's three per-lobe cusps are cusps like any
    other.

  ## PLACEMENT IS NOT ENOUGH — interpolate IN the analytic coordinate

  Node placement and interpolant FORM are separate, and placement alone is a
  half-fix. If nodes are placed at arc-length points but the spline is still
  built in RAW `theta` (knots at those thetas, cubic in theta between them),
  the between-node error keeps the cubic-in-theta form, which does not match
  the envelope's actual structure (Airy near a fold, the 2/3-power Pearcey
  near a cusp). The error comes straight back between the nodes.

  The complete fix is a COORDINATE CHANGE: interpolate in the coordinate where
  the demodulated envelope is SMOOTH. Then a polynomial spline captures it AND
  uniform nodes in that coordinate are well-placed by construction — placement
  and form become one act. The package already has one axis done right:
  `u = sqrt(eta)` is not node placement, it is a coordinate change — the chart
  splines IN `u`, and the fold's sqrt-branch is smooth in `u`. Every axis below
  must be read as "spline in this coordinate", not merely "sample at these
  points". The principled coordinate is the catastrophe-uniformizing one (Airy
  argument for a fold, Pearcey arguments for a cusp); arc-length / curvature is
  the cheap first approximation of it.

  CAVEAT on the F042 evidence: the 2.2x measured there came from arc-length
  NODE PLACEMENT with the chart still interpolating in `theta`, so it is the
  half-fix, not the ceiling. Splining in arc-length (or the uniformizing
  coordinate) should beat it and, more importantly, remove the residual
  between-node error this section is about.

  ## REUSE the existing uniformizing maps — do NOT reinvent them (DRY)

  The catastrophe-uniformizing coordinates are ALREADY computed and single-
  sourced in the uniform-arm evaluators. The collocation build must import and
  reuse these, not derive an arc-length approximation of them:

  - FOLD (the near-caustic tube axis): the signed Airy control
    `xi = (3 w Delta_tau / 4)^{2/3}` in `_airy_fold.py` (the `xi` computed at
    ~line 438, `Delta_tau` from the merging pair's Fermat-delay separation).
    The envelope is smooth in `xi`; `u = sqrt(eta)` is its `w`-independent
    shadow, which is WHY the incumbent `u` axis works. The principled tube
    coordinate is `xi` itself.
  - CUSP: the Pearcey controls `x = delta_par * sqrt(w) / sqrt(|C4|)`,
    `y = delta_perp * w^{3/4} / |C4|^{1/4}` in `_pearcey_cusp.py`
    (~lines 686-687), with `delta_par`/`delta_perp` the source offsets on the
    soft/hard axes from `nearest_caustic_point` and `C4` the normal-form
    coefficient. These are the smoothing coordinates near a cusp.

  - MACRO-SADDLE WEDGE EDGE: `s = sqrt(theta_max − theta)` with
    `theta_max = (1/2) arcsin(lam / |gamma|)`. Measured 2026-07-30 (F044): the
    edge is a REGULAR point of the caustic that the `theta` parametrization
    makes look singular — `y` and `dy/ds` are both finite and nonzero in `s`,
    while `|y'| ~ dtheta^{-1/2}` and `|y''| ~ dtheta^{-3/2}` in `theta`. Any
    saddle axis that runs to a wedge turnaround must be gridded and splined in
    `s`, or its last nodes chase a divergence that is not there. This is the
    same reparametrising move as `u = sqrt(eta)` on the fold axis, and it
    applies to BOTH the saddle tube arcs and the lobe-interior charts, whose
    lobe-local `theta_local` sweeps the same turnarounds. Build 1d deletes
    `_WEDGE_EPS` on the strength of this; the coordinate itself is owed here.

  Arc-length `int caustic_speed dtheta` is the CHEAP STAND-IN, valid only where
  no catastrophe dominates (mid-arc, away from cusps). Where a fold or cusp
  governs the local structure, use its uniformizing map. A single authoritative
  representation per coordinate (Part 0 / DRY): the tube chart, the uniform
  arms, and the collocation grid must all read the SAME `xi` and `(x, y)`, or
  the surrogate and the arm it falls through to disagree about where the
  structure is.

  ## ORDERING — this moves UP, ahead of the driver measurements (F042)

  Originally scheduled as a step-9 (train-once) prerequisite. F042 shows that
  is too late: step 2 and step 4 of [[lensing_caustic_relative_coordinates]]
  are driver MEASUREMENTS of held-out eps and node cost, and F042 proves both
  are coordinate-dependent (eps 0.059 uniform-theta vs 0.027 arc-length at the
  same nodes). Running those measurements on uniform-theta grids derives the
  tube-fraction constants (`f_max`, `f_floor`) and the far-zone crossover from
  PLACEMENT ARTIFACTS, not physics. So the `theta` (and `eta`) interpolation
  coordinate must be settled AFTER step 1's cascade is complete (1a-1d) and
  BEFORE step 2's first measurement — not before step 9.

  ## DECOMPOSITION — three sequential builds, ordered by what each unblocks

  Surveyed 2026-07-30. `_build_tube_chart` grids `u = sqrt(eta)` (already the
  coordinate change, the axis done right) and `theta_grid =
  linspace(arc.theta_lo, arc.theta_hi, n_theta)` (raw). `TubeChart` stores the
  four axes plus cubic B-spline coefficients and knot vectors, so the
  interpolation VARIABLE for that axis is literally `theta`. Splitting by
  which measurement each sub-build unblocks, rather than doing all three at
  once (the brief-discipline rule: >~3 WPs means sequential builds):

  - **1e-tube — blocks step 2.** The tube chart's `theta` axis, shared by both
    parities. This is the only piece step 2's tube-fraction sweep needs.
  - **1e-farfield — blocks step 4.** The exterior charts' `rho`/`theta_c`
    axes, per-parity, remembering the saddle's scalar-additive `rho` is a
    different coordinate, not a mirror of the astroid's directional one.
  - **1e-lobe — blocks only step 9.** `_build_lobe_chart`, macro-saddle only.
    Its lobe-local `theta_local` sweeps the wedge turnarounds, so it is also
    where `s = sqrt(theta_max - theta)` (F044) applies.

  Only 1e-tube gates the next measurement, so the sequence is not blocked on
  all three landing.

  ## THE SERVE-SIDE COST — the design question 1e-tube must answer first

  A coordinate change is cheap at BUILD time (sample at `theta(s)`, spline in
  `s`) and not obviously cheap at SERVE time: a query arrives as `theta` and
  the spline now wants `s = int_{theta_lo}^{theta} |y'| dtheta'`. Computing
  that quadrature per evaluation would put an integral in the likelihood's hot
  path, which is not acceptable.

  The fix is for the CHART TO CARRY ITS OWN AXIS MAP: bake a fine monotone
  `theta -> s` table (or 1-D spline) into the chart at build time, alongside
  the existing axes. That keeps one authoritative representation of the
  coordinate (Part 0 / DRY), costs one extra 1-D evaluation per serve, and
  makes the map serializable and testable on its own. NOTE this changes the
  chart SCHEMA, so it needs a `contracts_changelog.d/` fragment and a
  `DATA_CONTRACTS.yaml` update — harmless today precisely because nothing is
  trained yet (the window in this fragment's parent), and expensive after
  step 9. Another reason 1e comes before training, not after.

  ## SIZING the stored theta -> s map — revisit before training, not after

  1e-tube ships `N_map = 2001` nodes per chart. Measured 2026-07-30 (clean
  `h^2` convergence, strict monotonicity everywhere, both parities):

  | N | astroid rel err | saddle rel err | storage/chart |
  |---|---|---|---|
  | 101 | 2.42e-5 | 8.55e-5 | 1.6 KiB |
  | 201 | 6.05e-6 | 2.14e-5 | 3.1 KiB |
  | 501 | 9.68e-7 | 3.42e-6 | 7.8 KiB |
  | 2001 | 6.05e-8 | 2.14e-7 | 31.3 KiB |

  The round-trip tolerance is `1e-6`, so **2001 is conservative, not
  necessary** — `501` clears it with ~3x margin at a quarter the storage. The
  table ships inside EVERY chart's npz, so the real cost is artifact size and
  it scales with the chart count, which is unknown until step 9. Size it then,
  rather than rediscovering the question after training.

  Do NOT reach for a higher-order rule. `scipy.integrate.cumulative_simpson`
  needs scipy >= 1.12 and this env is 1.11.4, and independently trapezoid is
  the better choice: for a positive integrand every increment is
  `(h/2)(f_i + f_{i+1}) > 0`, so monotonicity is guaranteed BY CONSTRUCTION —
  which the `np.interp` inversion and the map's strict-monotonicity assertion
  both depend on. Simpson fits parabolas and carries no such guarantee.
  Accuracy is close to beside the point anyway: the map is a COORDINATE, and
  the same table places the build nodes and maps the serve query, so a smooth
  error cancels exactly between them.

  ## What each axis should count against

  All four scales are now computable in closed form from the step-1 cascade;
  none needs a sweep.

  1. **`theta` -> ARC LENGTH.** `ds = |y'| dtheta`, and `caustic_speed = |y'|`
     is exact. `|y'|` vanishes at cusps and varies by orders of magnitude
     along an arc, so uniform-`theta` nodes are strongly non-uniform in the
     geometry they sample. Better still, count against CURVATURE: `ds / R_c`
     is dimensionless, so equal-`ds/R_c` spacing puts nodes where the caustic
     actually bends. Both ingredients ship already.
     MEASURED (F042, 2026-07-29): on a real saddle tube arc at `gamma = 1.55`,
     at the SAME `n_theta = 4`, an arc-length grid (`int caustic_speed dtheta`)
     gives held-out eps 0.027 vs uniform-theta's 0.059 — 2.2x better at
     identical node count — and is insensitive to the arc-bound shift that
     swings uniform-theta +-23%. This is the concrete evidence for the whole
     fragment: the code currently grids `theta` uniform in `theta`
     (`_build_tube_chart`, `theta_grid = linspace(theta_lo, theta_hi, n_theta)`)
     while `u = sqrt(eta)` already got the analytic treatment. The `theta`
     axis is the next one to convert, and F042 is its motivating measurement.
  2. **`eta` -> the fold's own variable.** `u = sqrt(eta)` is the one axis
     where the right instinct was already applied — it absorbs the fold's
     square-root branch — but empirically rather than derived. The principled
     version is uniform in the Airy argument `xi = (3 w Delta_tau / 4)^{2/3}`,
     which is what the envelope actually oscillates in; `u = sqrt(eta)` is its
     `w`-independent shadow. Deriving it explains WHY the incumbent works and
     says what to do where it does not.
  3. **`w` -> the envelope's variation scale.** Uniform in `log w` is a guess.
     The envelope varies on `w * Delta_tau`; nodes should follow that. Note
     the serving path ALREADY does the adaptive version correctly — the
     leave-one-out envelope refinement (`_LOO_SEED_NODES = 8`, stop `4e-3`,
     ceiling `_LOO_MAX_NODES = 48`, node count config-independent) places
     nodes by measured error. That is the empirical form of this principle;
     the analytic form should agree with it and explain its node counts.
  4. **`gamma` -> caustic-relative.** Uniform in `gamma` cannot be right when
     the caustic's extent varies 28x across the prior and DIVERGES at the
     parity wall (F036). The natural measure is the same caustic-relative
     coordinate the whole redesign moves to.

  ## Why this is not premature optimisation

  It is not about node COUNT or speed. Uniform nodes in the wrong measure put
  resolution where the function is flat and starve it where the function
  moves, so the held-out eps is set by the worst-resolved region — which is
  how a chart lands at eps 0.4..2.2 while its neighbours sit at 1e-2 (the
  saddle deltoid-arc failure recorded at `_SADDLE_CUSP_WIDTH_SAFETY`). The
  response then was to widen an exclusion window, i.e. to serve less. Placing
  nodes by the local scale is the alternative to refusing.

  ## Acceptance

  - Each axis's node measure is stated as a dimensionless ratio of
    analytically-computed local quantities, with one O(1) constant.
  - The constant is calibrated ONCE against held-out eps, not per-axis-tuned;
    a per-chart fudge is the bug class this replaces.
  - Held-out eps at fixed node COUNT improves, or the node count needed for a
    fixed eps drops. Report both; either alone can be gamed.
  - The analytic `w`-node rule reproduces the LOO-adaptive node counts the
    serving path already measures, to within its own stopping tolerance. If
    it does not, the analytic scale is wrong — the LOO result is the
    measurement.
  - The chart INTERPOLATES in the analytic coordinate, not merely samples at
    analytic points: held-out eps is INSENSITIVE to a small arc-bound shift
    (the F042 knife-edge is gone). Placement-only cannot pass this; it is the
    acceptance that separates a coordinate change from node placement.
  - The uniformizing coordinates are IMPORTED from `_airy_fold.py` (`xi`) and
    `_pearcey_cusp.py` (the Pearcey `(x, y)` controls), not re-derived: a test
    asserts the collocation coordinate equals the arm's own control to machine
    precision where they overlap. A second copy of `xi` or `(x, y)` is a DRY
    violation and a future drift bug, not an implementation detail.
