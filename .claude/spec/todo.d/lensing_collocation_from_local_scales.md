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
