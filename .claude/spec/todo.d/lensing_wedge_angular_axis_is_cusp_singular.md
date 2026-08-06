---
section: Backlog
---

- **THE WEDGE ANGULAR AXIS IS CUSP-SINGULAR — and the arc-length remap makes it
  WORSE. Fix is `u ~ theta^(2/3)`** `[→ spec]` — Professor review + measurement,
  2026-08-06. CONSOLIDATES three earlier fragments that reached wrong diagnoses
  (coordinate-relocates-difficulty, cusp-diffraction, exclusion-strips); their
  measurements are folded in below and the refuted hypotheses are listed so
  they are not re-attempted. git is the archive.

  ## Mechanism

  The astroid's CUSPS sit exactly at `theta_wedge = 0` and `pi/2` — the wedge's
  angular EDGES. There

      r_caustic(0.3, theta) = 0.52623 - 0.663 * theta^(2/3)     (soft axis)
                            = 0.71714 - 1.360 * d^(2/3)         (hard axis)

  and because the wedge radius is NORMALISED, `r = |y| / r_caustic(gamma,
  theta)`, that 2/3 power contaminates EVERY radius along the axes — not just
  the neighbourhood of the cusp POINT at `r = 1`. Chart velocity diverges as
  `theta^(-1/3)`; every smooth physical field becomes 2/3-Hoelder IN CHART
  COORDINATES. At fixed CHART `r`, `tau_c` drops 0.087 over `theta` in
  [0, 0.05] with local slope 15 -> 6 -> 3; the same quantity at fixed PHYSICAL
  radius is linear with slope 0.108. The implied carrier error is
  `delta_tau ~ 0.05-0.09`, matching the value back-computed from the w-linear
  eps growth. The 2.05x hard/soft coefficient ratio is the measured asymmetry.

  ## The arc-length remap AMPLIFIES it

  The chart's angular spline axis is `s`, built by integrating `caustic_speed`
  over theta. `caustic_speed` vanishes LINEARLY at a cusp (measured 1.58e-4 at
  `theta = 1e-4`), so `s - s(axis) ~ theta^2` and the envelope behaves as
  `f(s^(1/3))` — a WORSE exponent than raw theta.

  Measured, 1-D transverse cut at chart `r = 0.455`, band
  `theta in [1e-4, 0.2]`, same samples, only the abscissa differs:

  | angular coordinate | 5 nodes | 9 nodes | 17 nodes |
  |---|---|---|---|
  | `s` (arc length — WHAT THE CHART USES) | 6.11e-2 | 4.88e-2 | 3.86e-2 |
  | `theta` raw | 1.17e-2 | 7.11e-3 | 4.15e-3 |
  | **`u = theta^(2/3)`** | 6.88e-4 | **2.85e-4** | 4.44e-4 |

  `u` is **171x** better than the shipping `s` axis and reaches the `ffin`
  baseline (3.42e-4). It flattens past 9 nodes because it has hit the
  engine/spline noise floor. The shipping `s` axis is 6.9x WORSE than doing
  nothing.

  ## What this means

  **The exclusion strips recommended earlier are unnecessary.** They worked by
  keeping tiles away from a coordinate singularity that should simply be
  removed. Fixing the axis reclaims the 12.7% of the quadrant they cost AND
  buys two orders of accuracy.

  ## Work

  - Replace the wedge chart's angular spline axis: `s` (arc length) ->
    `u ~ (angular distance to the NEARER axis)^(2/3)`, piecewise about
    `theta = pi/4`. The exponent is gamma-universal; only the coefficient
    varies, so the map is cheap and needs no new table beyond what
    `_WedgeCausticMap` already holds. Give it a new `axis_schema` tag so stale
    `s`-axis artifacts hard-refuse at load.
  - Independently, FIX THE TILER (below) — the two are orthogonal and both are
    needed.
  - Re-measure the full-quadrant chart. With the axis cured, one angular
    column may suffice; if not, subdivide in `u`, not in `theta`.

  ## SEPARATE DEFECT — the tiler cannot subdivide at all

  `_wedge_interior_tiles(r_extent, n_per_side)` (surrogate_training.py:2313)
  hardcodes `theta_center = half_theta = pi/4` and `j = 0`: ONE angular column
  spanning the full `[0, pi/2]` at every radius, with NO angular subdivision
  and NO adaptive subdivision on eps failure. Its docstring justifies this by
  the carrier being smooth across the `pi/4` diagonal — a non sequitur:
  smoothness at the diagonal says nothing about angular RESOLUTION, and
  nothing about the edges at 0 and `pi/2` where the failure actually is.

  The deeper problem is the missing FEEDBACK. `ffin`'s 106 charts were an
  eps-driven loop (tile, measure, split where it fails); the wedge path
  replaced it with "record a ladder-served gap", so the tiler cannot discover
  it needs more tiles and cannot fail toward correctness. That is why this
  stayed invisible until the eps distribution was inspected a day later.

  ## Chart-level measurements (the evidence this rests on)

  Retired `ffin` path vs the wedge path, same region, production config:

  | interior path | charts | median eps | PASS (bar 5e-2) |
  |---|---|---|---|
  | `ffin` (retired) | 106 | 3.42e-4 | 106/106 |
  | wedge, as shipped | 12 | 5.38e-1 | **0/12** |

  Localisation — same band, same radial tile, same `n_theta` = 7, varying ONLY
  the angular tile POSITION:

  | angular tiles | tile TOUCHING an axis | INTERIOR tile |
  |---|---|---|
  | 4 | 1.29e-1 | **3.82e-4** |
  | 8 | 6.48e-2 | **3.27e-4** |
  | 12 | 3.96e-2 | — |

  Interior tiles are ~340x better at identical width and node count, and are
  already at the noise floor. Axis-adjacent tiles converge at exactly FIRST
  ORDER in width (3.93e-1, 2.34e-1, 1.29e-1, 6.48e-2, 3.96e-2) — the classic
  signature of an edge singularity, now identified as the `theta^(2/3)` cusp
  above.

  Exclusion strips were measured as a workaround before the cause was known.
  The HARD axis binds, ~6x worse than the soft at every width (0.10 rad:
  7.64e-3 hard vs 1.21e-3 soft; the 2.05x singular-coefficient ratio is the
  same asymmetry). **Strips are NO LONGER RECOMMENDED** — they worked by
  keeping tiles away from a singularity that the `u` axis removes outright,
  and they cost 12.7% of the quadrant.

  ## Hypotheses TESTED AND REFUTED — do not re-attempt

  Recorded because each cost real time and each is superficially plausible.

  1. **"The coordinate relocates the difficulty; revert to `ffin`."** Wrong.
     Radial converges at 1.31e-4 on FIVE nodes; transverse away from the axes
     at p ~ 3.6. Only the ANGULAR map near the edges is bad.
  2. **Cusp DIFFRACTION (Pearcey territory).** Wrong locus: the failure is at
     `r ~ 0.45`, nowhere near the cusp POINT at `r = 1`. The cusp enters
     through `r_caustic(theta)` in the NORMALISATION, not through the local
     field.
  3. **A poisoned boundary NODE.** The grid does place nodes exactly on the
     degeneracy (`min delay gap = 0.000e+00`, `y1 = 3.9e-17`) and the engine
     refuses 1e-9 away. But offsetting the node changes nothing:
     3.9271e-1 (node on axis) vs 3.9270e-1 (1e-6 offset). Not the mechanism.
  4. **The delay-degenerate mirror pair's interference BEAT.** Real and
     measured (`d_tau ~ 0.021 per degree` off the axis, ~2.8 cycles across the
     span at `w_max`) but NOT binding: at 30 nodes/cycle a cubic spline should
     reach ~1e-6, while axis-adjacent tiles sit at 4e-2 and converge at first
     order.
  5. **SACR-C switch / carrier / ordering defects.** All three refuted against
     the code — see "Theory is UNAFFECTED" below.

  ## Theory is UNAFFECTED

  SACR-C is an exact algebraic GAUGE, not an asymptotic statement:
  `E := exp(-i w tau_c) (F - sum_j exp(i w tau_j) S_j H_j)` telescopes exactly
  for ANY weights, so E absorbs whatever the trials miss by construction.
  Nothing requires distinct `tau_a`; the images remain isolated with
  non-degenerate Hessians and stationary phase is valid at each. Three
  hypotheses were checked against the code and all are WRONG: the switch is
  per-channel `smootherstep(w * |tau_a - tau_c|, 0.5, 4.0)` and never keys on
  pairwise gaps (the driver's fear describes the RETIRED F008 full-cluster
  rule); image ordering is a lexsort by (polar angle, radius) with delay
  entering nowhere; and although the nearest caustic foot IS doubly degenerate
  on an axis, the mirror feet share a delay EXACTLY and nothing E-visible
  consumes the critical ANGLE.

  Recalibration worth keeping: on these tiles E is NOT a small near-critical
  residual — at `w_max` = 8.93 three of four images are unswitched
  (`|tau_a - tau_c|` ~ 0.04-0.16 means the switch does not start until
  `w ~ 12`), so E carries most of the image content. "Resolved subtracted,
  near-critical folded in" describes the `w -> infinity` limit, not this band.

  ACCEPTANCE: with the `u` axis, axis-adjacent tiles reach median eps at or
  below `ffin`'s 3.42e-4 with NO exclusion strip; the tiler subdivides
  angularly and on eps failure; and the full-quadrant chart count is
  materially below `ffin`'s 106.
