# Build — cusp-adapted wedge angular axis, and a tiler that can subdivide

## Mission

Two orthogonal defects, both measured, both in the astroid-interior wedge path.
Fix them together because the tiler must subdivide in the NEW coordinate.

1. The wedge chart's ANGULAR SPLINE AXIS is cusp-singular, and the arc-length
   remap makes it worse. Replace it with a cusp-adapted `u`.
2. `_wedge_interior_tiles` emits ONE angular column over the full `[0, pi/2]`
   at every radius, with NO angular subdivision and NO adaptive subdivision on
   eps failure. Give it both.

## Measured facts (do NOT re-derive; each cost real engine time)

The astroid's CUSPS sit exactly at `theta_wedge = 0` and `pi/2` — the wedge's
angular EDGES. There:

    r_caustic(0.3, theta) = 0.52623 - 0.663 * theta^(2/3)      (soft axis)
                          = 0.71714 - 1.360 * d^(2/3)          (hard axis)

The exponent 2/3 is EXACT (constant ratio over 4+ decades) and gamma-universal;
only the coefficient varies. Because the wedge radius is NORMALISED,
`r = |y| / r_caustic(gamma, theta)`, that 2/3 power contaminates EVERY radius
along the axes — not merely the cusp POINT at `r = 1`. Chart velocity diverges
as `theta^(-1/3)`.

The current angular axis is `s` (arc length), built by integrating
`caustic_speed` over theta. `caustic_speed` vanishes LINEARLY at a cusp
(1.58e-4 at `theta = 1e-4`), so `s ~ theta^2` and the envelope behaves as
`f(s^(1/3))` — a WORSE exponent than raw theta.

1-D transverse cut of the real `partition.envelope`, chart `r = 0.455`,
`theta in [1e-4, 0.2]`, IDENTICAL samples, only the abscissa differing:

    angular axis                       5 nodes   9 nodes   17 nodes
    s (arc length, WHAT SHIPS)         6.11e-2   4.88e-2   3.86e-2
    theta (raw)                        1.17e-2   7.11e-3   4.15e-3
    u = theta^(2/3)                    6.88e-4   2.85e-4   4.44e-4

`u` is 171x better than the shipping axis and reaches the `ffin` baseline
(median 3.42e-4, the retired path this replaced). It flattens past 9 nodes
because it has hit the engine/spline noise floor.

Chart-level context: interior tiles ALREADY pass (3.82e-4); only tiles touching
an axis fail (1.29e-1). eps grows LINEARLY in w (delta_tau ~ 0.05-0.09), is
immune to gamma and w node counts, and node exactness is 6.33e-16.

## The SACR-C theory is NOT at fault — do not touch it

SACR-C is an exact algebraic GAUGE: `E := exp(-i w tau_c)(F - sum_j exp(i w
tau_j) S_j H_j)` telescopes exactly for ANY weights. Nothing requires distinct
`tau_a`. The switch is per-channel `smootherstep(w * |tau_a - tau_c|, 0.5, 4.0)`
and never keys on pairwise gaps. Image ordering is a lexsort by (polar angle,
radius); delay enters no ordering, pairing or dedup. `channels.py`,
`operator.py` and `geometry.py` are OUT OF SCOPE.

## WP1 — cusp-adapted angular axis

Replace the wedge chart's angular spline coordinate.

`u = d^(2/3)` where `d` is the angular distance to THE CUSP THAT TILE IS NEAR.
Split the wedge at `theta = pi/4`: a tile in `[0, pi/4]` uses `d = theta`; a
tile in `[pi/4, pi/2]` uses `d = pi/2 - theta`. Do NOT use "distance to the
nearer axis" as a single global map — `min(theta, pi/2 - theta)` has a KINK at
`pi/4`, which would trade an edge singularity for an interior one. Per-tile
monotone maps have no such kink, and the carrier IS smooth across `pi/4`
(the original ruling, independently reconfirmed).

- The map is gamma-independent (only the exponent matters for smoothness), so
  it needs no new table beyond `_WedgeCausticMap`.
- Train and serve MUST use the same map. Train/serve skew is the recurring bug
  class in this repo.
- New `axis_schema` tag so a stale `s`-axis artifact HARD-REFUSES at load.
  Follow the existing `_WEDGE_AXIS_SCHEMA` pattern and its validator.
- Retire the `theta_to_s` arc-length path for the WEDGE chart only. The
  far-field chart's own arc-length map is untouched.

## WP2 — a tiler that can subdivide

`_wedge_interior_tiles(r_extent, n_per_side)` (surrogate_training.py:2313)
hardcodes `theta_center = half_theta = pi/4` and `j = 0`.

- Emit ANGULAR columns, not one. Minimum two (the `pi/4` split WP1 requires).
- Subdivide ADAPTIVELY on eps failure, the way the exterior still does via
  `_subdivide_farfield_tile`. A wedge tile that fails its bar must SPLIT, not
  become a ladder-served gap.
- Subdivide in `u`, not in `theta`: equal steps in `u` are what the spline
  sees.
- Keep the radial rows.

WHY THIS MATTERS, and the failure mode to avoid: the previous build's plan
removed subdivision ("record a LADDER-SERVED GAP ... NO tile subdivision") and
its Simplifier trimmed cusp/admission logic, on the reasoning that the carrier
is smooth across `pi/4`. That reasoning was sound and IRRELEVANT — smoothness
at the diagonal says nothing about angular RESOLUTION, and nothing about the
EDGES at 0 and `pi/2` where the failure is. Removing the eps feedback loop is
why the defect stayed invisible for a day: a tiler with no feedback cannot
discover it needs more tiles and cannot fail toward correctness. DO NOT trim
the subdivision. If a Simplifier proposes it, reject and say why.

## Scope

IN — the wedge chart's angular axis and its schema tag; `_wedge_interior_tiles`;
adaptive subdivision for wedge tiles; serve-side consistency; tests.

OUT — `channels.py` / `operator.py` / `geometry.py` (theory is sound); the
far-field and tube charts and their arc-length maps; the macro-saddle lobe
path; the exterior `(s,d)` degeneracy (separate, recorded); any training run.

## Acceptance

1. An axis-adjacent wedge tile reaches eps at or below the `ffin` baseline
   3.42e-4 with NO exclusion strip. This is the whole point — report the
   number against 4.88e-2, the current value at the same geometry.
2. `_wedge_interior_tiles` emits more than one angular column, and a tile that
   fails its eps bar SUBDIVIDES rather than being recorded as a gap. Show a
   test where a deliberately-too-coarse tile splits and then passes.
3. Train and serve use the same angular map: node exactness stays at machine
   precision, and a served value off-node matches a fresh engine evaluation.
4. A stale `s`-axis artifact hard-refuses at load with a named error.
5. Full suite green, driver-verified post-build. The 11 known-red
   serving-ladder tests are PRE-EXISTING and deselected by the tree gate —
   not yours.

## Constraints

- Branch `claude-dev`.
- **Every domain-test description MUST name its target suite file** (F057).
- Assign test suites to DISJOINT shards; a previous plan was rejected at the
  gate for two shards claiming one file.
- Keep the WP count at or below 3.
- Slow tiers stay empty in-build; fast synthetic oracles only; no training run.
- Assert VALUES against an oracle and a tolerance, never which branch produced
  them. No `git show HEAD` oracle.
- Report the eps DISTRIBUTION (p50/p90/max) and the WORST-SAMPLE LOCUS in any
  new diagnostic, never the bare max. A max-metric summary hid this defect for
  a day.
