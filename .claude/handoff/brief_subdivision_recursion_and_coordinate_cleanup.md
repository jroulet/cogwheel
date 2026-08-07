# Build — bounded subdivision recursion, exact `r_caustic`, and two contract fixes

## Mission

Four small fixes in the coordinate/tiling layer, all measured, all independent
of each other. They are batched because they touch the same files and the layer
should be opened once, not four times.

1. There are TWO subdividers duplicating one algorithm, and BOTH are
   SINGLE-LEVEL: a tile needing two halvings gets one and is abandoned. Unify
   them into ONE generic subdivider and give it bounded recursion — once.
2. `r_caustic` inverts a smooth monotone map by scanning 720 points. Root-find
   instead.
3. The wedge's `u` map is stored in fields named for ARC LENGTH.
4. `_wedge_cusp_axis_map` returns a silently COMPLEX array out of domain.

## Measured facts (do NOT re-derive; each cost real engine time)

### 1. Single-level subdivision

Both `_subdivide_wedge_tile` and `_subdivide_farfield_tile` are documented
"Single-level, no recursion": a child that still fails the eps bar becomes a
ladder-served gap instead of being split again.

Astroid interior, band 0, `gamma_mid = 0.495`, measured against the SHIPPING
`_subdivide_wedge_tile`:

    parent r    packed    residual gaps
    0.277        4/4      --
    0.455        4/4      --
    0.633        3/4      6.50e-2
    0.811        2/4      6.70e-2, 5.95e-2

13/16 children clear. The three that do NOT are MARGINAL — 1.19x, 1.30x and
1.34x the 5e-2 bar — while each halving has been buying 2-5x. One more level
clears them.

Totals so far: 18 charts at median 5.47e-4 in ~10.5 min, against the retired
`ffin` path's 106 charts at 3.42e-4 in 100.7 min.

The EXTERIOR very likely has the same cap: it shows 84% subdivision children
AND 35 of 57 charts still failing the 1e-3 bar — numbers that only sit together
if every marginal tile gets exactly one halving and is then abandoned.

DO NOT replace this with a cleverer initial tiling. Adaptive subdivision adds
resolution where it is needed BY CONSTRUCTION; an asymmetric seed would only
reduce round count while over-tiling where the first pass already passes.

### 1b. The two subdividers duplicate one algorithm — UNIFY, do not patch twice

`_subdivide_farfield_tile` (247 lines) and `_subdivide_wedge_tile` (213 lines)
share their entire SKELETON: iterate candidate children, `_load_or_build` each,
`_gate_chart` it, pack or record, accumulate a `children_summary`, return a
`{parent_tag, region, ..., children}` dict. Measured 21 identical statements
and the same control flow throughout.

What genuinely differs is only:
  (a) HOW child boxes are computed — the far-field halves caustic-fixed
      `(rho, theta_c)`; the wedge halves `(r, u)` with the angular split at the
      u-midpoint mapped back to theta;
  (b) WHICH build function is called (`_build_farfield_chart` vs
      `_build_wedge_chart`, with their own kwargs).

Those are PARAMETERS, not separate algorithms. Adding recursion to both
separately means writing the same loop twice and letting them drift.

The duplication has ALREADY caused a defect: there is no `_subdivide_lobe_tile`
at all, so a gated macro-saddle lobe tile becomes a ladder-served gap with no
recourse. A generic subdivider would have given the lobe path one for free.

So: extract ONE subdivider parameterised by a child-box splitter and a build
callable, put the bounded recursion in it ONCE, and have both existing call
sites use it. Shape it so the LOBE path can adopt it without further
refactoring — but WIRING the lobe is explicitly OUT of scope for this build
(it needs its own admission and testing); just do not preclude it.

### 2. `r_caustic` scans instead of root-finding

`r_caustic(gamma, theta, *, kappa=0.0, n_sample=720)` returns the distance to
the caustic along a SOURCE-plane direction. The caustic is available exactly
from `critical_point(gamma, theta_lens)`, but that parametrises by the
LENS-plane angle and `phi(theta_lens)` has no closed-form inverse — hence the
scan. Scanning 720 points to invert a smooth monotone map is the wrong method.

    r_caustic(0.9, pi/2)                     = 5.67376
    |critical_point(0.9, pi/2).source|       = 5.69210   -> 0.32% ERROR
    (agrees to 5 decimals at gamma = 0.3; the error grows with gamma)
    200 evaluations                          = 1.85 s

This error propagates straight into the wedge radial coordinate
`r = |y| / r_caustic(gamma, theta)`.

A CLOSED FORM WAS TRIED AND IS WRONG — do not retry it. The Chang-Refsdal
caustic is NOT the algebraic astroid `(x/A)^(2/3) + (y/B)^(2/3) = 1`; fitted at
the axes it errs 0.5% at gamma=0.2, 3.5% at 0.495, and 21% at 0.9.

EXACT ORACLE that does hold: `r_caustic(gamma, theta_waist) == gamma` at every
gamma tested (0.200, 0.300, 0.495, 0.700, 0.900 — dead on), where
`theta_waist = argmin_theta r_caustic`. The rejected astroid form does NOT
reproduce it.

### 3. Arc-length field names

After the cusp-axis change the wedge's angular coordinate is `u = d^(2/3)`, but
it is stored in `theta_to_s` / `s_grid` and validated by the SHARED
`_validate_theta_to_s`. Correct for that build (serve stays coordinate-agnostic;
`axis_schema` disambiguates at load) but the name now records the symbol's
first use rather than its role — the same failure already recorded for
`FarFieldChart`.

NOT a one-line rename: `_validate_theta_to_s` is shared with the tube,
lobe-interior and far-field maps, which genuinely DO hold arc length.

### 4. Out-of-domain silence

`_wedge_cusp_axis_map(theta_lo, theta_hi, 'high')` computes
`(pi/2 - theta)**(2/3)`. For `theta > pi/2` that takes a negative base and
returns a SILENTLY COMPLEX array — no raise, no clamp. The failure surfaces
frames later inside `np.interp` as an unrelated-looking cast error.
`theta_wedge > pi/2` is meaningless in a folded quadrant.

## Scope

IN — ONE generic subdivider replacing the two, with bounded recursion and
per-tile achieved depth reported; `r_caustic` root-find; wedge field/validator
renaming; the `_wedge_cusp_axis_map` domain guard; tests.

OUT — the `u = d^(2/3)` coordinate itself (settled, working); the exterior
polar re-chart and the D2 fold (separate, sequenced after this); the centre
tile's `CarrierDiscontinuityError` at `r -> 0` (a distinct polar-singularity
problem); `channels.py` / `operator.py` (SACR-C theory is sound); any training
run.

## Work

- **Unify + recurse**: extract one generic subdivider taking a child-box
  splitter and a build callable; both existing call sites use it. Bounded
  depth lives there ONCE (subdivide until the child clears or a cap is
  reached). Record the ACHIEVED depth per tile in the chart report so a
  runaway is visible and the census can attribute cleared vs still-gated
  windows. Pick the cap so the measured interior case (needs 2) is comfortably
  inside it; do not make it unbounded. Behaviour for existing far-field tiles
  must be UNCHANGED at depth 1 — pin that with a test, so the unification is
  provably a refactor plus a new capability, not a silent change.
- **`r_caustic`**: replace the scan with a `brentq` inversion of the exact
  parametrisation. Keep the signature; `n_sample` may stay as an ignored
  deprecated kwarg if callers pass it.
- **Naming**: either give the wedge its own validator plus `theta_to_u` /
  `u_grid`, or neutralise the shared names to `theta_to_axis` / `axis_grid`
  with meaning carried entirely by `axis_schema`. The second is DRYer and
  matches how the loader already works, but is a serialized-field change for
  every chart class and needs a version bump on each — prefer the first unless
  you are doing the second anyway.
- **Guard**: refuse out-of-domain `theta` in `_wedge_cusp_axis_map` with a
  named error, at the boundary.

## Acceptance

1. With recursion, the three marginal interior gaps (6.50e-2, 6.70e-2,
   5.95e-2) CLOSE. Report the achieved depth per tile and the final chart
   count against the 18-chart / 106-chart figures above.
2. `r_caustic` agrees with `|critical_point(...).source|` to machine precision
   at the axes, satisfies `r_caustic(gamma, theta_waist) == gamma` to ~1e-12,
   and a 200-call benchmark is at least 10x faster than 1.85 s.
3. No field or validator in the WEDGE path is named for arc length; the
   arc-length users (tube, lobe, far-field) keep theirs; serve stays
   coordinate-agnostic; a stale artifact still hard-refuses on `axis_schema`.
4. `_wedge_cusp_axis_map` raises a named error for `theta` outside
   `[0, pi/2]`, and a test asserts it rather than discovering it downstream.
5. Full suite green, driver-verified post-build. The 11 known-red
   serving-ladder tests are PRE-EXISTING and deselected by the tree gate.

## Constraints

- Branch `claude-dev`.
- **Every domain-test description MUST name its target suite file** (F057),
  and test suites must be assigned to DISJOINT shards — a plan was rejected at
  the gate this session for two shards claiming one file.
- `test_lensing_ppgo_bandsplit.py` also exercises `_wedge_interior_tiles`; it
  was missed by both shards last build and turned the tree gate red. Check the
  full consumer set of anything you change.
- Keep the WP count at or below 3.
- Slow tiers stay empty in-build; fast synthetic oracles only; no training run.
- Assert VALUES against an oracle and a tolerance, never which branch produced
  them. No `git show HEAD` oracle.
- Report the eps DISTRIBUTION (p50/p90/max) and the WORST-SAMPLE LOCUS in any
  new diagnostic, never a bare max.
