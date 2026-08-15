# Build: cusp-coincident tile edges crash the lobe-exterior axis map

## Mission

The 7a smoke run (2026-08-15 ~02:20, first end-to-end reach of the
lobe-exterior training path after F081 unblocked its tiler) crashed in
`surrogate._lobe_cusp_axis_map` (:682 at HEAD c661d62) via
`LensAmplificationSurrogate.from_lobe_exterior_engine` (:4048):

    ValueError: side='right' requires cusp_angle (3.270275691376951e-16)
    > theta_hi (3.552713678800501e-16).

Both numbers are ZERO at machine precision: a lobe-exterior tile's upper
theta edge lands exactly ON the theta = 0 cusp ray and the axis map's
guard is a strict float inequality between two machine-zeros, 3e-17
apart. F079's boundary-defect family: exact-boundary configurations at
the wrap/cusp rays, unreachable until the F081 fix made the lobe tiler
emit tiles. Fix the boundary semantics so a cusp-coincident edge is
handled deliberately; the trainer must complete the smoke config
end-to-end.

## Facts

1. The failing pair differs by 2.8e-17 — pure float noise on "the tile
   edge IS the cusp ray". The guard's question ("is the cusp strictly
   beyond the edge, so the 2/3-power map has room?") is the right
   question asked with the wrong arithmetic at coincidence.
2. Decide the SEMANTICS first (Professor): when a tile edge coincides
   with a cusp ray within tolerance, either (a) the tile abuts the cusp
   and the axis map should treat the edge AS the cusp (degenerate-but-
   valid: the u = d^(2/3) map anchored at the edge), or (b) the tiler
   should never emit a cusp-coincident edge (snap the edge off the cusp
   by the wrap-aware angular distance, house mod-2pi idiom) — pick from
   what the neighboring-tile geometry needs (no gap, no overlap), state
   why, and apply ONE of them; both is over-engineering.
3. Tolerance, if one is introduced, must be a DIMENSIONLESS ratio of
   local scales or a documented float-noise bound (~1e-12 abs on an
   O(1) angle is defensible as pure representation noise), never a
   tuned physical fudge — the F041/Part-0 discipline; the part0
   absorber guard will flag `_EPS`-suffixed module constants, so either
   allowlist with the float-noise justification or use a local literal
   with the reason comment.
4. Reproduction is cheap and engine-free for the GUARD (call
   `_lobe_cusp_axis_map` with the logged values); the full smoke
   (~1 h, engine) is the DRIVER's post-build acceptance, not in-build.
   In-build: a synthetic tile with a cusp-coincident edge exercises
   from_lobe_exterior_engine's axis-map call without the engine if the
   fixture permits; else pin the axis-map guard directly.

## Scope

IN: the boundary semantics fix (one of fact 2's options) in
`_lobe_cusp_axis_map` and/or the lobe-exterior tile emission; a value
pin for the coincident-edge case (the map's output at coincidence, or
the tiler's snapped edge — assert VALUES); both parities' lobe paths if
both can emit cusp-coincident edges (check the deltoid's 3-cusps-per-
lobe rays).
OUT: the deltoid far-field redesign (separate fragment); any training
run; the F081 machinery (landed); other axis maps unless the same
strict-inequality-at-coincidence shape exists there — GREP for the
pattern (`requires cusp_angle`/similar guards in surrogate.py) and
REPORT siblings, fix only if identical.

## Acceptance

- The exact logged configuration no longer raises; the coincident-edge
  behavior is pinned as a VALUE test with the chosen semantics stated.
- Sibling guards audited (report: same shape or not).
- Full fast suite green. The driver re-runs the smoke config post-build
  as acceptance (NOT in-build).

## Constraints

Branch claude-dev; fragments ([housekeeping] unless the semantics choice
moves a documented boundary — then [→ spec]); values-not-paths; no
engine calls in-build; escalate rather than iterate on any surprise.
