# Build: tube angular axis graduation — the analytic delay-uniformized coordinate

## Mission

The tube chart's angular axis is the LAST ungraduated chart axis: wedge
and lobe axes spline in the cusp-adapted `u = d^(2/3)` (V3 schemas), but
the tube still splines in arc length `s` — the sanctioned "cheap
stand-in", valid only where no catastrophe dominates. F083 measured the
consequence: at w ~ 52 the demodulated envelope swings 5.3x with 8
extrema over a 1.34-rad arc (the fold pair's beat), producing textbook
nodal error 0.40 at n_theta = 7 (error ~0 at nodes, 0.19-0.44 between);
48 brute-force s-nodes reach eps 0.0237. Graduate the axis to the
ANALYTIC fold-family uniformizer and make the node count derived, not
tuned. This blocks tube training and the (f_max, f_floor) sweep.

## The transformation (fully analytic — owner direction; derive nothing empirically)

For a fold, `Delta_tau(theta, eta) = c(theta) * eta^(3/2)` in closed
form, `c(theta)` the local fold-strength coefficient from the step-1
analytic cascade — the SAME `Delta_tau` that `_airy_fold` already
computes for the Airy argument `xi = (3 w Delta_tau / 4)^(2/3)`
(computed near its line ~438 from the merging pair's Fermat-delay
separation). The uniformizing angular coordinate is
`s'(theta) ∝ Delta_tau(theta, eta_ref)` (equivalently the integrated
fold coefficient), IMPORTED from the same authoritative source as `xi`
— ONE `Delta_tau` in the tree (the collocation fragment's DRY rule; a
second derivation is the violation this program exists to prevent). In
`s'` the beat has constant angular frequency by construction; uniform
nodes are optimal placement; the count is the Nyquist requirement
`w_max * Delta_tau-span / (2 pi)` oscillations times a points-per-period
factor from cubic-spline approximation theory (~6-8 per period at 1e-2)
— NO measured constant anywhere in the coordinate.

CONSISTENCY PIN (free machine-check of the derivation): approaching a
cusp, the fold coefficient's scaling makes `s'` asymptote to the
`d^(2/3)` law — the same coordinate the wedge/lobe axes use. Pin
`s' -> d^(2/3)` at the arc ends.

FALSIFICATION (not calibration): the F083 ladder — the uniformized axis
must beat or match the brute 48-s-node baseline (eps 0.0237 at
gamma=0.4 astroid, pilot density elsewhere) at FEWER nodes; adaptive
refinement against the held-out bar engages ONLY if the closed form
under-predicts, and that engagement is itself a reportable finding.

## Scope

IN:
1. `_build_tube_chart` splines the angular axis in `s'`; the chart
   carries the `theta -> s'` table exactly as `theta_to_s` does today
   (fine monotone table, `N_map = 501` per the sizing note — not 2001);
   serve side maps the (folded) query angle through the same table.
   Schema bump + hard-refusal of stale tube schemas (the V3/V5
   pattern); contracts fragment.
2. The Nyquist node-count rule replaces the bare `n_theta` for tube
   charts (config keeps a CAP, not a count); `engine_budget` raised to
   match (the 24-node build already trips 400).
3. `_heldout_eps` silent-skip fix: unserved held-out points are
   REPORTED as coverage (count + where), never silently dropped —
   F083's blind spot. Record the ~40% arc-end shell that cannot serve
   (nearest-point crosses the cusp): decide shrink-the-shell vs
   route-to-adjacent-arc-via-fold, from the geometry, and say why.
4. NO-EXPLOSION GATE in the tiling census: per-region proposed nodes
   vs the information-content estimate (Nyquist x spline factor,
   closed form); exceeding it by more than a small stated factor is a
   flagged coordinate/representation defect. Engine-free.
5. Both parities (the saddle tube axis graduates identically; its
   measured envelope structure is milder but the same law applies).

OUT: the (f_max, f_floor) sweep (driver re-runs it AFTER this lands —
runner ready at /tmp/f_fraction_sweep.py); the residual-representation
endgame (follow-up fragment); the deltoid far-field redesign; lobe/
wedge axes (already graduated); any training campaign; serving-ladder
changes.

## Acceptance

- The `s'` table is DRY-imported from the `_airy_fold` `Delta_tau`
  source (a test asserts the collocation coordinate equals the arm's
  own control to machine precision where they overlap — the 1e-eta
  acceptance pattern).
- `s' -> d^(2/3)` cusp-limit pin green at arc ends, both parities.
- F083 falsification: gamma=0.4 astroid tube at the derived node count
  reaches eps <= 0.0237 with FEWER than 48 angular nodes (report the
  count and eps; the ladder is the baseline).
- `_heldout_eps` reports unserved-point coverage; no silent skips.
- No-explosion census check live and green on the current region set.
- Full fast suite green; stale tube artifacts hard-refuse by schema.

## Constraints

Branch claude-dev; fragments (closes
`todo.d/lensing_tube_angular_axis_graduation.md`; `[→ spec]` + contracts
fragment for the schema bump); values-not-paths; in-build tests FAST
(pilot-density charts, minutes); the F083 ladder numbers are handed in —
no in-build re-measurement beyond the falsification build+probe;
escalate rather than iterate on any surprise.
