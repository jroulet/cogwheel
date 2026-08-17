---
section: Backlog
---

- **TUBE ANGULAR AXIS GRADUATION — spline in the delay-uniformized
  coordinate; the node count becomes Nyquist, not a knob** `[→ spec]` —
  F083 + owner direction 2026-08-17. The tube theta axis is the LAST
  ungraduated chart axis: wedge and lobe axes spline in the cusp-adapted
  `u = d^(2/3)` (V3), but the tube still splines in arc length `s` —
  the collocation fragment's sanctioned "cheap stand-in", valid only
  where no catastrophe dominates. F083's measured pathology (envelope
  swinging 5.3x with 8 extrema over the arc at w ~ 52; nodal error
  0.40 at n_theta = 7) is oscillation in `w * Delta_tau`: the merging
  pair's delay separation varies along the arc, so the demodulated
  envelope's angular frequency is `w * |dDelta_tau/ds|` — invisible to
  arc length, uniformized by `ds' ∝ |dDelta_tau/ds| ds` (the
  fold-family uniformizing coordinate). THE TRANSFORMATION IS FULLY
  ANALYTIC (owner direction 2026-08-17): for a fold,
  `Delta_tau(theta, eta) = c(theta) * eta^(3/2)` in closed form, with
  `c(theta)` the local fold-strength coefficient from the step-1
  cascade — the SAME Delta_tau `_airy_fold` already computes for the
  Airy argument `xi = (3 w Delta_tau / 4)^(2/3)`. The uniformizing
  angular coordinate is `s'(theta) ∝ Delta_tau(theta, eta_ref)`
  (equivalently the integrated fold coefficient), IMPORTED from the
  same authoritative source as `xi` (the collocation fragment's DRY
  rule — one Delta_tau; a second derivation is the violation this
  program exists to prevent). In `s'` the fold-pair beat has constant
  angular frequency by construction; the node count is the Nyquist
  requirement `w_max * Delta_tau-span / 2pi` oscillations times a
  points-per-period factor from cubic-spline approximation theory
  (~6-8 per period at 1e-2) — NO measured constant anywhere in the
  coordinate. Build: (1) GRADUATE THE COORDINATE — spline the tube's
  angular axis in `s'` (chart carries the `theta -> s'` table exactly
  as `theta_to_s` does today; schema bump). CONSISTENCY PIN (free
  validation of the derivation): approaching a cusp the fold
  coefficient's scaling makes `s'` asymptote to the `d^(2/3)` law —
  the SAME coordinate the wedge/lobe axes already use; pin
  `s' -> d^(2/3)` at the arc ends as a machine check of the closed
  form. FALSIFICATION (not calibration): the F083 ladder — the
  uniformized axis must beat or match the brute-force 48-s-node
  baseline (eps 0.0237) at fewer nodes; adaptive refinement against
  the held-out bar engages ONLY if the closed form under-predicts; (2) raise engine_budget to match (the 24-node build
  already trips 400); (3) fix `_heldout_eps`'s silent-skip blind spot
  (unserved held-out points must be REPORTED as coverage, never
  silently dropped) and record the ~40% arc-end shell that cannot serve
  (nearest-point crosses the cusp — decide: shrink the constructed
  shell to the servable region, or route those queries to the adjacent
  arc's chart via the fold machinery); (4) THEN re-run the joint
  (f_max, f_floor) sweep (runner ready at /tmp/f_fraction_sweep.py,
  priced ~2.6-2.8 h at production density, w capped 60) on resolved
  charts — `_DEFAULT_F_MAX = 0.40` has no valid measurement behind it
  (F083) and `f_floor = 0.16` is already measured unsupported. Blocks
  tube training in the demand-sized campaign; independent of the demand
  census and the deltoid redesign.
