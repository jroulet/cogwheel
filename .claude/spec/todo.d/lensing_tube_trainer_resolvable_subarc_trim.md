---
section: Backlog
---

- **THE TUBE TRAINER MUST TRIM TO THE RESOLVABLE SUB-ARC — full-arc
  astroid charts fail the bar at production density** `[→ spec]` —
  measured 2026-08-17 ((f_max, f_floor) sweep on beat-free charts,
  `.claude/handoff/f_fraction_sweep_results.json`, 108 points): astroid
  full-arc tube charts read eps 0.13–0.85 across EVERY (f_max, f_floor)
  at production density (n_theta=7), while the F083 falsification proved
  eps 4.3e-3 at n_theta=10 on the TRIMMED sub-arc of the same gamma=0.4
  band — the full cusp-to-cusp arc has near-cusp zones where
  `_merging_fold_pair` refuses, build nodes zero-fill, and spline knots
  spread over dead regions. The saddle legs are healthy (eps 0.003–0.14;
  post-F081 lobe-edge arcs resolve throughout), so this is
  positive-parity-specific. The trim ALGORITHM already exists, derived
  and tested, in `cogwheel/tests/test_lensing_tube_beat_free.py`
  (`_f083_shared_tube`: binding-corner (gamma_hi, eta_max) Delta_tau
  profile, low knee at `_F083_DTAU_FRAC` of peak, stand inward off the
  steep-rise/turnover ends) — PROMOTE it into `surrogate_training` (the
  test then imports the production helper, DRY) and have
  `_build_tube_chart`/`_tube_training_arcs` train on the trimmed span.
  The excluded near-cusp zones are the cusp/Pearcey arm's serving
  domain (the owner's earlier question answered: yes, something else
  serves there); the serve-side F_ref probe already declines queries in
  the unresolvable zone, so no coverage hole — but verify the
  boundary interval (last resolvable knot to the fence) does not
  interpolate into zero-filled rows (the Professor's silent-r=0 watch
  item). BLOCKS the training campaign's astroid tube legs and the
  demand-sized tiling design's tube theta-spans; the astroid
  (f_max, f_floor) constants are re-measured on trimmed arcs (driver
  sweep, in progress). Sequence: with/before the tiling design.
