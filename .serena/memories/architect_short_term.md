Last session: 2026-08-04 production batch. Clean.

2026-08-06 brief_wire_interior_wedge_chart:
- from_wedge_engine + _from_wedge_fixed + wedge NPZ round-trip ALREADY
  complete in surrogate.py (build 56a223a). Brief is STALE claiming it's
  missing. Only surrogate_training.py wiring is genuinely absent.
- _interior_admission MUST BE KEPT: live exterior-tiler dependency
  (surrogate_training.py:3949) + 5 test suites. Brief's "interior-only,
  move/delete" premise is WRONG.
- _farfield_interior_tiles genuinely dead after swap -> DELETE (but ported
  by 2 test suites: exterior_windows, ppgo_bandsplit).
- Professor rulings: 1 angular column [0,pi/2] (carrier smooth through
  pi/4, empirically confirmed by test_lensing_wedge_dd_arclength), uniform
  n_per_side radial rows, r_min>0, r_extent capped below 1 by tube shell
  (leave Airy edge to tube); in-build eps gate = ABSOLUTE floor <5e-2 +
  chart-count (ffin relative baseline is driver post-build since ffin is
  deleted).
- Simplifier trims: no 2D tiler, inline/minimal radial split; no verify-
  only WP.
- Single Coder WP (all edits in surrogate_training.py; multiple WPs on
  _train_band_charts would conflict). _heldout_eps annotation add =
  biggest risk (G3).
- Gated/flip wedge tiles -> ladder-served gap (mirror LOBE, NOT ffin
  subdivision).
