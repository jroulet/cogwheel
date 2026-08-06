Last session: 2026-08-04 production batch. Clean.

2026-08-06 WP1 wire InteriorWedgeChart into training (surrogate_training.py ONLY):
- Retired ffin path: positive-parity astroid interior now tiled by
  _wedge_interior_tiles (ONE angular col theta_wedge in [0,pi/2], center pi/4
  half pi/4, uniform radial rows floored at _WEDGE_R_MIN=1e-2, r_extent capped
  <1) + _build_wedge_chart -> LensAmplificationSurrogate.from_wedge_engine
  (definition=INTERIOR_SACR_C; DD-cap + arc map applied INSIDE engine). Build
  loop region=='wedge_interior' branch: inline held-out via chart.wedge_map +
  _from_wedge_fixed; CarrierDiscontinuityError/gated -> ladder-served gap, NO
  subdivision (mirrors lobe).
- DELETED _farfield_interior_tiles (no prod caller). _build_farfield_chart +
  _subdivide_farfield_tile now exterior-only (dropped `definition` param;
  both subdivide call sites pass interior_admission=None). _heldout_eps:
  is_farfield_label = isinstance(chart, FarFieldChart) (wedge -> else-branch);
  annotation + _load_or_build annotation gained | InteriorWedgeChart.
- KEPT _interior_admission UNCHANGED (live exterior-tiler dep at ~L4014 via
  exterior_admission; brief was WRONG). admission var stays None on positive
  parity path (annotated assign L4182); only consumed in parity!=1 saddle path.
- chart_types provenance (L3462) LEFT coarse tube/farfield: InteriorWedgeChart
  -> 'farfield' same bucket as LobeInteriorChart (out-of-scope census label,
  no crash/dispatch impact). _gate_chart type-agnostic (kind='interior').
- Smoke: ast.parse OK + import OK. UNVERIFIED: no training run (per WP1
  no-exec constraint) -> wedge training path UNVERIFIED end-to-end.
- OWED TEST BREAKAGE (Test Dev): test_lensing_exterior_windows.py ~L2079 and
  test_lensing_ppgo_bandsplit.py ~L89/L620 import/use deleted
  _farfield_interior_tiles.

2026-08-06 WP2 wedge tiler waist-split columns + adaptive u-subdivision
(surrogate_training.py ONLY):
- Imports: added _wedge_theta_waist, _wedge_cusp_axis_map to the existing
  `from cogwheel.lensing.surrogate import (...)` block.
- _wedge_interior_tiles: signature now (gamma, r_extent, n_per_side); returns
  5-tuples (center, half, i, j, axis_origin). Emits TWO angular columns per
  radial row split at _wedge_theta_waist(gamma) (NOT pi/4): low col
  [0, waist] j=0 axis_origin='low'; high col [waist, pi/2] j=1
  axis_origin='high'. NO cusp-edge exclusion strip (u=d^(2/3) absorbs cusp
  scaling; centre-exclusion _WEDGE_R_MIN has no angular analog). Radial rows
  unchanged (uniform in [_WEDGE_R_MIN, r_extent]).
- _build_wedge_chart: added axis_origin:str|None=None kwarg, threaded verbatim
  into from_wedge_engine(axis_origin=...). Single-sourced; engine asserts it vs
  its own midpoint-vs-waist derivation.
- NEW _subdivide_wedge_tile (single level, no recursion; mirrors
  _subdivide_farfield_tile but in wedge (r,u) coords). Halves r at plain
  r-midpoint AND theta at the u-MIDPOINT mapped back to theta via
  _wedge_cusp_axis_map + np.interp(u_mid, u_fine, theta_fine) (NEVER
  theta-midpoint -> near-cusp child narrower in theta). Up to 4 children, each
  rebuilt via _build_wedge_chart carrying parent axis_origin, re-gated on
  interior bar (_gate_chart('interior')). 3-way eff_w_nodes (tile override ->
  interior_w_nodes_per_decade -> w_nodes_per_decade); passes RESOLVED
  eff_w_nodes to _build_wedge_chart (NOT raw None -> avoids exterior-density
  fallback). Passing->charts; gated/carrier-flip->ladder_served_gap recorded
  not packed. Returns summary w/ 'packed' count + 'theta_split'.
- Call site (~4569): compute gamma_rep = median(_log_reach_gamma_axis(band,
  config.n_gamma,'gamma')) — EXACT match to from_wedge_engine's rep_gamma so
  tiler waist == engine waist (no assert misfire). Pass gamma_rep first.
  admitted dict gains 'axis_origin'. tile-unpack loop `for center,half,i,j in`
  -> `...,j,axis_origin in`. Main tile loop binds axis_origin=tile.get(
  'axis_origin') (None non-wedge), build_wedge closure passes it.
- Wedge gated branch (~4825): REPLACED ladder_served_gap+continue with
  subdivided=True + _subdivide_wedge_tile(...) call; ladder_served_gap =
  (subdivision['packed']==0). Carrier-flip except path LEFT as ladder gap
  (out of WP2 eps-feedback scope).
- Smoke: ast.parse OK + import OK + numeric: gamma=0.3 -> 4 tiles, waist
  0.71011 (0.90x pi/4), axis_origin matches derived both cols, low-col u-split
  theta=0.251 vs theta-mid 0.355 (narrower near cusp). Diagnostics clean.
  UNVERIFIED: _subdivide_wedge_tile build_child path (engine call) NOT run per
  no-training-run constraint -> subdivision end-to-end UNVERIFIED.
- OWED TEST BREAKAGE (Test Dev): test_lensing_interior_wedge_chart.py
  (WedgeInteriorTilesContractTestCase, ...CapFalsificationTestCase,
  WedgeTilesSelfFalsificationTestCase, _unpack_tile ~L2168,
  test_single_angular_column_spans_full_wedge premise now WRONG — 2 cols) and
  test_lensing_ppgo_bandsplit.py L633 all call _wedge_interior_tiles with OLD
  2-arg (r_extent,n) + unpack 4-tuples; need (gamma,r_extent,n) + 5-tuples.

2026-08-06 WP1 cusp-adapted wedge angular axis (surrogate.py ONLY):
- Schema bump v1->v2: _WEDGE_AXIS_SCHEMA='wedge_caustic_relative_v2',
  _KNOWN_WEDGE_AXIS_SCHEMAS={v2} (v1 DROPPED). Stored in chart's OWN meta
  (_chart_to_npz wedge branch already uses _WEDGE_AXIS_SCHEMA; load-path
  ~L4613 validates via _KNOWN_WEDGE_AXIS_SCHEMAS -> v1 artifact hard-refuses,
  no code change needed there). Provenance dict also carries v2.
- New helper _wedge_theta_waist(gamma): minimize_scalar(r_caustic, bounds
  (1e-4,pi/2-1e-4), 'bounded', xatol=1e-6). Oracle r_caustic(g,waist)==g
  verified <1e-14 at g=0.2/0.3/0.495/0.7/0.9 (waists 0.735..0.552 match
  brief table). Added `from scipy.optimize import minimize_scalar`.
- New helper _wedge_cusp_axis_map(theta_lo,theta_hi,origin)->(theta_fine,
  u_fine): u=d^(2/3). low: u=theta^(2/3)-theta_lo^(2/3), theta=(u+theta_lo
  ^(2/3))^(3/2). high: u=(pi/2-theta_lo)^(2/3)-(pi/2-theta)^(2/3), theta=
  pi/2-clip(base_lo-u,0,None)^(3/2). UNIFORM-IN-u fine grid (linspace u,
  invert to theta), _FARFIELD_ARC_MAP_SIZE=2001 nodes (NO new constant).
  Endpoints forced exact (theta_fine[0]=lo,[-1]=hi). Both monotone incr,
  u_fine[0]=0 -> passes _validate_theta_to_s unchanged.
- from_wedge_engine: added kwarg axis_origin:str|None=None (keyword-only,
  backward compat; prod caller surrogate_training.py:2991 unaffected).
  REPLACED arc-length block (rep_gamma median / caustic_speed /
  cumulative_trapezoid / vstack / interp) with waist-classified u-map:
  derived_origin='low' if theta_mid<=theta_waist else 'high'; if axis_origin
  supplied, assert ==derived_origin (train/serve skew guard) else ValueError.
  theta_to_s=vstack([theta_fine,u_fine]); s_grid=interp(theta_wedge_grid,
  theta_fine,u_fine). No cumulative_trapezoid survives in wedge path (import
  still used by _caustic_arclength_map exterior).
- UNTOUCHED (verified): DD-cap block, _validate_theta_to_s, _evaluate_chart
  serve branch (coordinate-agnostic on theta_to_s).
- Smoke: ast.parse OK + import OK + helper numerics OK. UNVERIFIED: no
  from_wedge_engine end-to-end run (no-training-run constraint) -> full
  wedge u-map training path UNVERIFIED end-to-end.
- OWED TEST BREAKAGE (Test Dev): test_lensing_wedge_dd_arclength.py is
  entirely about the RETIRED arc-length axis (imports caustic_speed +
  cumulative_trapezoid; test_arc_length_teeth_linear_would_fail compares
  theta_to_s row1 to cumulative_trapezoid(caustic_speed)). Row1 is now
  u=d^(2/3), NOT arc-length s -> those assertions must be retargeted to the
  cusp-adapted map. Any test pinning schema=='..._v1' or loading a v1
  artifact must expect hard-refuse. _NODE_EXACT_TOL already 1e-7 in
  test_lensing_surrogate_lobe.py (Test-Dev-owned, no action).
