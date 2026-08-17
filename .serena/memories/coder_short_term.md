# Coder Short-Term Observations

## 2026-08-17 WP4 tube census no-explosion band tightened to capped Nyquist (tiling_census.py)
- Replaced the loose static tube 'nodes' upper band (1,10**9) with a DYNAMIC
  capped-Nyquist edge. New module helper `_tube_n_theta_ceiling(st, box,
  parity, gamma_mid, structure, tube_arcs, arc_r_min, max_eta_max, config)`
  ENGINE-FREE mirrors `_build_tube_chart`'s n_theta sizing per arc:
  w_max = st._capped_w_range(box, parity, structure.caustic_reach +
  max_eta_max)[1]; per arc eta_max_arc=config.f_max*arc_r_min[idx];
  st._tube_delay_map(gamma_mid, arc, eta_max_arc) -> TV=s_fine[i_hi]-s_fine[i_lo];
  ceiling_arc = min(config.n_theta_cap, max(st._TUBE_MIN_THETA_NODES,
  ceil(st._TUBE_NYQUIST_PPP*w_max*TV/(2pi)))); return MAX over arcs. Safety
  consts PPP/floor + cap READ from st (verified PPP=8 floor=4), NEVER re-typed
  (mirror-currency). Falls back to config.n_theta_cap when no arc / all refuse
  / w-cap LensDomainError (loosest edge, never false-positive EXPLOSION).
- gamma_mid == production rep_gamma: rep_gamma=float(np.median(gamma_grid)),
  gamma_grid=linspace over band => median==midpoint==0.5*(band0+band1). Exact.
- _BandCtx: added non-default field `tube_n_theta_ceiling: int` after
  max_eta_max (constructed via keywords in _build_band_ctx, always passed).
- _build_band_ctx computes ceiling after min_eta_max, stores on ctx.
- _census_region: for region=='tube', ceiling=max(ctx.tube_n_theta_ceiling
  over contexts, default n_theta_cap); nodes_high = total_tiles * ceiling *
  n_u * n_gamma * _w_nodes; nodes_band=(1, nodes_high); verdict_nodes runs
  against it. Because n_nodes = total_tiles*n_theta*n_u*n_gamma*w_nodes and
  every factor except n_theta is shared, n_nodes>nodes_high IFF
  config.n_theta>ceiling -> EXPLOSION. total_tiles==0 -> n_nodes==0 ->
  _verdict(0,(1,0)) count==0 -> SILENT_EMPTY preserved. Verdict enum + IN_BAND
  semantics unchanged; NO new gate (reused verdict_nodes). record
  expected_bands now reports the dynamic tube nodes band (not stale 1e9).
- NO fast-suite regression expected: smoke default config.n_theta=4;
  ceiling >= floor(4) always => 4<=ceiling => IN_BAND for the smoke tiling.
  Only a genuinely over-provisioned n_theta (> most node-hungry arc's capped
  Nyquist) trips EXPLOSION.
- Verified PARSE_OK, IMPORT_OK, helper+field present, production consts
  accessible. Did NOT run census/test suite (Coder remit). Test Dev: pin
  (a) served smoke tube -> IN_BAND (n_theta=4 <= ceiling), (b) inflate
  config.n_theta above ceiling -> EXPLOSION, (c) empty tube (0 arcs) ->
  SILENT_EMPTY, (d) ceiling reads st._TUBE_NYQUIST_PPP/_TUBE_MIN_THETA_NODES
  (mirror not re-typed), (e) engine-free: mpmath never imported / no
  evaluate call during census run.

## 2026-08-17 WP3 _heldout_eps counts unserved held-out points as coverage misses (surrogate_training.py)
- Added module const `_HELDOUT_COVERAGE_MISS_EPS = 1.0` just before
  `_heldout_eps` (a "total miss": surrogate delivers nothing where the true
  envelope is finite -> rel error unity; exceeds every per-kind eps bar).
- `_heldout_eps` NEW contract: a point with a VALID FINITE reference that the
  guard stack refuses to serve (`not served`) is now a COVERAGE MISS
  (sets has_coverage_miss=True) instead of silent `continue`-skip. It is kept
  OUT of the `errors` list; at the end `eps = max(errors)` then, if
  has_coverage_miss, `eps = max(eps, 1.0)`. Points with NO reference (engine
  refusal / farfield ghost-gate GhostDomainError / non-finite env_true) STILL
  skip and are NOT counted (no ground truth => not a coverage question).
- nan-iff-zero-served preserved: `if not errors: return nan` runs BEFORE the
  miss-fold, so an all-miss band (valid refs, zero served) returns nan (errors
  empty) -> `_reprovision_w_nodes._eps_for` -> None -> early 'engine_refused',
  NO loop. Float return type + return annotation unchanged (verified).
- NO fast-suite regression expected: pilot/smoke tube fixtures have whole arc
  servable (i_lo=0/i_hi=n-1 per WP1 memory) => zero misses => case reduces to
  old `max(errors)` byte-identically. Only genuine refused-tail (F083) charts
  hit the miss path -> eps=1.0 -> gated 'eps_above_bar' (the intended flag) /
  reprovision 'bar_not_cleared'. Both callers (_reprovision_w_nodes,
  chart-registration via _gate_chart/_chart_gated) handle finite-or-nan
  without crash; termination preserved (finite descent range + early returns).
- Verified: PARSE_OK, IMPORT_OK, const=1.0, return_annotation float, params
  unchanged; isolated 4-case scalar logic check green. Did NOT run test suite
  (Coder remit). Test Dev: add pins for served-only==old-value, served+miss
  ->penalty, all-miss->nan, no-ref->nan.


## 2026-08-17 WP2 TubeChart schema rename theta_to_s -> theta_to_s_prime (surrogate.py)
- Renamed TubeChart dataclass field theta_to_s -> theta_to_s_prime; field,
  from_values param (theta_to_s_prime: np.ndarray | None = None), _assemble
  param, class docstring (axes now (log w, gamma, u, s')), and serve-path
  np.interp in _evaluate_chart all updated. Identity fallback (theta_to_s_prime
  is None -> np.vstack([theta_grid, s_grid])) KEPT; documented that identity now
  means "s'==theta as a degenerate delay coordinate".
- Added module consts after _KNOWN_LOBE_AXIS_SCHEMAS: _TUBE_AXIS_SCHEMA =
  'tube_delay_tv_v1', _KNOWN_TUBE_AXIS_SCHEMAS = frozenset({_TUBE_AXIS_SCHEMA}).
  Added _validate_tube_axis_schema(tag, artifact_label) thin wrapper delegating
  to shared _validate_axis_schema (mirrors wedge/exterior-polar/lobe pattern).
- _chart_to_npz tube branch: meta['axis_schema']=_TUBE_AXIS_SCHEMA;
  arrays write prefix+'theta_to_s_prime'. _chart_from_npz tube branch:
  _validate_tube_axis_schema(meta.get('axis_schema'), f'chart {index}') BEFORE
  assemble (HARD-REFUSE: None/stale/unknown tag -> ValueError naming required
  schema + retrain instruction); reads prefix+'theta_to_s_prime'. No fallback
  to identity on stale tag.
- KEPT helper name _validate_theta_to_s (referenced by
  test_lensing_wedge_dd_arclength.py); only its docstring made
  coordinate-agnostic. Renaming would break a sibling test.
- _build_provenance is NOT a per-chart axis_schema surface (surrogate.py's is
  exterior-polar-specific; training's is build-wide) — the load-bearing refusal
  path is per-chart NPZ meta, which is updated. WP "How" mention of
  _build_provenance is boilerplate; no tube-specific provenance builder exists.
- surrogate_training.py WP1 caller updated for rename: _build_tube_chart local
  theta_to_s_prime = np.vstack([theta_fine, s_fine]); from_values call passes
  theta_to_s_prime=theta_to_s_prime (s_grid= kept alongside).
- LATENT (WP1 scope, not WP2): if i_lo>0, theta_to_s_prime[0][0]
  (theta_fine[0]) != theta_grid[0] (theta_fine[i_lo]) could trip
  _validate_theta_to_s row0-start check; smoke fixtures have i_lo=0 so green
  now. Flag for Test Dev when non-trivial refused tails appear.
- Verified: py_compile PARSE_OK; import + behavioral sanity SANITY_OK
  (theta_to_s_prime present / theta_to_s absent in NPZ keys;
  _TUBE_AXIS_SCHEMA=='tube_delay_tv_v1'; _validate_tube_axis_schema raises
  ValueError for None/stale/wrong, accepts correct tag). Did NOT run test
  suite (Coder remit). Test Dev must migrate any tube test passing
  theta_to_s=/reading .theta_to_s to theta_to_s_prime, and arc-length-oracle
  uniformity tests to the delay coordinate.

## 2026-08-17 WP1 delay-uniformized tube angular coordinate (surrogate_training.py)
- Added `_tube_delay_map(gamma, arc, eta_ref, n_map=501)` returning
  (theta_fine, s_fine, i_lo, i_hi): s'=cumulative_trapezoid(|d Delta_tau/d
  theta|, theta, initial=0). Delta_tau=0.5*(tau_minus-tau_plus) via IMPORTED
  `_merging_fold_pair` (chang_refsdal._airy_fold) — NO re-derived formula
  (DRY pin). matrix=geometry.macro_matrix(gamma) (beta=kappa=0, tube frame);
  find_images len!=4 or pair None -> NaN. Guards: all-NaN raise, interior
  refusal hole raise, non-finite/non-monotone s' raise.
- `_fill_cusp_tails`: A3 law Delta_tau~d^{2/3} => g=Delta_tau^{3/2} LINEAR in
  theta; linearly extrapolate g from servable boundary+inner neighbour,
  Delta_tau=max(g,0)^{2/3}. Cusp-location-free (root of g is the cusp). Fills
  refused tails so s' spans full arc; servable subrange [i_lo,i_hi] marks
  where pair resolved.
- `_build_tube_chart`: replaced `_tube_arc_length_map` node placement with
  `_tube_delay_map`. N_theta = min(n_theta_cap=32, max(4, ceil(PPP=8 * w_max *
  TV / (2pi)))). w_max = w_range[1] (NO module tube w cap; passed from caller
  via tube_w_range=_capped_w_range(...); documented). TV = s'[i_hi]-s'[i_lo]
  at eta_max (largest Delta_tau, conservative over-count). Nodes uniform in s'
  over [s_lo,s_hi]; endpoints pinned to theta_fine[i_lo]/[i_hi] (shrink-shell,
  no routing). theta_to_s = vstack([theta_fine, s_fine]) (WP2 renames field).
  4-node floor = _validate_axis cubic-spline minimum (surrogate.py).
- TrainingConfig: +n_theta_cap=32, engine_budget 400->2048. Constants added:
  _TUBE_DELAY_MAP_SIZE=501, _TUBE_NYQUIST_PPP=8, _TUBE_MIN_THETA_NODES=4.
- `_tube_arc_length_map` KEPT INTACT (still used by build_tube adequacy
  diagnostic + WP1 falsification baseline).
- Smoke fixtures: whole arc servable (cusp windows exclude |y'|->0 cusps) =>
  i_lo=0/i_hi=n-1 => endpoints=arc bounds, so existing endpoint tests
  (theta_grid[0]==theta_lo etc) stay green; shrink only activates on a real
  refused tail. Test Developer must update arc-length-oracle uniformity
  tests to the delay coordinate.
- UNVERIFIED: did not run geometry/engine to exercise _tube_delay_map end to
  end (verification is Test Dev/Inspector scope); parse+import+signature
  checks pass. Import adds no cycle (_airy_fold imports only geometry).
