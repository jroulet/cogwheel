# Test Dev Short-Term Observations

## 2026-08-18 (WP1 tiling_plan Spec7 engine-free whole-tool, +4 tests -> 36, 23.8s)
- EXTENDED test_lensing_tiling_plan.py with EngineFreePlanRunTestCase — the
  ONLY suite here that runs the real tp.run() (all 6 prior shards are
  helper-level, engine-free by never calling run). setUpClass runs
  tp.run(n_samples=_ENGINE_FREE_N_SAMPLES=60, seed=0) ONCE under an ExitStack
  arming the four wave doors (ChangRefsdalChannels.evaluate,
  _schwinger.f_schwinger, _schwinger._f_schwinger_mpmath,
  mpmath.gauss_quadrature) each side_effect=unique sentinel Exception.
  Measured: n=40 ~12.7s/total_nodes=2580; n=60 setUpClass ~20s; whole file
  23.8s (well under 5min ceiling). 4 tests: wellformed report+positive
  total_nodes; every door call_count==0 (load-bearing); LIVE positive
  control re-arming doors and asserting each raises its sentinel + records
  call_count==1 (kills vacuous-patch-target); sentinel-disjointness from
  tuple(sg._REFUSAL_ERRORS)+(ValueError,ZeroDivisionError).
- MPMATH SUBSTITUTION (documented in class + module docstring): literal
  "mpmath never in sys.modules" is UNACHIEVABLE (mpmath imported at
  cogwheel.lensing package-import via _schwinger, before run() is called);
  load-bearing substitute is mpmath.gauss_quadrature door call_count==0 (no
  special-function EVALUATION entered). Same pattern as
  test_lensing_serve_route_census EngineFreeTestCase.
- POSITIVE-CONTROL ergonomics: mock.patch.object(Cls,'evaluate',side_effect=
  Door()) replaces the attr with a MagicMock callable with ZERO args, so
  `ChangRefsdalChannels.evaluate()` / `_schwinger.f_schwinger()` /
  `mpmath.gauss_quadrature()` all raise the sentinel regardless of real
  signature — no need to construct valid engine args to prove traps armed.
- WP pure-additive (git ??: tiling_plan.py/test/scripts only); no production
  edit, no sibling suite imports tiling_plan => NO backward-compat audit.
  Net +4 tests (32->36), one new invariant (whole-tool engine-free).

## 2026-08-18 (WP1 tiling_plan shard 2: w-edge/annulus-gauge/escalation, +16 tests -> 32, 6.05s)
- EXTENDED test_lensing_tiling_plan.py (16->32) at helper level, still no
  run()/engine. Three new classes: MeasuredWAxisEdge, AnnulusGaugeRoundTrip,
  EscalationTripwire. Added `from cogwheel.lensing import ppgo_map` + consts
  _ANNULUS_ROUNDTRIP_RTOL=1e-6, _RETIRED_ANNULUS_CAP=2.40.
- SPEC 4 (_measured_w_range(records,region,band,box,parity)): filters
  route=='engine_residual' matching region+gamma_band, returns
  (exp(min log_w_min), exp(max log_w_max), 'measured'); empty -> box.w_range
  + 'prior_box_fallback'. Synthetic records w_hi 38/51.6/60 per region;
  each region gets its OWN edge (kills blanket 60). exp(log(38))=37.9999...
  so assertAlmostEqual places=9, NOT assertEqual. Pinned per-(region,band)
  isolation (exterior 60 must not leak into lobe_exterior 38) + min/max agg.
- SPEC 5 (_annulus_record(st,ctx,region)): astroid 'exterior' ->
  gauge='caustic_rho', rho_inner=exclusion_rho, rho_outer=rho_outer_region,
  caustic_reach=st._scalar_caustic_reach(gamma_mid). GENUINE round-trip
  (INDEPENDENT oracle): physical=rho_outer*caustic_reach; then
  ppgo_map.caustic_rho(gamma,physical) recovers rho_outer EXACTLY (0.0 diff,
  gate 1e-6) because caustic_rho(g,|y|)=|y|/caustic_geometry(g,0)[0] and
  st._scalar_caustic_reach==caustic_geometry(g,0)[0]. Teeth: 2*reach breaks
  round-trip. Saddle 'lobe_exterior' -> gauge='rho_lobe', rho_inner=1.0,
  prior_demand_rho_outer_lobe==_SADDLE_LOBE_DEMAND_RHO_OUTER=20.2 (assert
  >2.40 kills retired cap); rho_lobe->physical (deltoid) is OUT OF SCOPE per
  production docstring, so no saddle round-trip. Interior regions -> None.
- SPEC 6 (_escalation_verdict(total_calls, per_region, total_nodes)):
  per_region={key:{'region_nodes':n}}; share=n/total_nodes; escalate iff
  total_calls>_ESCALATION_CALL_LIMIT(5e5) OR max_share>_ESCALATION_REGION_
  SHARE(0.40). STRICT >: at exactly 5e5 calls (nodes=62500, /4 regions to
  keep shares 0.25) NOT escalate; +4 nodes -> escalate w/ 'total_calls...
  exceeds limit' reason. share=exactly 0.40 (nodes 4,3,3) NOT escalate.
  Cost currency: SECONDS_PER_CALL==0.0903 EXACT, _LABELS_PER_NODE==8 (tiling
  _census), total_calls=nodes*8, wall_clock_s=total_calls*0.0903 (built in
  build_plan, not verdict). benign 15000 nodes -> 120000 calls pins LPN==8.
- WP is PURE-ADDITIVE (git status: only ?? tiling_plan.py/test_.../scripts/
  tiling_plan.py; no existing module modified) => NO backward-compat audit
  triggered, no sibling suite depends on anything changed. Standalone file,
  no process global. Net +16 tests; each pins a distinct invariant.

## 2026-08-18 (WP1 tiling_plan campaign-cost predictor, test_lensing_tiling_plan.py NEW, 16 tests, 6.76s)
- ENGINE-FREE at helper level: never call tiling_plan.run() (triggers 10k
  serve_route_census + full tiling_census). Drove the 3 axis-sizing helpers
  (_gamma_resolution/_n_gamma_in_band, _n_theta_for_span, _n_w_nodes), the
  demand gate (_residual_by_region_band, _plan_band, _plan_region) and the
  reach oracle directly. st via functools.lru_cache(_load_production_modules).
- INDEPENDENT REACH ORACLE for Spec 2: st._scalar_caustic_reach ==
  surrogate._caustic_reach == ppgo_map.caustic_geometry(g,0)[0] (CLOSED-FORM
  u-candidate extremization). Independent oracle = max over theta of
  geometry.r_caustic(g,theta) (polar sweep over critical_point) — genuinely
  disjoint path. ASTROID reach is ON-AXIS (theta=0, a grid point) so rel
  ~1e-16 (asserted <1e-9). SADDLE reach is OFF-AXIS (deltoid extremum) — a
  coarse 91-pt polar grid misses it by 6-9%, so r_caustic identity/oracle
  checks are ASTROID-ONLY; saddle-side monotonicity carried by production
  _gamma_resolution (closed form, exact both sides).
- MONOTONE-TOWARD-WALL currency: astroid ascending grid [0.70..0.95] res
  STRICTLY DECREASES [0.129->0.036]; saddle ascending [1.05..1.30] res
  STRICTLY INCREASES [0.037->0.725] (= decreasing toward wall). Use grids
  >=0.70; a non-monotone bump lives at astroid g~0.5-0.6 (below the window).
  n_gamma equal-span far (0.5,0.6)=1 vs near-wall (0.85,0.95)=3.
- SPEC 1 (kill hardcoded counts): _ExpReachST._scalar_caustic_reach=exp(k*g)
  makes gamma resolution EXACTLY g-independent (res = C_GAMMA*s/sinh(k*s)), so
  doubling band span at fixed anchor doubles n_gamma with resolution provably
  fixed (assertAlmostEqual res1==res2). Invariant: n2>n1 (kills flat count)
  AND |n2-2*n1|<=1 (ceil slack). theta/w laws identical shape via stub config
  SimpleNamespace(w_nodes_per_decade, interior_w_nodes_per_decade).
- SPEC 3 gate: patch tp._band_tile_geometry -> (4,8,[]); SimpleNamespace
  stubs for st(_scalar_caustic_reach)/box(w_range)/ctx(band,gamma_mid,
  exclusion_rho,rho_outer_region)/config. _plan_band label via
  serve_route_census._gamma_band_of(ctx.gamma_mid, gamma_edges) -> pick
  gamma_edges=[0.4,0.6] + gamma_mid=0.5 => '0.4-0.6'. Empty residual_lookup
  => (None,'gated_no_demand'); {key:5} => 'planned', band_nodes ==
  spatial_total*n_gamma*n_w. _plan_region over [served_ctx,demand_ctx]:
  region_nodes == demand band ONLY, n_bands_gated_no_demand==1.
- HOOK NOTE: `cat >> file <<EOF` appends are hook-BLOCKED intermittently
  (first two worked, third blocked same session). Use
  mcp__serena__insert_after_symbol on the last class for large appends.
- No production edits; standalone new file; ran only the new suite (16/16).

## 2026-08-18 (WP1 F083 arc-trim promotion, test_lensing_surrogate_training.py)
- CALL-ORDER INJECTION for a pure-arithmetic scan helper: `_trim_tube_arc`'s
  inner `_delta_tau` calls `_tube_source -> geometry.macro_matrix ->
  _frame_delays -> _merging_fold_pair` once per scan node IN ORDER. Force the
  first three to succeed (finite source, 4 images) so `_merging_fold_pair` is
  the SOLE Delta_tau source, then patch it with a call-counter mapping call
  index i -> synthetic dtau[i] (None => NaN node). Lets you inject a bespoke
  Delta_tau(theta) profile engine-free and pin the knee/peak/standoff affine
  arithmetic BIT-EXACT (assertEqual). peak_val=5.0 => 0.6*5.0==3.0 exact
  float64 so the knee crossing lands exactly on a grid sample.
- SELF-FALSIFICATION for peak selection: swap dtau[peak] with dtau[peak+1]
  (rise/knee untouched) so nanargmax moves one index; assert the returned
  bracket DIFFERS (assertNotEqual) AND matches the peak+1 affine form. A
  wrong-knee mutation would be inert here since knee is unchanged; the peak
  swap is the load-bearing mutation.
- PARITY GATE pin: saddle band (1.1,1.2) parity=-1 => `_trim_tube_arc` returns
  the SAME object (assertIs) field-identical; astroid (0.2,0.4) parity=1 =>
  span STRICTLY narrower (1.298->0.567), other fields (branch/inward_sign/
  image_count/cusp_windows) unchanged, bounds within original arc. eta_max
  computed the loop's way: `config.f_max * _min_curvature_radius(band, arc,
  config.n_caustic_samples)`.
- Spec C (off-span boundary-decline serve probe) ALREADY COVERED by
  test_lensing_tube_d2_fold.py::TubeThetaInframeClosedFormTestCase
  ::test_returns_none_when_no_image_touches_frame (off-frame gauge angle ->
  _tube_theta_inframe None; reads the trimmed theta_grid fence) and
  test_lensing_surrogate.py _tube_serves cusp-exclusion declines. Recorded +
  skipped (parsimony); both suites owned by other runs.
- 6 new tests, +6 collected (180->186), file collects clean in 6.3s; full-file
  run exceeds 240s tool budget (driver's job).

## 2026-08-18 (WP1 F083 arc-trim promotion, test_lensing_tube_beat_free.py)
- DRY RE-POINT: retired the fixture-local `_f083_delta_tau` helper + inline
  80-pt knee scan + 3 constants (_F083_DTAU_FRAC/_LO_STANDOFF/_HI_STANDOFF) in
  `_f083_shared_tube`; replaced with a single `arc2 = _trim_tube_arc(band,
  arc, eta_max, parity=1)`. Production trim constants match the retired
  fixture literals byte-for-byte, so arc2 (and the eps bar) are unchanged.
  Existing `test_trimmed_run_refused_no_build_nodes` (asserts fx.refused==0)
  now guards the PRODUCTION trim = drifted-core LOUD-failure invariant.
- BOUNDARY-DECLINE PROBE teeth without the eta confound: full surrogate.serve
  would decline via the ETA gate (partition.caustic_distance 0.0127 <
  eta_floor 0.0254 near the excised sliver), giving zero teeth vs a
  trim-revert. Drove `_tube_serves(chart, gamma, log_w_min, log_w_max,
  eta_mid, excised_theta, image_count, require_fref=False)` DIRECTLY with an
  in-band eta_mid so the theta gate is the sole decliner. Trim-revert teeth:
  same excised theta on a `types.SimpleNamespace(theta_grid=[untrimmed_lo,
  untrimmed_hi])` frame returns `_tube_theta_inframe` not-None. Harness teeth:
  interior_theta serves True. excised_theta = 0.5*(frame_hi+untrimmed_hi).
- Parsimony ledger: retired 1 helper fn + 1 inline scan + 3 constants; added
  1 class / 1 method. Net test count +1 (23->24). 5 fixture-sharing tests
  green in 63.25s.
