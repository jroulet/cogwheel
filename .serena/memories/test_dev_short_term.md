# Test Dev Short-Term Observations

- 2026-08-14 (INS-1-001 disposition, tiling_census census-vs-trim gap):
  EXTENDED test_lensing_tiling_census.py (+1 class `PpgoTrimIndependenceTestCase`,
  +1 self-falsification method; 24->26 tests, 62.5s). Finding text explicitly
  says "no new tests needed beyond run()'s output schema coverage" for the
  docstring/`ppgo_trim_modeled`-field fix (that's a Coder-owned production
  change, not yet landed -- confirmed `ppgo_trim_modeled` absent from both
  tiling_census.py and the test file at audit time). Audited existing schema
  coverage: the one `result['schema']=='tiling_census_v1'` assertion is the
  only schema pin and carries no exact-keyset assertion, so a future field
  addition is forward-compatible with zero edits needed there. Chose to pin
  the ACTUAL load-bearing structural invariant behind the finding's
  "conservative upper bound, never an underestimate" disposition instead:
  booby-trapped `ppgo_map.get_certified_ppgo_map` and
  `surrogate_training._apply_ppgo_trim` (mock.patch.object, AssertionError
  side_effect, engine-door-trap idiom) and confirmed `tc.run()` completes
  without calling either -- proves the census's per-region counts are
  structurally independent of whatever certified map is installed in-process,
  not merely independent by the current absence-of-a-call-site. Self-
  falsification twin proves both primitives ARE reachable/raise when called
  directly (trap has teeth, not testing an inert function). No other
  pre-existing test file references `tiling_census` (new module, first build
  cycle) so no cross-file backward-compat audit was needed.

- 2026-08-14 (tiling_census WP1 shard-2): EXTENDED test_lensing_tiling_census.py
  +3 classes/+2 self-falsification methods (12->24 tests, 54.6s). Q1 arc census:
  astroid detected==4/trained==1 (F079 fundamental-domain fold), saddle
  detected==6/trained==min(6,max_tube_arcs); teeth via TrainingConfig(
  max_tube_arcs=2) -> saddle trained->2, astroid stays 1. Q4 SPEC DIVERGENCE
  (flagged in report): spec claimed "contained True for every admissible region
  OR any False carries a reason" — FALSE on smoke config: EVERY region reports
  contained=False and those non-deferral Falses carry NO reason (reason only on
  floor=None deferrals). Did NOT assert contained==True; pinned the durable
  invariant instead — containment predicate is arithmetically FAITHFUL to
  reported bounds (independent _recompute_contained), plus closed-form oracles:
  astroid ceiling==min(480,60/sqrt(s)), saddle tube floor/ceiling==58/148
  (ppgo_map.SADDLE_WALL / _SADDLE_W_CEILING), saddle far-field floor==
  (2e4*K)**(1/3) via from-scratch geometry.macro_matrix->find_images->
  ppgo_error_estimate recompute (K=sum sqrt|mu||c3|), matched census exactly
  (70.186...). SELF-ESTIMATE: self_estimate_seconds == surrogate_training.
  _self_estimate(config,regions) EXACT passthrough; scoped run==scoped direct
  AND != unscoped; aggregate_call_count==sum(per_region n_nodes)*8; cross_check
  ratio within _CROSS_CHECK_FACTOR=5000. Self-falsification: 10x floor coeff ->
  floor*10**(1/3)~2.15x diverges; synthetic inside/outside/below/None entries
  discriminate the containment predicate. Oracle recipe for saddle floor:
  source = centroid + direction*1.2*r_max, w_min=box.w_range(-1)[0].

- 2026-08-14 (tiling_census WP1): NEW suite test_lensing_tiling_census.py
  (12 tests, 38.5s). Module cogwheel/lensing/tiling_census.py is engine-free
  by construction (run() is geometry-only, ~6s/call). Pins: (1) ENGINE-FREE —
  namespace absence (`not hasattr(tc,'ChangRefsdalChannels')`) + booby-trap
  all four amplitude doors via mock.patch.object raising AssertionError:
  channels.ChangRefsdalChannels.evaluate (class-method patch = catch-all
  regardless of import binding), _schwinger.f_schwinger,
  _schwinger._f_schwinger_mpmath. (2) THIN-CALLER — patch
  st._farfield_exterior_tiles to drop one tile on FIRST call only (state
  flag) => exterior:+1 n_tiles delta==1, n_nodes delta == n_rho*n_theta_c*
  n_gamma*int(w_nodes_per_decade*2)=512 (independent config-grid oracle, NOT
  a census helper). NOTE: spec named _farfield_tiles but that tiler feeds
  ONLY Q2/Q3 diagnostics, not any per_region count — used it as the NEGATIVE
  control (patching it leaves all per_region counts unmoved). (3) VERDICT —
  tc._verdict pure classifier: count==0 or <low -> SILENT_EMPTY, >high ->
  EXPLOSION; the explicit `count==0` clause is load-bearing when low==0
  (0<0 is False). Drove 0/huge through run() on exterior:+1, reused cached
  baseline (module-level _BASELINE dict) as the IN_BAND leg to save a run.
  Baseline smoke run: exterior:+1=19 tiles/9728 nodes, lobe_interior:-1
  naturally SILENT_EMPTY (0 tiles). Anti-vacuity base _CensusTestCase with
  self._observe()/tearDown; CensusSelfFalsificationTestCase proves traps +
  zero-guard teeth. No sibling-suite backward-compat risk (new module, no
  signature/constant changes; no prior test referenced tiling_census).
