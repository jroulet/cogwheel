# Test Dev Short-Term Observations

- 2026-08-15 (F081 Part B per-arc lobe-edge shell, test_lensing_tube_d2_fold.py):
  EXTENDED +2 classes/6 tests (SaddleLobeEdgeShellTestCase x4,
  SelfFalsification x2); 24/24 green 7.82s, engine-free. Pin 8: the tube
  shell scalar consumers pass is f_max*MIN(arc_r_min) (lobe-edge shell), NOT
  the retired band-wide f_max*MAX(arc_r_min). WP changed the CALLER's choice
  of which eta_max scalar to hand `_saddle_lobe_admissions` and the far-field
  `exclusion_rho` inner-edge test, NOT the callee signatures/formulas. Derived
  fixture from REAL SADDLE_BAND geometry: band_caustic_structure ->
  _tube_training_arcs(structure,-1) -> _min_curvature_radius per arc =
  arc_r_min [0.3987, 9.1559] (~23x anisotropy, non-vacuous), min_eta_max
  0.1595 / max_eta_max 3.6624. Witness = synthetic circular lobe
  (_circular_lobe, exact +x-axis sample so interior probe nearest dist is
  exact d), d=sqrt(min*max)=0.7643 with min<d<max; admits() True under
  min_eta_max, dataclasses.replace(admission, eta_max=max_eta_max).admits
  False -> clean flip. corridor_half==_INTERLOBE_CORRIDOR_ETA_SCALE*min_eta_max
  (asserted !=*max). exclusion_rho leg: base=1.0+reach_max-coord_radius_min
  via _coordinate_radius_bounds, child_rho=0.5*(min+max) clears min inner
  edge, not max. Self-falsification: equal shells never flip (flip is pure
  fn of shell size); feeding max_eta_max to _saddle_lobe_admissions BOTH
  widens corridor to max AND excludes witness (reverted-code double
  signature). Diagnostic output/saddle_lobe_edge_shell_witness.png. Sibling
  max_tube_arcs/exclusion_rho premises in test_lensing_tiling_census.py,
  test_lensing_caustic_cusps.py, test_lensing_surrogate_training.py are
  OWNED BY OTHER RUNS + use own eta_max fixtures on unchanged signatures ->
  not edited (scope discipline).

- 2026-08-14 (F081 saddle tube fundamental-arc trim, test_lensing_tube_d2_fold.py):
  PORTED + EXTENDED (10->18 tests, 5.8s, engine-free). RETIRED
  `max_tube_arcs` knob: `_tube_training_arcs` now 2-arg `(structure,parity)`.
  Replaced `test_saddle_reconsumes_max_tube_arcs_slice` (dead arcs[:1]/[:20]
  premise) and the astroid knob-loop; added `test_astroid_rejects_the_third_
  positional_argument` (3-arg -> TypeError, proves knob path gone). NEW
  Pin 6 SaddleOrbitPartitionSelectionTestCase + SelfFalsification: saddle
  band (1.1,1.15) -> 6 arcs mids [-0.3065,0.3065,0.0,2.8351,3.4481,3.1416]
  -> 2 reps [-0.3065,0.0]. Orbit count COMPUTED via independent union-find
  oracle (`_independent_orbit_labels`, my own `_circular_gap`=abs((a-b+pi)%
  2pi-pi), NOT production `_circular_angular_distance`) = 2 == production;
  each arc D2-equiv to exactly one rep; reps pairwise non-equiv. Teeth:
  mock.patch surrogate_training_module._circular_angular_distance -> 1e9
  => 6 reps (identity trim). NEW Pin 7 SaddleServeCoveragePreservation +
  SelfFalsification: build minimal TubeCharts per arc (theta_grid spans
  arc; envelope arbitrary — `_tube_theta_inframe` only reads frame ends),
  sweep 720-pt ring [0,2pi), fundamental served-set SUPERSET of all-6
  incumbent (0 violations, both serve 126/720). Teeth: drop rep0->32,
  rep1->50 violations. Astroid band (0.35,0.45)->4 arcs->1 rep [0.137,1.478]
  brackets pi/4. Diagnostic plots saddle_orbit_partition.png /
  saddle_serve_coverage.png. SCOPE: test_lensing_tiling_census.py (Q1
  trained=len(rep.tube_arcs)) and test_lensing_caustic_cusps.py
  (arcs[:config.max_tube_arcs]) still hold dead knob premises but are OWNED
  BY OTHER RUNS — NOT edited; grep confirms only my file's intentional
  TypeError test uses the 3-arg call form.

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
