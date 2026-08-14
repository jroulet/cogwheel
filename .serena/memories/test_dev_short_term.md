# Test Dev Short-Term Observations

## 2026-08-14 Born reachability-lift reachability suite (test_lensing_born_analytic_reachability.py, NEW, 14 tests green 4.2s)

- Pins the LIFTED first-class Born rung `LensedRelativeBinningLikelihood._born_residual_analytic`
  reached via `_amplification_coefficients` (distinct from the BURIED `_surrogate_coefficients`
  Born slot already pinned by test_lensing_born_residual_wiring.py). Probe pattern: bind the REAL
  unbound methods (`_amplification_coefficients`, `_born_residual_analytic`, `_lens_params`,
  `_reduce_dense_kernels`, `_image_delays`, `_ppgo_band_split`, `_ppgo_cell_ceiling`,
  `_ppgo_cell_coords`) onto a minimal `_BornAnalyticProbe` carrying only the attrs those methods
  read (no engine/waveform). Anti-vacuity tearDown (`n_checks>0`), self-falsification class.
- LIFTED-vs-BURIED GATE DIFFERENCE: buried rung's Born slot serves at `rho > 1.0`; the lifted rung
  serves at `rho > 2.0` (AND kappa==0 AND beta==0 AND born_chart.covers). Fixture: gamma=0.5
  astroid, kappa=0/beta=0, rho=3.0 (abs_y derived via caustic_geometry reach, NOT a pinned literal).
- GENUINE PRODUCTION DEFECT FOUND + PINNED AS RED-WHEN-FIXED TRIPWIRE (must be FLAGGED to
  driver/Architect for a production fix): `_born_residual_analytic`'s band-split branch (fires when
  a certified ppGO map gives w_trust strictly inside the band) sub-slices the frequency grid
  (`partition_ns.w = chart_w`, 31 of 64 nodes) but hands FULL-length geometry
  (`saddle_kernels`/`delays`, 64) to `born_carrier_from_partition`. That call
  (likelihood.py:2362) runs its OWN internal reconstruct_farfield -> `_switched_setup`
  (_gauge.py:650) validates `saddle_kernels.shape==(w.size,4)` -> raises
  `ValueError: ...expected saddle_kernels of shape (31, 4)...got (64, 4)`. Confirmed against REAL
  geometry_partition + exact production arithmetic, and production-reachable (dispatcher has NO
  try/except around the Born call -> the raise propagates). The buried rung avoids it by serving
  the Born slot over FULL dense_w and masking the reconstructed ENVELOPE, never `w`.
- Because the raise makes the Architect's MAP BAND-SPLIT positive invariant (`max|k_split-k_nomap|>0`)
  impossible to serve, I encoded `test_band_split_serve_raises_shape_defect` (assertRaises ValueError,
  'saddle_kernels' substring, on BOTH direct rung and dispatcher) as a governance tripwire: green now,
  flips RED the instant production stops raising, at which point restore the positive invariant. Did
  NOT weaken to pass (asserts the CURRENT contract) and did NOT skip. Kept the two premise/negative
  MAP tests that DO hold (w_trust lands strictly inside band == max(1.5*20,22)=30; beyond-wall cell
  keeps whole band byte-identical).
- NULL-SPLIT IDENTITY holds byte-exactly (np.array_equal, max|A-B|==0.0): no-map serve ==
  w_trust>=w_hi-map serve == direct whole-band rung. band_split never fires there, so the defect is
  not on the null path. SERVE-PATH TRACE holds: dispatcher==direct rung byte-exact; two different
  charts differ (max_diff>1e-12); no-chart/kappa!=0/beta!=0/rho<=2 all return None (fall through).
- BACKWARD-COMPAT AUDIT (step 7) CLEAN: grepped all cogwheel/tests for `_born_residual_analytic`,
  `born_residual_chart`, `BornResidualChart` — only my new file + test_lensing_born_residual_wiring.py
  (buried rung, unaffected by the lift) reference them. Neighbors regress-clean:
  born_residual_wiring + ppgo_bandsplit = 96 passed, 4 skipped, 13.3s.
- SELF-FALSIFICATION teeth: corrupt one served k0 float (`corrupt.flat[0]+=1.0`) -> np.array_equal
  goes False while identical copy stays True (byte-equality oracle discriminates); two SAME-scale
  charts serve max_diff==0.0 (proves the >1e-12 SERVE-PATH threshold discriminates content);
  null-split identity with DIFFERENT chart scales A vs B breaks array_equal (identity not vacuous).


## 2026-08-14 Born auto-attach fact-7 re-points (3 files, all green)

- SCOPE = "ENUMERATED EXISTING TEST RE-POINTS (fact 7)": re-point 3 pre-existing
  classes to the WP-C/WP-B construction+serialization contract; NONE deleted.
- RE-POINT 1 test_lensing_surrogate.py DefaultSurrogatePathTestCase: default
  construction now AUTO-ATTACHES the shipped Born chart (cogwheel/data/
  born_residual_chart.npz EXISTS) -> added test_default_construction_auto_attaches
  _born_chart asserting `born_residual_chart is not None` AND
  `_born_residual_chart_is_default is True`. test_default_surrogate_attribute_is_none
  STILL HOLDS (WP-C attaches the CHART, not the amplification_surrogate — surrogate
  stays None). Softened "exact path" -> "default path" wording. 3 passed isolation.
- STEP-7 AUDIT SURFACED ORACLE CONTAMINATION (same file, applied the fix): the two
  LnlikeAccuracyTestCase `cls.exact` constructions (now lines ~1651 and ~1938) had
  NO born_residual_chart arg -> WP-C default silently auto-attaches, so the accuracy
  ORACLE stopped being engine-pure. Added explicit `born_residual_chart=None`+comment
  to BOTH (replace_in_files literal, expected_count=2). This was flagged in prior
  shards as "owned by that file's owner"; I own the file this run, my re-point-1
  docstring asserts the oracle is engine-pure, so the fix is entailed. Whole file
  128 passed (88s).
- RE-POINT 2 test_lensing_born_residual_wiring.py NoChartByteIdentityTestCase: bound
  the LIFTED rung `_born_residual_analytic` onto `_BornResidualProbe` (was only the
  buried `_surrogate_coefficients`). Added test_lifted_born_intercept_falls_through
  _without_chart — TEETH via POISON lens battery {None -> TypeError, {} -> KeyError,
  raising-mapping -> RuntimeError}: method's first 2 lines are
  `born_chart=self.born_residual_chart; if born_chart is None: return None` BEFORE
  reading lens[...], so a clean None return with a poison lens proves the chart guard
  short-circuits first (not an incidental crash / upstream guard). Each poison fails
  differently so no single swallow makes all 3 pass. 4 passed isolation, 35 whole file.
- RE-POINT 3 test_lensing_ppgo_bandsplit.py _PpgoTestCase.setUp: added
  `self.addCleanup(set_certified_ppgo_map, get_certified_ppgo_map())` — captures
  INCOMING process-global map, restores LIFO after tearDown even if anti-vacuity
  assert raises. Fixes the F078 reset-to-None-clobbers-outer-map leak (subclasses did
  set_certified_ppgo_map(None), which CLOBBERS rather than RESTORES). All subclasses
  already call super().setUp() (comparisons attr dependency) so inheritance is safe.
  Added GlobalMapSaveRestoreDisciplineTestCase (2 tests) with TEETH: run a throwaway
  _PpgoTestCase subclass that installs a DIFFERENT synthetic map and never resets,
  through a real unittest.TestResult while an OUTER map is installed, assertIs the
  global back to OUTER after; raising-inner variant proves restore survives a failing
  body. _synthetic_map(parity='positive'|'saddle', ...) — parity is a STRING not code.
  2 passed isolation, 64 passed/4 skipped whole file.
- CROSS-FILE (F078 diagnostic "run in-suite to catch leakage"): ppgo_bandsplit +
  born_residual_wiring + born_analytic_reachability together under DEFAULT xdist =
  130 passed 4 skipped, no cross-file map leak. born_analytic_reachability alone 31
  passed (unmodified, confirms the "shipped Born w-box declines on event band" fact
  my docstrings rely on).

## 2026-08-14 WP1 D2 serve-fold + astroid fundamental-arc reduction (test_lensing_tube_d2_fold.py, NEW, 14 tests green 4.1s)

- Suite pins WP1's three durable invariants against `surrogate._fold_caustic_theta`,
  `surrogate._tube_serves` (via `serve`), and `surrogate_training._tube_training_arcs`.
- SPEC-1 "np.array_equal across ALL FOUR octants" DOES NOT HOLD as written — measured:
  negation-only octant pair (+,+)/(+,-) folds to EXACTLY theta0 (negation is bit-exact
  in IEEE-754) so served arrays are BIT-IDENTICAL (np.array_equal, maxabs 0.0); but the
  pi-reflection octants (-,+)/(-,-) fold through `math.pi - theta`, a subtraction that
  rounds by <=1 ULP (~2.2e-16 rad), so served arrays differ ~2.5e-16 (saddle) to
  ~6.7e-16 (astroid). Split the assertion: `np.array_equal` for NEGATION_ONLY_OCTANTS,
  `np.allclose(rtol=1e-9,atol=1e-11)` + documented 1-ULP reason for PI_REFLECTION_OCTANTS.
  This matches the handoff acceptance ("exact bit-equality if fold precedes all arithmetic,
  else a stated near-machine bound with the reason"). Do NOT weaken to a blanket allclose —
  the negation pair genuinely IS bit-exact and that is the stronger teeth.
- OCTANT->PHYSICAL-THETA INVERSE FOLD (so all 4 octants fold back to the same fundamental
  theta0): (+,+)->theta0, (+,-)->-theta0, (-,+)->pi-theta0, (-,-)->pi+theta0. Pick theta0
  generic (astroid 0.6, saddle -0.24) — NON-cusp-window, NON-diagonal — and eta inside the
  tube band (QUERY_ETA=0.02 in [0.005,0.05]); eta is D2-invariant so it passes unfolded.
- SPEC-3 CAUSTIC-GAUGE CORRECTION (coder-confirmed): astroid cusps sit on the AXES
  {0,pi/2,pi,3pi/2} in caustic gauge, so the fundamental tube arc brackets pi/4, NOT the
  source-plane pi/2 (pi/2 is a CUSP). At gamma=0.4 the 4 detected arcs give exactly one
  bracketing pi/4 ([0.137,1.478]); 1.478<pi/2 so a pi/2 selector returns ZERO — asserted
  as an extra guard. Saddle path: `_tube_training_arcs(structure,-1)` returns ALL arcs
  unchanged (reduction is astroid-only).
- SELF-FALSIFICATION teeth: (Spec1) module-level `_fold_without_s2_branch` mutant dropping
  the y2_eig<0 leg, applied via `mock.patch.object(surrogate_module,'_fold_caustic_theta',...)`
  -> (+,-)/(-,-) octants fold to wrong theta, `_theta_into_frame` maps out of arc range,
  not served (zeros) -> diverge. (Spec2) `_swapped_fold` applying y2 reflection under y1
  sign breaks >=1 octant. (Spec3) SimpleNamespace(arcs=survivors) with pi/4-arc removed ->
  selector returns EMPTY (never picks a neighbor). Every self-falsification test still calls
  self._count() to satisfy the `_TubeD2TestCase` anti-vacuity tearDown.
- BACKWARD-COMPAT AUDIT (step 7) CLEAN: grepped ALL of cogwheel/tests for both WP1 symbols
  (`_tube_training_arcs`, `_fold_caustic_theta`) — ONLY the new file references either.
  `test_lensing_caustic_cusps.py` ASTROID_EXPECTED_ARCS=4 asserts `detect_caustic_structure`
  DETECTION count (unchanged by WP1; my own suite re-confirms 4 detected arcs pass); WP1's
  `_tube_training_arcs` is a downstream SELECTION (astroid 4->1 for tube TRAINING only) that
  does not touch detection. `test_lensing_surrogate_training.py` has NO max_tube_arcs
  assertion (only a docstring tube-name mention). No signature/constant/domain break.

## 2026-08-14 WP2 cusp-arm-coverage retirement (test_lensing_surrogate_training.py)

- Deleted the two dead D2 monkeypatch classes
  `CuspArmCoverageParityGateSelfFalsificationTestCase` +
  `CuspArmCoverageParityGateSelfFalsificationSelfFalsification` (former
  lines 5995-6252) and the `_WP2_CUSP_*` fixture block (former 3615-3625).
  Both classes set/read `surrogate_module._CUSP_ARM_COVERAGE` /
  `_SADDLE_CUSP_ARM_COVERAGE`, which WP2 DELETED from cogwheel/lensing/
  surrogate.py (grep-clean confirmed) -> they would AttributeError at
  setUp/attr-read. Pure deletion is correct here: the parity-gated cusp
  shrink they guarded no longer exists (F079/WP2), and the surviving
  `_tube_serves` full-window contract is covered by OTHER suites
  (test_lensing_surrogate.py TubeCuspWindowExclusion*, per prior memory).
- Regex block-delete pattern (two occurrences of the restore line inside
  the block): anchored the needle from the class header via DOTALL
  non-greedy to a captured (# dashes / # SHARD D:) trailer and re-emitted
  only that captured trailer. Anchoring to the trailing SHARD-D comment
  avoids matching the first restore-line occurrence.
- Saddle regression witnesses stay BYTE-PASSING unmodified (proves saddle
  path unchanged): StableGammaBandsF041TestCase arc-COUNT-equal-across-gamma
  assertion (~2046, `_F041_LABEL_GAMMAS`), WedgeCoverageNoShrinkTestCase +
  `_WP1_GOLDEN_STRUCTURE`/`_WP1_GOLDEN_SPAN` (line 2174/2182/2360),
  AstroidByteIdentityTestCase, SaddleThetaToUMutation* (D1 neighbor).
- Full file green: 103 passed, 77 skipped, 0 failed in 823s (13m43s;
  PRE-EXISTING slow tail = engine-backed SaddleExteriorHeldoutOracle +
  SaddleCuspUCoordinateRoundTrip, TRAIN_TIER skips; my change only removed
  tests). WORKTREE cwd is /home/tejaswi/Work/cogwheel-claude-dev (NOT
  cogwheel/); Serena shell defaults there.

## 2026-08-14 cusp-arm coverage-shrink retirement (test_lensing_surrogate.py)

- WP2 deleted `_CUSP_ARM_COVERAGE`/`_SADDLE_CUSP_ARM_COVERAGE` from
  cogwheel/lensing/surrogate.py (grep-clean confirmed) and `_tube_serves`
  now uses `residual = delta_theta` (FULL cusp window, no parity dispatch,
  no shrink). Deleted the two coverage-shrink classes
  `TubeCuspWindowParityGatingTestCase` (+ its
  `test_parity_gating_cusp_window.png` plot test) and
  `TubeCuspWindowParityGatingSelfFalsificationTestCase` (former lines
  5056-5302). File is now grep-clean of both retired identifiers.
- Pure deletion would have STRANDED the "inside cusp window -> refuse"
  branch: surviving `_tube_serves` tests (overlap_band / dispatch
  determinism) only drive SERVED queries, none inside a cusp window. Added
  a parity-free replacement `TubeCuspWindowExclusionTestCase` (3 tests) +
  `TubeCuspWindowExclusionSelfFalsificationTestCase` (1) that certify the
  surviving full-window contract: in-window (dist<delta_theta) refuses,
  beyond admits, flip at exactly delta_theta (strict `<`). Fixtures DERIVED
  from `chart.cusp_windows[0]` on `_multichart_fixture().charts[0]`
  (parity+1 tube, window (0.2,0.1), theta_grid [0.2,1.2]); gamma=0.35,
  eta=0.01 clear box/eta/u/image-count gates so the cusp loop is reached.
- TEETH via `dataclasses.replace(tube, cusp_windows=())` NOT
  `mock.patch.object`: `TubeChart` is `@dataclass(frozen=True, eq=False)`,
  so setattr on the instance raises FrozenInstanceError; replace re-invokes
  __init__ with precomputed arrays + empty windows cleanly. Empty windows
  -> the in-window query now serves (proves the exclusion loop is
  load-bearing, not an unrelated gate).
- Full file green: 127 passed in 87.68s. New exclusion suite alone 6 passed
  in 3.8s.
- SIBLING BREAKAGE (out of my scope, FLAGGED not fixed): WP2's constant
  deletion still leaves `test_lensing_surrogate_training.py` referencing
  `_CUSP_ARM_COVERAGE`/`_SADDLE_CUSP_ARM_COVERAGE` (monkey-patch/restore
  classes ~6002-6252 per prior memory) which will AttributeError at
  runtime. Owned by another Test Dev run.

## 2026-08-14 F079 cusp-arm coverage constant removal (test_lensing_cusp_arm_coverage.py)

- DOCS-ONLY EDIT: WP2 deleted `_CUSP_ARM_COVERAGE`/`_SADDLE_CUSP_ARM_COVERAGE`
  from surrogate.py; my job was to purge both literal identifiers from
  test_lensing_cusp_arm_coverage.py (docstring line 8 + RETIRED block) and
  add an F079 retirement record. All refs were PROSE (docstring/comments) —
  no live assertion in this file ever touched the constant, so 6/6 pass
  unchanged (11.2s). Grep-clean confirmed: 0 tokens of either identifier
  remain in the file.
- F079 RETIREMENT GROUNDING (read `_tube_serves` at surrogate.py:2848-2870):
  production now excludes the tube over the FULL cusp window (residual =
  delta_theta, no angular subtraction; inline comment says "no certified
  angular serve boundary to subtract post-F074"). Two reasons the constant
  was dead: (a) WRONG UNITS — subtracted an image-plane polar offset from a
  critical-curve parameter-angle window half-width; (b) 0/64 production
  windows changed serve/refuse decision post-F074 (eta-floor + w-floor gates
  already decide every cusp query).
- BACKWARD-COMPAT AUDIT FLAG (owned by OTHER Test Dev runs, NOT fixed):
  surrogate.py + all chang_refsdal production modules are grep-clean of both
  constants (deleted). BUT two sibling suites still reference
  `surrogate_module._CUSP_ARM_COVERAGE`/`_SADDLE_CUSP_ARM_COVERAGE` and will
  raise AttributeError at runtime: test_lensing_surrogate.py (lines ~5059-
  5296: D2a/D2b classes, direct attr reads + mock.patch.object) and
  test_lensing_surrogate_training.py (lines ~6002-6252: monkey-patch/restore
  swap classes). These must be retired/rewritten by whoever owns those files.

## 2026-08-14 Born reachability shard 3: auto-attach fallback + JSON round-trip (test_lensing_born_analytic_reachability.py, +12 tests, file 31 green 22.6s)

- EXTENDED the 19-test suite (did NOT rewrite): +AutoAttachFallbackToNoneTestCase (3),
  +JsonRoundTripBornChartTestCase (6, RB + marginalized), +ReachabilityFallbackSelfFalsification
  TestCase (3). These are the FIRST tests in the suite that build a REAL LensedRelativeBinning
  Likelihood/LensedMarginalizedExtrinsicLikelihood (earlier shards used the engine-free
  _BornAnalyticProbe). Neighbors clean: born_residual_wiring+ppgo_bandsplit 96 passed 4 skipped.
- SENTINEL CONTRACT (`_AUTO_BORN_CHART = object()` in likelihood.py): 3-way construction intent —
  argument-omitted sentinel (auto-load shipped artifact, REFUSE-TO-NONE on OSError/ValueError/
  KeyError with RuntimeWarning 'Born-residual chart unavailable', mirrors use_certified_ppgo_map)
  vs explicit None (pure-engine opt-out) vs explicit chart. RB records
  `self._born_residual_chart_is_default = (arg is _AUTO_BORN_CHART)`. Marginalized class stores the
  raw sentinel/None/chart VERBATIM and forwards to inner `_engine` (single auto-load lives there).
- Spec-1 fallback teeth: patch `mock.patch.object(BornResidualChart,'load',side_effect=OSError)` ->
  construction SUCCEEDS, born_residual_chart is None, is_default flag PRESERVED, warning emitted;
  fallback `_amplification_coefficients` serve is np.array_equal to explicit-None serve (Born rung
  is the ONLY chart-dependent branch). subTest over all 3 caught exception types.
- Spec-2 JSON round-trip house idiom: `obj.to_json(tmp, overwrite=True)` + `utils.read_json(tmp)`
  (dir arg, NOT literal json.dumps). get_init_dict 3-way: default -> POP key (re-defaults +
  re-auto-loads on reconstruct, serves byte-identically); explicit None -> emit None verbatim;
  in-memory chart -> NotImplementedError, assertion `'source path' in str(e)`. Round-trip
  serve-identity teeth is DETERMINISTIC-MATCH (shipped Born w-box [5,60] can't fire in the event
  band [15,1024]) -> re-attach teeth = key-omission + reconstructed is_default True + not None.
- MARGINALIZED None/in-memory branches EXERCISED ENGINE-FREE (avoid a 2nd ~13s build): real
  override run on `object.__new__(LensedMarginalizedExtrinsicLikelihood)` stub with
  `mock.patch.object(MarginalizedExtrinsicLikelihood,'get_init_dict',_fake_base)`; zero-arg super()
  resolves via MRO which REQUIRES a genuine LM instance (hence object.__new__, not bare object).
  Default marginalized build is one functools.lru_cache(maxsize=1) ~12.6s build shared across tests.
- Fixtures: event `data.EventData.gaussian_noise(...HLV, seed=20260717)` + inject IMRPhenomXPHM +
  WaveformGenerator.from_event_data + uniform _DF_BIN=4Hz fbin; _CBC_PAR_DIC (18 keys, m1=60/m2=45),
  _MAIN_LENS (kappa=0/beta=0). Measured: RB build ~0.08s, RB serve fast, marg build ~12.6s.
- SELF-FALSIFICATION teeth for the NEW invariants: (a) fallback serve k0.size>0 AND a 1.0 perturb
  flips array_equal False (equality is over non-trivial content, not two empty arrays); (b) the
  3-way branch keys on DISTINCT objects — `_AUTO_BORN_CHART is not None`, in-memory chart is neither
  (else default/None indistinguishable); (c) the init-dict key CAN appear (explicit-None emits it)
  so the default's assertNotIn is a real branch decision not a never-present key.
- SPEC-3 FLAG (owned by another run, NOT edited): confirmed real — test_lensing_surrogate.py
  LnlikeAccuracyTestCase.setUpClass `cls.exact` (lines 1901-1903, + 1354/1611/1614/1715/1824/1895/
  1898) constructs LensedRelativeBinningLikelihood WITHOUT born_residual_chart=None, so it now takes
  the auto-attach default; the Architect's Spec-3 one-line re-point (add born_residual_chart=None +
  inline comment) keeps the engine-reference oracle engine-pure. Must be applied by that file's owner.
- BACKWARD-COMPAT AUDIT (step 7) CLEAN: grepped all cogwheel/tests for
  `_AUTO_BORN_CHART|born_residual_chart|_born_residual_chart_is_default`. Only my suite +
  test_lensing_born_residual_wiring.py (local _BornResidualProbe(born_residual_chart=None) default,
  never touches the sentinel or the RB constructor — passed in the 96) reference the symbol. No
  sibling constructs either likelihood with a born_residual_chart arg, so the sentinel-default
  constructor change breaks nobody except the Spec-3 oracle above.

## 2026-08-14 Born reachability shard 2: byte-identity battery + loader refusal (test_lensing_born_analytic_reachability.py, +5 tests, file 19 green 4.2s)

- EXTENDED the existing 3-shard suite (did NOT rewrite): added
  ByteIdentityBatteryTestCase (2 tests) + BornResidualChartLoaderRefusalTestCase
  (3 tests). Neighbors clean: born_residual_wiring + ppgo_bandsplit = 96 passed,
  4 skipped, 12.9s.
- SPEC-1 (byte-identity battery) FAST-TIER SUBSTITUTION: the spec asks
  serve-with-chart == serve-with-chart=None (np.array_equal) for 5 gate-miss
  draws, but the DOWNSTREAM route for gate-miss draws needs the exact seed
  engine the probe cannot supply. Substituted the engine-free DECISIVE form:
  the Born rung `_born_residual_analytic` is the ONLY chart-dependent branch of
  `_amplification_coefficients`, so a chart-attached None PROVES every
  downstream float64 input is identical to the no-chart route. Battery
  (label,gamma,rho,kappa,beta), rho->|y| via LIVE caustic_geometry not literals:
  (a) interior rho=0.5, (b) exterior rho=1.5 (<2 gate), (c) covers=False rho=6.0
  (>rho_grid max 5.0), (d) saddle gamma=1.3 (covers=False, gamma_grid max 0.8),
  (e) kappa=0.1 AND beta=0.1. Per-draw boolean table asserted complete + all
  True. ISOLATION test proves the kappa/beta veto is the gate: reference cell
  (kappa=beta=0, rho=3) SERVES (served_ref True premise) so the flip-only-
  kappa/beta declines are non-vacuous — the silent-accuracy bug the gate blocks.
- SPEC-2/3 (loader refusal) — engine-free, write npz manually via np.savez
  (no save() method on BornResidualChart). Import `_SCHEMA,_content_hash` from
  born_residual_chart. Corrupt-hash: one-ULP flip real_coeffs.flat[0] via
  np.nextafter but store ORIGINAL hash (+ inline assertNotEqual that the flip
  actually changed the hash, else fixture inert) -> load raises ValueError with
  'train_born_residual.py' AND 'hash'. Schema: parametrize {missing key,
  wrong string 'born_residual_v0'} -> both raise naming regen script + 'schema'.
  POSITIVE CONTROL (teeth): valid npz round-trips byte-for-byte (proves writer
  produces a loadable artifact, so refusals are the corruption not a bad
  fixture). tmpdir via tempfile.mkdtemp + addCleanup(shutil.rmtree,path,True).
- "explicit-path construction propagates loudly" satisfied by load(path)
  raising (constructor evaluates BornResidualChart.load before build); the
  auto-attach swallow-to-None-with-warning is a SEPARATELY-OWNED description
  (needs full likelihood build, too heavy for fast tier) — documented in the
  class docstring, not re-tested here.
- BACKWARD-COMPAT AUDIT (step 7) CLEAN: grepped all cogwheel/tests for
  `BornResidualChart.load|born_residual_v1|_content_hash|train_born_residual`.
  Only my suite references BornResidualChart.load. The `_content_hash` hit in
  test_lensing_ppgo_bandsplit.py is imported from ppgo_map (with _hash_scalars,
  _SCHEMA_VERSION) — DIFFERENT module, no collision with the born loader.
- All 5 new tests call self._count() (the `_BornReachTestCase` anti-vacuity
  tearDown fails on 0 comparisons — including in the battery subTest loop).

## 2026-08-14 caustic_cusps_wrap_fix (test_lensing_caustic_cusps.py)

- WP1 `_find_cusps` wrap fix: astroid theta=0 cusp window was inflated by the
  linear span across a PERIODIC index walk (~1.5*pi -> half-width ~4.5 rad),
  swallowing the two fold arcs adjacent to theta=0, so
  `detect_caustic_structure(gamma,+1)` shipped only 2 arcs while still
  reporting 4 cusps. `detect_caustic_structure` now cross-checks surviving-arc
  count vs `_EXPECTED_ARCS={1:4,-1:6}` and raises `CausticTopologyError`
  (message contains 'fold arc') on mismatch.
- ARCHITECT SPEC PUSHBACK (encoded measured reality): Architect assumed the
  astroid is 4-FOLD symmetric in cusp windows (theta=0 delta ~0.11-0.13
  matching the median of the other three). MEASURED: 2-FOLD symmetry — the
  x-axis pair (theta=0, theta=pi) share a BIT-IDENTICAL window; the y-axis
  cusps (pi/2, 3pi/2) FLOOR to `_CUSP_MIN_HALFWIDTH`=0.05 at gamma>=0.4.
  Healthy theta=0 half-width measured 0.094 (gamma 0.2/0.4), 0.141 (0.7),
  0.236 (0.9) — NOT 0.11-0.13. Wrote the invariant that actually holds:
  theta=0-window == theta=pi-partner within SAMPLING_STEP + not-an-outlier +
  sane ceiling 0.5, NOT the false 4-fold-equal median claim.
- Golden re-freeze (Spec C): GOLDEN_INWARD_SIGN astroid rows 2-tuples ->
  4-tuples `(-1,-1,-1,-1)` at gamma 0.2/0.4/0.7/0.9; signs DERIVED from
  geometry (sign(fold_dir . serve_normal)==-1, worst |dot| 0.298 at gamma=0.2)
  AND independent 4-real-image census, not copied-to-pass. Saddle 6-tuples
  unchanged.
- Topology-guard teeth without an engine campaign:
  `unittest_mock.patch.dict(st._EXPECTED_ARCS, {1: 3})` (auto-restores) makes
  the 4-arc astroid raise CausticTopologyError while cusp count still matches.
  Needed to add `from unittest import mock as unittest_mock` (file only had
  `from unittest import TestCase, main, skip, skipUnless`).
- ANTI-VACUITY BIT ME: the self-falsification test asserted but forgot
  `self._count()`, so the `_CuspTestCase.tearDown` anti-vacuity guard failed
  with '0 not greater than 0'. Every test in a `_CuspTestCase` subclass must
  call `self._count()` at least once — including self-falsification tests.
- WP2 BACKWARD-COMPAT (FLAGGED, out of my scope): WP2 removed
  `_CUSP_ARM_COVERAGE`/`_SADDLE_CUSP_ARM_COVERAGE` from surrogate.py. My
  imported module surrogate_training.py is CLEAN of them (my suite passed
  fully, 49/49). But THREE sibling suites still reference the deleted
  constants and will break: test_lensing_surrogate.py (D2a/D2b @ ~5058-5295),
  test_lensing_surrogate_training.py (~6001-6251), test_lensing_cusp_arm_
  coverage.py (docstring/comments only, may still run). Owned by other Test
  Dev runs — reported, not fixed.
- RUNTIME: full test_lensing_caustic_cusps.py = 49 passed in 413s (6:53),
  OVER the 5-min fast-tier file ceiling — but PRE-EXISTING (dominated by
  UniversalFMaxTestCase + DiagnosticPlotTestCase eps/universality chart-build
  classes). My 6 new tests are ~3-4s each. Serena shell caps at 240s: run the
  full file via `nohup ... > /tmp/log 2>&1 &` then poll with `pgrep` (Bash
  allowed) + `sleep` via Serena; `-q` buffers output until the very end.

</content>
