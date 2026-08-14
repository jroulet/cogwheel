# Test Dev Short-Term Observations

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
