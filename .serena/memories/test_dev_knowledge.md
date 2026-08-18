- Smoke/staging fixture design: use named tuple constants for every
  threshold/seed; derive fixture grids from those constants so rescaling
  doesn't scatter magic numbers. A "shared generator" pattern (one
  setUp builds a heavy chart then all tests in a class share it) is the
  standard approach for expensive builds; "smoke=True" switch is the
  standard way to gate expensive sub-steps.
- Pin tests anchor produced values against a single stored reference —
  they are sensitive to layout/method renames but NOT to refactors that
  preserve the computation. Remove or update them proactively when
  renaming code paths (stale pins -> silent false greens or loud false
  reds).
- Reachable-red tests must call the shipping code path, not a
  reconstructed version (Build 8h-b5); a guard isolated by fixing an
  unrelated parameter at its trivial-satisfy point ensures that guard
  alone trips.
- Never mix "self-oracle" (same-function) and independent-oracle
  assertions within the same test; mark them distinct test methods.
- Smoke-scale (4×4×4 grid) eps for tube charts is ~0.4-0.45
  (interpolation sparsity); the production bar <0.05 applies to
  12×8×12 or denser grids — never gate production accuracy against a
  smoke-fixture measurement.
- For a GATE-CONTRACT SWAP build (old TrainingConfig field moved to
  explicit function arg), the Test Dev PORT pattern is:
  (1) add module-level constants for the old field's values (e.g.
  `_ETA_MAX = 0.05`, `_ETA_FLOOR = 0.02`), (2) remove the dead field
  from any TrainingConfig() constructor calls in the test, (3) pass the
  constants as explicit kwargs to every affected function call site,
  (4) rename any pin tests referencing the old config field to reference
  the new constant. One file at a time; verify each to completion before
  the next.
- A production assertion (`assert f_max < 0.5`) that replaces a
  skip-guard (`if eta_max > ...`) means: no chart is ever skipped; tests
  must delete the old guard-fires test and add both a universality test
  (ratio max/min < 10 across bands) and an assertion-fires test
  (f_max=0.55). Add self-falsification helpers confirming each has teeth.
- cogwheel lensing gotchas: ChangRefsdalChannels needs a >=2-pt strictly-
  increasing positive w grid (no scalar fixtures); _lens_dic has beta as the
  4th positional (pass lens params by keyword). Mass-sheet twin lnL invariance
  needs t_geo_twin = t_c - dt_ms - xi*(t_min_B - t_min_A)/2pi (read t_min from
  a throwaway eval). Unlensed-injection near-truth reference = LIGHTEST lens,
  source OFF the caustic centre (y=(0,0) -> -inf). Census: saddle(det<0)
  signed=-2, positive(det>0) signed=0. `ChangRefsdalChannels(w).reset()`
  mutates IN-PLACE and returns None (NOT chainable) — call `ch = Chang
  RefsdalChannels(w); ch.reset(); ch.evaluate(gamma=, y=, beta=, kappa=)` (or
  `ch.geometry_partition(...)`) as separate statements, never chained.
- Neighbor-suite reds from drift: report, don't touch. Fully revert probe/
  mutation edits (verify by read-back + pattern search). Shell gate: plainest
  command shape (`python -m pytest <file> -q`) from the WORKTREE root; retry a
  bare denial once, a reasoned denial binds. Heavy lensing suites run together
  -> MemoryError; run one file at a time.
- A test-only change (no production edits) cannot regress an unrelated
  slow/heavy suite that doesn't import the touched symbols — verify via a
  grep for zero imports/references, then skip running it.
- For self-contained HEAD functions (no module-level state), AST-extract just
  the FunctionDef (`ast.get_source_segment`) and exec in a minimal namespace.
- To fixture near-singular/underflow behavior of a quantity LINEAR in its
  inputs, solve for a nullspace combination rather than hand-tuning.
- Before using a config as a dispatch/ladder probe fixture, verify it's
  actually served by the TARGET arm and not preempted by a higher-priority
  arm — check which arm served bit-for-bit.
- TOOLING: huge test files break the built-in Edit/Read tools — use Serena
  `replace_content` (relative path) + `read_file` ranges +
  `get_symbols_overview`. `tests/output/` is hook-blocked from `list_dir`/
  `find_file` — verify generated plots via `pathlib.glob` in the conda env.
  The bash hook also blocks `cat >>` (the word "cat"): append via Serena
  `insert_at_line` (0-based; insert at line == linecount to append). Serena
  `replace_content` can also intermittently reject a large multi-line repl
  ("Field required repl missing" pydantic error) on identical args — just
  retry as-is, don't reformat/shrink the repl; it can succeed verbatim on
  the next attempt.
- SDK caps inlined short-term memories at 24KB (tail-kept); earlier entries
  survive only in git history, not the prompt.
- When production adds a new REQUIRED positional field to a serialized-
  artifact constructor, update every test helper that rebuilds/re-saves that
  artifact with ALL fields — otherwise load() raises KeyError before reaching
  the intended validation error, silently invalidating the premise.
- To unit-test a method refactored out of a free function into an instance
  method, bind the REAL methods onto a lightweight stateless probe class
  (class attrs) and call as instance methods — preserves `self` dispatch.
- When picking the worst-case band edge for a coverage/admission fixture,
  check which direction is actually worst: exterior admission truth-sets are
  worst at the band edge with the LARGEST caustic reach, interior at the
  SMALLEST — they can point opposite ways (Build 8h-b6).
- To isolate one gate's teeth when tightening a threshold also reshapes an
  upstream derived quantity, re-run the gate function directly on the SAME
  fixed tile/sample set with only the threshold changed, rather than
  regenerating the fixture.
- Content-stable digest for a saved .npz: raw-byte hashing is flaky (zip
  member timestamps aren't reproducible) — hash sha256 over the LOADED
  arrays sorted by (name,dtype,shape,tobytes) instead; guard the digest
  helper itself with a save-twice-matches test.
- Golden/history-free regression recipe: build the frozen fixture once via
  a THROWAWAY generator script, print frozen bits as `float.hex()` literals
  (or a content digest), bake the literals into the test, then DELETE the
  generator — no HEAD import, no self-oracle, immune to future drift.
- Before assuming interpolation/new code is the runtime cost, profile:
  a coarse test grid can be slow because each cell re-invokes an expensive
  shared primitive (e.g. a caustic-reach sweep) that has nothing to do with
  what's under test — reduce grid density (keep odd counts to preserve an
  exact symmetry point) rather than optimizing the wrong function.
- Guard-bypass pattern for legacy fixtures broken by a new production guard:
  mock.patch the guard for the legacy path, but pair it with a reachable-red
  test that calls the SAME entry point unpatched and asserts the guard fires
  for real; the guard's own dedicated teeth-test lives in its owning suite.
- Don't trust a brief's stated sign/direction for a fix — measure it. A
  brief can claim "move outward/harder" while the code's own inline comment
  and the measured values show the opposite; encode the invariant that
  matches MEASURED behavior (e.g. "never easier than before") and flag the
  brief's direction error separately, rather than encoding the brief's claim.
- TWO-STAGE oracle for a numerically-differentiated closed form: STAGE-1
  validate the TRANSCRIBED curve itself against the shipping function's own
  output (tight tolerance, e.g. <=1e-13) BEFORE STAGE-2 high-precision
  differentiation of that curve — a single-stage oracle that only
  differentiates can hide a wrong transcribed curve (a wrong sign/constant
  survived a full round this way, F038).
- Served-values insensitivity gate: perturb the analytic input by measured
  physical increments and assert the served output moves less than the
  accuracy bar; if the perturbation instead pushes the config out of the
  served region into refusal (no "served-but-over-bar" window exists), the
  gate's teeth become "still served (not None)", and value-sensitivity must
  be demonstrated separately via a deliberately-wrong-value fixture.
- mpmath dps cross-check discriminator: increasing oracle precision (e.g.
  40dps -> 60dps) should NOT necessarily shrink a correct implementation's
  residual once the residual is float64-limited (oracle precision >> float64)
  — a dps-INVARIANT residual confirms the closed form is right, not wrong;
  the pass/fail discriminator is residual MAGNITUDE against the mixed
  atol/rtol gate, never "shrinks with dps".
- MEASURE-ZERO EDGE STRADDLE: before encoding a brief's demand for
  unconditional behavior (e.g. "always raises") at an exact analytic
  boundary, measure BOTH sides numerically first — float round-off can
  straddle the boundary sign (e.g. a discriminant lands <=0 on one side,
  ~+1e-15 on the other), making the unconditional claim false on one side.
  Encode the honest disjunction the data actually supports (e.g. raises OR
  the quantity exceeds a divergence floor) and flag the brief's/docstring's
  unconditional claim as aspirational rather than silently picking a side
  (Build 1d wedge edge).
- PORTING TESTS TO A NEW COORD SYSTEM: tests that exercise behaviors tied
  to the OLD coordinate system (e.g. rho=1-on-caustic normalization, a
  gamma=1.0 refused-point-in-chart assumption) are UNPORTABLE — remove
  them rather than weakening. The surviving tests should explicitly certify
  the STILL-EXISTING scalar path. Removal is the correct outcome; note it
  with a RETIRED comment above the dead constants.
- When a closed-form improvement (e.g. 720-scan -> analytic) doesn't change
  served values at shipped gammas (scan already converged), the
  insensitivity gate's TEETH come from demonstrating the improvement at
  UN-SHIPPED near-wall gammas where the scan was wrong — assert the closed
  form is closer to the dense-scan oracle there.
- A SPEC.md literal can be TRUNCATED with error >> 1e-9 (e.g. 5.692100 vs
  exact 5.692099788303083, error 2.1e-7) — gate tight tests on the EXACT
  closed form (1e-9); use the SPEC literal only for a loose straddle check
  (1e-6) that the tolerance margin dwarfs the truncation error.
- InteriorWedgeChart TEST PATTERNS (test_lensing_interior_wedge_chart.py,
  40 tests, 8 classes): (1) CoordinateRoundTrip: verify _to/_from_wedge_
  fixed inverse to ~1e-15 AND D2 symmetry (all 4 quadrants identical).
  (2) NpzRoundTrip: 14 fields max|diff|=0.0 + spline evaluation bitwise.
  (3) WedgeServesGuard: 1 accept + 8 independent refusal gates. (4)
  SelectChartDispatch: select returns wedge + _evaluate_chart finite +
  D2 reflected identical + wrong image_count refuses. (5) Carrier
  ContinuityGate: smooth passes, jump raises, NaN no false flip, engine
  tile passes. (6) EnvelopeAccuracy: nodes succeeded count + spline
  reproduces nodes to 1e-10 + fresh engine matches. KEY FINDINGS:
  (a) _validate_axis requires >= 4 nodes per axis (2-node grids infeasible).
  (b) _WedgeCausticMap gamma_nodes MUST equal chart.gamma_grid exactly.
  (c) envelope_definition tag must be 'interior_sacr_c_envelope' for NPZ
  round-trip (validated against KNOWN_INTERIOR_DEFINITIONS).
- fold_ppgo_correction TEST PATTERNS (test_lensing_fold_ppgo_correction.py,
  23 tests, 7 classes): (1) MonotoneImprovement: use w=5..15 where fold
  divergence dominates (NOT w>25 where diffractive error < Airy residual).
  (2) LargeXiNoOp: rho=3.5 triggers structural fallback (b3=0) — test as
  byte-identical. (3) AxisAngleCorrection: 4-40% oscillation (carrier
  phase interference). (4) UniformErrorEstimateRelaxation: xi=0 returns
  0.0 (exact on fold), xi<0 returns None. (5) FallbackIdentity: scalar vs
  array paths differ by ~1 ULP (5e-17j) due to FP reduction order — assert
  byte-identity against the ARRAY path (which is what the internal
  _fallback() computes). (6) Schwinger ceiling at w=60 prevents using
  ChangRefsdalChannels as oracle for w>=100 near the caustic.
- ppGO EXTRAPOLATION TEST PATTERNS (test_lensing_extrapolate_floor.py,
  10+4 tests): spec's geomspace(1,60,24) with bar=1e-4 hits BOTH R² guard
  and MAX_RATIO guard (beat aliasing) — use geomspace(10,2000,50) for clean
  power-law fit (R²=0.91, ratio=0.53). Engine-backed tests gated by
  COGWHEEL_TRAIN_TIER. Bandsplit test uses stubbed error=0.0 so
  extrapolation never triggers (backward compat preserved).
- INTERIOR_W_NODES_PER_DECADE TESTING: the Architect spec's falsification
  claim ("WNPD=6 fails the 0.05 bar at gamma=0.65") is FALSE at the
  existing smoke geometry — measured eps is 0.0002 even at WNPD=6 because
  SACR-C envelope is dominated by spatial smoothness. Replace with a wiring
  test + node-count test proving the field is load-bearing (different node
  counts) and correctly wired.
- The s(theta(s)) np.interp round-trip is ~0 for ANY monotone table — its
  teeth come from the strictly-increasing/endpoint assertions PLUS a
  MISMATCHED-row round trip (forward uses s*1.05, inverse uses s) which
  yields a detectable error of ~0.05; use the mismatched-row test to give
  the bound reachable-red teeth.
- Positive control for coordinate accuracy: fit a chart at raw theta and
  assert the served relative error >> 0.20 (large) — this proves the
  coordinate choice is load-bearing; keep it as a separate red-when-wrong
  companion to the arc-length accuracy (green-when-right) test.
- A production accuracy phenomenon (e.g. knife-edge F042 bound-shift
  sensitivity) may be unreproducible at smoke scale (7-node tile vs 12+
  nodes at a cusp-adjacent tile) — encode the MEASURED reality: assert the
  SWING (sqrtedge swing < 0.01 across ±0.01rad shifts, measured ~0.003)
  rather than claiming a dramatic sensitivity the smoke fixture cannot show.
- SkipTest guard for a git-show-based oracle: guard with `if not
  subprocess.run(['git','show','HEAD:file'],…).returncode==0` so the
  test degrades gracefully in detached/fresh checkouts.
- Parity-wall gate checklist for a closed-form reach function: (a) exact-
  point refusal at wall, (b) both nextafter(wall,±) finite, (c) wall
  tracks lam not hardcoded, (d) over-critical (lam<=0) refuses, (e) reach
  diverges monotone approaching wall from both sides, (f) scalar wrapper
  bit-identical to full function.
- A single draw from prior sits ~25-30 nats ABOVE lnL_marg (extrinsic
  Occam), so the consistency gate is a LOWER bound. Get in-support vectors
  under Fixed*Prior by sampling the unit cube until lnposterior is finite.
- Phase-loss: np.exp(1j*x) range-reduces accurately — float64 loss lives in
  the w*tau MULTIPLICATION; demos need irrational-scaled factors or synthetic
  inputs checked vs an independent oracle.
- GATE ORTHOGONALITY WITNESS: to prove two guards are NOT subsumed by each
  other, find a config where one fires and the other does NOT. For the ghost
  decay/separation pair this requires SADDLE parity (gamma>1): at positive
  parity (gamma<1), Im(tau_c)>=0.4 physically implies separation>=1.2
  (exhaustive scan), so no positive-parity config can pass the decay gate
  while failing the separation gate. The orthogonality witness must use a
  saddle-parity config (e.g. gamma=5.0). The gate function itself need not be
  parity-aware; the constraint is on the fixture.
- PROTECTIVE REFUSAL TEST PATTERN: to certify a gate that subtracts a
  term is genuinely protective (never makes things worse when admitted):
  build the full partition for a REFUSED config, bypass the gate to
  force-compute the subtracted term, and assert mean|residual_with_term|
  > mean|residual_without_term| — the gate refuses exactly the configs
  where the subtraction worsens the result.
- FREQUENCY-INDEPENDENT GATE SKEW TEST: to prove a gate cannot skew
  between train and serve, show the gate function contains no w: test
  that both a train-time w-grid and a serve-time w-grid give the SAME
  admit/refuse decision for the SAME geometric config; optionally confirm
  ghost values are bit-identical on shared w-points.
- FOLD-PPGO HANDOFF TEST PATTERNS (test_lensing_fold_ppgo_handoff.py):
  (1) Gate-refusal fixture: the xi < 4.0 refusal regime requires rho CLOSE
  to caustic (e.g. rho=0.7 at gamma=0.5 gives xi~2.15 at w=45) — near-axis
  theta alone does NOT force xi<4 at rho=0.3 (delta_tau stays 0.26-0.33
  for all angles). (2) Fine gate fixture: gamma=0.85, rho=0.5 near meta
  morphosis: xi~5.85 (coarse admits), error~0.066 >> 1e-4 (fine refuses;
  margin ~660x). (3) DefaultPathUnaffectedTestCase: mock_chart must be IN
  surrogate.charts for _chart_index identity match — select_chart returning
  a chart takes priority over fold block. (4) CensusRecordsPpgoFold: use
  M_lens=20e6 Msun (w_min~49500, xi~531, error~8.2e-5 < 1e-4) with patches
  select_chart→None + get_certified_ppgo_map→None; use REAL geometry_partition.
  Backward-compat: existing test fixtures with w < 20 have xi < 4.0 even
  at rho=0.3, so the coarse gate refuses and falls through — no regression.
- REACHABLE-RED SUBDIVIDER MOCK PATTERN (Build subdivision_recursion_and_
  coordinate_cleanup, 2026-08-07): to test the generic bounded-recursion
  `_subdivide_tile` without engine access, mock
  `surrogate_training._load_or_build` + `surrogate_training._gate_chart`
  so the REAL subdivider function runs end-to-end against synthetic
  gate outcomes (stubborn-gap vs no-gap), rather than reconstructing the
  recursion logic in the test. Depth-cap teeth: a stubborn gap that never
  decays below bar must terminate at `MAX_SUBDIVISION_DEPTH` as a
  'recorded_gated' leaf, never crash or infinite-loop.
- CLOSED-FORM U-MIDPOINT ORACLE FOR WEDGE SUBDIVISION: the child boundary
  `theta_split` in `_subdivide_wedge_tile` is the u-midpoint image, NOT the
  theta-midpoint — verify via a closed-form oracle
  `_u_midpoint_theta(theta_lo, theta_hi, side)` (low side:
  `(0.5*(tl**(2/3)+th**(2/3)))**1.5`; high side mirrored about pi/2) that
  independently reproduces `_wedge_cusp_axis_map`'s u=d**(2/3) convention;
  match to ~1e-9, not 1e-16 (production rounds the returned split to 6dp).
- OPTIONAL-TO-MANDATORY NPZ KEY RENAME BREAKS IDENTITY-PATH FIXTURES: when
  a rename (e.g. theta_to_s->theta_to_u) also flips the field from
  optional-on-load (old reader: `data[key] if key in data else None`) to
  UNCONDITIONALLY REQUIRED, any existing test fixture built on the
  identity/None path (map omitted) now hard-refuses at load with a
  KeyError referencing the NEW field name — audit every synthetic-chart
  setUpClass for map=None fixtures and rebuild them with a real map (e.g.
  via the production axis-map helper) rather than widening the loader
  back to optional.
- LOBE U-COORDINATE MIGRATION TEST PATTERNS (Build lobe_cusp_coordinate,
  2026-08-08): migrated test_lensing_surrogate_lobe.py (73 tests) +
  test_lensing_lobe_subdivision.py + test_lensing_wedge_dd_arclength.py:
  theta_to_s/s_grid -> theta_to_u/u_grid on ALL lobe references, retired
  V1 identity-path tests (theta_to_s=None unsupported), SQRTEDGE ->
  U_COORD (u = d**(2/3)), `_engine_lobe_fixture` now calls from_lobe_engine
  with cusp_angle from tile cusps. Added 6 new classes/30 tests:
  CarveOutRetirement (2), LobeCuspAxisMap (10), CuspAdjacentRoundTrip (2),
  LobeSchemaHardRefuse (7), UAxisNodeExact (3), OpenCuspEdgeProbe (1) +
  4 self-falsification classes. Key measurements: theta_to_u is REQUIRED
  under lobe_caustic_relative_v1 (no identity fallback on load — second
  instance of the OPTIONAL-TO-MANDATORY NPZ KEY pattern, audit every
  synthetic setUpClass for map=None fixtures and rebuild with a real
  axis-map helper); U-axis B-spline reproduces stored u-nodes to 1e-7
  (both test files); open-cusp smoke-scale 4x4x4 chart ~7% error near cusp
  (gate at 0.10; production bar 1e-3 only at 12+ nodes); rho_lobe must be
  <=0.5 at cusp edge for eta > DEFAULT_CAUSTIC_FLOOR=0.05 (rho=0.95 at
  cusp edge gives eta~0.0001). 10 golden-value tests need re-freeze (D2
  fold skips, pre-existing).
- LOBE SUBDIVISION + CARRIER-FLIP + GHOST-SADDLE TEST PATTERNS (Build
  saddle_forensics, 2026-08-08, test_lensing_lobe_subdivision.py, 19 tests):
  (1) LobeCuspProximityTestCase — near-cusp tile refused, far-from-cusp
  probes clear eta_max, near-cusp proximity witness; LobeCuspSelfFalsification
  — deep-interior same-cusp-ray tile admitted. (2) LobeSubdivisionTestCase —
  children clear the bar, packed>=1, additive keys, admission predicate;
  LobeSubSelfFalsification — stubborn gap packed=0. (3) GhostKernelSaddleTestCase
  — finite non-trivial kernel + multi-source sweep on SADDLE parity (gamma>1);
  GhostSaddleSelfFalsification — wrong shape -> ValueError, origin ->
  GhostDomainError. (4) LobeCarrierFlipRefusalTestCase — mock
  `_build_lobe_chart` raising CarrierDiscontinuityError; `_subdivide_lobe_tile`
  catches it and records children as result='carrier_flip' with
  admission='admitted' + carrier_flip_detail, packed=0, terminating (NON-wedge
  style — the structural refusal is never recursed, unlike the wedge
  subdivider's ladder-served-gap leaf); LobeCarrierFlipSelfFalsification —
  normal build packs children. (5) ppGO above-ceiling suite extension:
  test_lensing_ppgo_above_ceiling.py (15 tests) exercising the w>150
  engine-intercept rung.
- EXTERIOR-POLAR CUSP-ADAPTED U COORDINATE TEST PATTERNS (Build
  exterior_polar_cusp_coordinate, 1a97bbd, 2026-08-08): added 6 classes/
  22 tests in test_lensing_surrogate_training.py — BuildFarfieldPositive
  ParityCuspAdapted (5: theta_to_u shape/monotonicity/endpoint-exactness),
  BuildFarfieldHighSideCuspAdapted (4: high-side origin, monotone,
  zero-start, endpoints), BuildFarfieldCuspOriginSelfFalsification (4:
  correct vs wrong theta_to_u differ, coefficients differ),
  BuildFarfieldSaddleExteriorUnchanged (2: parity=-1 -> theta_to_u=None),
  SubdividedChildrenCuspAdapted (2: parity pass-through to children),
  BuildFarfieldCuspAdaptedSelfFalsification (5: domain raises, non-uniform
  spacing, identity not byte-identical). Flipped
  FieldExposureTestCase.test_exterior_polar_uses_caustic_fixed_axes
  assertNotIn->assertIn: ExteriorPolarChart now has optional
  theta_to_u (nn.Array | None) — an assertNotIn on an optional field is a
  stale negative; flip to assertIn and update the docstring. INS-3-003 fix:
  `_train_tile`/`_train_exterior_chart` hardcoded origin='low' -> shared
  `_exterior_cusp_axis_map(theta_c_grid, gamma_band, n_gamma)` that MIRRORS
  production `_build_farfield_chart` origin (waist at median log-spaced
  rep_gamma; falls back to (None,None) raw-theta when theta_c range is
  outside [0,pi/2] or degenerate, so the domain guard never fires from a
  fixture). STRADDLING stays low (0.7698<0.7766), EXTERIOR/OVERSIZED now
  high (0.7854>0.7766).
- MIRROR-PRODUCTION-ORIGIN TEST HELPER PATTERN (INS-3-003, same build):
  when a test fixture must reproduce a production coordinate/origin
  decision (waist-based origin, median rep_gamma, null fallback for
  unrepresentable tiles), copy the PRODUCTION rule into a shared test
  helper — a hardcoded origin (e.g. 'low') is the defect; the shared
  helper is callable from EVERY fixture entry point (_train_tile AND
  _train_exterior_chart) so no path diverges.
- CLASS-DELETION DROPS TEST CLASSES SILENTLY (2026-08-08): the 0a31fcf
  polar re-chart deletion ALSO dropped `DefinitionTagLoaderRefusalTestCase`
  — its 8 loader tests were silently absorbed into `_legacy_single_box_
  arrays` dead code. When a chart-class deletion commit lands, grep for
  dropped TEST classes whose coverage silently migrated into dead helpers;
  restore the class, and retire tests for paths production now hard-refuses
  unconditionally (e.g. every legacy single-box tag at surrogate.py:3849).
- CUSP PPGO FAST RUNG TEST PATTERNS (test_lensing_airy_fold.py, 2026-08-08,
  cusp_ppgo_high_w WP1, 5 classes / 13 tests): do NOT assert golden
  agreement of the ppGO rung against the Pearcey uniform form — the rung
  delegates to `fold_ppgo_correction` (an Airy FOLD correction applied in a
  cusp context), so the rung value is NOT the asymptotic limit of the
  Pearcey form (measured 12-36% difference where both paths serve; the
  approximation is asymptotically sound only via the shared geometric-image-
  sum limit at large R). Assert the CONTRACTED properties instead:
  structural rung firing, finite/deterministic output, small-R refusal,
  w-floor gate, finiteness guard (all NaN/Inf variants), saddle parity,
  self-falsification. Document the fold-vs-cusp design gap in the class
  docstring. Fixture measurements: rung fires at w>=500 for astroid
  (radius=98 > r_ppgo_min=54 at the provisional const), w>=200 for saddle
  (radius=29 at w=100).
- 2D FOLD-CARRIER TEST PATTERNS (test_lensing_exterior_polar_fold.py,
  58 tests, Build exterior_2d_fold_carrier, 2026-08-10): port from 1D:
  rho_carrier -> rho_u_carrier shape (n_rho, n_theta_c), schema tag
  exterior_polar_rho_u_carrier_v2, NPZ key chart0_rho_u_carrier. (1) 1D->2D
  backward-compat tests use a CONSTANT-in-u carrier (broadcast of
  rho_u_carrier[:,0]) because byte-identical claims require zero u-
  variation. (2) angular distance helper: abs(np.angle(np.exp(1j*(p1-p2))))
  — naive abs(diff) fails on 2pi-wrapped differences. (3) magnitude
  invariance: compare served magnitude with-k vs without-k on the SAME
  rho_log_axis (shared spline-on-log interpolation budget ~1e-3), NOT vs
  an analytic oracle — interpolation error dwarfs pure-phase-rotation error
  (~5e-13). (4) ghost-kernel delay-match probe must use MEDIAN across all
  gamma-band gammas (production _compute_rho_u_carrier stores median), not
  first-match. (5) import path for geometry is
  cogwheel.lensing.chang_refsdal.geometry (NOT cogwheel.lensing.geometry).
  (6) theta_to_u at 4 nodes gives ~22% piecewise-linear interp error in u
  (~37x worse off-grid accuracy than raw-theta axis) — for smoke-scale
  carrier tests use a carrier bilinear in (rho, theta_c) directly;
  u-coordinate accuracy is separate from the carrier demodulation mechanism.
- GATED TESTS HIDE STALE FIELD REFS (TRAIN_TIER, 2026-08-10): after a
  production field rename, COGWHEEL_TRAIN_TIER-gated classes don't run in
  the fast tier and silently keep stale references (DT-10 referenced
  chart.rho_carrier post-rename; fast tier stayed green). Grep gated test
  classes for the old field name as part of the port.

- SADDLE CUSP-ADAPTED U TEST PATTERNS (Build saddle_exterior_full_treatment,
  2026-08-10; test_lensing_surrogate_training.py + test_lensing_surrogate.py +
  test_lensing_farfield_envelope.py): (1) SaddleCuspUCoordinateRoundTrip —
  build a saddle exterior chart with the cusp at the tile boundary, verify
  theta_to_u (2,N) N>=100, monotone, endpoint-exact to 1e-12, round-trip
  ~1e-14*max(u_grid). KEY FINDING: `_build_farfield_chart`'s cusp-boundary
  detection is FRAGILE TO FLOAT PRECISION (the boundary can be missed) — the
  test builds an INDEPENDENT theta_to_u as fallback when detection misses.
  (2) Mutation self-falsification builds TWO separate charts via from_engine
  for the same tile (with vs without theta_to_u): eps_with=1.98e-4 vs
  eps_without=3.32e-05 at smoke scale (4x4x4, rho=3.0) — the spec's "2x
  worse" claim does NOT hold at smoke scale (the u-coordinate benefit needs
  more nodes); encode the MEASURED reality with softer assertions (both
  differ measurably, ratio ~6x). (3) CuspArmCoverageParityGateSelfFalsification
  (fast tier, synthetic tube charts): monkey-patch _CUSP_ARM_COVERAGE->0.0
  (positive falsely refuses) and _SADDLE_CUSP_ARM_COVERAGE->0.07 (saddle
  falsely serves); constants restored in finally. TubeCuspWindowParityGating
  (+ SelfFalsification) uses the pre-built _multichart_fixture() charts. (4)
  Far-field envelope accuracy/serving: F-norm eps ~7e-5 median on a 5x5
  grid; E-norm eps USELESS for far-field (~1e-4 denominator) — use F-norm;
  at smoke scale cusp-adapted vs raw-theta indistinguishable (ratio ~1.0).
  (5) Global-replace collateral: a variable rename (denom -> f_denom) can
  accidentally hit SIBLING test helpers (StraddlingTileTrainabilityTestCase
  line 453 + _chart_eps line 1492 were caught and fixed) — audit for
  displaced/duplicate fragments after any global replace.
- PEARCEY RESIDUAL TABLE TEST PATTERNS (Build zero_quadrature_pearcey,
  2026-08-11, test_lensing_levers.py): ConsultPearceyRefusalTestCase
  (table=None -> None; table inside -> value / outside -> None; mock pearcey
  -> no live quadrature), PearceyTableSchemaMigrationTestCase (0.2.0
  round-trips, 0.1.0 raises ValueError), test_explicit_residual_reconstruction
  (P_asymp + spline resid == table.evaluate). Removed
  test_consult_routes_outside_box_to_live_quadrature (WP-2 killed the
  fallback). KEY FINDING: P - P_asymp is DISCONTINUOUS across a caustic
  crossing — PearceyTableCertificationTestCase (3 tests) blew to 1.9e+09
  spline error sweeping across it; residual certs must stay INSIDE one
  topology region. `import warnings` at top level; the git-grep helper
  `_git_grep_cogwheel` needs -E (extended regex) and excludes
  cogwheel/tests/ so docstring mentions don't self-trigger.
- GIT-GREP DEAD-CODE DELETION GATE TESTS (2026-08-11, DeadCodeDeletionGate
  TestCase in test_lensing_levers.py): gate that deleted symbols stay
  deleted via `git grep -E 'name'` over NON-test cogwheel/ + an AST absence
  check on the module-under-test (demodulate/remodulate/_carrier_phase/
  _dominant_stationary_point); `_SPLIT_BASE` legitimately retained in
  _pearcey_cusp.py. Pair with a self-falsification test grepping a LIVE
  symbol to prove the pattern has teeth.
- MPMATH OVERLAP-BAND CROSS-AGREEMENT (2026-08-11, OverlapBandDdMpmath
  AgreementTestCase in test_lensing_schwinger.py): DD-vs-mpmath at w=60,
  8 pts (gamma' ∈ {0.3,0.7,1.3,1.5} x y ∈ {(0.3,0.2),(0.7,0.4)}), gate
  < 5e-10, worst measured 5.6e-11 at gamma'=1.5, y=(0.3,0.2). Tolerance
  note: DD path at w=60 has e^{pi*60/4} ~ 3e20 amplification limiting DD
  accuracy to ~1e-10. Self-falsification: mock ceil->-10 (dps=20) + relax
  _CERTIFICATION_TOL->100 -> cross-agreement > 1e-4 (proves teeth).
- SPEC GATE-MARGIN ESTIMATES CAN BE COPY-PASTE ERRORS (2026-08-11, ppGO
  resolution gate): the spec claimed w*delta_min~1.9 at w=500 for the
  saddle fixture but measured delta_min=0.644 gives 322 >> 4.0 — saddle
  sources always resolve at w>=50. _merging_fold_pair returns None for
  dual-saddle 2-image sources, making the resolution gate the SOLE
  admission criterion there. Prove gate teeth by INFLATING the threshold
  (->1000 blocks, ->0 admits, resolved w=20000 still admits) rather than
  trusting the spec's refusal scenario.
- _CUSP_VERTEX ROUTING FIX DOMAIN TESTS (2026-08-11, test_lensing_airy_fold
  .py, 11 tests / 4 classes): interior cusp source serves via BOTH table
  and live quadrature (route 'pearcey'); table-live agreement to 1e-5
  relative (measured ~1e-7); _cusp_vertex returns source-plane-closest
  astroid cusp (seed_theta-independent); exterior ppGO route unaffected;
  cleared-table still serves, corrupted _cusp_vertex violates the distance
  gate. Fixture _CUSP_FIXTURES[0] = (0.5, 0.20, 0.25π). PRE-EXISTING (not
  these tests): 8 vertex tests red at HEAD — the coder's WP1 _cusp_vertex
  change returns a finite wedge-tip vertex where old code returned None at
  wedge-edge configs (multi-candidate source-distance selection).

## 2026-08-12 builds (saddle interior-ring fix, saddle rho-origin misclassification, ppGO mid-w + MINUS_GHOST)

- SADDLE INTERIOR IS A RING, NOT A DISK (Build "fix _is_interior
  discriminator", test_lensing_airy_fold.py): the origin-ray-based
  `r_caustic` check fails (LensDomainError) at certain gap angles (e.g.
  gamma=1.3, beta=0.37, ~46deg/226deg) where the source still has 4
  images — old `_is_interior` (r_caustic + |source|<r_caustic) mis-read
  these as exterior/errored. Fixed via an image-count discriminator
  (`len(images)>=4`). The saddle deltoid interior is a RING (4 images ->
  2 at the inner boundary -> 4 again in an annular lobe -> 2 past the
  outer boundary), not a simple disk — never assume monotone
  interior/exterior by radius alone for saddle parity.
- SADDLE ORIGIN-RHO MISCLASSIFICATION BACKWARD-COMPAT (Build "fix saddle
  origin-rho misclassification", test_lensing_saddle_rho_guards.py, 5
  parity-gated guard sites across likelihood.py/surrogate_census.py/
  ppgo_map.py): adding a `gamma>1 AND image_count==2 -> 'born'` census
  guard breaks pre-existing tests (test_lensing_born.py,
  test_lensing_surrogate_census.py) that had pinned the OLD broken
  saddle 2-image classification as 'out-of-box' — before treating
  pre-existing suite reds as alarming, verify a backward-compat break is
  the INTENDED fix (it pins the bug being fixed, not a regression). All
  guard sites are parity-conditional; positive-parity path stays
  byte-identical.
- ppGO MID-W + MINUS_GHOST CHART TEST PATTERNS (Build "lower ppGO radius
  gate + MINUS_GHOST exterior chart", test_lensing_ppgo_midw_and_minus_
  ghost.py): (1) lowering `_R_PPGO_ERROR_CONST` opens the ppGO rung for
  BOTH parities at mid-w; on-axis 4-image sources with
  `_merging_fold_pair=None` and `w*delta_min=0` still correctly fail the
  resolution gate and fall through to Pearcey, byte-identical to
  old-code fallthrough. (2) A MINUS_GHOST-labeled farfield chart
  (force_minus_ghost=True) produces a genuinely DIFFERENT envelope than
  the same tile's KERNEL_SUM chart (max|diff|>1e-15) — proves ghost
  subtraction is non-trivial. (3) MINUS_GHOST serve round-trip (chart
  eval -> ghost re-add -> reconstruct_farfield) reconstructs to ~5e-3 at
  smoke scale vs a 2e-2 bar; omitting the ghost re-addition step changes
  the reconstructed F (self-falsification teeth). (4) SYMMETRIC
  TRAIN/SERVE GATE REFUSAL: near a cusp vertex (Im(tau_c) and separation
  gates both marginal), `farfield_envelope_from_partition(MINUS_GHOST)`
  (train) and `farfield_ghost_term` (serve) raise the SAME
  GhostDomainError for the same source — confirms the ghost-admission
  gate decision is train/serve-consistent; the plain KERNEL_SUM label
  (no ghost subtraction) succeeds for the identical source, showing
  refusal is ghost-label-specific, not a general source refusal.

## 2026-08-13 (saddle_above_ceiling_serving, tier-1 shards A/B/C)

- MAGICMOCK FIXTURES HIDE A NEW ATTRIBUTE READ, AND THE CRASH LOOKS LIKE A
  PRODUCTION BUG. When new production code reads an attribute the fixture's
  `MagicMock` never set, the mock happily returns another MagicMock;
  `np.asarray()` turns it into a 0-d array and the failure surfaces as
  `IndexError: boolean index did not match indexed array` far from the
  cause. Measured 2026-08-13: the tier-1 census block added
  `np.asarray(geom.delays)[real_mask]`, and
  `test_lensing_saddle_rho_guards.CensusBandSplitMirrorIntegrityTestCase`
  crashed because its mock set `real_mask` but not `delays`. RECURRED
  2026-08-14 with `geom.images` in place of `geom.delays` (same family) —
  when auditing a mocked geometry object after any new production attribute
  read, complete the mock with ALL attributes the new code touches, never
  add a defensive length/shape check to the hot path.
  CORRECTED — an earlier version of this entry called that a
  "reused-vs-fresh partition" PRODUCTION hazard. It is not. Driver measured
  real `geometry_partition` across corridor, lobe-interior, far,
  near-origin, astroid and parity-edge sources: `delays` and `real_mask`
  are equal length ((4,)) in every case, and no mismatch is producible.
  The fix is to complete the MOCK, never to add a defensive length check to
  a hot path. Before promoting a mock-only failure to a production defect,
  reproduce it against real objects — a fixture crash is evidence about the
  fixture until you show otherwise.
- DOMAIN CONFOUND WITHIN A GATE-ADMITTED SET: when certifying a new
  rung's accuracy bar, the raw resolvability/admission-gate-admitted set
  can still contain a SEPARATE confound (e.g. caustic proximity) that
  dominates the error and blows the spec's claimed percentile (p90) —
  measure accuracy stratified by the confound before trusting the
  gate-admitted set as the rung's certified domain; a config that is
  gate-admitted but served wrongly (a "leaky-gate witness") is a real
  spec discrepancy to flag, not a test bug to paper over. Certify the
  rung's ACTUAL contract domain (e.g. resolvable AND far-from-caustic),
  not the raw gate's nominal domain.
- ZERO-ENVELOPE SELF-FALSIFICATION HAS NO t_min TEETH: for a rung that
  reconstructs with a ZERO complex envelope, a wrong-t_min self-
  falsification mutation is inert (0*exp(...)=0 regardless of gauge,
  i.e. the reconstruction is gauge-independent bit-exactly) — pick a
  different mutation (e.g. a wrong switch/kernel term) to give the
  self-falsification teeth on such rungs.

## 2026-08-13 (ppgo_interior_certificate, fold_exterior_ghost)

- PAIR THE FRAMES BEFORE SCORING ANYTHING (standing ritual; two phantom
  errors on one day — a 6e-2 and a 130%, both fake, both from comparing
  across demodulation frames). Before ANY relative-error claim from a new
  harness, run a PAIRING GATE: score `operator.geometric_amplification`
  against the oracle at a KNOWN-RESOLVED config (e.g. gamma=0.5,
  y=(1.8,0.9), w=40) and require < 1e-2. Only then trust the harness.
- F069-SAFE ORACLE RECIPES: positive parity — the mass-sheet/eigenframe
  reconstruction (`operator._mass_sheet_map`) fed to `_schwinger
  .f_schwinger`; at kappa=0, beta=0 that collapses to `f_schwinger(w,
  source, gamma)` DIRECTLY. Saddle parity — `_saddle_mass_sheet_map` +
  `f_schwinger` (validated to 0.0e0 including kappa=0.2). NEVER use
  `F_op` above w=60 (self-oracle), and NEVER use
  `channels.evaluate().exact_total` as a reference for arm outputs without
  first proving the t_min pairing — it lives in the min-subtracted frame.
  Keep oracle points at w <= 60 to stay on the exact DD path (>60 is
  mpmath, >150 hard-refuses).
- A MUTATION TEST NEEDS A REGIME WHERE THE MUTATED TERM IS ABOVE TOLERANCE:
  ghost sign teeth (minus vs plus) are INERT at the serve band because the
  term decays as exp(-w Im tau_c) (~1e-23 at w~90) — run them at w=12 where
  the term is ~4e-4. Conjugation teeth resolve at any w (they flip decay
  into growth). Check the magnitude of what you are mutating before
  believing a passing teeth test.
- PROVE A GUARD LOAD-BEARING WITHOUT FABRICATING PHYSICS: a "4-image spoof"
  (2 real + 2 decoy images) dies on the Morse census guard in
  `geometric_amplification` (signed-magnification sum != sign(detA)-1 ->
  LensDomainError). Instead monkeypatch the PRIMITIVE the guard consumes
  (`_merging_fold_pair` returning non-None on a genuine exterior census) so
  the shipping code path stays intact and the input stays physical.
- DERIVE DOMAIN FIXTURES FROM THE LIVE CLOSED FORM: place sources as
  `rho * r_caustic(gamma, theta) * [cos, sin]` (rho<1 interior/4-image,
  rho in [0.9,0.99] near-caustic, rho>1 exterior/2-image) — a hand-picked
  (gamma, y) that "looks interior" is routinely 2-image exterior and
  surfaces as a confusing shape mismatch, not a clear failure.
- MONOTONE-TOWARD-REFUSAL GUARDS ARE BACKWARD-COMPATIBLE BY ARGUMENT:
  a new `count != 4 -> refuse` guard cannot flip None -> non-None, and
  existing "fires" tests all use 4-image configs where it is a no-op — so
  the audit is a grep for the fixture domain plus one confirming run, not a
  full re-derivation. Still run the neighbour suites: the argument covers
  assertions, not fixture premises.
- TEST-COUNT CHANGES RESHUFFLE XDIST `--dist loadfile` SCHEDULING AND CAN
  EXPOSE LATENT MODULE-GLOBAL LEAKS (F078, measured): adding/removing tests
  changes which files share a worker, and a suite that installs a PROCESS
  GLOBAL (e.g. the Pearcey table) then leaks it into an unrelated file that
  happens to land on the same worker. Any suite installing a process global
  owes a save/restore at MODULE scope. A "new" failure that appears only
  under xdist after a count change is this, not a physics regression.

## 2026-08-14 (saddle serve-gate: certificate/governance test patterns)

- CERTIFICATE-GATE TEST PATTERN: for a gate admitting via
  `safety*estimate<=bar`, (1) bracket the exact flip point by scanning the
  gate's own continuous input (e.g. w_lo) for where
  `safety*estimate==bar`, and confirm the flip matches within tol; (2) if
  the estimate is an EXACT asymptotic power law (e.g.
  sum(sqrt|mu||c3|)/w_min**3), assert the log-log slope pins to the exact
  exponent (e.g. -3 to ~1e-6), not just "decreasing".
- NON-REGRESSION GOVERNANCE PAIR PATTERN (for a monotone-only-tightens
  constant, e.g. an admission floor that may only rise per the false-
  admit/false-refuse asymmetry): pin a FROZEN anchor literal (deliberately
  NOT re-reading the live module constant — else the comparison is
  vacuous against itself) plus a second tripwire mirroring the "flips red
  once fixed" pattern; add a self-falsification test that re-runs the
  identical comparison against a synthetic regressed value inside
  assertRaises, proving the check has teeth without ever mocking the
  bound constant.
- BOUNDARY-WITNESS SEARCH (generalizes to any admission gate): to find a
  witness genuinely near a gate's admission boundary without a pinned
  literal, step outward from a geometric anchor (e.g.
  `nearest_caustic_point`) along the anchor-to-source ray and search for
  the first offset the LIVE production gate admits; confirm boundary-
  exactness by re-checking the immediately preceding offset is refused
  by the same live gate (not a cached/reconstructed one).

## 2026-08-14 (born_residual_wiring reachability suite)

- DECISIVE SUBSTITUTION VIA SOLE-DEPENDENT-BRANCH: when a spec asks for an
  engine-heavy A==B identity that can't be cheaply reproduced, and exactly
  ONE branch of the served function depends on the new collaborator (here
  `_born_residual_analytic` is the ONLY chart-dependent branch of
  `_amplification_coefficients`), proving "chart-attached with a None-chart
  input" reproduces the no-chart route byte-for-byte is a legitimate
  engine-free substitute for the full A==B pin — document why the
  substitution is decisive, not just convenient.
- CROSS-AGENT INDEPENDENT DISCOVERY: Test Dev independently reproduced the
  same band-split ValueError defect Coder's INS-2-001 fixed, via a
  from-scratch battery rather than reading the fix — corroborating
  evidence a fix's regression coverage is real, not just self-reported.


## 2026-08-14 (F080 ppGO saddle rho<1 per-cell relaxation)

- MIRROR TEST VIA REAL BOUND METHODS, NOT REIMPLEMENTED PRIMITIVES: when
  testing a census/likelihood mirror, bind the test probe's unbound methods
  to the PRODUCTION module's globals (`__globals__`) so
  `get_certified_ppgo_map`/`caustic_rho`/etc. resolve to the real shipping
  functions without needing a full engine instance — avoids re-deriving the
  primitives from scratch, which is the class of bug a mirror test exists
  to catch.
- EXACT-EDGE-KEYED ALLOWLISTS ARE BACKWARD-COMPAT BY CONSTRUCTION: a guard
  keyed on exact float64 grid-edge equality is unreachable from any
  synthetic-fixture suite using different edges — before flipping old
  tests, check whether their fixtures can even reach the new allowlisted
  cell; if the edges never match, no pre-existing test needs updating.


## 2026-08-15 (F081 saddle tube fundamental training)

- DOUBLE-SIGNATURE SELF-FALSIFICATION: for a fix that changes TWO coupled
  behaviors from a single wrong-currency input (widens the corridor AND
  admits/excludes a witness), feeding the OLD wrong value (max_eta_max
  instead of min_eta_max) should produce a "double signature" test failure
  (both symptoms flip together) — stronger self-falsification than a
  single-assertion mutation test.
- PARTITION/CLUSTERING PREDICATES NEED AN INDEPENDENTLY-CODED ORACLE, NOT
  A RE-CALL OF PRODUCTION: verifying a count invariant on a grouping
  algorithm (e.g. 6 arcs -> 2 D2-orbit reps) requires an independently
  written classifier (e.g. a from-scratch union-find via
  _circular_gap/_independent_orbit_labels), never a call into the same
  production clustering function under test — this is a distinct pattern
  from "oracles must call shipping code" (that rule is for NUMERICAL
  physics oracles, not structural/algorithmic correctness of the thing
  being tested).


## 2026-08-15 (lobe cusp edge-coincidence test patterns)

- BOUNDARY-TRICHOTOMY SUBTEST SWEEP PATTERN: for a coincidence-tolerance
  guard (edge vs interior-straddle), a single parametrized test
  (`test_boundary_trichotomy_at_edge`) sweeping subTests over
  {exterior, on_edge, hair_inside, straddle} x {both sides} at fixed
  theta_lo/theta_hi pins the `<`/`<=` flip guard far more cheaply than one
  test per case — exterior/on-edge/hair-inside must build a strictly-
  increasing map with bit-exact endpoints, straddle must raise. Watch for
  a "hair_inside" offset landing inside half-ULP of the edge (e.g. 2e-17
  vs half-ULP(0.9)=5.55e-17) — it rounds TO the edge exactly, which still
  exercises the tolerance branch validly, just not the "just inside" case
  it was meant to probe; note this explicitly rather than assuming the
  offset always resolves as intended.
- CALLER-PATH COVERAGE FOR A LEAF-FUNCTION GUARD: when a fix lives in a
  leaf helper (`_lobe_cusp_axis_map`), add a companion test exercising the
  actual caller (`_lobe_child_boxes`) with the SAME coincident-edge
  fixture, engine-free — confirms the guard's return value flows correctly
  through the real call site (side selection, box count, split point) and
  that the interior-straddle case still propagates its ValueError through
  the caller, not just the leaf.
- SELF-FALSIFICATION FOR A ULP-WIDTH TOLERANCE: an offset of 1e-6 (vastly
  larger than an 8-ULP-at-this-scale band of ~1.8e-15) inside the edge
  must still raise — this is the cheapest self-falsification check for any
  newly-introduced small-tolerance guard: confirm the tolerance band is
  narrow relative to a "how would a real regression look" magnitude, not
  just that a few chosen literals happen to pass.
- AUDIT PRE-EXISTING STRADDLE TESTS FOR TOLERANCE-BAND OVERLAP: before
  shipping a new ULP-scale coincidence tolerance, grep all pre-existing
  `assertRaises` sites on the guarded function and confirm their offsets
  (here 0.1-0.7 rad, both surrogate_lobe and lobe_subdivision suites) are
  far above the new band — cheap and prevents silently flipping an
  existing regression test into a false negative.

## 2026-08-17 build (serve_route_census)
- ENGINE-FREE PROOF PATTERN (reusable): booby-trap every exact-wave/engine
  door (evaluate, f_schwinger, _f_schwinger_mpmath, mpmath fn) with a
  UNIQUE sentinel Exception that is explicitly NOT a subclass of the
  production module's caught-refusal tuple; assert both call_count==0 on
  each door AND sentinel-disjointness from the catch tuple (so a future
  widening of that tuple that would swallow the sentinel is itself caught).
- CAUSTIC-REACH FIXTURE TRICK: when no caustic_reach() helper exists,
  derive it as `1/ppgo_map.caustic_rho(gamma, 1.0, 0.0)`.
- Route/kind D2-invariance tests should drive the REAL classify_draw via a
  memoized `_classify_env()` triple (load/frequency-grid/band-edges) shared
  with run(), never a reimplementation — same "call shipping code" rule
  applied to route-kind vectors, not just numeric oracles.
- Gate-bounds-wrong-object regression witnesses (F069/F074 family): the
  omitted-term/certificate estimate is often MAXIMAL exactly ON a symmetry
  axis (e.g. cusp axis, angle=0) — place the witness there with a large
  magnitude parameter, not at a generic off-axis point, or the "huge
  estimate" case under test won't actually be huge.

## 2026-08-17 build (tube beat-free / D2 fold / Nyquist coordinate — multi-shard)
- SPY BINDING TARGET (reusable, hit independently across builds): to
  intercept a helper the code under test calls, patch the MODULE ATTRIBUTE
  the calling code actually dereferences (e.g. `mock.patch.object(
  surrogate_module, '_tube_f_ref', ...)`), not the helper's own defining
  module — a same-name import elsewhere is a silent no-op.
- ENGINE-FREE SUBSTITUTION WHEN THE REAL ORACLE REFUSES NEAR A SPEC'S
  DEMANDED REGION: if a literal spec (e.g. an asymptotic power-law near a
  cusp) lands where the real production function legitimately refuses,
  feed the SAME shipping function a synthetic input engineered to land in
  that region rather than dropping the spec, and confirm it reconstructs
  the closed form — extends "measure, don't trust the brief" to the case
  where the oracle is unreachable at the literal spec's demanded point.
- SHARED EXPENSIVE BUILD PAID ONCE PER FILE: cache a heavy chart build
  behind a module-level `functools.lru_cache(maxsize=1)` helper shared
  across multiple TestCase classes in the same file, to keep suite runtime
  under the ceiling when several specs need the same expensive fixture.
- STRAY/ORPHAN TEST FILE ACROSS DIVERGENT BRANCHES: an untracked (`??`)
  test file from a sibling/abandoned design branch can fail at COLLECTION
  (ImportError for a symbol absent on the current branch) and poison a
  full-suite collect-only count; confirm it predates your build (grep the
  missing symbol at HEAD) and flag for the owner to retire/relocate rather
  than treat it as a regression you caused.

## 2026-08-17 (driver fixture-repair session — consolidated from tube_test_investigation)
- LEGACY SYNTHETIC FIXTURES GO STALE WHEN A SERVE GATE GAINS A
  SOURCE-DEPENDENT TERM: fixtures that bypass the production builders keep
  encoding pre-gate expected values and break (or silently mis-measure)
  once serving consults the source. The fix pattern is to THREAD A REAL
  production-derived source through the fixture and RE-DERIVE expected
  values through the LIVE serve path, keeping golden literals bit-identical
  — third instance of the rebuild-with-real-data family (see the
  OPTIONAL-TO-MANDATORY NPZ KEY and guard-bypass entries); prefer this over
  mock-patching the gate whenever a real source is obtainable.
- STRUCTURAL-PROBE FLAG FOR SELECTION PRECEDENCE: when the precedence
  branch under test requires a physical double-match that is unsatisfiable
  with real inputs, disable the physical requirement via the builder's
  structural flag (`require_fref=False`) rather than fabricating physics —
  it isolates the selection-order logic as its own testable invariant.
- A CONSTANT FRAME/ORIGIN OFFSET IS INVISIBLE TO NODE-EXACT ROUND-TRIPS:
  it round-trips EXACTLY at build nodes yet leaks a slowly-growing
  OFF-node error — pair every node round-trip suite with an off-node
  (inter-node midpoint, interpolation-only, no extrapolation past end
  nodes) sweep against a normalized error bar; that sweep is what catches
  frame-consistency bugs (e.g. tau_bar vs tau_c) the round-trip cannot.
- DERIVE THE SERVABLE SUB-DOMAIN FROM THE LIVE GATE PREDICATE, never a
  pinned literal (drift -> spurious fail on a non-servable region OR
  spurious pass on a padded one), and guard held-out accuracy sweeps with
  refused_count == 0 so a sample landing on a zero-filled/refused node
  cannot silently contribute a garbage measurement.


## 2026-08-18 (INS-3-001, low_w_diffractive_rung)
- "THE GATE MOVES, NOT THE TEST": when a driver ruling's acceptance text
  assumes a companion production fix (owned by a DIFFERENT, not-yet-run
  agent) has already landed, and your own file read + live measurement
  prove it has not, write the test to the CORRECTED expectation anyway and
  let it be honestly RED at authoring time — never soften it (no
  expectedFailure, no loosened bound) to force a green. Document the
  measured value and the root-cause file:line in the docstring/change
  report so the pin flips green with zero further edits once the real fix
  lands.

## 2026-08-18 (born_certificate census-mirror shards)
- ENGINE-FREE CENSUS FIELD-SWAP PATTERN: to unit-test one classify_draw
  branch in isolation, call the real `_load_production_modules()` then
  `dataclasses.replace()` it swapping ONLY the fields the scenario needs
  (e.g. born_chart, dimensionless_frequency, ppgo_band_split,
  ppgo_cell_ceiling, diffractive_bottom_ceiling) -- every other field
  (real geometry_partition, shipped `_band_split_mask`, other intercepts)
  stays production, so earlier intercepts still return before any wave
  door is touched. Cheaper and more faithful than reimplementing the
  whole classify_draw dispatch with mocks.
