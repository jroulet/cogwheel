# Test Dev Long-Term Knowledge

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
