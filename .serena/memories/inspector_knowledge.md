# Inspector Long-Term Knowledge

- To certify an index-clamp or bounds trick, find the degree/step
  invariant bounding index motion and show out-of-range table entries
  are provably zero — don't just confirm "it didn't crash."
- When a truncation/series-length heuristic is rescaled, check it
  against the real peak-term location and magnitude-scaling law, not
  the caller's original proxy variable.
- When a spec discloses a certified sub-range narrower than the code's
  nominal domain, confirm the gap is honestly written up (SPEC +
  FINDINGS); carry as an open defect, never close by widening tolerances.
- Confirm test oracles are non-circular: independent re-derivation; look
  for (or add) an AST guard forbidding the module-under-test's names.
- Pre-existing environment-only collection errors are out of scope for a
  focused review — note, don't chase.
- Run the mandated mutation check yourself when reviewing new test
  suites (perturb the load-bearing constant/branch, confirm red).
- When a "goes to X" test fails from legitimate physics at a nonzero
  offset, fix via a dedicated fixture where the closed form truly is X,
  paired with a contrast test predicting the excluded case.
- When sibling suites each rebuild the same buggy primitive, prefer a
  shared upstream source — flag as stylistic, not blocking.
- njit voids Python-level monkeypatching: new njit cores must expose
  .py_func and falsification tests must go RED through that chain (F010).
- A scatter/weight-vector reduction replacing a bilinear form is a real
  accumulation-order change: require re-certification vs an independent
  oracle at the ORIGINAL tolerance plus solo-vs-batch certify-XOR-refuse
  decision identity.
- Single-path delegation (scalar wrapping batched core) means existing
  suites auto-exercise new code; a dedicated new test module instead of
  editing existing suites is a benign plan deviation.
- A build that REMOVES named constants/mechanisms almost always leaves
  stale the exact SPEC sentence naming them — check that paragraph. Doc
  staleness is a flag-to-Librarian finding, not a Coder defect; a build
  the plan expected to touch SPEC but didn't = flag to Librarian. CARVE-
  OUT: retired-mechanism text surviving in COMPLETED.md / completed.d /
  changelog fragments is append-only HISTORY describing a past build —
  correctly frozen, NOT staleness. Only the canonical surfaces (SPEC.md,
  FINDINGS.md, DATA_CONTRACTS.yaml, docs/source) can go stale.
- A GATE-CONTRACT swap (retiring one admission criterion for another)
  breaks every sibling suite that encoded the OLD contract, not just the
  changed module's own new test — require the fix to re-key EVERY named
  required-green suite, and verify by running each file to completion
  (Build 8h-d1, INS-d1-001; confirmed across two passes).
- Reconcile a suite's result against the file's ACTUAL test-method count
  (passed + xfailed + failed must equal it, HEAD count == worktree count
  unless methods were added/removed). A pass/fail count carried over from
  your own earlier pass can be a miscount — re-derive it, don't trust it.
- Re-reviewing a byte-identical diff: still re-run the suite + import
  probes and re-derive the key identity by hand.
- DATA_CONTRACTS covers serialized/shipped artifacts only; new fields on
  an in-memory dataclass need no contract change (pickle __getstate__
  still deserves a round-trip probe). An OFFLINE-ONLY artifact likewise
  needs no entry yet — revisit if a file is actually shipped/consumed.
- A serve/approximation gate must include EVERY parameter axis the
  approximation was trained at. A missing axis (surrogate has no kappa
  axis, trained kappa=0) that is harmless only because production pins
  that axis is still a LATENT correctness violation of the conservative-
  serve contract — flag it, don't wave it through on "non-triggering in
  production". (Build 8h-d1 landed the matching beta!=0 guard.)
- Refusal-net reviews: trace EVERY dispatch route to the boundary —
  scalar lnposterior AND the sampler's prior.unfold_apply wrap must both
  route through the override; except must name specific refusal types,
  never bare (Build 4).
- Check the approved plan JSON before flagging a constraint deviation: an
  explicitly documented/approved deviation is not a finding.
- A documented @expectedFailure efficiency aspiration is a property, not
  a bug — but verify all non-finites are exact -inf with zero NaN.
- A crash that aborts tuple-unpacking (e.g. an arity change) MASKS every
  downstream content assertion in that test process. After the crash-fix,
  the FIRST full green run is the real content review — never close a build
  on "unpack fixed"; run the whole suite to completion and grep the summary
  line (a background wrapper's exit_code can be the trailing echo).
- When a producer label is redefined (e.g. far-field envelope), stale test
  ORACLES and reconstruction HELPERS that still reference the OLD label/path
  are the likely failure — distinguish from a production bug via a node-exact
  test: if served==new-label to machine precision but the oracle uses the old
  label, it's a test bug; production is fine.
- Builds run in the sibling worktree cogwheel-claude-dev; `cd` to the
  main tree is hook-blocked — run Bash from the worktree cwd.
- A partial/targeted re-run that greenlights the NAMED finding can HIDE a
  sibling regression from the SAME edit. When a build resolves an
  emulation-accuracy finding by RELOCATING a shared fixture (e.g. moving
  the source to a larger |y|), the relocation changes downstream PHYSICS
  (larger image separation -> larger relative delays) and can break unrelated
  tests — most likely those hitting the RB delta_t_max binning limit. Always
  run the WHOLE changed test file(s) and audit every reuse of a relocated
  shared fixture. A shared lnlike fixture whose relative delay sits ~1e-4
  under delta_t_max is a latent trip-wire: flag edge-margin fixtures as
  design fragility even when the base config passes.
- A guard-clause refactor that ends a loop body on `continue` inside the
  guard is a red flag: check the fall-through (non-guarded) tail still
  packs/returns — a deleted pass-case tail is a silent regression.
- After a production-only edit, re-run PRE-EXISTING suites over UNCHANGED
  test files too — a production change can regress suites that never
  touched the diff; the failure text usually points straight at the
  deleted/changed line.
- A shared 'DRY' helper invoked independently by two accessors can still
  trigger its expensive underlying computation (e.g. a full geometry
  sweep) TWICE per request — code-level dedup doesn't guarantee runtime
  dedup; flag redundant re-invocation of expensive shared derivations as
  a design finding even when functionally correct (Build 8h-b).
- When verifying an added ceiling/boundary mechanism, confirm the
  degenerate "fully accepted / no restriction" case algebraically forces
  the new boundary equal to the old default (e.g. ceiling=wall) — this is
  how byte-identical-to-HEAD behavior is actually proven.
- Before crediting a fix against its named acceptance-gate test, confirm
  the test's fixture actually ROUTES THROUGH the fixed code path — a
  correct production fix (e.g. in a tiler) can leave the named gate red
  if the test builds via a different entry point that bypasses the fixed
  component entirely (Build 8h-b6, INS-1-001).
- Validate a persisted schema/version tag from the artifact's OWN stored
  meta (e.g. the chart's saved attribute), not from a provenance wrapper
  that may be rebuilt minimally on reload — the wrapper can silently lose
  the tag while the artifact's own meta still round-trips correctly.
- When a serve dispatch chain gains new optional context (e.g. eigenframe
  coords) threaded through select/evaluate, check ALL consumers of that
  chain (census, reporting, not just the primary serve path) also thread
  it — a consumer left behind silently undercounts/misclassifies once an
  artifact exercising the new path ships (Build 3, saddle lobe-serve).
- A clean/advisory-only working tree (no .py/SPEC/DATA_CONTRACTS diff)
  means nothing NEW to certify, but does NOT auto-close a carried finding —
  re-read the actual SPEC paragraph and the actual code symbol byte-for-
  byte each pass to confirm the divergence still holds before deciding
  resolved vs still-open.
- When a spline is fit on a reparametrized axis (e.g. s) and serve maps
  through an interp table (theta→s, ~6e-9 error at 2001 nodes), the
  node-exact tolerance must budget for the interp error — widening
  _NODE_EXACT_TOL from 1e-10 to 1e-7 is justified; verify the budget
  arithmetic is commented at the constant's definition.
- After a production interface change (e.g. a TrainingConfig field moved
  to an explicit function arg), verify ALL callers across ALL suites, not
  just the directly-touched test file — a gate-contract swap can break
  every sibling suite that passed args via the old config field.
- SHARED-PHASE-HELPER CONVENTION: when two code sites demodulate/
  remodulate with the SAME phase function (e.g. `_frame_phase(w, t_min)`),
  both must import and use the SHARED helper — one side using np.exp
  inline while the other calls the helper is a convention violation even
  if libm handles the large-argument range-reduction correctly. Flag as a
  finding (non-blocking if functionally safe), require import+use of the
  shared helper on both sides (INS-12-001 pattern).
- EXCEPTION COVERAGE AT DEGENERATE INPUTS: when an except clause names
  ValueError+LensDomainError for a caustic-geometry call, audit whether
  degenerate parameter values (e.g. gamma=0 for caustic_rho) can raise
  ZeroDivisionError instead — LensDomainError IS-A ValueError but
  ZeroDivisionError is NOT; a missing ZeroDivisionError catch is a latent
  production crash at the gamma=0 boundary (INS-11-001 pattern).
- CHART COVERAGE CONTRACT: a `covers(gamma, rho)` gate that checks only
  2 axes but not the w-axis is correct IFF the chart's w-band coverage is
  a training-driver-responsibility contract (chart built to span the full
  w band). Flag if the contract is only implicit; it is not a production
  bug but a design fragility to document.
- InteriorWedgeChart REVIEW CHECKLIST (Build interior_wedge_chart, 40 tests
  PASS): (1) Coordinate math: D2 fold abs(y1),abs(y2) -> theta=atan2 is
  correct quotient. Round-trip residual < 1e-15. (2) Axis ordering:
  coefficients (log_w, gamma, r, theta_wedge) contract consistently.
  (3) NPZ persistence: all 14 fields round-trip bitwise (max|diff|=0.0).
  (4) select_chart dispatch: wedge is lowest priority (tube>farfield>lobe>
  wedge). (5) Duck typing: chart.log_w_grid/gamma_grid used by census
  _chart_log_w_range, _chart_index, _is_band_edge, heldout_envelope_eps —
  all present on InteriorWedgeChart. (6) Validator:
  _validate_wedge_caustic_map checks gamma equality, theta span [0,pi/2],
  finite positive r_table. (7) OPEN: DATA_CONTRACTS.yaml + SPEC.md do NOT
  describe InteriorWedgeChart (Librarian scope).
- FOLD-ppGO MOCK PATTERN (INS-c8-001): when production replaces
  `geometric_amplification` with `fold_ppgo_correction` via a DEFERRED
  import, existing test mocks on geometric_amplification miss the new path.
  Fix: additionally patch `_airy_fold_module.fold_ppgo_correction` (the
  MODULE OBJECT, not the string) so the deferred import picks up the stub.
  Pattern: `mock.patch.object(_airy_fold_module, 'fold_ppgo_correction',
  _stub)`.
- INTERIOR TRAINING CONFIG WIRING: `_subdivide_farfield_tile` must mirror
  the main tiler's eff_w_nodes 3-way logic (tile override -> interior uses
  config.interior_w_nodes_per_decade -> else config.w_nodes_per_decade).
  A stale ternary in the subdivision path is a recurring bug pattern
  (INS-2-001).
- DD-CAP FORMULA REVIEW (Build wedge_followup, 56a223a): when reviewing a
  DD-product ceiling, verify the code uses `r_grid[-1]` (r_max), NOT r_min.
  r_max gives the tightest conservative global bound (`w_max * r_max *
  reach_max <= DD_MARGIN`). A brief that says r_min is a brief error — the
  code's deviation is correct and should be noted as an approved deviation,
  not a finding. Confirm `_log_w_grid()` is called AFTER the cap (not before)
  so the grid is built from the already-capped w_range.
- DD-CAP SUCCESS-RATE INTERPRETATION: do NOT assess the DD cap by measuring
  what fraction of training nodes succeed. The cap's job is to prevent
  `w * |y| > DD_MARGIN` (impossible requests); whether nodes pass the engine's
  independent Schwinger ceiling (~w~60 at large |y|) is a separate question.
  Tests should verify the FORMULA invariant (`w_max * r_max * reach_max <=
  DD_MARGIN` to float64 precision) and that `w_max` was actually reduced below
  the requested ceiling when the cap is binding. A 6% success rate at the
  correct cap is physics-expected and not a defect.
- ARC-LENGTH MAP REVIEW CHECKLIST (Build wedge_followup, 56a223a): (1)
  `theta_to_s` shape is (2, N) with N >= 100 (typically 2001). (2) Row 0 spans
  `[theta_wedge_grid[0], theta_wedge_grid[-1]]` EXACTLY (to ~12 digits) — if
  not, `np.interp` extrapolates and can produce wrong s values. (3) Row 1 starts
  at 0.0 and is strictly increasing (verify min positive diff). (4) Map is
  genuinely nonlinear (max residual from linear fit >> 1e-4 — confirms caustic
  curvature near cusps). (5) Grid-node accuracy after remap < 1e-9 (budget:
  ~6e-9 interp error at 2001 nodes). (6) Self-falsification: a perturbed
  `theta_to_s` should measurably degrade served accuracy vs. fresh engine.
- AXIS NODE COUNT CONSTRAINT: `_validate_axis` requires >= 4 nodes per axis for
  cubic spline interpolation. Brief/spec suggestions of n=3 grids
  (e.g. gamma=[0.25,0.45] → 2 nodes) are infeasible. Always use >= 4-node grids
  in test fixtures, and flag any spec that suggests fewer.
- LOCAL CONSTANT DUPLICATION PATTERN (INS-w3-001): a local variable
  `_ARC_MAP_NODES = 2001` that duplicates the value of a module-level constant
  `_FARFIELD_ARC_MAP_SIZE = 2001` is a trivial finding (harmless, but should
  reference the module constant). Flag as trivial/non-blocking when reviewing
  similar patterns.
- BRIEF-VS-CODE APPROVED DEVIATION: when production code correctly deviates from
  the build brief for physical correctness reasons (e.g. using r_max instead of
  brief's r_min for conservative bounding), document as an approved deviation in
  the review findings, not as a code defect. Explicitly note "correct deviation
  from the brief" so future reviewers don't revert it.
- BRIEF ACCEPTANCE CRITERION GAP (INS-1-001 pattern): when a brief's stated
  acceptance metric (e.g. "dropped fraction < 1e-3") is NOT achieved by the
  implementation, distinguish between a code defect and a brief estimation
  error — if the code correctly implements what was asked (e.g. reduce
  threshold to 0.005) and the residual drops are genuine physics (real
  topology boundaries narrower than min_width), classify as a brief
  estimation error, not an implementation defect. Carry forward as a
  non-blocking open issue with a note; do NOT demand the code be changed to
  match an aspirational metric that the physics does not allow.
- SADDLE-LOBE INTERIOR FOLD FALLTHROUGH: fold-ppGO correctly falls through for
  saddle-lobe interior (gamma > 1) because `_merging_fold_pair` returns None
  when no (Morse 0, Morse 1) adjacent pair exists. For saddle-lobe configs that
  DO have a fold pair the correction is valid and will serve correctly — the
  image_count==4 census restriction is physically correct.
- STALE BRIEF DETECTION — 0-WP ESCALATION PATTERN (Architect): before
  designing a multi-WP build, verify that the brief's claimed state of the
  code is actually current (grep/find_symbol the named symbols). Two
  confirmed stale-brief patterns this session: (a) `brief_saddle_born_carrier`
  — all 5 in-scope items had shipped in commits 31ee133 + 65eebcb; escalated
  as 0-WP. (b) `brief_analytic_cusp_serving` — build 1c shipped in b9c3ed6;
  escalated as 0-WP. Both saved the full cost of a multi-WP design cycle.
  Trigger: architect cross-checks brief's "TODO" list against HEAD before
  writing a plan — if the named symbols/constants are already present and
  the named tests pass, the brief is stale.
- INTER-LOBE CORRIDOR PROBE RESULT: for saddle-regime gammas (1.1–2.0),
  the inter-lobe corridor has 0.0% area overlap with either lobe interior.
  The two deltoid lobes are STRICTLY separated (lobe A at x ∈ [-1.52,-0.59]
  at gamma=1.1; lobe B symmetric). Despite corridor width/sep ratio being
  up to ~17% near the gamma=1 bifurcation, the corridor (x ∈ [-0.16,+0.16])
  is entirely outside both lobes. Region 2 CLOSED. No code change needed;
  the inter-lobe exact-engine fallback is purely an efficiency non-issue.
- CENSUS DRY-RUN REVIEW CHECKLIST: (1) classification gates must be
  structural-only (no engine calls, no chart artifact on disk); (2) born
  = rho>1 subsumes farfield; (3) ppgo_fold uses actual `_merging_fold_pair`
  (not a proxy); (4) each draw gets a fresh geometry object (label-
  continuation safety); (5) n_freq=2 is sufficient for structural-only;
  (6) 100% structural coverage = zero draws reaching `exact_engine` residual
  bucket confirms production training can proceed. If any non-trivial
  fraction lands in exact_engine, review gate ordering before launching.
- GENERIC SUBDIVIDER REVIEW (Build subdivision_recursion_and_coordinate_
  cleanup, 2026-08-07): when reviewing a bounded-recursion tile subdivider
  that replaces two near-duplicate single-level subdividers, confirm (a)
  legacy summary/report keys are preserved verbatim, with only ADDITIVE
  new keys (e.g. achieved_depth); (b) a field like 'packed' now counting
  the FULL subtree rather than just the immediate level is an intentional,
  brief-endorsed behavior change — verify it's INTENDED, not a silent
  regression; (c) a structural-refusal branch (e.g.
  CarrierDiscontinuityError) is caught and treated as a terminal gap, never
  recursed.
- WEDGE FIELD RENAME CONSUMER CHECK (same build): after renaming a
  dataclass field consumed via an isinstance-dispatched evaluate (e.g.
  Tube/Lobe/FarField keep theta_to_s, only Wedge is renamed to theta_to_u),
  grep ALL test files for the OLD field name RESTRICTED to the renamed
  TYPE only — sibling types that legitimately keep the old field name are
  correctly unaffected, not a coverage miss.
- POST-STRAND REGIONS-FILTER AUDIT (2026-08-07, remeasure_v3): reviewed
  surrogate_training.py regions filter + guard_slow_operation +
  _self_estimate + train_lens_surrogate.py --regions + 52 new tests
  (28 regions + 24 guard), all green, 64 existing surrogate-training tests
  green, no regressions. Invariants: regions=None preserves BYTE-IDENTICAL
  all-regions behavior (astroid + saddle); regions is keyword-only with
  None default; all 5 train() callers forward-compatible;
  guard_slow_operation refuses over-budget without the slow-tier env and
  admits with one. DELIBERATE asymmetry (documented, not a defect):
  `_self_estimate(..., ())` falls back to the FULL estimate (conservative,
  never undercharges) while `_train_band_charts(regions=())` builds
  nothing. Tests are value-asserting with reachable-red (self-falsification
  classes corrupt their contracts and prove the checks trip).
- PROBE ARTIFACT READING (driver probes, post-strand): after an NPZ load,
  the in-memory chart.provenance LACKS heldout_eps — a probe reading
  in-memory provenance reports NaN/missing eps and can be misread as a
  coordinate failure. Read heldout_eps from the NPZ provenance. An all-NaN
  reading from a config that does NOT match the production tiling
  (e.g. gamma_band_halfwidth 0.48 vs production 0.04) is a probe-config
  artifact, not a coordinate verdict.
- CLASS-DELETION FOLLOW-UP DEFECTS: after a chart-class deletion commit
  (0a31fcf, ~1064 lines) that shipped with "2 surrogate test failures
  remain (post-commit triage)", the follow-up defects were STALE
  REFERENCES — a stale FarFieldChart reference + a (s,d) docstring
  (5859a78, defects D1+D2). When a deletion commit lands with known red
  tests, grep the WHOLE tree (not just the deleted module) for the dead
  class name and its old coordinate vocabulary — the residual failures
  are usually references, not logic.