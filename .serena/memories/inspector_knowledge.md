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
- QUOTA-DEATH SALVAGE RE-AUDIT (2026-08-08, lobe_cusp_coordinate build):
  the build died at inspector-17 (quota exhaustion) after coder-16 fixed
  revision-2 findings; the code was salvaged as b18e6a8 UNVERIFIED. The
  salvage audit re-ran ALL changed-file suites to completion (149 pass /
  16 pre-existing skip, 0 fail) and re-derived every invariant by hand:
  single schema tag `lobe_caustic_relative_v1` (no old constants anywhere),
  zero `theta_to_s` in ANY lobe code path (remaining refs = Tube/FarField
  only), `_LOBE_ARC_MAP_SIZE` + `_LOBE_CUSP_EXCLUSION_DISTANCE` deleted,
  lobe_cusps threaded through all tiers (train -> tile dict -> subdivision
  children -> nearest_cusp -> child boxes -> build -> from_lobe_engine),
  both from_lobe_engine paths (cusp-adapted + raw-theta fallback) work,
  `_lobe_cusp_axis_map` both sides u_fine[0]~0 monotone endpoint-exact.
  Lesson: a quota-killed build's salvage commit is UNVERIFIED — a green
  partial pass from before the death does not certify it; re-audit the
  whole changed surface from scratch. See `mem:lobe_interior_chart`.
- EXTERIOR-POLAR CUSP-ADAPTED U COORDINATE REVIEW (2026-08-08, Build
  exterior_polar_cusp_coordinate, 1a97bbd): all 8 test files fast-tier
  green except two defects. RESOLVED carry-overs: INS-3-003
  (`_train_tile`/`_train_exterior_chart` hardcoded origin='low' -> shared
  `_exterior_cusp_axis_map` mirroring production waist-origin, null
  fallback for unrepresentable tiles) and INS-3-004 (sentinel block
  rewritten as explicit 3-case contract). NEW BUG INS-4-001: the wedge
  branch of `_chart_from_npz` was changed to `data.get(prefix+
  'theta_to_u')` (soft None fallback) but InteriorWedgeChart V3 REQUIRES
  theta_to_u — the soft read silently loads a corrupt V3 artifact with
  None and breaks `test_v3_missing_theta_to_u_raises_keyerror`. The
  `.get()` fallback is correct for the exterior-polar loader (optional
  field) and acceptable for the lobe loader (pre-existing latent trap),
  but WRONG for the wedge loader — per-kind required-vs-optional
  decision, not blanket-applied. NEW BUG INS-4-002: three test classes
  unskipped from the polar re-chart skip use
  `definition=ch.INTERIOR_SACR_C` with from_engine, which validates the
  tag against `KNOWN_FARFIELD_DEFINITIONS` — the interior tag is not in
  the far-field set, so every test raises ValueError; the unskip was
  premature (re-apply the skip or add non-farfield definition support to
  from_engine). DESIGN INS-4-003/004 (Librarian scope): SPEC.md +
  DATA_CONTRACTS.yaml still cite the OLD 'exterior_polar_rho_theta_c'
  tag and lack theta_to_u — carried to the doc-sync phase.
- TEST-CLASS-UNSKIP VALIDATES DEFINITION TAGS: when re-enabling test
  classes that were skipped during a chart migration, verify every
  `definition=` tag they pass to from_engine is still in the production
  KNOWN_*_DEFINITIONS set — a migration that removes a tag makes the
  unskipped class raise ValueError at construction (INS-4-002 pattern).
- CUSP PPGO FAST RUNG REVIEW (2026-08-09, Build cusp_ppgo_high_w): all 13
  ppGO tests pass, INS-6-001/002 resolved. Verified: r_ppgo_min formula
  (50*1/(0.05/10))^(2/3) ~ 464.16 consistent with test comments;
  fold_ppgo_correction signature matches the call; LensDomainError caught,
  non-finite guarded, falls through to Pearcey; rung positioned BEFORE the
  existing Pearcey uniform path (correct — ppGO is faster); _W_PPGO_FLOOR
  prevents ppGO at low w; existing tests unaffected (control radii << 464);
  all new tests increment n_checks (anti-vacuity); mock patching on the
  same module object; self-falsification verifies guard teeth.
  PRE-EXISTING (not actionable, not ppGO-induced): the two slow tests
  `test_moving_error_const_threshold_flips_a_fixed_node` and
  `test_served_node_is_bit_identical_to_the_cusp_arm` time out via
  `_grid_served` -> `F_op_grid` -> mpmath quadrature at w=80 (Professor
  confirmed the ppGO gate fails these configs on both branches —
  r_ppgo_min ~25x the radius at the low-const setting). Note, don't chase.
- 2D (rho, u) FOLD-CARRIER REVIEW CHECKLIST (Build exterior_2d_fold_carrier,
  re-review PASS, 2026-08-10): (a) demod broadcast
  exp(-1j*w*carrier[None,None,:,:]) shape (n_w,1,n_rho,n_theta_c) x
  envelope (n_w,n_gamma,n_rho,n_theta_c); (b) NaN fill order along u
  (axis=1) then rho (axis=0), zero-order hold at boundaries, all-NaN->None;
  (c) NPZ backward compat: data.get('rho_u_carrier') then
  data.get('rho_carrier') -> np.broadcast_to 1D->2D; (d) V4+V5 both in
  _KNOWN_EXTERIOR_POLAR_AXIS_SCHEMAS, _chart_to_npz AND _build_provenance
  write V5; (e) grep stale `_compute_rho_carrier`/`rho_carrier` in
  production AND tests — the ONLY surviving 'rho_carrier' is the backward-
  compat NPZ load key; (f) from_engine continuity gate + k_chart BOTH use
  the 2D-demodulated envelope when fold_carrier=True; (g) serve re-
  modulation at the interpolated u (after theta_c->u map), never raw
  theta_c.
- STILL OPEN -> Librarian (INS-1-002/003, doc staleness, NOT code defects,
  2026-08-10): SPEC.md ~line 63 and DATA_CONTRACTS.yaml ~line 199 still
  call exterior_polar_rho_log_carrier_v1 "the ONLY known tag" and describe
  rho_carrier as 1D (n_rho,) — stale since V5 2D shipped; code verified
  correct, OVERRIDE to Librarian doc-sync (recurring rule).

- SADDLE EXTERIOR FULL TREATMENT REVIEW (2026-08-10, Build
  saddle_exterior_full_treatment, 238d21e, re-review 3rd pass — ALL code
  changes correct, NO new findings): (1) `_deltoid_cusp_axis_map` mirrors the
  wedge/lobe cusp-adapted map pattern: correct 2/3 exponent for gamma-
  universal cusp-reach scaling, straddle->None, [0, pi/2] validation raises
  ValueError, np.clip FP guard + explicit endpoint pinning safe. (2)
  `_build_farfield_chart` parity==-1 branch activates ONLY on boundary cusp
  rays (nearest == theta_lo or theta_hi) — an interior nearest cusp would
  straddle and return None; falls through to theta_to_u=None otherwise. (3)
  `_tube_serves` parity dispatch (coverage = _SADDLE_CUSP_ARM_COVERAGE if
  chart.parity==-1 else _CUSP_ARM_COVERAGE) is correct; _SADDLE_CUSP_ARM_
  COVERAGE=0.0 is load-bearing (nonzero would admit queries the Pearcey arm
  cannot serve for saddle parity). (4) `_chart_from_npz` wedge branch KEEPS
  hard data[prefix+'theta_to_u'] (KeyError) — correct, NOT changed to .get();
  exterior-polar + lobe branches keep .get() (optional field). (5) Docstring
  updates _exclude_ghost_dominated / _needs_fold_carrier ('Positive-parity
  only' -> 'Both parities') correct; no stale 'Positive-parity only' refs in
  surrogate_training.py. (6) Fast tier: test_lensing_surrogate.py 123 pass,
  test_lensing_surrogate_training.py 113 pass/90 skip (train-tier gated),
  test_lensing_farfield_envelope.py 36 pass/28 skip. (7) Measurement script
  scripts/measure_saddle_cusp_arm_coverage.py functional but UNTRACKED
  (post-build calibration tool). STILL OPEN -> Librarian: INS-1-001 — SPEC.md
  lines 72-76 ('macro-saddle parity==-1 exterior interpolates on raw theta_c,
  no map') now stale: the code builds cusp-adapted theta_to_u maps for saddle
  exterior tiles; doc-staleness flag, NOT a Coder defect (recurring rule).
  PRE-EXISTING (not this build): INS-1-002/003 exterior_polar_rho_log_
  carrier_v1 'ONLY known tag' staleness since V5 2D carrier shipped.
- PPGO RESOLUTION GATE REVIEW (2026-08-11, Build operator_routing_one_home,
  findings NONE): the dual gate correctly separates fold-pair nodes (Morse
  0,1: fold_ppgo_correction valid regardless of resolution) from saddle-only
  nodes (Morse 2,3: only valid above w*delta_min >= 4.0). _PPGO_RESOLUTION_
  GATE=4.0 matches operator.RHO_END; delta_min from pairwise SORTED delays,
  len<2 -> 0.0 (conservative); _merging_fold_pair inside try/except (can
  raise LensDomainError) correct. Fold-pair + resolved-saddle nodes are
  BYTE-IDENTICAL to pre-change; only unresolved saddle nodes are newly
  refused (the brief's failing configs, w*delta_min ~1.90 < 4.0). Self-
  falsification teeth verified: gate->1000 blocks at w=500, gate->0 admits,
  resolved w (20000) still admits. Gate: 128 passed / 11 skipped / 2 xfailed
  (operator + fast_path + airy_fold); all ppGO classes pass. PRE-EXISTING
  (not this diff): 8 vertex-related tests in test_lensing_airy_fold.py red
  at HEAD from the separate _cusp_vertex routing-fix build (already
  committed); INS-1-001/002/003 doc staleness carried to Librarian.

## 2026-08-12 build (lobe_exterior region wiring, pass-3 PASS)

- LOBE_EXTERIOR REGION WIRING FINAL REVIEW (Build lobe_exterior_region_
  wiring, pass-3 PASS): INS-7-001 (stale doc comment claiming lobe_exterior
  NPZ theta_to_u followed the wedge hard-read convention) resolved by a
  doc-only reword; re-confirmed all wiring (default region tuples, per_region
  cost dict, packing/admissions gates, build-loop dispatch, CLI --regions
  choices) correct and unchanged since pass-1/2. NEW -> Librarian (INS-5-001,
  doc staleness, not a Coder defect): SPEC.md + DATA_CONTRACTS.yaml have ZERO
  mention of lobe_exterior / lobe_interior / wedge_interior even though this
  build makes lobe_exterior a PUBLIC --regions CLI choice + NPZ
  kind='lobe_exterior' + training-region contract — the spec never names the
  region vocabulary the trainer/CLI now exposes (bidirectional divergence,
  joins the INS-1-001/002/003 doc-staleness lineage carried to Librarian).

## 2026-08-13 (saddle_above_ceiling_serving, pass 3, still open)

- MANIFEST TRUST TRAP: a "files actually changed" manifest that LISTS a
  test file does NOT mean it was edited — it can be untracked and
  byte-identical to a prior red state. ALWAYS re-run the named
  acceptance-gate files directly (don't just diff-stat them); never trust
  the manifest, or an agent's claim that a signature/fixture skew was
  fixed, without a fresh pytest run reproducing green.

## 2026-08-13 (ppgo_interior_certificate, fold_exterior_ghost — both PASS)

- FIRST QUESTION FOR ANY GATE: *which object's error does this estimate
  bound?* Four defects in one day were all the same shape — a check that
  measures something other than the error it claims to bound (F069 the
  estimate decayed while the true error stayed flat; F070 the clamp licence
  keyed on the label; F074 the radius gate bounded the error of the object
  the rung REPLACED; F076 the resolution gate read the wrong image pair).
  A gate is only reviewable against the SERVED object's own asymptotics
  plus a calibration run against an F069-safe oracle; "it never
  over-certified on our grid" is not an answer.
- REAL-PAIR BLINDNESS: any gate computing min-gap / xi / resolution /
  delta_tau over REAL images is structurally blind exterior to a caustic,
  where the merging pair is COMPLEX (and saddle mirror pairs give
  delta_tau exactly 0). Flag it on sight; the exact discriminator is the
  real-image COUNT (4 interior / 2 exterior, both parities).
- MIRROR FIDELITY: a census / training mirror of a production rung must be
  a FAITHFUL mirror — same predicate, same estimate call, and the safety
  constant BOUND from the production module rather than re-typed — and the
  superseded machinery must be gone with zero dangling refs. Check the
  mirror in the same review as the rung; it is the classic laggard.
- HANDOFF-ONLY ROUTING SILENTLY NO-OPS: a finding routed to another agent
  can come back unfixed with the file untouched. Confirm the target file is
  actually modified (` M` in status) as well as green; when the same
  finding survives a second pass, direct execution explicitly rather than
  re-routing a third time.
- A PINNED FIXTURE LITERAL AT A PHYSICS BOUNDARY IS SELF-GUARDING: a pinned
  source that must stay 4-image is not a silent-strand risk when its OWN
  test goes red the moment it leaves that domain (the fold refuses ->
  assertIsNotNone fails). Domain boundaries are physics, not movable
  constants — don't flag them alongside genuine stale pins.
- CONTAMINATED-BUT-CONSERVATIVE TRAINING ARTIFACTS ARE A DRIVER RETRAINING
  ADVISORY, NOT A CORRECTNESS DEFECT: when a physics fix retroactively
  invalidates labels drawn through the defective path, decide the DIRECTION
  first — if the contamination can only cost coverage/perf and can never
  over-certify, carry it to the driver as a retraining advisory instead of
  blocking the build. Bound the contaminated set exactly (parity x band x
  the w-nodes that actually enter the affected band); a producer whose grid
  tops out below the band is provably CLEAN.

## 2026-08-14 (symmetry_tie_c3_admission, re-review PASS)

- MANIFEST TRUST TRAP (reconfirmed): a task manifest can list files as
  "changed" that are actually git-deleted, while a new untracked file it
  never mentions is the real addition — always `git status` + a fresh run
  to verify, never trust the manifest string. This session: manifest
  listed 3 test files as changed; git showed them DELETED, with a 4th
  untracked file (the real replacement suite) unmentioned.
- A GATE-CONTRACT SWAP that also changes the gate's ARGUMENT SIGNATURE
  (3-arg -> 4-arg here) requires verifying every call site by name
  (`find_referencing_symbols` + a grep for the function name string) —
  a stale caller on the old signature fails loudly (TypeError) at
  collection/call time, unlike a same-signature semantic drift which can
  fail silently; still worth the explicit pass since a partial re-key
  (production fixed, census mirror left old-signature) is the recurring
  laggard failure mode in this codebase.

## 2026-08-14 (born_residual_wiring, INS-3-001)

- ARTIFACT-VS-SPEC COVERAGE CLAIM: verify a doc's coverage claim (e.g.
  "covering the far exterior on both parities") against the ACTUAL trained
  artifact's grid axes (gamma_grid/rho_grid/log_w_grid), not just the
  prose describing intent — a rewritten doc entry can still assert
  coverage the shipped npz's grid never included (here: astroid-only,
  gamma_grid all <1.0, no saddle node, despite the doc saying "both
  parities"). Bidirectional finding: retrain to match spec OR narrow the
  spec to match the artifact — direction is a triage call, not an
  Inspector call.


## 2026-08-14 (certified_map_guard_relaxation, F080, final pass)

- STALE-DOCSTRING-AFTER-GUARD-REMOVAL: when a build removes a guard and
  rewrites SOME sibling test docstrings, a missed sibling can still pass
  but for a DIFFERENT reason than its stale docstring claims (latent
  vacuity) — sweep every docstring naming the removed mechanism, not just
  the one the finding cites, and re-verify any numeric claim (e.g. why a
  test passes) against a fresh run rather than trusting the prose.


## 2026-08-15 (saddle_tube_fundamental_training, F081, pass-2)

- DEFERRED FIX MUST RE-DERIVE, NOT DELETE OR LOOSEN: when a prior pass
  defers 3 test-file findings to a later run ("owned by other runs"), that
  deferral rationale is itself invalid — the correct fix re-derives each
  broken expectation from the live production selector (e.g.
  st._tube_training_arcs(structure, parity)) inside the SAME build that
  changed the selector, not a delete-and-move-on. Confirmed green via
  fresh collect-only + targeted suite runs, not by trusting the diff.


## 2026-08-15 (lobe_cusp_axis_edge_tolerance, PASS)

- Reviewed the `_lobe_cusp_axis_map` edge-coincidence ULP-tolerance fix
  (surrogate.py): signature unchanged (4 positional); both production
  callers (from_lobe_engine, from_lobe_exterior_engine) and
  `_lobe_child_boxes` set `side` from the tile CENTRE via
  `_lobe_nearest_cusp`, guaranteeing the far-edge distance is > 0 — the
  negative-base complex-power regime is UNREACHABLE via production
  callers (noted, not a finding). Sibling audit confirmed correct:
  `_wedge_cusp_axis_map` has no cusp-vs-edge guard (cusp pinned to domain
  edge by construction); `_deltoid_cusp_axis_map` already handles the
  coincident/straddle case via `None` + non-strict branch. New test
  suites cover endpoint bit-exactness, monotonicity, the boundary
  trichotomy (exterior/on-edge/hair-inside->map, straddle->raise) on both
  sides, and self-falsification. Full targeted + neighbor suites green
  (139 passed, 10 skipped train-tier). No SPEC/DATA_CONTRACTS impact.
- RESOLVED (verify before re-flagging): the carried-forward INS-1-002/003
  "exterior_polar_rho_log_carrier_v1 'ONLY known tag' staleness since V5
  2D carrier" item is confirmed closed — Librarian's short-term memory
  independently found SPEC.md/DATA_CONTRACTS.yaml already correctly
  describe the V4/V5 two-tag set and the 2D rho_u_carrier, fixed by some
  earlier untracked pass. Do not re-open this pair from memory alone;
  re-verify with a fresh grep if it resurfaces.

## 2026-08-17 (tube beat-free recovery, multiple passes)
- REFUSING-DEFAULT NEW-ARG PATTERN: a serve guard gaining a NEW required
  input whose DEFAULT triggers refusal (e.g. NaN->decline) fails every
  un-updated caller SILENTLY (no TypeError, just always-False/refuse) —
  `find_referencing_symbols` on the guard is mandatory, including
  census/diagnostic mirrors and direct unit-test callers; sweep EVERY


## 2026-08-18 (low_w_diffractive_rung / serve_route_census WP3, INS-5-001)
- CONTRACT-WIDENING LAGGARD: widening a shared enum/contract (e.g.
  SERVE_ROUTES 8->10, or a route gaining a new detail field like a
  w_split-carrier) silently strands PRE-EXISTING tests in a file OUTSIDE
  the build's changed-file manifest that pin the old count / old
  iff-contract. Always re-run the schema/census invariant test file
  (often a different file than the one carrying the new behavior) after
  any SERVE_ROUTES/ROUTE_KINDS/per-record-detail change — a green
  behavior suite does not cover the census invariant suite.
- MagicMock-hides-new-attribute recurred a 3rd+ time this build (two
  separate stub/probe classes missed binding a NEW self.<method> call
  added mid-build: `_diffractive_bottom_ceiling` on `_BornAnalyticProbe`,
  `_engine_envelope_below_split` on the above-ceiling gate stubs) —
  whenever a WP adds a new `self.<method>` call inside an existing method,
  sweep every stub/probe class that binds unbound production methods for
  that same class, not just the caller the WP names.

## 2026-08-18 (INS-1-002, born_carrier_certificate review)
- DELIBERATE-SCOPE PROBE EXCEPTION to the MagicMock-hides-new-attribute
  lineage: an engine-free-BY-DESIGN probe legitimately not binding a new
  engine-dependent method (e.g. `_engine_envelope_below_split`) is NOT
  automatically the same defect -- check whether the route that method
  serves is even reachable engine-free before flagging; if not, it's a
  documented gap for a future engine-aware probe, not a stub-hides-
  attribute finding.

## 2026-08-19 (tiling_plan pass-4 PASS)
- DELIVERABLE-ARTIFACT-ABSENT IS NOT AUTOMATICALLY A DEFECT: when a build's
  target output (e.g. a `.claude/handoff/*.json`) doesn't exist on disk yet
  but the function that would produce it (`run()`/`build_plan()`) is
  exercised end-to-end by a passing test and its logic is independently
  verified, carry the missing artifact forward as a DRIVER run-the-CLI
  step, not a code defect — don't block the build on an operator action.

## 2026-08-20 (diffractive_certificate_fit_interior_fix, pass-3)

- NUMERIC-CLAIM DOCSTRING SWEEP AFTER A COEFFICIENT RE-BAKE: when a re-bake
  updates SOME docstring 'measured ~Nx' values, sweep EVERY docstring
  carrying a measured-ratio claim for that surface — the caustic-feature
  self-falsification docstring (~1.66x -> ~1.91x) is the laggard that
  survives even a meticulous sibling pass (test still passes; docstring
  only).
- OVER-CLAIM IN A TEST NAME/DOCSTRING AFTER A FIXTURE REFACTOR: a method
  named '..._at_calibrated_cell' whose fixture refactor widens coverage to
  an EXTRAPOLATED cell re-introduces the exact over-claim the earlier
  docstring lineage cleaned up — the assertion may be structurally correct
  while the NAME over-claims; rename to keep wording == coverage.
- FENCE-WITNESS MANAGEMENT PATTERN (reusable): when a new refusal band
  lands, name the witnesses now inside it via a module constant (e.g.
  NEAR_FOLD_DECLINED_WITNESSES) and skip them in the sweep, adding a
  dedicated test that asserts the band IS declined — don't silently drop
  witnesses or claim the sweep covers them.

## 2026-08-20 (diffractive_wall_nearfold_chart, pass-4 FINAL)

- HASH-CONTRACT COMPLETENESS VERIFICATION (INS-2-002, RESOLVED both sites):
  a correctness-critical field folded into an artifact content hash must be
  byte-identical at EVERY site (train bake, load recompute, test helper —
  identical float64 bytes, identical field ORDER), and the tamper test must
  premise-assert the tampered field is non-trivial (`declined_mask.any()`
  and `not .all()`) so a future all-False collapse can't silently make the
  tamper a no-op. Round-trip/rehashed-tamper-loads-cleanly positive
  controls + non-vacuous tamper-refuses all green (36 passed); train script
  self-check asserts `np.array_equal(loaded.declined_mask, declined_mask)`.
- NEW-CHANGE AUDIT confirms (no new findings): serve reconstruction identity
  verified vs the test oracle — `sqrt(mu_pure) = lam*sqrt_mu` cancels the
  `1/lam`, so `F = mass_sheet_phase*prefactor_c(w)*sqrt_mu_full*r_pure`
  holds exactly; mass_sheet_phase `exp(0.5j*w*(log(lam)-kappa*s))` matches
  the oracle; reconstruction tail (demod by t_min -> reconstruct_farfield ->
  _reduce_dense_kernels -> _image_delays) byte-identical to
  `_low_w_diffractive_serve`. `_AUTO_LOW_W_CHART` sentinel + get_init_dict
  handling mirrors the born_residual_chart pattern exactly. rho_dir =
  caustic_rho(abs(gamma'),s,theta) is a FRESH LOCAL — it does NOT rebind the
  outer scalar `rho` gauge (INS-2-001). SERVE_ROUTES 11->12 widening is
  laggard-safe: the census invariant test derives from dynamic
  `src.SERVE_ROUTES` (no hardcoded 11).
- RESOLVED -> Librarian: the SPEC.md "near-fold shell DECLINED" staleness +
  missing DATA_CONTRACTS.yaml entry (INS-1-003) were fixed by the in-DAG
  Librarian (chart entry + spec_version/schema_version bumps + completion
  fragment). Do not re-open from memory alone.
- ADVISORY (driver): shipped `cogwheel/data/low_w_diffractive_chart.npz`
  still absent — full bake is a DRIVER post-build step; the content hash now
  covers declined_mask, so bake AFTER the hash-fix commit.

## 2026-08-21 (low_w_chart_cusp_fallback RE-REVIEW, pass 3 FINAL)

- RE-VERIFY RESOLVED FINDINGS IN CODE + GREP, not from memory: on a re-review pass, re-check each previously-open finding against the live code (INS-2-001 dead fields gone — only the 9 live fields remain; INS-2-002 scalar accessor dropped — only `cusp_uniform_reference_grid` remains, consumed by `_pearcey_cusp_reference` -> `fold_cusp_reference`), re-run the full changed-file suites, and re-derive the key identity by hand.
- GRID-DEPENDENT GUARD RATIO NOTE (non-finding): the non-vanishing guard min|F_ref|/max|F_ref| is computed over the bake w_grid at train time and over the likelihood dense_w at serve — so a serve/bake decline ASYMMETRY is possible, but the failure mode is a SAFE decline (-> exact engine); per-node F_ref values are grid-independent. If a future census shows unexpected Pearcey-cell declines at serve, this asymmetry is the first explanation to check.
- `cusp_uniform_reference_grid` solves geometry ONCE per cell, loops only w, calls `_cusp_uniform_at_w` per node WITHOUT the ppGO rung / F074 gate / calibration certificate; `_consult_pearcey(x,y,None)` -> live quadrature. The non-vanishing guard catches cluster_sum->0 collapses; max==0 -> nan -> the isfinite guard declines.


## 2026-08-21 (low_w_shell_born_extension, pass 5 FINAL)

- CROSS-GAUGE SINGLE-SOURCING (INS-3-001 -> INS-4-001 lineage): a "same
  constant (no gap/overlap)" claim spanning two rungs is FALSE whenever the
  two boundaries live in DIFFERENT rho gauges (scalar reach vs directional)
  — identical float values (both 1.4) do NOT make two physical surfaces
  equal. Verify the GAUGE of each "rho" before accepting "no gap/no
  overlap" prose in ANY surface; value-equality pins (assertEqual) are
  correct but do not license surface-equality prose.
- PLAN-LISTED-SPEC-EDIT MISS: when the plan lists SPEC.md +
  DATA_CONTRACTS.yaml as expected-to-change but the build edits neither, a
  green code diff does NOT certify the doc surfaces — sweep them explicitly
  whenever a build renames a data-product tag/route AND changes a gate
  constant (recurring INS-1-xxx lineage, now INS-5-xxx).
