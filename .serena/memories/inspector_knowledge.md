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
