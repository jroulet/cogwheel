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
  staleness is a flag-to-Librarian finding, not a Coder defect. A build
  the plan expected to touch SPEC but didn't = flag to Librarian.
- Re-reviewing a byte-identical diff: still re-run the suite + import
  probes and re-derive the key identity by hand.
- DATA_CONTRACTS covers serialized/shipped artifacts only; new fields on
  an in-memory dataclass need no contract change (pickle __getstate__
  still deserves a round-trip probe). An OFFLINE-ONLY artifact (surrogate
  .npz, trained but not shipped/consumed by pipeline scripts) likewise
  needs no entry yet — revisit if a file is actually shipped/consumed.
- A serve/approximation gate must include EVERY parameter axis the
  approximation was trained at. A missing axis (e.g. surrogate has no
  kappa axis, trained kappa=0) that is harmless only because production
  pins that axis is still a LATENT correctness violation of the
  conservative-serve contract — flag it, don't wave it through on
  "non-triggering in production".
- Refusal-net reviews: trace EVERY dispatch route to the boundary —
  scalar lnposterior AND the sampler's prior.unfold_apply wrap must both
  route through the override; except must name specific refusal types,
  never bare (Build 4).
- Check the approved plan JSON before flagging a constraint deviation: an
  explicitly documented/approved deviation (e.g. d_app deferral to Build
  5) is not a finding.
- A documented @expectedFailure efficiency aspiration (prior-width
  refusal fraction) is a property, not a bug — but verify all non-finites
  are exact -inf with zero NaN before accepting.
- Builds run in the sibling worktree cogwheel-claude-dev; `cd` to the
  main tree is hook-blocked — run Bash from the worktree cwd.
