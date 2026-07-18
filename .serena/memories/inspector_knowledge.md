# Inspector Long-Term Knowledge

- To certify an index-clamp or bounds trick, find the degree/step
  invariant bounding index motion and show out-of-range table entries
  are provably zero — don't just confirm "it didn't crash."
- When a truncation/series-length heuristic is rescaled, check it
  against the real peak-term location and magnitude-scaling law, not
  the caller's original proxy variable — else it truncates silently
  inside the nominally certified domain.
- When a spec discloses a certified sub-range narrower than the code's
  nominal domain, confirm the gap is honestly written up (SPEC +
  FINDINGS); carry it as an open defect, never close by widening
  tolerances.
- Confirm test oracles are non-circular: they must independently
  re-derive the expected value, not call production code; look for (or
  add) an AST guard forbidding the module-under-test's names.
- Pre-existing environment-only collection errors are out of scope for
  a focused review — note them, don't chase them.
- Run the mandated mutation check yourself when reviewing new test
  suites (perturb the load-bearing constant/branch, confirm red).
- When a "goes to X" test fails from legitimate physics at a nonzero
  offset, fix via a dedicated fixture where the closed-form value truly
  is X, paired with a contrast test predicting the excluded case.
- When sibling suites each build an independent reproduction of the
  same buggy primitive, prefer a shared upstream source — flag the
  asymmetry as stylistic, not blocking.
- njit compilation voids Python-level monkeypatch instrumentation: new
  njit cores must expose .py_func, and falsification tests must be
  shown to actually go RED through that chain (FINDINGS F010).
- A scatter/weight-vector reduction replacing a bilinear form is a real
  accumulation-order change: require re-certification vs an independent
  oracle at the ORIGINAL tolerance plus solo-vs-batch certify-XOR-refuse
  decision identity (no cross-node convergence-state leakage).
- Single-path delegation (scalar API wrapping the batched core) means
  existing suites auto-exercise new code; a dedicated new test module
  instead of editing existing suites is a benign plan deviation.
