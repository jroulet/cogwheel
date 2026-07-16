# Inspector Long-Term Knowledge

- To certify an index-clamp or bounds trick, find the degree/step
  invariant that bounds how far an index can move per operation, and
  show that out-of-range table entries are provably zero (so the clamp
  never contributes wrongly) — don't just confirm "it didn't crash."
- When a truncation/series-length heuristic is rescaled, check it
  against the real peak-term location and magnitude-scaling law of the
  series, not the caller's original proxy variable. A heuristic sized on
  the wrong scale (e.g. |zz| instead of the true cancellation exponent
  w*sqrt(s)) undersizes exactly when the two diverge, truncating
  silently inside the nominally certified domain.
- When a spec discloses a certified sub-range narrower than the code's
  nominal domain (e.g. oracle-certified only to part of a wider nominal
  band), confirm the gap is honestly written up (SPEC + FINDINGS) rather
  than silently present. That gap is a real defect to carry forward as
  open, not something to close by widening tolerances.
- Confirm test oracles are non-circular: they should independently
  re-derive the expected value via a definitional form (cross-checked
  against the production formula by an identity), not call the
  production function itself. Look for (or add) an AST guard that
  forbids the module-under-test's names inside fixture builders.
- Pre-existing, environment-only collection errors (e.g. an optional
  external dependency not installed) are out of scope for a focused
  review of a specific change — note them, don't chase them.
- Run the mandated mutation check yourself when reviewing new test
  suites (perturb the load-bearing constant/branch, confirm red) rather
  than trusting that green tests imply correctness.
