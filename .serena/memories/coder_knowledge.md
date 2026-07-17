# Coder Long-Term Knowledge

- "Coders write, downstream verifies": when the sandbox denies shell /
  `ast.parse` / runtime-import checks, don't block delivery on them.
  Verify everything checkable read-only (line-length via pattern search,
  forbidden-import/name scans, signature cross-checks via find_symbol),
  state plainly what's UNVERIFIED, and let Inspector/Test Dev do runtime
  verification. Don't retry a denied shell call repeatedly in one task.
- Numerical-series pitfall: when a brief says to share one computed
  quantity "across all k", check whether that shared quantity itself
  overflows/underflows before adopting it — the fix is often a
  reciprocal-binomial-style factorization that stays O(1)-scaled. Verify
  via an algebraic identity, not just "matches the headline numbers."
- Large-magnitude phase/angle arguments (e.g. w*s/2) can lose precision
  in an ordinary float64 `exp(1j*angle)` even inside a double-double
  series — reduce mod 2*pi in double-double before exponentiating.
- Before reusing a "shared" primitive across modules, check for
  redundant re-derivation bugs at the call site (e.g. re-normalizing
  already-normalized weights). Exactness tests can stay green through
  this bug class — cross-check the arithmetic by hand.
- Accuracy tests near coordinate singularities need a scale-aware error
  bound (~eps times the summed terms' magnitude), not a flat absolute
  epsilon. Assert both, with a canary that would go red if the flat gate
  ever passed spuriously.
- When a module chooses between two branches gated by different criteria
  (e.g. wave vs. geometric optics), a refusal/exception from one branch
  is independent of the other's gating — don't catch/fallback across
  that boundary; letting it propagate preserves certified-domain
  guarantees.
- Mutation testing (deliberately break each design decision, confirm the
  suite goes red) catches tests that are structurally blind — e.g. a
  fixture built from the same code path it's meant to verify.
- A flat, parameter-independent "floor" (constant across many decades of
  frequency/mass) can be an exact closed-form limit (e.g. macro-image
  magnification as w->0) rather than noise — verify independence before
  "fixing" it with a short-circuit, which can inject a real discontinuity
  and destroy an already-correct exact limit.
- If forced to author both a module's source AND its tests under
  re-dispatch (handoff to a dedicated test author not honored), flag it
  explicitly for mandatory independent review of oracle/mutation
  independence — self-authored tests risk circularity.
- Neighbor/nearest-comparison bugs: filtering a candidate set to only
  "real"/resolved members before computing a nearest-neighbor distance
  can silently miss a virtual/placeholder member that is actually
  nearest — check the spec's definition of the comparison set (full
  cluster vs. real-only) before restricting it.
