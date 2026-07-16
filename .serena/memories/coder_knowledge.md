# Coder Long-Term Knowledge

- "Coders write, downstream verifies": when the sandbox denies shell /
  `ast.parse` / runtime-import checks, don't block delivery on them.
  Verify everything checkable read-only (line-length via pattern search,
  forbidden-import/name scans, signature cross-checks via
  find_symbol), state plainly what's UNVERIFIED (e.g. "ast.parse denied
  — read-back shows valid syntax, no undefined names"), and let
  Inspector/Test Dev do runtime verification. Don't retry a denied shell
  call repeatedly in one task.
- Numerical-series pitfall: when a brief says to share one computed
  quantity "across all k" (or similar), check whether the obvious shared
  quantity itself overflows/underflows before adopting it — the fix is
  often to share a term that stays O(1)-scaled (e.g. a reciprocal-
  binomial factorization) instead, even when the brief doesn't warn
  about it. Verify the factorization via an algebraic identity check,
  not just "it matches the brief's headline numbers."
- Large-magnitude phase/angle arguments (e.g. built from a product like
  w*s/2) can lose precision in an ordinary float64 `exp(1j*angle)` even
  when the surrounding series is computed in double-double — reduce such
  angles mod 2*pi in double-double before exponentiating, rather than
  adding trig helpers to the primitives module.
- Before reusing a "shared" primitive across modules (e.g. a gauge/
  projection routine), check for redundant re-derivation bugs at the
  call site — e.g. normalizing already-normalized weights a second time
  silently reintroduces float error. Exactness tests can stay green
  through this class of bug, so cross-check the arithmetic by hand, not
  just by test-passing.
- Accuracy tests near coordinate singularities (e.g. near-fold blowup)
  need a scale-aware error bound (roughly eps times the magnitude of the
  terms summed), not a flat absolute epsilon — a flat 1e-12 gate can
  fail by orders of magnitude while the scale-aware bound holds with
  large margin. Assert both, with a canary that would go red if the flat
  gate ever started passing spuriously.
- When a module chooses between two computational branches gated by
  different criteria (e.g. wave vs. geometric), a refusal/exception from
  one branch's sub-component is independent of the other's gating
  logic — don't add a catch/fallback across that boundary; letting it
  propagate preserves certified-domain guarantees.
- Mutation testing (deliberately break each design decision one at a
  time, confirm the suite goes red) is the standard way to catch tests
  that are structurally blind — e.g. a fixture built from the same code
  path it's meant to verify stays green even when that path is wrong.
