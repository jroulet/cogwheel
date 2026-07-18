# Coder Long-Term Knowledge

- "Coders write, downstream verifies": if the sandbox denies shell or
  runtime checks, verify what's checkable read-only, state plainly what
  is UNVERIFIED, and let Inspector/Test Dev run the rest; don't retry
  denied shell calls repeatedly.
- Numerical-series pitfall: a quantity shared "across all k" can itself
  overflow/underflow — prefer a reciprocal-binomial-style factorization
  that stays O(1)-scaled; verify via an algebraic identity.
- Large phase arguments lose precision in float64 exp(1j*angle) even
  inside a double-double series — reduce mod 2*pi in double-double first.
- Before reusing a "shared" primitive, check the call site for redundant
  re-derivation (e.g. re-normalizing normalized weights); exactness
  tests can stay green through this — cross-check arithmetic by hand.
- Accuracy tests near coordinate singularities need a scale-aware bound
  (~eps * summed-term magnitude), plus a canary for the flat gate.
- Don't catch/fallback across independently gated branches (wave vs
  geometric optics): let refusals propagate to preserve certified-domain
  guarantees.
- Mutation-test each design decision (break it, confirm red) to catch
  structurally blind tests (fixture built from the path under test).
- A flat parameter-independent "floor" can be an exact closed-form limit
  (macro magnification as w->0), not noise — verify before
  short-circuiting.
- If forced to author both source and its tests, flag for mandatory
  independent review of oracle/mutation independence.
- Nearest-neighbor over a filtered "real-only" candidate set can miss a
  virtual/placeholder member that is actually nearest — check the
  spec's comparison-set definition (full cluster vs real-only).
- numba njit freezes module-level constants and binds compiled callees
  at compile time — monkeypatched globals/primitives are silently
  ignored; instrument tests through the full .py_func chain (F010).
- fastmath=False is load-bearing wherever error-free transforms
  (two-sum/two-prod, double-double) exist — FMA contraction breaks
  them. njit only pure float64/complex128 loops; keep domain
  validation, scipy special functions, and refusal logic in Python.
- Batched-API refactor: collect branch-subset indices in the loop, one
  batched call outside, scatter back; guard the empty subset; a refusal
  raising inside the batch before scatter keeps refusal symmetry. Make
  the scalar API delegate to the batched core (one certified path).
- Replacing numpy/BLAS reductions with explicit njit loops changes
  accumulation order (pairwise -> sequential); ULP-level equivalence
  claims need re-certification in deep-cancellation regimes.
