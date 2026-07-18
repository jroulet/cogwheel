# Coder Long-Term Knowledge

- "Coders write, downstream verifies": if the sandbox denies shell/runtime
  checks, verify what's checkable read-only, state UNVERIFIED items
  plainly; don't retry denied calls repeatedly.
- Verify a plan's "code-pinned"/"already exists" claim by grep/find_symbol
  before building on it; if the primitive is absent and supplying it needs
  out-of-scope design or new physics, BLOCK and escalate — never fabricate
  an unverifiable oracle to satisfy a WP (Build 3e refusal pattern).
- Never author certification gates for your own WP code. When retiring
  superseded tests (finding sanctions "or retire"), leave a loud OWED list
  naming the replacement gates Test Dev must author; mutation-test design
  decisions you can, and flag oracle independence for external review.
- After a regex/lookahead-anchored method retirement, re-parse the file
  (ast.parse): a 4-space `def` anchor matches as substring inside an
  8-space nested def, leaving a silent IndentationError.
- Cubic splines are C2: put a NODE on each C2 kink, never a segment break —
  fresh not-a-knot BCs at an artificial edge are strictly worse (falsified
  12-100x); matched-C1 segmentation only restricts the global spline
  space, adds no resolution.
- Interpolate only the single smooth object (e.g. a demodulated envelope);
  rebuild analytic/switched parts in closed form at dense samples.
- LOO adaptive node refinement: held-out error from a few-nearest-OTHER-
  node local fit, normalized by the gate's own error currency, is a
  conservative self-certifying stop; hard-code the threshold, no config
  knob.
- Include the known worst-case point in every seed/coarse grid so engine
  refusals fire unswallowed on the first eval (refusal symmetry with the
  exact path).
- Numerical series: a quantity shared "across all k" can itself overflow —
  prefer a reciprocal-binomial-style O(1)-scaled factorization; verify via
  an algebraic identity. Large phases lose precision in the w*tau
  MULTIPLICATION — reduce mod 2*pi (double-double if needed) first.
- Before reusing a "shared" primitive, check the call site for redundant
  re-derivation; exactness tests can stay green through it.
- Accuracy tests near coordinate singularities need a scale-aware bound
  (~eps * summed-term magnitude) plus a canary for the flat gate.
- Don't catch/fallback across independently gated branches: let refusals
  propagate to preserve certified-domain guarantees.
- A flat parameter-independent "floor" can be an exact closed-form limit
  (macro magnification as w->0) — verify before short-circuiting.
- Comparison sets: a filtered "real-only" nearest-neighbour can miss the
  actually-nearest virtual member — check the spec's set definition.
- numba: njit freezes module globals/callees at compile time (test via the
  full .py_func chain, F010); fastmath=False is load-bearing where
  error-free transforms exist; njit only pure float64/complex128 loops,
  keep validation/scipy/refusals in Python. Explicit njit loops change
  accumulation order vs numpy/BLAS — re-certify in deep-cancellation
  regimes. Batched refactor: gather indices, one batched call, scatter
  back; guard empty subsets; scalar API delegates to the batched core.
