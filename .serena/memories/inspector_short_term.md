# Inspector Short-Term Observations

## 2026-07-18 — Review of Build 3c (weight-vector batched wave-branch contraction)

Scope: uncommitted tree in cogwheel-claude-dev. Build 3c = STEP toward
few-ms lensed lnlike. WP1 replaces the per-node 85x85 bilinear
`_contract_orders` with a w-INDEPENDENT per-order weight-vector reduction
(`_weight_vectors` builds v[n,j] = scatter-add of
z_powers[a]*table[n,a,b]*zbar_powers[b] over gather idx(a,b,n); each node's
order becomes a length-dim dot v[n].derivs). New batched njit core
`_contract_grid`, single certification path `_grid_certified`, lean public
`F_op_grid`, and scalar `F_op` now DELEGATES to `_grid_certified([w])`.
WP2 wires `channels._exact_total` to call `F_op_grid` ONCE on the
wave-branch node subset. Source files changed: operator.py (big),
channels.py (import+_exact_total). New test: test_lensing_batched_operator.py.
likelihood.py correctly UNTOUCHED (plan said "expected none").

### Verified GOOD (value/contract preserving)
- Weight-vector regrouping is an exact reordering of the SAME (a,b) sum:
  contribution = sum_j v[n,j]*derivs[j] = sum_{a,b} z[a]*table[n,a,b]*
  zbar[b]*derivs[idx(a,b,n)]. Matches old two-stage col/row reduction term
  for term; only accumulation order differs (re-certified vs mpmath).
- Coeff recurrence coeff*= 1j*gamma_scaled/(2w*order) == (1j*g/2w)^n/n!.
  positive_total, max_term, last_ratio, converged, _MIN_ORDER/
  _SERIES_TOLERANCE/_CONSECUTIVE_SMALL small-term stop all identical to old
  scalar path. All four F005 refusals (non-finite total, cancellation ratio,
  truncation tail, contraction round-off guard) BYTE-unchanged in
  `_grid_certified` per-node loop; overflow-safe frexp/ldexp power-of-two
  rescaling preserved per node. Reconstruction (mass_sheet_phase*phase_scaled
  *total/lam) unchanged.
- Zero-coeff skip in `_weight_vectors` is bit-safe (0*finite=0); defensive
  idx clamp treats clamp case identically to old code (spurious-contribution
  parity), and the gather invariant (nonzero coeff => half_sum+n<=dim-1)
  is the SAME pre-existing invariant.
- channels WP2: branch decision stays per-node (cancellation_exponent +
  select_branch), carrier exp(-1j*w*t_min) applied elementwise, geometric
  nodes inline unchanged, CancellationError propagates unswallowed from the
  batched call (refusal symmetry with brute path intact). No F_op consumers
  left broken (scalar F_op only defined, not imported, outside tests).
- New suite F002-clean: mpmath oracle built from mpmath.hyp1f1 + integer
  (u,v) monomial ladder, AST guard forbids production names; F010 py_func
  chain present on both `_contract_grid` and `_weight_vectors`
  (.py_func has no .signatures).

### Tests RUN (all green at ORIGINAL tolerances)
- test_lensing_batched_operator.py: 15 passed (28s)
- test_lensing_operator.py + test_lensing_fast_path.py: 42 passed (59s)
- test_lensing_likelihood.py + channels + gauge: 74 passed (45s)
- test_lensing_dd.py + hyp1f1 + geometry: 61 passed (17s)
Import clean; F_op_grid in __all__.

### FINDINGS: none (NOW-introduced). VERDICT PASS.

### Notes / carry-forward
- The two engine falsification tests my Build-3b memory flagged RED
  (dd SplitterSensitivity, hyp1f1 LadderComplexity) are now GREEN — fixed &
  committed in a prior build. F010 idiom held.
- Benign plan deviation: plan expected edits to test_lensing_fast_path.py /
  test_lensing_operator.py; test_dev instead ADDED a dedicated
  test_lensing_batched_operator.py. Cleaner; existing suites still exercise
  the batched path via scalar delegation (BatchedEquivalenceTestCase).
- Timing gate MS_CEILING=0.175s is the plan's arithmetic-derived floor
  (~108ms x1.6), NOT the 10ms target; SPEEDUP_MIN raised 3.0->8.0. Build is
  a STEP; named next lever = Lever B (3D post-contraction surrogate, Build 4).
- SPEC.md not in changed set; new test module unreflected -> spec/doc hook
  will need a fragment before commit (Librarian/commit-step item).

### Bug patterns (carry forward)
- njit-inlined primitives break Python monkeypatch/sensitivity tests unless
  patched through the FULL .py_func chain (F010). New njit cores here expose
  .py_func and the falsification tests confirm they can go red.
- A weight-vector scatter-reduction that replaces a bilinear form is a legit
  accumulation-order change: re-certify vs an INDEPENDENT oracle at the
  original tolerance + check the certified-XOR-refuse decision doesn't flip
  solo-vs-batch (cross-node convergence-state leakage).
