# Build 1: Chang–Refsdal lens engine (`cogwheel/lensing/`)

## Mission
Create the wave-optics lens engine that later builds will wrap into a lensed
likelihood: image geometry, amplification evaluation, and the topology-stable
four-component decomposition F(w) = sum_a e^{i w tau_a} K_a(w), as a new
cogwheel subpackage with cogwheel-grade API, docs, and accuracy tests.

This brief is CONTEXT AND REQUIREMENTS — you (the Architect) own the plan.
Consult the Professor: read Serena memory `professor/microlensing_chang_refsdal`
first (the design manual), and `professor/likelihood_and_inference` for the
relative-binning context. The physics source is the manuscript + tested
prototype in `.claude/spec/lensing_paper/` (tex + `code/` + `data/`;
`code/test_topology_stable.py` and `code/test_external_convergence.py` pass —
run them to see the contracts the prototype satisfies).

## Scope
IN: `cogwheel/lensing/` subpackage:
- `chang_refsdal/geometry.py` — macro matrix A(kappa, gamma, beta); quartic image
  solver (+ Newton polish, axial/degenerate branches); delays, Hessians, signed
  magnifications, Morse indices; stationary-phase kernels H_a with the C1, C2
  coefficient polynomials; critical-curve / nearest-caustic-point utilities.
  (Port/adapt from prototype `chang_refsdal_geometry.py`; paper Appendices A, B.)
- `chang_refsdal/operator.py` — contour-free amplification F_op(w, y; gamma, beta,
  kappa): point-mass 1F1 seed + source-derivative operator series + EXACT
  mass-sheet rescaling for kappa (paper Sec. II.B, Appendix C; prototype
  `chang_refsdal_operator.py`).
- `chang_refsdal/channels.py` — topology-stable 4-label decomposition: label
  continuation (assignment on lens-plane markers, virtual labels at nearest
  critical point), smooth switch S_j, cluster residual projection; exposes
  (tau_a, K_a(w)) and the exact total (paper Secs. V–VI; prototype
  `chang_refsdal_global_tracking.py`, `chang_refsdal_topology_stable.py`,
  `exact_gauge_partition.py`).
- Tests in `cogwheel/tests/` (project convention — NOT a top-level tests/).

OUT (later builds): any waveform/likelihood/prior integration; delay-grid
summaries; sampling. Do not touch existing cogwheel modules except possibly
`cogwheel/__init__.py` if subpackage registration is the project convention.

## Hard requirements
1. **Accuracy first (tolerance-based tests, the build gate):**
   a. Operator F_op vs an mpmath high-precision oracle over a documented (w, y,
      gamma, kappa) domain covering 2-image, 4-image, near-fold, near-cusp:
      relative error <= 1e-10 (document any tighter/looser domain edges).
   b. Exact reconstruction: |sum_a e^{i w tau_a} K_a - F_op| <= 1e-12 relative,
      everywhere, including across fold and cusp crossings.
   c. Continuity: port the prototype's fold/cusp label-continuity tests.
   d. Mass-sheet identity as a test: F(kappa != 0) equals the remapped
      F(kappa = 0) expression to ~1e-13.
   e. Quartic geometry vs the prototype's validation data
      (`.claude/spec/lensing_paper/data/quartic_geometry_validation.csv`).
2. **mpmath is ORACLE-ONLY** (test-time). Production paths: numpy double
   precision. Implement a fast complex 1F1(1 - iw/2 + k; 1 + k; -iws/2) kernel
   (Maclaurin + Kummer transform; k-ladder; asymptotic branch for large |z| if
   needed on the tested domain). Design it numba-compatible (plain loops/arrays,
   no object mode) even if @njit is deferred; document the intended hot path.
   The high-w regime should short-circuit to stationary-phase kernels (no
   special functions) via the switch machinery.
3. **API for Build 2** (the consumer is a relative-binning likelihood): a single
   entry-point class (e.g. `ChangRefsdalChannels`) constructed from lens params,
   exposing vectorized `delays()` -> tau_a (dimensionless) and `kernels(w)` ->
   K_a array, plus `amplification(w)` for brute-force reference use, plus
   convergence/series diagnostics. Continuation state must support "evaluate at
   a nearby parameter point" (path from previous point) AND a deterministic
   reset convention for far points (paper Appendix D). Document the label-
   permutation invariance of the total.
4. **Conventions**: dimensionless w = 8 pi G M_L (1+z_L) f / c^3; angles in
   point-mass Einstein-radius units; positive-parity domain 1 - kappa > |gamma|
   enforced with a clear error. Record the macro-saddle exclusion as a
   documented limitation.
5. **Style**: cogwheel idiom (module docstrings WHAT/WHY, numpy-style API docs,
   no dead code). Spec workflow applies: todo fragment exists
   (`.claude/spec/todo.d/` — lensing program); on completion this build should
   add its completed.d fragment and a SPEC.md changelog fragment ([-> spec]:
   new subpackage). No new disk data products expected (if you create any cached
   table, register it in DATA_CONTRACTS.yaml + changelog fragment).
6. **Performance sanity** (assert in tests, generous bounds): single F_op
   evaluation at moderate w in double precision <= 10 ms; a K_a(w) evaluation
   over a 10-node w-grid <= 100 ms. (These guard against accidental mpmath
   in the hot path — the prototype's known weakness.)

## Helpful notes (not directives)
- The prototype is CORRECT but research-grade: mpmath at 60-70 dps everywhere,
  dataclass sprawl, duplicated fold/cusp code paths. Expect to restructure, not
  transliterate. Its tests encode the ground truth — port their assertions.
- The operator series' D_beta^n representation-building (monomials in z, zbar,
  radial-derivative order) is clean in the prototype and worth keeping.
- Numerical cancellation hot spots the paper flags: strong cancellation on the
  continuation path near large intermediate terms (~1e-11 worst case); series
  strain as gamma' -> 1 and at high w. Bound the tested domain accordingly and
  document it.
