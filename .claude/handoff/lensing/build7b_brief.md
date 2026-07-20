# Build 7b — Saddle-domain channel/likelihood/prior integration

## Mission

Lift the interim saddle refusals and make negative-parity (macro-saddle)
hosts SAMPLEABLE end to end. The engine is DONE (Builds 6/7a: geometry,
`_schwinger.py`, parity dispatch in `F_op`/`F_op_grid`, census guard,
crash-class refusals) — this build extends the CONSUMING layers:

1. **Channel layer**: `channels.evaluate` currently raises at the top on
   saddle hosts ("engine-supported but not yet available in the
   channel/likelihood layer"). Extend the SACR-C channel construction to
   the saddle domain. The design authority's Build-S2 prescription is
   INLINED below (§"Binding S2 scope and gates") — plan from THIS BRIEF;
   the full research note is supplementary reading for coders during
   execution, and planning must NOT block on delegated whole-note reads
   or subagent file-access verification. Saddle images carry Morse
   phase `e^{-i pi/2}` per saddle; the wave branch is `f_schwinger`
   through the existing operator dispatch (NOT a new evaluator).

## Binding S2 scope and gates (inlined verbatim from the research note §11)

In scope: `channels.py` (saddle-domain `evaluate`; virtual labels on
the nearest lobe; crossing-scenario fixtures built from
geometry+_gauge only, F002), `likelihood.py` (branch plumbing only —
the envelope/LOO machinery is parity-blind), `lensing/prior.py`
(domain description: either positive-parity-only unchanged, or the
two-domain prior with the parity-boundary refusal band mapped to
lnL = -inf at proposal level).

Fast in-build gates:
1. SACR-C node-count gate N <= 30 for eps < 1e-3 on the research
   Sec. 8 anchor set (2-decade windows);
2. identity residual <= 1e-13;
3. max|S H| <= 2 on fold/cusp crossings at eta = +-0.002, <= 4 on the
   random scan — and per the research's residual-risk list, the scan
   gate must MEASURE the saddle-side constant rather than assume the
   positive-parity 1.3 (two scan configs measured 2.4-2.8: bounded,
   harmless, but gate-by-measurement);
4. lobe-jump spot check (kernel-ratio continuity along one path
   crossing lobes);
5. RB lnL vs brute force on one saddle config within the standard
   tolerance.
Post-build, driver-verified (NOT in-build): 25-config saddle scan,
warm-lnlike timing, full-suite regression.

Out of scope per the research: the v-plane evaluator, lam <= 0
(Type III), any change to positive-parity operator/1F1/refusal
constants, ratio-layer speedups.
2. **Waveform/likelihood/prior**: remove the constructor guard in
   `LensedWaveformGenerator.__init__`; extend the sampled lens
   coordinates so the sampler can propose BOTH parities (the parity
   branch label is physical and MST-invariant — the research fixes the
   parameterization; do not invent one), and widen the positive-parity
   shear range beyond 0.45 (enabled by the 7a fallback: the engine now
   certifies or refuses-by-name across the strong-shear band).
3. **PE-layer band-limit**: proposals whose node set exceeds the
   Schwinger ceiling must resolve to a NAMED refusal before any QMC
   work (the Build 5 refusal-precedes-coherent-score contract), mapped
   to -inf by the posterior net. No silent truncation of the w band.
4. **Rescued-node envelope accuracy gate (REQUIRED precondition)**: at
   a rescued strong-shear config (m_lens x2 on the ratio suite's
   CANCELLATION_CONFIG family) the SACR-C interpolated paths agree with
   each other but differ from `lnlike_bruteforce` by 0.94 nats — the
   envelope layer is NOT yet certified in the newly-opened region. Add
   a tolerance-gated test (ratio AND direct vs bruteforce, small
   deterministic config set spanning rescued gamma') and FIX the
   envelope resolution there if it fails (LOO node policy, not new
   architecture). The sampler must not explore a region the
   interpolation layer hasn't been gated in — on either parity.

## Out of scope — hard fences

- NO changes to `_schwinger.py`, `geometry.py`, or the operator parity
  dispatch (engine certified; consume it). Exception mirrors 7a: if a
  one-line input-domain POLICY guard demonstrably blocks the layer
  extension, document the deviation in the plan for the file gate.
- NO surrogate, NO homogenization, NO Airy patch, NO v-plane (durable
  todos exist; Build 8 program).
- NO changes to the marginalized path's QMC/coherent-score machinery
  (Build 5 certified; the lensed fold consumes it unmodified).
- Positive-parity certified outputs stay byte-identical (the 7a pins
  in the suites must keep passing untouched).

## Measured facts (pre-answered — do not re-derive)

- Schwinger warm cost 30-125 ms/point, linear in w, ceiling w <= 60
  certify-XOR-refuse; oracle error 9.1e-14 (w=20) .. 1.6e-11 (w=59.9).
  CONSEQUENCE: saddle-domain and rescued strong-shear lnlike evals are
  10-100x slower than the 9.8 ms positive-parity hot path. That is
  ACCEPTED for this build (the surrogate owns speed, Build 8); do not
  burn WP budget optimizing it, but DO keep the per-eval node count on
  the existing LOO-adaptive policy (ceiling 48).
- Interim guards to lift: `channels.evaluate` (top-of-function parity
  guard) and `LensedWaveformGenerator.__init__` (explicit
  `1 - kappa > |gamma|` check). Tests currently PIN both refusals:
  `test_lensing_waveform.py::ConstructionValidationTestCase::
  test_macro_saddle_raises_lens_domain_error` and the channels-layer
  saddle guard test — the Test Developer reconciles them to the new
  contract (construction succeeds; evaluation certified-or-refused).
- Saddle census: {1,1} (2-image) / {0,1,1,1} (4-image), index sum -2;
  fold/cusp degenerate censuses pass the guard with a near-critical
  witness and route to the wave branch; the geometric branch refuses
  exactly-singular metrics by name (F015).
- Mass-sheet: kappa is ELIMINATED from the sampled space (Build 4);
  gamma is reduced shear, identity transform, currently
  `range_dic = {'gamma': (0.0, 0.45)}` in
  `cogwheel/lensing/prior.py::UniformReducedShearPrior` with a
  docstring explaining the old 0.45 headroom rationale — that
  rationale is OBSOLETE post-7a; the new bound and the saddle-branch
  parameterization come from the research note.
- d_app convention (Build 5, documented): sampled distance is
  apparent; `d_L = d_app * sqrt(mu_macro)` deferred to post-analysis.
  Saddle-branch mu_macro = 1/|det A| with det A < 0 — the SIGN goes to
  the Morse phase, the MAGNITUDE to d_app; verify the existing
  documentation stays correct rather than re-deriving.
- Refusal vocabulary at the posterior net (closed set, built at raise
  time — do NOT hoist): LensDomainError, CancellationError,
  SchwingerCertificationError, LensedBinningError.
- Full suite baseline at 83d75dc: 367 passed + 2 xfailed, 0 failed.

## Acceptance (build-level)

1. A saddle-host end-to-end lnlike: `LensedWaveformGenerator`
   constructs, the likelihood evaluates finitely on a resolved
   2-image saddle fixture (the research's gamma'=1.3, y=(0.4,0.3)
   configuration), and the value is gated against an INDEPENDENT
   image-sum/wave oracle per the research S2 gates (fast, few-eval).
2. The rescued-node envelope gate (item 4) passes at a stated
   tolerance on both parities' strong-shear configs; its paired
   falsification (an under-seeded envelope grid must exceed the
   ceiling) proves it can go red.
3. Above-ceiling proposals refuse by name BEFORE the coherent score
   (spy-test idiom from `RefusalContractTestCase`) and map to -inf
   through `LensedPosterior`.
4. Prior round-trips (transform/inverse_transform) cover both
   parities; folding declarations remain valid (check the astroid
   quadrant symmetry claim against the DELTOID caustic — the saddle
   caustic has 3 cusps, NOT the astroid's 4; do not silently inherit
   positive-parity folds).
5. In-build tests are FAST (synthetic/few-eval; Schwinger calls
   budgeted — a handful per test, never sweeps). Full suite green is a
   POST-BUILD driver step. Sampling runs are NOT build gates.

## Constraints

- Spec/TODO workflow applies (todo fragment at plan time; completion +
  SPEC row + FINDINGS updates at close; fragments rendered).
- Tests: stdlib unittest in `cogwheel/tests/`, tolerance-based
  accuracy assertions, AST-guarded independent oracles (F002), F010
  falsifiability for every new gate.
- numba compatibility on any hot-path addition.
- The inlined S2 gate list above is binding; if a gate is infeasible
  as written, put a WP-less NOTE in the plan summary AND still emit
  the feasible WPs — never an empty plan (a zero-WP plan is
  structurally rejected by the pipeline gate, not read as an
  escalation).
