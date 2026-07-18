# Professor review — 2026-07-18 Build 3c batched-contraction verdict: PASS
- Ran 4 fast lensing suites (cogwheel-newlal py): batched_operator 15/15, operator 22/22,
  fast_path 20/20, likelihood 19/19 = 76 pass, no tol widened, no CancellationError-grid change.
- Oracle accuracy scatter: rel-err grows exp with cancellation exponent L (1e-16@L~0 ->
  ~1e-10@L~44) then REFUSES beyond — XOR contract honored, no returned point over 1e-10 ceiling.
  Single-vs-batch refusal decisions identical (zero flips); value agreement <1e-14 => no
  cross-node leakage in the reordered accumulation.
- F010 falsification live: _SERIES_TOLERANCE=1.0 and zeroed half_sum both drive oracle gate RED
  via py_func chain (gate non-vacuous).
- Timing (crown, 1 thread): caustic 1.2ms, engine 38ms, contraction 1.6ms (subdominant),
  lnlike 41ms < 175ms ceiling, speedup 33.9x > 8.0 floor. brute ~1.4s here (lighter than
  the ~15s predicted machine) but structural speedup gate still clears comfortably.
- Heavy full-posterior sampling validation operator-deferred (not run per turn budget).

# Professor short-term — 2026-07-18 Build 3c consultation (session 2, final)

## Session: Four authoritative test specs for Build 3c WP1+WP2

### Delivered prescriptions (all four items complete)

1. **TIMING GATE**: MS_CEILING = 0.175 s (108 ms floor x 1.6 margin, arithmetic-derived);
   SPEEDUP_MIN = 8.0 (from 3.0; ~139x measured, ~17x margin); structural gates lead
   (contraction < engine, speedup > 8x), absolute ms is secondary regression guard.

2. **RE-CERTIFICATION**: Batched-vs-oracle on FOP_GRID + CERT_LS union at FOP_RTOL=1e-10;
   refusal-decision identity (zero flips, scalar-vs-batched); scalar-vs-batched value
   agreement to 1e-13; F002 oracle independence (mpmath, no shared substrate).
   Diagnostic: rel-error-vs-L scatter, boolean refusal map, scalar-vs-batched scatter.

3. **F010 SELF-FALSIFICATION**: Two perturbation sub-tests under py_func-chain idiom:
   (a) _SERIES_TOLERANCE set to 1.0 (truncates to 1 term, ~4% wrong at gamma=0.20);
   (b) half_sum zeroed (collapses bilinear form to scalar, wrong at any nontrivial config).
   Both must make the oracle gate go RED. py_func chain must cover the ENTIRE new njit
   call chain.

4. **EQUIVALENCE**: All existing gates stay green at ORIGINAL tolerances -- no new
   equivalence tests needed because WP1 is single-path (existing tests automatically
   exercise the batched code). Enumerated: (a) RB-vs-brute on 5 _LENS_CONFIGS at
   max(1.5, 1e-2*|bf|), (b) bit-identical determinism, (c) null-safe interp < 1e-3,
   (d) macro limit 1e-8, (e) zero-noise floor 1e-2 / 1.5e-2, (f) near-cusp pin,
   (g) refusal symmetry, (h) F005 certification band. Plus one new one-liner:
   F_op(scalar_w) == F_op_grid([scalar_w])[0].

### Code paths read this session
operator.py: _contract_orders, F_op, select_branch, cancellation_exponent,
  all threshold constants
channels.py: _exact_total, _coarse_w_node_grid
_hyp1f1.py: point_mass_g_derivatives, _shared_numerator, _ladder_core,
  _ladder_sum, _carrier, prefactor_c
likelihood.py: _DEFAULT_KERNEL_NODES=100
test_lensing_operator.py: OperatorOracleTestCase, ContractionCertificationTestCase,
  SelfFalsificationTestCase, _oracle_amplification, _oracle_series, ORACLE_CONFIGS
test_lensing_fast_path.py: NumbaOperatorPreservationTestCase, FewMsTimingTestCase,
  CrownAccuracyAnchorTestCase, CoarseNodeInterpolationTestCase,
  SelfFalsificationTestCase, _oracle_fop, all FOP_* constants, MS_CEILING, SPEEDUP_MIN,
  _LENS_CONFIGS, _CROWN, MACRO_LIMIT_*, INTERP_NULLSAFE_CEIL, RB_FLOOR_REGRESSION
test_lensing_likelihood.py: BruteForceAgreementTestCase, ContractionTimingTestCase,
  DeterminismTestCase, SelfFalsificationTestCase, _LENS_CONFIGS, all constants
FINDINGS.md: F010 (numba compilation voids Python-level instrumentation)
build3c_brief.md, build3c_plan_approved.json
