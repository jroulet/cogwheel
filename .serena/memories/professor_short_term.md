# Professor Short-Term — Build 3b fast-path REVIEW (2026-07-17, supersedes prior advisory)

Ran fast tests in `cogwheel/tests/test_lensing_fast_path.py` (env cogwheel-newlal,
single-thread). Production `_DEFAULT_KERNEL_NODES = 40` (likelihood.py:148 — NOT raised).

## Results (base=40 shipped default)
- CausticSearchPreservationTestCase (WP1): PASS — dist vs brute-force oracle rel<1e-10,
  branch invariance holds. ✓
- NumbaKernel/OperatorPreservationTestCase (LEVER1): PASS — mpmath-oracle accuracy within
  cancellation law / F_op<1e-10, bit-identical repeat, F005 refusals fire. ✓
- CrownAccuracyAnchorTestCase (RB-vs-brute EVERY config): PASS all 5 (incl. kappa, the old
  3.44-nat leak now within max(1.5,1e-2|bf|)); symmetric LensDomainError on macro saddle;
  zero-noise F->1 floor; macro-mag limit; near-cusp regression. ✓ (lnlike-decisive gate)
- FewMsTimingTestCase: PASS — speedup 85.7x (>>3), contraction 2.5ms < engine 124ms, lnlike
  137ms < 250ms ceiling. caustic-search 1.9ms slightly over soft "<1ms" print (informational).
- SelfFalsificationTestCase: PASS — n=4 grid epsilon>1e-3 (gate can go red); positive control
  uses CONVERGED_NODES=128 (NOT the production default).
- CoarseNodeInterpolationTestCase (production interp gate): **FAIL** — two-image null-safe
  epsilon=2.758e-2 = 27x over 1e-3 ceiling at base=40 (docstring: near-cusp 3.7e-3, kappa
  3.5e-3, rotated-shear 1.8e-3 also over; only four-image 6.7e-5 passes).

## Verdict: FAIL (one required gate red; central Build-3b deliverable unshipped)
Spec required epsilon<1e-3 for EVERY config AND the production default to pass as the
self-falsification positive control. Neither holds: Coder did NOT raise
`_DEFAULT_KERNEL_NODES` 40 -> ~85-128. Physics is genuine, not a test defect: at prod rho~20,
two-image 2.76e-2 -> deltaL ~ rho^2*eps ~ 11 nat worst-case (>>1.5-nat RB gate). Tests are
domain-CORRECT and honest (null-safe metric well-posed at nulls; gate not widened). Fix is a
one-line production change (raise base to ~85-128, log-spacing insufficient at 40). The
lnlike-decisive RB-vs-brute gate passes on THIS low-rho fixture only.
