# Build Brief: Resolve the `select_branch` / served-routing disagreement in test_thresholds_have_one_home

## Mission

Adjudicate and fix the pre-existing disagreement between `select_branch` and
what the operator grid ACTUALLY serves, which fails
`cogwheel/tests/test_lensing_operator.py::BranchGateTestCase::
test_thresholds_have_one_home` at HEAD. This is the last unresolved item in
`lensing_serving_ladder_guards_are_red.md` (the 2nd of 2 remaining).

## The failure (measured 2026-08-11, driver)

Saddle node `gamma=1.2, kappa=0.0, y=(0.04, 0.03), beta=0.7, w=500`:
- `select_branch(w, delta_min, inf, eta) == 'wave'` because
  `w*delta_min = 1.90 < RHO_END = 4.0` (resolution leg fails). delta_min =
  3.8e-3 (real-image delay separation), eta = 0.84 ≥ ETA_MIN_GEOMETRIC=0.3.
- BUT `_observed_branch` (the test's no-mock probe: served value
  bit-equal to `geometric_amplification` ⇒ 'geometric') reads **'geometric'**
  in the full-sweep context, so the test asserts `'geometric' != 'wave'`
  and fails.
- Root cause (driver trace): the grid routes this node through
  `select_branch` → 'wave' → `_uniform_arm_value` → the CUSP arm's **ppGO
  rung** serves `fold_ppgo_correction`, whose value is 1 ULP from
  `geometric_amplification` in isolation (measured
  `served=...04207724j` vs `geom=...04207727j`, diff 2.8e-17). After the
  sweep's prior nodes warm numba/JIT, the two become bit-identical, so the
  test's bit-identity probe mislabels the cusp-ppGO serving as 'geometric'.
- Same for `gamma=2.0, w=500` and `gamma=1.2, w=1000` (3 total mismatches).

## The physics question (Professor to adjudicate FIRST)

This is a routing/bit-identity correctness question, not a test-parameter
fix. Options:

(a) **The grid is wrong**: at `w*delta_min < RHO_END` the node is
    UNRESOLVED — the geometric-sum serve (via cusp ppGO's
    `fold_ppgo_correction`) is inaccurate there, and the cusp ppGO rung
    should NOT serve it. The fix would be a production gate (e.g. the cusp
    ppGO rung must not fire below resolution), and the test's 'wave'
    expectation is correct.

(b) **The test probe is stale**: the cusp ppGO rung legitimately serves
    the geometric image sum at large R (ppGO = geometric limit), so its
    value being bit-equal to `geometric_amplification` is EXPECTED, and the
    test's bit-identity discriminator can no longer distinguish "geometric
    rung served" from "cusp ppGO served the same limit". The fix would be
    to sharpen `_observed_branch` (e.g. distinguish the rung by a
    tolerance, or by which arm served) — NOT a production change.

(c) A hybrid: the cusp ppGO rung's `fold_ppgo_correction` at an UNRESOLVED
    node may be the same physical failure mode the fold arm's fence guards
    against (F028/F032 measured 60-267% wrong on well-resolved above-ceiling
    configs; the resolution gate RHO_END exists precisely because the
    geometric sum is wrong below it). If so, option (a) is the correct
    physics and the cusp ppGO rung needs a resolution guard mirroring the
    fold arm's admission.

**The Professor must decide (a), (b), or (c) with measured evidence before
the Coder codes.** Consider: `fold_ppgo_correction` is the fold arm's own
serve object — does the fold arm itself refuse this node (delta_min small)?
If the fold arm would refuse it (unresolved), the cusp arm serving the SAME
`fold_ppgo_correction` via ppGO at the same node is inconsistent and should
also refuse.

## Measured facts (at HEAD 72028a2)
- Failing node: `gamma=1.2, kappa=0.0, y=(0.04,0.03), beta=0.7, w=500`
  (also w=1000; and gamma=2.0, w=500). `w*delta_min=1.90 < RHO_END=4.0`.
- Grid served value (isolation): `0.33952075833590023+0.11196425404207724j`
  (cusp ppGO rung). `geometric_amplification`: `...04207727j` — 1 ULP apart.
  After sweep warmup they are bit-identical (state-dependent).
- `select_branch` legs: resolution `w*delta_min >= RHO_END`, cancellation
  `L > L_MAX`, eta `>= ETA_MIN_GEOMETRIC`. RHO_END=4.0, L_MAX=48,
  ETA_MIN_GEOMETRIC=0.3. Saddle passes `L=inf` (no saddle analogue).
- cusp ppGO rung gate (recent build): `radius >= r_ppgo_min AND
  w >= _W_PPGO_FLOOR AND nearest.distance >= _ETA_MAX_FOLD`. No resolution
  check currently.
- `_observed_branch` discriminator: `served == geometric_amplification`
  (bit-exact) ⇒ 'geometric'; else 'wave'; LensDomainError ⇒ 'geometric'
  (census guard); SchwingerCertificationError ⇒ 'wave'.

## Acceptance
1. `test_thresholds_have_one_home` passes with a non-zero comparison count.
2. The chosen option is physics-justified and the fix is refusal-conservative:
   - if (a)/(c): the cusp ppGO rung refuses UNRESOLVED nodes
     (`w*delta_min < RHO_END`), and `test_refusal...`/ppGO tests still pass;
     `_PPGO_ASTROID_SOURCE` (a resolved astroid fixture) must still serve.
   - if (b): `_observed_branch` correctly identifies the rung without
     breaking the other ~40 one-home checks (positive + saddle parity, the
     eta-leg-live assertion).
3. `python -m pytest cogwheel/tests/test_lensing_operator.py cogwheel/tests/test_lensing_fast_path.py cogwheel/tests/test_lensing_airy_fold.py -q --no-header --timeout=120 --timeout-method=thread` green.
4. Professor's adjudication is recorded in the build report (which option,
   and the measured evidence).

## Constraints
- Fast tests only. Refusal-conservative.
- Do NOT weaken the bit-identity contracts elsewhere (byte-identity tests
  are load-bearing). If (b), the change is local to `_observed_branch`'s
  discriminator logic and must not disturb `select_branch`'s one-home
  property.
- Investigate the state-dependence (isolation vs in-sweep bit-equality) —
  understand WHY before choosing; it may indicate a real determinism issue
  (numba warmup changing a served value) that itself needs a fix.
