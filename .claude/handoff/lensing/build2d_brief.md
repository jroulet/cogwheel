# Build 2d — Engine small-w short-circuit: close Build 2 green

## Mission

Fix the Chang–Refsdal engine's small-w inaccuracy (the last real defect
behind Build 2's 3 remaining test failures), flip one channels test whose
premise the Build 2c switch fix voided, and bring the full suite green at
the ORIGINAL tolerances. This is a single-focus build; keep it shallow.

## Scope fences

IN: `cogwheel/lensing/chang_refsdal/operator.py` (and, only if the analytic
correction lives naturally there, `_hyp1f1.py`); the three failing tests
(`test_lensing_likelihood.py` zero-noise floor pair,
`test_lensing_channels.py` flat-gate guard) plus the small-mass floor test
in `test_lensing_waveform.py` whose mass range Build 2c restricted.

OUT (do not touch): the channel switch, `_gauge.py`, `channels.py`,
`likelihood.py` (RB is fully green: near-cusp, two-image, timing ~7x),
norm/data contractions, moment orders, refusal thresholds/`MAX_ORDER`
(the band-edge refusal at w=20, gamma=0.3, tail 2.665e-10 vs 1e-10 target
seen during probing is CORRECT engine behavior — strong-shear order
widening remains a separate ticket). NO tolerance widening anywhere.

## Measured facts (pre-answered; do not re-derive)

1. Engine small-w floor: max|F−1| over the band is FLAT at 2.062e-2 for
   M_L = 1e-4 … 1e-12 Msun (w_hi from 1.77e-5 down to 1.77e-13). Physics
   requires |F−1| = O(w) (the leading lensing correction is a pure phase;
   |F|−1 is second order). Cause: the operator shear-series prefactor
   gamma/(2w) is numerically singular as w→0, so the series returns a
   spurious ~2% amplitude instead of collapsing to unity.
2. This floor is the WHOLE story of the zero-noise failures: on zero noise
   (d = h0, asd_drift pinned 1) the floor is exactly
   0.5*<|F−1|^2>*(h0|h0); measured (h0|h0)=570.7968, predicted floor
   0.1214, measured brute-force floor 0.1214 at BOTH M_L=1e-6 and 1e-10
   (identical — mass-independent). RB reads 0.1307; the 0.0093 difference
   is the known F007 RB template asymmetry, well under the 0.01 gate once
   the engine floor is gone. Template construction and normalization are
   measured innocent.
3. Prescription (Professor consult, 2026-07-17): a small-w asymptotic
   short-circuit — below a w threshold return F = 1 + the analytic leading
   correction (O(w) term), instead of evaluating the singular series. The
   threshold must be chosen so the short-circuit and the full series agree
   at the crossover (continuity check belongs in the tests). Expected
   result: |F−1| → physics (~w), zero-noise floors → ~1e-11, both zero-
   noise tests and the un-restricted small-mass floor test pass at their
   ORIGINAL gates (ZERO_NOISE_TOL=0.01 unchanged).
4. Channels flat-gate test (`test_flat_gate_fails_where_the_targets_
   diverge`): asserts on-caustic kernel divergence >1e12 that the Build 2c
   switch fix eliminated by design (parked virtual label co-located with
   the source → switch 0 → bounded cluster residual). Professor ruling:
   FLIP it — assert the fixed gauge keeps on-caustic kernels BOUNDED, do
   NOT pin the buggy real-only switch. Calibration (measured, fixed gauge,
   w in [5,20]): worst sum|K_a| = 4.27 and recon error <= 5e-16 across
   two-image / four-image / sheared / convergent / three on-caustic /
   three cusp configs — a 1e3 ceiling gives two orders of margin and the
   buggy switch variant blows it (falsifiable regression guard for the
   switch fix).
5. Small-mass floor test (`test_lensing_waveform.py`): Build 2c restricted
   the mass sweep to w >~ 1e-3 citing this engine gap as a deferred
   ticket. With the fix, restore the tiny masses (down to the original
   M_L=1e-12) and assert the roundoff-floor behavior the test originally
   intended; the deferred-ticket docstring note should be replaced by a
   pointer to the fix.

## Environment facts

- Suite interpreter: /Users/tejaswi/miniconda3/envs/cogwheel_310/bin/python
- Ignore test_gw_prior/test_posterior/test_waveform (pre-existing
  IMRPhenomXODE optional-dep gap, not this build's concern).
- Current suite state (2026-07-17, HEAD): 174 passed + 193 subtests,
  3 failed (the two zero-noise floors, the channels flat-gate guard).
  Full run costs ~1h15m; the crown likelihood suite dominates.

## Acceptance (build-level)

The full suite (minus the XODE trio) is green at original tolerances:
the zero-noise pair passes with the 0.01 gate unchanged, the flipped
channels guard passes in the fixed gauge and goes red under the real-only
switch variant, the small-mass floor test passes over its restored range,
and every currently-green test stays green (including the crown gates and
the ~7x timing margin). FINDINGS gains the measured small-w story
(mechanism, table, fix) per repo convention, with changelog fragments
rendered.
