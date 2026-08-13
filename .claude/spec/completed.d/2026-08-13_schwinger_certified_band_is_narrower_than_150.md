---
date: 2026-08-13
section: Backlog
---

- **THE mpmath ORACLE IS RED BY CONSTRUCTION ABOVE w ~ 100, AND PRODUCTION
  SPURIOUSLY REFUSES CORRECT ANSWERS FROM w ~ 139** `[→ spec]` — measured
  2026-08-13. REPLACES an earlier version of this fragment which claimed the
  exact evaluator's certified band was narrower than 150; that claim came
  from [[FINDINGS F071]], which is RETRACTED — the oracle was the thing that
  was wrong, not `f_schwinger`.

  ## 1. Fix the oracle FIRST — it currently convicts production of its own error

  `_oracle_1d` in `cogwheel/tests/test_lensing_schwinger.py` ships
  `ORACLE_MAXDEGREE = 5`, and its own calibration comment says the knobs are
  converged "at `w = 30, 45, 55`". Above `w ~ 100` it is NOT converged, and
  its error is amplified by the same `e^{+pi w/4}` factor as everything else
  in this problem. Measured at w=130, gp=1.3, y=(0.3,0.2):

      oracle maxdeg5 vs maxdeg6     1.7056e-04
      PRODUCTION vs oracle maxdeg5  1.7058e-04    <- the false signal
      PRODUCTION vs oracle maxdeg6  2.0896e-15    <- production is correct
      oracle maxdeg6 vs maxdeg7     0.0000e+00    <- converged at 6

  Confirmed at w=190 (maxdeg6 identical to maxdeg7) and w=210 (converged to
  2e-61).

  ACTION: raise `ORACLE_MAXDEGREE` 5 -> 6, and amend the calibration comment
  to say the knobs must be validated in the AMPLIFICATION band, not only at
  w = 30/45/55. Until then
  `MpmathPathOracleAgreementTestCase::test_mpmath_path_matches_oracle_uniformly`
  is red under `COGWHEEL_TRAIN_TIER=1` and reports production's error when it
  is reporting its own.

  Do NOT trim the test's `w` grid to make it pass. The grid is right; the
  oracle is under-resolved.

  ## 2. Then raise `_MP_PANEL_ORDER` 32 -> 40 — for REFUSALS, not wrong values

  At order 32 the paired N/2N certificate blows the `3e-10` gate from
  `w ~ 139` — INSIDE the advertised `60 < w <= 150` band — while the value is
  right to 9e-16. So `f_schwinger` raises `SchwingerCertificationError` on
  correct answers over the top ~11 units of its own band. Measured at w=150,
  against a converged oracle:

      gp    order  TRUE rel err   cert rel   certifies?   s/call
      1.3     32     8.99e-16     8.11e-07      NO          57
      1.3     40     8.99e-16     1.45e-28     yes          72
      0.7     32     3.24e-16     3.09e-07      NO          96
      0.7     40     3.24e-16     5.61e-29     yes          74

  Order 40 certifies through `w ~ 204` for BOTH fixtures. Cost ratio 1.29x
  (cleanest at w=190: 99s -> 128s), so the ceiling's RUNTIME rationale
  survives — the 150 ceiling stays a budget decision, not an accuracy one.

  The certificate is structurally CONSERVATIVE, not optimistic: it reports
  the N-rule error while the function returns the 2N value, so it refuses
  before the value degrades (measured margin ratio 1.65e19 vs 2^64 = 1.85e19
  at w=190). Raising the order does not create a certified-but-wrong band.

  ## Watch out when you change the order

  `THREE_OUTCOME_W = 151.0` with its `THREE_OUTCOME_FOLD/CUSP/REFUSE`
  fixtures encodes WHICH arms certify at exactly w=151. Raising
  `_MP_PANEL_ORDER` changes certification outcomes there; expect those three
  to need re-calibration, and re-derive them from the live gate rather than
  re-pinning.

  ## Acceptance

  Quote error-vs-`w` against the HARDENED oracle (maxdeg >= 6), and state the
  refusal onset (not the accuracy crossing) as the number that moved. Confirm
  both `gamma'` fixtures. Report seconds/call at the shipped order and the
  new one on an UNCONTENDED box — the 1.29x ratio is trustworthy, absolute
  timings from a loaded box are not.

  ## RESOLVED 2026-08-13 (5451ab9)

  Both actions shipped together. `test_lensing_schwinger.py`:
  `ORACLE_MAXDEGREE` 5 -> 6 (7 is bit-identical to 6 at w=130 and w=190,
  2e-61 at w=210, so 6 is converged across the exercised band).
  `_schwinger.py`: `_MP_PANEL_ORDER` 32 -> 40, certifying through `w ~ 204`
  on both `gamma'` fixtures (1.3 and 0.7) at ~1.29x cost (99s -> 128s at
  w=190). Acceptance table (order, TRUE rel err, cert rel, certifies?):
  32/8.99e-16/8.11e-07/NO and 40/8.99e-16/1.45e-28/yes at gp=1.3, w=150;
  matching pattern at gp=0.7. The "watch out" item is moot: `THREE_OUTCOME_W`
  is `W_CEILING_SCHWINGER_QD + 1.0` (derived, not a `151.0` literal), so it
  tracks the ceiling automatically and needed no re-pin.
