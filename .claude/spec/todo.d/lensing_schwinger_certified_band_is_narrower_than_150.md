---
section: Backlog
---

- **THE EXACT EVALUATOR'S CERTIFIED BAND IS `w <~ 110`, NOT 150 — AND IT
  CERTIFIES WRONG ANSWERS ABOVE THAT** `[→ spec]` — measured 2026-08-13 by
  the driver against the independent `_oracle_1d`. Full measurement and
  mechanism in [[FINDINGS F071]].

  `f_schwinger` is the reference every accuracy claim in this repo is
  measured against. In `(110, 150]` it RETURNS (certifies) values wrong by up
  to **1.7e-4** against its own suite's 1e-10 bar. The error grows as
  `e^{+pi w/4}` — fitted `d(ln err)/dw = 0.785` vs `pi/4 = 0.7854` — because
  a fixed absolute quadrature floor is amplified by the `1/Gamma(iw/2)`
  prefactor. The paired N/2N certificate cannot see it: both rules share the
  floor and the amplification.

  ## The decision, which is NOT taken here

  Three options, and they are not equivalent:

  1. **Lower `W_CEILING_SCHWINGER_QD` to ~110** and refuse above it. Honest
     and cheap. COST: it removes serving coverage in `(110, 150]`, and the
     cusp-exterior windows where the exact engine is the LAST rung would
     start refusing — `_MP_PANEL_ORDER`'s own docstring says protecting that
     coverage is why order-32 was chosen over order-24. Measure what actually
     falls off before choosing this.
  2. **Raise the working precision so the floor stays below the
     amplification.** `dps = 30 + ceil(w)` grows 1.0 decimal digit per unit
     `w` while the amplification needs `pi/4 / ln(10) = 0.341` digits per
     unit `w`, so dps is NOT the binding constraint — suspect the fixed
     `_MP_PANEL_ORDER = 32` composite rule instead. The fold investigation
     already found that raising it 32 -> 48 made a refusing node return at
     w=150. Measure error-vs-order at fixed `w = 130` before assuming.
  3. **Keep the band and make the certificate honest** — replace paired N/2N
     with a check that does not share the floor (e.g. a coarse independent
     high-dps spot check, or a Richardson estimate across ORDER rather than
     panel count).

  Option 2 is the most likely to preserve coverage AND correctness; option 1
  is the only one that is certainly safe today.

  ## What must be re-checked once the band moves

  - `_ppgo_above_ceiling`'s boundary tests pin agreement "at w=150" and at
    w=55/60. The w=150 anchors are inside the untrustworthy band.
  - Any arm-vs-engine comparison in `(110, 150]`. Note this COMPOUNDS with
    [[FINDINGS F069]]: in `60 < w <= 150` the positive-parity grid may return
    the ARM rather than the engine, so such a test can be comparing an arm
    against an arm above 110, and against a wrong engine below it.
  - `test_lensing_schwinger.py::MpmathPathOracleAgreementTestCase` — its
    `MPMATH_EXT_WS` currently reaches into the bad band; it is the test that
    FOUND this, and it should keep failing until the band is fixed rather
    than have its grid trimmed.

  ## Acceptance

  Report error-vs-`w` against `_oracle_1d` at the CHOSEN configuration, and
  state the ceiling as the `w` where the fitted `e^{pi w/4}` law crosses
  1e-10 — not as a round number. Do not trim the test grid to make the
  existing ceiling look earned.
