## 2026-08-20 (INS-2-002/003, fence-bake test fixes)
- PROBE BEFORE TRUSTING A FINDING'S PROPOSED FIX, EVEN ON A TEST: INS-2-003's
  literal prescription — "patch RHO_LO=0.0 AND DELTA to a large negative so
  the interior is served by the fit, then assert the fit value != ceiling"
  per-fixture — is RED by construction because `w_low_fit`'s fit path ends
  with `w_fit = min(w_fit, _DIFFRACTIVE_FIT_CEILING)`: the UNCERTIFIED fit
  extrapolates ABOVE the ceiling (w_fit 77–2479 unclipped) for 7 of the 12
  deep-interior fixtures (gamma in {0.2,0.3} x theta {0,pi/4,pi/2} x rho
  {0.3,0.5}), so those clip to exactly 60.0 == ceiling even with the fence
  gone. The honest fix: collapse BOTH fence constants, assert per-fixture
  `assertIsNotNone(raw)` (kills the original vacuity — the old shape
  returned None for 12/12) and aggregate `differed |= raw != ceiling` +
  final `assertTrue(differed)` (the fence is load-bearing at least
  somewhere; measured differed=True). The per-fixture != form is genuinely
  un-satisfiable and would be a flaky/false assertion, not a real test.
- Related: the old test patched ONLY RHO_LO=0.0, which routes interior
  fixtures into the SHELL branch (rho <= 1+DELTA -> None) — confirming the
  vacuity mechanism: to collapse the shell for rho<1 fixtures, DELTA must
  go below -1 (1+DELTA < 0), e.g. -10.0; DELTA=-0.4 leaves 1+DELTA=0.6 and
  rho=0.3/0.5 still declined.
- INS-2-002 was a pure docstring-number refresh in test_lensing_diffractive.py
  (raw over-prediction ~2x -> ~1.18x, smoke de-rate 0.5034 -> 0.844967 for
  the fenced bake). Verified with whitespace-flexible source-string asserts
  (the hyphen "de-" word-split across the line wrap makes a naive literal
  needle fail — normalize whitespace first). FullGridCertificateOracleTestCase
  test still green (35.8 s).
