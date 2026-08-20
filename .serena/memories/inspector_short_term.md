# Inspector Short-Term — 2026-08-20 diffractive_certificate_fit_interior_fix (pass 3, final)

Scope: re-review of the uncommitted deep-interior `w_low_fit` calibration build
(INS-3-001). Pass-3 is the docstring-only fix of pass-2's F1/F2 (= INS-4-001/
INS-4-002). Full diff: `_diffractive.py` provisional smoke coeffs + docstrings,
`scripts/fit_diffractive_certificate.py` smoke grid interior-inclusive + full
radii linspace(0.1,1.3,7), two test files (docstrings + CORNER_R 1.05->1.1 +
`_fixtures()` refactor removing `_CALIBRATED_GAMMA`).

## VERIFIED (fresh engine probes, re-measured this pass — nothing trusted)
- Fast tier green: 94 passed / 6 skipped (test_lensing_part0_mechanical +
  test_lensing_diffractive). `test_removing_derate_trips_overserve` RAN (38.87 s,
  matches its "~39 s" docstring despite the larger full grid) and passed;
  `test_dropping_caustic_feature_inflates_over_prediction` RAN and passed.
- Live over/under-serve probes reproduce the shipped docstrings EXACTLY:
  gamma=0.5/rho=0.2/cusp 1.1244x, gamma=0.5/rho=0.3/cusp 1.0233x (over-serve,
  red-by-design), gamma=0.2/rho=0.2/cusp 0.6850x, gamma=0.3/rho=0.2/cusp
  0.7719x (under-serve, conservative).
- Honest ceiling range "~4-41" CONFIRMED correct (max = 40.976 at
  gamma=0.2/rho=0.2/cusp; gamma=0.5/rho=0.3 = 4.121, gamma=0.3/rho=0.3 = 19.18).
  The ~4-34 -> ~4-41 change is a real re-measurement reflecting the deeper grid.
- CORNER witness rho = 2.188 (= "~2.19"), raw over-prediction 1.0062x (=
  "~1.01x"), r=1.1 in linspace(0.1,1.3,7), gamma=0.5 in linspace(0.05,0.5,6),
  theta is a fenced off-grid midpoint (1 match). All CornerRaw docstring claims
  verified.
- log(s)^2 coeff = -0.04816 (index 6 of 10-monomial degree-2 poly). derate 0.85
  == _HARD_DERATE_CEILING clamp; interior served by the fit below the ceiling.
- Grid arithmetic: smoke 227/178/49(21.6%)/44; full 1356/1014/342(25.2%)/250.
  Fence exclusion: gamma=0.2/r=0.2 cell 16/32 fenced, gamma=0.3/r=0.2 8/32 —
  confirms the INS-4-001 qualification is accurate.
- No stale refs: 0.844967 / 196/196 / 48/48 / 63/259 / 9e09bdd / ~7x /
  _CALIBRATED_GAMMA / singular "interior cell" all gone.

## RESOLVED
- INS-4-001: smoke-grid docstring now qualified "rho < RHO_LO in the smooth
  (off-cusp) directions, while the near-cusp thetas fall in the near-fold shell
  and are fenced out". Accurate.
- INS-4-002: `w_low_fit` docstring now "never over-serves on the calibration
  grid and its held-out off-grid midpoint probes; extrapolated off-grid points
  can over-serve". Accurate.

## FINDINGS (NEW, trivial — both docstring staleness from the coefficient re-bake)
- F1: `TestWLlowFitDeepInteriorServedByFit.test_deep_interior_served_below_
  ceiling_at_calibrated_cell` method NAME ("at_calibrated_cell") + docstring
  "Every calibrated deep-interior cell" — gamma=0.5 deep interior is
  EXTRAPOLATED (uncalibrated) per the class's own "Calibration is qualified by
  bake" note. Suggest rename to test_deep_interior_served_below_ceiling and
  drop "calibrated" from the docstring (the assertion is structural and
  correct; only the wording over-claims).
- F2: `test_dropping_caustic_feature_inflates_over_prediction` docstring
  "measured ... ~1.66x" is STALE — with the re-baked coefficients dropping the
  caustic feature raises the raw surface 1.9059x, not ~1.66x. The re-bake
  updated the sibling corner claim (~1.19x -> ~1.01x) but missed this one.
  Test still passes (asserts > 1.05x), so teeth unaffected; docstring only.

## CARRIED FORWARD (unchanged, NOT code defects)
- INS-1-003 (driver advisory): witness cells (gamma=0.2/0.3/rho=0.2/cusp,
  r~0.04-0.06) stay below the full-grid floor r=0.1; conservativeness there
  hinges on log(s)^2 = -0.048 (negative). Driver must verify after --scale full
  (sign flip would over-serve). Docstrings now document this explicitly.
- SPEC.md LOW-W DIFFRACTIVE RUNGS paragraph still describes only the ANALYTIC
  `w_low = (gamma'/2)*[...]` certificate, never `w_low_fit` — Librarian sync.

## PATTERN
- When a coefficient re-bake updates SOME docstring "measured" values, sweep
  EVERY docstring carrying a "measured ~Nx" claim for that surface — the
  caustic-feature self-falsification docstring (~1.66x -> ~1.91x) is the
  laggard that survives even a meticulous sibling pass.
- A test-method name/docstring that says "calibrated cell" after a `_fixtures`
  refactor widens coverage to an EXTRAPOLATED cell re-introduces the exact
  over-claim the INS-4-001/002 lineage cleaned up.
