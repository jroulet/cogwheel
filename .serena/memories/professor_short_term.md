# Professor short-term (serve_route_census INFERENCE REVIEW, 2026-08-17)

Reviewed the built engine-free serve-route census tests
(`cogwheel/tests/test_lensing_serve_route_census.py`, 28 tests) against the
design rulings in the prior consult. VERDICT: PASS.

## Run facts
- All 28 tests pass in 28s (cogwheel-newlal py3.10). Two run() calls
  (150x32 demand + 120x32 engine-free-under-4-patches); well under 5-min
  fast ceiling. Tests are STRUCTURAL/contract (no amplitude oracle, so no
  diagnostic plots emitted — consistent with the docstring).
- Direct census eyeball (seed=0, n=150): route_counts sum=150=n_samples
  (MECE ✓); surrogate=0 (no artifact ✓); saddle_c3=1, born_analytic=20,
  engine_residual=129, engine_refused=0. Residual split (gauge=caustic_rho,
  NOT rho_lobe ✓): born_chart_demand=0, near_caustic_tube=28, interior=101,
  undetermined=0 → sums to 129 = engine_residual ✓.

## Physics checks confirmed against first principles
- Residual dominated by interior(rho<=1, 4-image, 101) + near-caustic(28);
  exterior served cheaply as born_analytic → born_chart_demand=0 inside the
  residual is EXPECTED (rho>2 exterior draws are already born-served upstream).
- engine_refused=0 at this small scale is fine — per my prior ruling it must
  be REPORTED empirically, never asserted at 59% a priori. Bucket exists in
  schema; a hard-refusal (macro-saddle parity wall) simply wasn't drawn at
  n=150/seed=0. Not a concern.
- Saddle finite-but-huge c3: production ppgo_error_estimate ~4.8e15 (finite,
  not None); _SADDLE_FARFIELD_SAFETY=20, _SADDLE_FARFIELD_CERT_BAR=1e-3, so
  20*4.8e15=9.6e16 >> 1e-3 → real gate refuses, route=engine_residual, NOT
  saddle_c3. The F069/F074 "certificate certifies its own blow-up" mode is
  correctly closed (gate bounds safety*est, not est-is-finite).
- D2 sign-flip quadruple: route + node_route_kinds elementwise-invariant
  across (±y1,±y2) for both parities (astroid gamma<1, saddle gamma>1). IEEE
  sign flips are bit-exact so no ULP drift; gates key on |y|/caustic_rho.
- Engine-free guarantee: 4 unique door sentinels (evaluate, f_schwinger,
  _f_schwinger_mpmath, mpmath.gauss_quadrature) all OUTSIDE the caught tuple
  (refusal_errors+(ValueError,ZeroDivisionError)); run() completes with all
  door call_counts==0 → demand-map, not evaluator, holds.

## Operator-deferred
Heavy full-posterior / real-data lensed validation is NOT run here (out-of-band
ship gate). Verdict rests on the fast structural suite + direct census eyeball
+ code-constant invariants, all consistent.
