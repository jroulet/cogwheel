# Test Dev Short-Term Observations

## 2026-08-17 (serve_route_census D2 + saddle-finite-huge extension)
- EXTENDED test_lensing_serve_route_census.py (+2 classes, +3 self-falsif
  methods; 19->28 tests, 18.8s). Two Architect specs: (spec4) D2 sign-flip
  invariance — route AND node_route_kinds ELEMENTWISE identical across the
  4 IEEE sign-flips (±y1,±y2) of a source; (spec5) saddle finite-but-huge
  ppgo_error_estimate REFUSES (F069/F074 gate-bounds-wrong-object).
- FIXTURE DERIVATION: caustic reach has NO caustic_reach() fn — derive via
  reach = 1/ppgo_map.caustic_rho(gamma,1.0,0.0). Single-draw probes call
  the REAL src.classify_draw through a memoized _classify_env() =
  (_load_production_modules(), _frequency_grid, _gamma_band_edges) — same
  triple run() feeds classify_draw, so no reimplementation.
- D2 REP DRAWS (m=1e3, angle=0.4 rad so cos!=sin & both coords nonzero ->
  4 DISTINCT points): astroid_interior g0.5 rho0.5 -> engine_residual
  (kinds{exact_wave}); born_exterior g0.5 rho3 -> born_analytic (kinds());
  near_caustic g0.5 rho1.05 -> engine_residual (kinds{exact_wave,geometric});
  saddle_farfield_c3 g3.0 rho3 -> saddle_c3 (kinds()). Spans BOTH parities +
  3 routes. Intercept routes carry () node vector (trivially D2-equal);
  residual carries a 32-tuple that is elementwise-equal because freq nodes
  don't permute under a source sign flip.
- SPEC5 WITNESS: gamma=3, m=1e3, rho=1.001, angle=0 (ON cusp axis where the
  omitted-term est is MAXIMAL) -> est=4.76e15 (finite, not None), nimg=2,
  saddle_farfield_serves=False, route=engine_residual. Off-axis kills the
  magnitude fast (angle0.05 -> ~3e8), so the ~1e15 the spec cites needs the
  ON-AXIS placement + m~1e3 (est ~ w_min**-3, w_lo=2.48). est floor asserted
  at 1e9 (measured 4.8e15 = 6 orders margin). Load-bearing contrast: naive
  'est is not None' ADMITS while safety*est(=9.5e16) > bar(1e-3) REFUSES.
- TEETH: D2 via sign-keyed toy route (2 labels across quad); elementwise-vs-
  multiset via tuple!=perm but Counter==Counter; saddle-refusal via naive-vs-
  safe predicate disagreement on huge_est. NO production changes; neighbor
  test_lensing_saddle_rho_guards green (27).


## 2026-08-17 (serve_route_census demand-census suite)
- NEW SUITE test_lensing_serve_route_census.py (19 tests, 18s, leaf module —
  no other test imports it). Three Architect specs: MECE small-run (schema
  v1, every route in the 7-label SERVE_ROUTES, route_counts keyed exactly on
  SERVE_ROUTES, sum==n_samples, no-artifact->surrogate==0); residual-
  partition (born_chart_demand+near_caustic_tube+interior+undetermined ==
  route_counts['engine_residual'], split_gauge=='caustic_rho' NOT 'rho_lobe'
  = the F073 regression; rho>2->born, (1,2]->tube, <=1->interior); engine-
  free mock-to-raise (4 UNIQUE sentinel Exceptions on ChangRefsdalChannels.
  evaluate, _schwinger.f_schwinger, _f_schwinger_mpmath, mpmath fn — all
  OUTSIDE the census caught refusal tuple; run completes, all 4 door mocks
  call_count==0).
- ENGINE-FREE PROOF PATTERN: booby-trap every exact-wave door with a
  sentinel that is NOT a subclass of _load_production_modules().
  refusal_errors — a test explicitly asserts sentinel-disjointness so a
  future widening of the caught tuple that swallows the sentinel is caught.
  mpmath confined to _schwinger.py, so patching mpmath.gauss_quadrature
  cannot false-positive from an allowed analytic path. Measured shared
  150-draw: engine_residual=129, saddle_c3=1, born_analytic=20.
- INDEPENDENT ORACLE FOR A STRUCTURAL COUNT = from-scratch Counter re-tally
  of per-draw records + independent rho-ladder re-bin, never a re-call of
  the production aggregation. lru_cache(maxsize=1) memoizes the one shared
  run so the file stays at 2 run() calls total.
- Anti-vacuity tearDown (_comparisons>0) verified to have teeth via an
  out-of-band probe subclass; SelfFalsificationTestCase pins route-
  membership/exhaustiveness/residual-sum/gauge-regression teeth + door-
  wiring + sentinel-disjointness.
