# Test Dev Short-Term Observations

## 2026-08-20 (INS-2-002 declined_mask content-hash, LoadContractTestCase)

- INS-2-002: `declined_mask` (the INS-1-001 correctness-critical near-fold
  refusal mask) was stored in the npz but NOT in the content hash at either
  production site (load recompute + training script bake, both 7 fields) --
  a tampered/all-False mask loaded silently. Coder owns the 2-line production
  fix (add `declined_mask` to `_content_hash(...)` at
  cogwheel/lensing/low_w_diffractive_chart.py `load`'s `actual =` line AND
  scripts/train_low_w_diffractive_chart.py bake); Test Dev owns the
  LoadContractTestCase tamper case.
- TEST-SIDE MIRROR MUST ADVANCE WITH THE FIX: the helper `_save_chart_artifact`
  MUST store `declined_mask` AND hash it (8 fields) or the round-trip breaks
  POST-fix (7-field stored vs 8-field recompute). Flip is not optional --
  a helper frozen at the pre-fix 7-field hash strands the whole suite green-
  then-red on the day the fix lands. Keep the helper's byte layout == the
  POST-fix training script, always.
- AUTHORING-TIME STATE IS HONEST-RED: with the fix not yet landed, the
  round-trip + rehashed-positive-control tests FAIL (load recomputes 7 fields,
  refuses the 8-field artifact as stale) while the tamper tests PASS-for-the-
  wrong-reason (raise comes from the 7-vs-8 format mismatch, not tamper
  detection). The red round-trip IS the signal production hasn't adopted the
  contract. Probe confirmed flip-green: patched 8-field load accepts clean,
  refuses stale-mask, accepts rehashed mask; stored hash == 8-field recompute
  and != 7-field recompute. All tests flip with ZERO edits once the fix lands.
- New twin test in LoadContractSelfFalsificationTestCase: rehashed all-False
  mask loads cleanly (proves the tamper refusal is hash-bytes, not a shape/
  dtype guard on the mask). Premise asserts on the fixture mask (any & !all)
  guard the derived fixture against a future all-False collapse making the
  tamper a no-op.

## 2026-08-20 (census-mirror + de-rate self-falsification shards, same file)

- CENSUS MIRROR (serve_route_census.classify_draw -> 'low_w_diffractive_chart'): drive the REAL
  `classify_draw` via `dataclasses.replace(_load_production_modules(), low_w_chart=chart,
  dimensionless_frequency=lambda...)` (field-swap idiom from test_lensing_born_certificate). Witness =
  NEAR_FOLD_DECLINED_WITNESSES shape (gamma=0.3, beta=-1.1 at Y_REF=(0.8,0.4)): the census fixes
  kappa=beta=0, so y = exp(-1j*beta)*Y_REF directly (y=(0.00639,0.8944)); rho=_caustic_rho(0.3,s,theta)
  =1.247 inside [RHO_LO,RHO_HI]; farfield_w_floor ~1.24 so _CENSUS_W_GRID floor 0.05 < floor, ceiling 1.0
  << QD ceiling (intercept 3 skipped). Reuse _build_coverage_chart() (its box contains gamma'=0.3, rho=1.247).
  "served == counted" teeth: spy on LowWDiffractiveChart.covers (patch the CLASS, not the instance -- frozen
  dataclass __setattr__ raises FrozenInstanceError on mock.patch.object(instance,...)) -> exactly 1 call with
  (gp=0.3, rho=res.caustic_rho, w=_CENSUS_W_GRID); patch covers->False diverts route to 'engine_residual'
  (w_low_fit declines the shell). SERVE_ROUTES has 12 entries, 'low_w_diffractive_chart' before
  'diffractive_analytic'.
- DE-RATE SELF-FALSIFICATION: `_worst_serve_engine_ratio(derate)` module helper. GOTCHA: init `worst=0.0`
  NOT 1.0 -- with init 1.0 the conservative-derate side returns the INIT (1.0) because all ratios <1, making
  assertLessEqual(cons,1.0) vacuous against itself. unit_worst=1.5785 (>1 teeth), cons_worst=0.9091 (real).
  The existing test_unit_derate_overshoots_off_grid already pins unit>1; the new twin pins the FLIP (conservative
  derate restores one-sidedness).

## 2026-08-20 (low_w_diffractive_chart serve suite, test_lensing_low_w_diffractive_chart.py)

## 2026-08-20 (load-contract/coverage/fold shards, same file)

- LOAD CONTRACT engine-free: `_save_chart_artifact(path, chart, schema=, content_hash=, drop_keys=)` mirrors the
  training script's npz save format (derate stored as `np.array(chart.derate)` 0-d, provenance as
  `np.array(json.dumps(...))`, content_hash via the production `_content_hash` over the 7 stored fields).
  Tamper test: `_tamper_artifact(path, key, mutation)` re-saves with the ORIGINAL hash. TEETH = positive control:
  tamper with a FRESH hash loads cleanly (LoadContractSelfFalsificationTestCase) vs stale hash refuses.
- COVERAGE UNION: derive witnesses from the LIVE gate constants (RHO_LO/RHO_HI/_WALL_GAMMA_PRIME from
  `low_w_diffractive_chart`); the test-file's `WALL_GAMMA_PRIME=0.8` (fixture shear) != prod `_WALL_GAMMA_PRIME=0.5`
  (gate). The coverage chart's grid BOX must contain every witness (gamma' 0.1-0.95, rho 0.2-3.0) so the band
  predicate, not box containment, decides. `covers` reads RHO_LO/RHO_HI/_WALL_GAMMA_PRIME as module globals ->
  mock.patch.object(_lwd_module, ...) gives teeth.
- THETA D2 FOLD: chart coeffs = `cos(2*theta)` (even + pi-periodic, 8 theta nodes). Four octants fold to the SAME
  query point -> bit-identical evaluate() (allclose rtol/atol=1e-12). TEETH: no-fold RegularGridInterpolator queried
  at raw pi-theta extrapolates and diverges; + premise assert residual varies >0.1 in theta.

- SERVE INTERCEPTION PATTERN (reusable): to test `LensedRelativeBinningLikelihood._low_w_diffractive_chart_serve`
  engine-free, bind the UNBOUND method to a `types.SimpleNamespace` with `low_w_diffractive_chart=chart` +
  `_reduce_dense_kernels`/`_image_delays` lambdas (instance attrs are NOT descriptor-bound, so the lambdas get
  only their own args), and `mock.patch.object(likelihood, 'reconstruct_farfield', _capture)` to grab the
  `envelope` arg. `geom` MUST carry `delays`/`saddle_kernels`/`real_mask`/`t_min` even though reconstruct is
  patched -- the serve evaluates those `geom.*` args BEFORE the patched call. With `t_min=0` the captured
  envelope IS the re-modulated farfield F_serve (frame phase vanishes).
- FIXTURES MUST BE INTERIOR GRID NODES: the serve reconstructs gamma'/rho/theta from the source with ~1e-16
  float roundoff, so a fixture exactly ON a grid edge (rho=1.0 with rho_grid[0]=1.0) is pushed just outside
  and `chart.covers` (inclusive <=) returns False -> serve declines. Use interior nodes (index 1..n-2).
- scipy RegularGridInterpolator 'cubic' needs >= 4 points PER axis (2-node axis -> ValueError).
- NODE-EXACT ORACLE: an exact-residual chart (r_pure = f_pure*sqrt(1-gp^2)/prefactor_c) reconstructed through
  the serve equals `_engine_reference_kappa` to 1.66e-14 at nodes -- assert 1e-10 with 6000x margin. Building
  the 256-node chart costs ~8s (cache via module-level lru_cache shared by all test classes).
- DC anchor with a CONSTANT residual=1 chart isolates the re-modulation factors (no engine): |F_serve|/sqrt(mu_macro)
  == |prefactor_c(w)| EXACTLY and arg == arg(prefactor_c) + 0.5 w (log lam - kappa s). Self-falsification teeth
  magnitudes: doubled prefactor -> rel err |C(w)-1| (0.48@w=0.5, 6.0@w=8), unit anchor -> 0.40 at gamma'=0.8.
- `_born_factors` returns a 5-tuple (sqrt_mu, phi_geo, q2r, b1, a0); sqrt_mu == 1/sqrt(|lam^2-gamma^2|) == sqrt(mu_macro) exactly.
