# Coder Short-Term Observations

## 2026-08-18 (born_farfield_completion Closure #3 WP2 — census Route-2 mirror)
- serve_route_census.py: added thin helper `_born_trained_floor_route(mods,
  gamma, rho, host_mask, w_grid)->bool` (modeled on `_saddle_c3_route`),
  inserted into classify_draw Intercept 5 BETWEEN Route 1 (`covered and not
  trained_band_escape`->born_analytic) and the carrier-only cert. It mirrors
  WP1's production Route 2 EXACTLY: trained_floor=math.exp(float(born_chart.
  log_w_grid[0])) (artifact read, NOT literal), band_split_floor,below_floor=
  mods.band_split_mask(w_grid,trained_floor); engine_mask=host_mask&below_floor,
  chart_mask=host_mask&~below_floor; returns band_split_floor and
  engine_mask.any() and chart_mask.any() and born_chart.covers(gamma,rho,
  w_grid[chart_mask]). A True -> STILL born_analytic (partial serve); False
  (high-edge/disjoint escape) -> falls to born_carrier_only cert (Route 3).
  ZERO reimplemented decision logic — all accessors bound via _ProductionModules
  (born_chart, band_split_mask). No new SERVE_ROUTES enum (born_analytic
  already exists) so the schema/invariant count pins are unaffected.
- host_mask/chart_w already computed upstream in Intercept 5 (unchanged);
  the new helper reuses host_mask. covers(gamma,rho,w[chart_mask]) is the
  exact production applicability test (low-edge escape True only for strict
  sub-band). Verified AST OK, import OK, mpmath absent at import, helper
  present. Updated classify_draw docstring intercept-5 to a 3-route list.
- MEASUREMENT: in-build engine-free 10k A/B census (wp2_census_ab.py, scratch,
  deleted after) — A=post, B=_born_trained_floor_route forced False (pre-WP2).
  Booby-trapped ChangRefsdalChannels.evaluate + _schwinger doors; asserted
  mpmath never in sys.modules. Run at n_freq=8 (born/saddle intercept deltas
  are n_freq-INDEPENDENT per WP3 finding). MEASURED (10k seed-0 n_freq=8):
  born_analytic A=343 vs B=0 -> recovered=343 (3.430%), ALL Route-2, ALL
  astroid (saddle 0). engine_residual 4468->4212 (delta -256, ~5.7% engine-
  demand drop); diffractive_analytic 812->725 (delta -87). 256+87=343 exact
  (Intercept-5 born fires BEFORE Intercept-6 diffractive, so 87 recovered
  draws were already analytic/diffractive in B — a precedence reassignment,
  NET engine recovery is 256). born_carrier_only=0 (saddle far-field GO-
  served 0 -> closure #1 correctly ABSENT). ppgo_above_ceiling/saddle_c3/
  wave_refused UNCHANGED A vs B (WP2 touches only the Born intercept).
  CAVEAT (honest): run B born_analytic=0 shows the trained_band_escape
  host-band refinement (added to the census AFTER WP3's box-only-covers
  count of 611) drops ALL box-covered draws out of Route-1 at this config;
  Route-2 recovers 343 of them. The EXACT recovered count is NOT strictly
  n_freq-independent — Route-2's covers(gamma,rho,w[chart_mask]) depends on
  the sub-band node values, so at production n_freq=128 the boundary count
  may shift slightly (UNVERIFIED at n_freq=128; the qualitative result —
  few-% born_analytic recovery entirely from low-edge trained-floor
  escapers, engine_residual reduction, saddle-GO absent — is robust).

## 2026-08-18 (born_farfield_completion Closure #3 WP1 — trained-floor band split)
- `_born_residual_analytic` (cogwheel/lensing/likelihood.py) serve decision
  went from 2-way to 3-way. Route 1 fully-in-box = byte-identical HEAD
  (early return). NEW Route 2 TRAINED-FLOOR SPLIT: on `trained_band_escape`
  (box-covered but host sub-band escapes trained log_w), split the HOST
  region again at `trained_floor = math.exp(float(born_chart.log_w_grid[0]))`
  (artifact-read, NOT a literal) via a 2nd `_band_split_mask(dense_w,
  trained_floor)` call. Inverted polarity: engine BELOW (engine_mask =
  host_mask & below_floor, below_floor=dense_w<=trained_floor incl. the node),
  chart AT/ABOVE (chart_mask = host_mask & ~below_floor, strictly above).
  Route 2 fires IFF `band_split_floor and engine_mask.any() and
  chart_mask.any() and born_chart.covers(gamma, rho, dense_w[chart_mask])`
  — the covers() guard is the exact applicability test: True only for a
  low-edge escape (strict sub-band); high-edge/disjoint escape leaves
  chart_mask uncovered -> guard False -> Route 3. Route 3 = HEAD carrier-only
  certificate fall-through (unchanged). gamma>1 saddle rung untouched.
- KEY POLARITY/NULL-FALLBACK: chart residual is evaluated ONLY on chart_mask
  (`residual[chart_mask] = evaluate(dense_w[chart_mask], ...)`; zero
  elsewhere) — avoids the RegularGridInterpolator off-log_w-axis extrapolation
  garbage below trained_floor. engine envelope via
  `_engine_envelope_below_split(lens, dense_w, engine_mask)` (same
  FARFIELD_KERNEL_SUM gauge as the saddle-c3 rung uses). band_split_floor +
  chart_mask.any() guards reject the `_band_split_mask` all-True below_floor
  null-fallback (trained_floor outside band).
- `_born_reconstruct` gained OPTIONAL `engine_envelope=None, engine_mask=None`
  (backward-compatible; existing/test-probe callers unaffected). Overlay
  placed AFTER `envelope[~below_mask]=0.0`: `if engine_envelope is not None:
  envelope[engine_mask] = engine_envelope[engine_mask]`. engine region ⊆
  below_mask & above bottom_mask so it's neither zeroed by below_mask nor
  overwritten by Rung P — clean tier stitch. Null-residual byte-path
  untouched (engine_envelope None on Route 1/3).
- WHY covers(chart_mask) is exact & non-contradictory: if trained_floor <=
  host bottom (w_low), the whole host would be covered -> not
  trained_band_escape (contradiction), so a reached Route-2 low-edge escape
  has trained_floor strictly interior to host and engine_mask non-empty.
  Verified AST OK + import OK + reconstruct sig has engine_envelope/
  engine_mask.
- TEST-IMPACT (UNVERIFIED, for Test Developer): `_BornAnalyticProbe` in
  cogwheel/tests/test_lensing_born_analytic_reachability.py binds
  `_born_reconstruct`/`_diffractive_bottom_ceiling` etc. but does NOT bind
  `_engine_envelope_below_split` (nor `_evaluate_envelope` it calls). Any
  probe fixture that now triggers Route 2 (a low-edge escape draw) will
  AttributeError until the probe binds those two production methods. Route
  1/3 fixtures unaffected. Did NOT run the suite (downstream job).

## 2026-08-18 (born_farfield_completion WP3 — census carrier-only route)
- Added `born_carrier_only` to serve_route_census.SERVE_ROUTES (10->11,
  after `born_analytic`). `classify_draw` Intercept 5 now SPLITS on the
  shipped Born chart's BOX-ONLY `covers(gamma, rho)` (mirrors production
  `_born_residual_analytic`'s `covered` discriminator): in-box astroid ->
  born_analytic; beyond-box/macro-saddle -> born_carrier_only IFF the
  SHARED production predicate `_born_carrier_certificate_serves(lens, w_lo,
  w_hi, real_images)` admits (bound whole, NOT re-typed — it internally
  owns the omitted-term bar / min-sep backstop / saddle RHO_END fence;
  fires on BOTH parities). Certificate refusal or missing chart -> fall
  through to node pass. Chart loaded engine-free in `_load_production_
  modules` via `BornResidualChart.load()` (try/except OSError/ValueError/
  KeyError -> None, mirroring `_AUTO_BORN_CHART`). Two new _ProductionModules
  fields: born_carrier_serves, born_chart. ROUTE_KINDS unchanged (born
  intercepts emit `()` — whole-band, not a per-node kind). residual_demand
  split still on caustic_rho (untouched). Verified: AST OK, import OK,
  mpmath NOT in sys.modules at load, 11 routes.
- KEY DESIGN: the pre-WP3 census OVER-counted born_analytic (astroid-only
  chart never covers gamma>1 saddles, yet old code labelled any gamma!=0
  rho>2 as born_analytic). Loading the chart + box-only covers() fixes this
  and is exactly production's discriminator. D2 fixture (0.5,3.0) stays
  born_analytic because covers(0.5,3.0)=True (empirically confirmed).
- MEASURED (10k seed-0, full 20-1024Hz band): born_carrier_only=0/10000
  (0.000%), pos-parity 0 / saddle 0. NOT a wiring bug — 928 beyond-box/
  saddle draws DO reach `mods.born_carrier_serves`, but ALL 928 fail the
  cert: 20*born_carrier_omitted_term(w_hi) ranges 0.67..15154, all >> bar
  1e-3, because the omitted term is LINEAR in w and the full-band ceiling
  w_hi is too high for a 1e-3 carrier-only truncation. Honest mirror: prod
  declines carrier-only at these full-band ceilings and falls to engine.
  born_analytic=611 (was over-counted pre-WP3 when saddle rho>2 counted as
  born_analytic; now box-only covers() keeps only astroid in-box). Runtime
  mpmath_in_sys_modules=False (engine-free confirmed). Census is SLOW
  (~35min for 10k@n_freq=128); Intercept-5 born classification is
  n_freq-INDEPENDENT (geomspace endpoints fixed) so diagnostics ran at
  n_freq=8 ~16x faster with identical born counts. scripts/serve_route_
  census.py only doc-touched (breakdown is route_counts-driven).


## 2026-08-18 (born_farfield_completion WP2)
- Lifted the Born gate in cogwheel/lensing/likelihood.py: beyond-box /
  whole-saddle far-exterior queries now attempt a certificate-gated
  CARRIER-ONLY serve (residual identically ZERO) instead of refusing to
  the engine. New module-level `_born_carrier_certificate_serves(lens,
  w_lo, w_hi, real_images)` mirrors `_saddle_farfield_analytic_serves`:
  refuse on kappa/beta!=0 or gamma==0; cert `_SADDLE_FARFIELD_SAFETY *
  born_carrier_omitted_term(w_HI,...) <= _SADDLE_FARFIELD_CERT_BAR`
  (evaluated at band CEILING w_hi — omitted term is LINEAR in w, OPPOSITE
  to the saddle-c3 gate's w_lo); `_saddle_min_image_sep >= 0.05` backstop;
  saddle-only (gamma>1) fence `w_lo * _real_delay_min_separation >=
  RHO_END(4.0)`. No new tolerance constants (reused the saddle-gate trio).
- Factored the reconstruction TAIL into `_born_reconstruct(self, lens,
  dense_w, geom, residual, below_mask, bottom_mask)` — partition_ns ->
  born_carrier_from_partition -> f_total=carrier+residual -> Rung P
  (bottom_mask, HypergeometricDomainError->None) -> ppgo -> envelope=
  (f_total-ppgo)*exp(1j w t_min), zeroed above split -> reconstruct_farfield
  -> _reduce_dense_kernels -> (delays,k0,k1,geom). Decision+zeroing stay
  INLINED at the two call sites in `_born_residual_analytic` (NOT owned by
  the tail): in-box passes evaluate() residual; carrier-only passes
  np.zeros(...,complex). `f_total = carrier + residual` allocates fresh so
  Rung P mutation is safe; zeros branch reduces to bare carrier byte-clean.
- Byte-identity invariants preserved & how: (a) in-box = `covered and not
  trained_band_escape` -> evaluate residual (HEAD tail reproduced exactly);
  (b) tube rho<=2 -> `return None` BEFORE geom solve (split off from the old
  combined `rho<=2 or not covers` site A); (c) born_chart is None -> None
  first; (d) kappa/beta!=0 -> None. `trained_band_escape` reproduces HEAD
  site B (`covered and host_mask.any() and not covers(...,chart_w)`) then
  routes to the cert instead of unconditional None.
- ORDERING NOTE: a beyond-box NOT-covered query now solves geom (needed for
  the cert's real_images + saddle fence) where HEAD returned None pre-geom;
  a geom LensDomainError now propagates unswallowed — this is
  outcome-preserving because the seed engine path below would raise the
  IDENTICAL error for the same geometry (caller does NOT wrap the rung;
  sibling `_saddle_farfield_analytic` + low-w rung share this convention).
- Verified: AST parse OK, module import OK, both new symbols present;
  born_carrier_omitted_term/_real_delay_min_separation/macro_matrix/RHO_END
  signatures confirmed via inspect; no duplicated reconstruction tail left
  in `_born_residual_analytic` (single born_carrier_from_partition call at
  the _born_reconstruct site; the 2395 occurrence is the untouched buried
  surrogate rung). UNVERIFIED: no test suite run (downstream job) — in-box
  byte-identity and carrier-only numerics not exercised here.

## 2026-08-18 (born_farfield_completion WP1)
- Added `born_carrier_omitted_term(w,y1,y2,gamma,beta=0,kappa=0)` to
  cogwheel/lensing/chang_refsdal/_born.py (after `born_amplification`):
  carrier-relative truncation CERTIFICATE = `math.hypot(a0, 0.5*w*b1)/q2r`
  from `_born_factors` DIRECTLY (parity-agnostic, certifies BOTH parities;
  saddle gamma=1.3 returns finite, does NOT hit born_amplification's
  positive-parity BornDomainError guard). a0/b1 enter the certificate ONLY,
  never the serve carrier (F009/F025).
- GOTCHA: the spec's stated `if q2r == 0.0: return inf` post-call guard is
  UNREACHABLE at the true origin — `_born_factors` raises ValueError via
  `math.log(r0_sq)` (r0_sq==0 <=> q2r==0) BEFORE returning. Wrapped the
  `_born_factors` call in `except ValueError -> return math.inf` (its only
  ValueError, r0_sq is a sum of squares so ==0 only at origin); kept the
  post-call q2r==0 check as defense-in-depth. gamma==0 guarded up front.
  Verified: origin & gamma0 -> inf; formula & b1-a0 invariant match to
  float eps for both det_a signs.
