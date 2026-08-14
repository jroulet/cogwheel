# Test Dev Short-Term Observations

## 2026-08-14 (INS-2-001 eta-floor non-regression governance guard)

- INS-2-001 re-flagged the SAME breach as INS-1-001 (near-floor eta band
  [0.5, 0.784) served at O(1e-1) error): the finding's suggested fix is a
  Coder-side floor raise, explicitly out of Test Developer scope, and
  `_SADDLE_ETA_FLOOR` was STILL 0.5 (unraised) when this task landed.
  Confirmed `SaddleTier1NearFloorEtaAccuracyTestCase` (added same day,
  earlier session) already fully documents the breach via the correct
  design: 2 `@expectedFailure` accuracy assertions + one undecorated
  tripwire (`test_reports_worst_near_floor_locus`) that PASSES today and
  is designed to FAIL the moment a future floor raise fixes it — re-ran,
  confirmed unchanged (46 passed, 2 xfailed, ~27s).
- ADDED the one genuinely missing piece: a NON-REGRESSION governance pair
  (`SaddleTier1EtaFloorNonRegressionTestCase` +
  `...SelfFalsificationTestCase`, 3 tests) guarding against the floor
  being accidentally LOWERED below its last-certified value (per the
  Professor asymmetry: false-admit=silent lnL bias, false-refuse=engine
  time only — the floor may only rise). Anchor `_ETA_FLOOR_ANCHOR_2026_08_14
  = 0.5` is a FROZEN literal (deliberately NOT re-reading the live
  constant, else the comparison is vacuous against itself) plus a second
  tripwire `_INSPECTOR_FLAGGED_WORST_EDGE = 0.784` mirroring the same
  "will flip red on fix" pattern from the constant side. Self-
  falsification proves the comparison has teeth without mocking the
  already-bound module constant (re-runs the identical assertion against
  a synthetic regressed value inside assertRaises). Pure Python constant
  checks, no lensing calls, microseconds each.
- Full file: 49 passed, 2 xfailed, ~21s (was 46/2). Re-confirmed (not
  new, unrelated, out of scope) the pre-existing collection errors in
  test_lensing_saddle_gauge.py and test_lensing_saddle_tier1_refusal.py
  (`ImportError: cannot import name '_SADDLE_FARFIELD_RHO_FLOOR'`) — an
  earlier port build's leftover, owned by other runs.

## 2026-08-14 (INS-1-001 near-floor eta accuracy witness, saddle_tier1)

- ADDED `SaddleTier1NearFloorEtaAccuracyTestCase` to
  test_lensing_saddle_tier1_accuracy.py (INS-1-001: `_SADDLE_ETA_FLOOR=0.5`
  sits below the measurement script's own worst failing edge eta=0.784 at
  gamma=2.0; the admitted near-floor sub-band was UNCERTIFIED — no existing
  test drove a REAL gate-admitted source there against the exact engine at
  the production bar. `SaddleTier1FarFromCausticAccuracyTestCase` only
  draws rho>=2.0 (eta>>floor); T5 EtaFloorBoundaryBite uses SYNTHETIC eta
  with T1's delays — gate-flip mechanics only, no accuracy content).
- NEW WITNESS-SEARCH HELPER `_near_cusp_first_admitted(gamma, w_grid,
  offsets)`: generalizes the existing `_near_cusp_eta_below_floor` REFUSAL
  search to the ADMISSION side — steps outward from
  `geometry.nearest_caustic_point(...)` along the caustic-to-source ray and
  returns the FIRST offset the FULL production gate
  (`_saddle_farfield_analytic_serves`) actually ADMITS. This is the
  closest-to-refusal admitted witness, hence genuinely near the floor BY
  CONSTRUCTION (not a pinned eta literal). Swept `NEAR_FLOOR_GAMMAS =
  linspace(GAMMA_LO, GAMMA_HI, 8)` (reused existing file constants), all 8
  located, eta range measured [0.5035, 0.5339] — inside
  `[_SADDLE_ETA_FLOOR, NEAR_FLOOR_ETA_CAP=1.5*floor)` and well below the
  Inspector's flagged eta=0.784 failing edge.
- MEASURED SEVERITY (zero-envelope tier-1 serve vs exact
  `ChangRefsdalChannels(w).evaluate(...).exact_total` oracle, over the
  cheap w-band): **p90(err) ~= 4.735e-2 (~47x over P90_TOL=1e-3)**,
  **max(err) ~= 1.455e-1 (~14.5x over OUTLIER_TOL=1e-2)**, p50~4.88e-3.
  Confirms INS-1-001 quantitatively.
- `@unittest.expectedFailure` DESIGN CHOICE for the two production-bar
  accuracy assertions (p90, max): documents the real defect without
  hard-failing the suite (a Coder fix — raising the floor — is out of Test
  Developer scope per the finding's own text); flips to "unexpected
  success" automatically if a future floor raise resolves it, which is the
  correct signal to promote both to plain assertions. A THIRD, undecorated
  test (`test_reports_worst_near_floor_locus`) asserts the CURRENT breach
  state passes today and will itself start FAILING the moment the bug is
  fixed — a tripwire that forces re-triage of the expectedFailure pair
  rather than letting them silently rot as permanently-skipped-looking
  green xfails.
- "BOUNDARY, NOT INTERIOR" TEETH: for each witness, independently verified
  the offset immediately preceding the located first-admitted offset is
  ITSELF refused by the full gate (recomputed from the same caustic-point
  ray, not cached) — proves the deterministic search lands exactly at the
  admission boundary, not an arbitrary interior admitted point. All 8/8
  witnesses confirmed boundary-exact.
- FULL FILE RUN: 46 passed, 2 xfailed, 1 warning in ~27s (well inside the
  5-min fast-tier file ceiling); zero regressions to the other 46
  pre-existing tests.
- RE-CONFIRMED (not new) BACKWARD-COMPAT AUDIT: test_lensing_saddle_gauge.py
  and test_lensing_saddle_tier1_refusal.py STILL error at collection
  (`ImportError: cannot import name '_SADDLE_FARFIELD_RHO_FLOOR'`) from the
  earlier eta-gauge port build — unrelated to and unaffected by this
  session's change, unfixed, out of scope (belongs to their owning runs).

## 2026-08-14 (saddle_tier1 T6 shard — census mirror == live serve)

- EXTENDED test_lensing_saddle_tier1_accuracy.py +2 classes/5 tests (42
  total green ~30s). T6 = single-source-of-truth: census admit/refuse ==
  live serve gate. TWO layers: (L1 structural teeth) assertIs
  `census._saddle_farfield_analytic_serves is
  likelihood._saddle_farfield_analytic_serves` — same imported gate OBJECT;
  (L2 behavioural plumbing) drive the REAL
  `surrogate_census.characterize_sample` on T1/T2/T3/T4-on/T4-off witnesses
  and assert counted==live for each, with the witness set spanning BOTH
  admit and refuse (`any(live) and not all(live)`) so the equality isn't
  vacuously all-refuse.
- CENSUS f->w INVERSION: `dimensionless_frequency(f,M,z)` is LINEAR in f, so
  `f_grid = w_grid / dimensionless_frequency(1.0, M, 0.0)` reconstructs the
  chosen w_grid EXACTLY for any positive M (used M=1e6). census verdict must
  check `record.category == 'saddle-farfield-analytic'` (NOT just
  record.served) — T3 (4-image) is served by the ppgo_fold interior handoff
  first, which would false-positive a bare served check.
- `LensAmplificationSurrogate([], {})` RAISES ValueError (empty charts
  forbidden) — cannot build an empty surrogate to force census fallthrough.
  Pivot: `_decoy_surrogate()` = ONE positive-parity TubeChart with gamma box
  [0.30,0.50] that never contains a gamma>1 saddle query, so
  `select_chart`->None and census falls through to the saddle gate as
  intended. TubeChart.from_values envelope axis order is
  (n_log_w, n_gamma, n_u, n_theta) — meshgrid indexing='ij' with log_w
  FIRST; a (gamma,u,theta,log_w) build raises a shape ValueError.
- SELF-FALSIFICATION teeth for T6: `test_retired_rho_gauge_would_diverge_on_T1`
  computes retired_verdict = `caustic_rho(T1_GAMMA,|y|) >= RHO_PLACE_FLOOR`
  (2.0) and asserts assertNotEqual(retired_verdict, live_eta_verdict) on the
  T1 witness (rho=1.903 REFUSES under old floor, eta=1.994 SERVES) — proves a
  stale-rho census drift would be caught.
- BACKWARD-COMPAT AUDIT (step 7, REPORTED not fixed — out of scope): two
  sibling suites ERROR AT COLLECTION on the retired
  `_SADDLE_FARFIELD_RHO_FLOOR` (removed from likelihood.py):
  test_lensing_saddle_gauge.py (import + `SADDLE_RHO = _..._FLOOR + 0.02`)
  and test_lensing_saddle_tier1_refusal.py (import line 109). Both encode the
  OLD 3-arg `(real_delays,w_lo,rho)` gate sig; need the eta port by their
  owning runs. Confirmed via `pytest --collect-only` = 2 collection errors.

## 2026-08-14 (saddle_tier1 T3/T4/T5 shard — eta/tie/fence teeth)

- EXTENDED test_lensing_saddle_tier1_accuracy.py +3 classes/15 tests (37
  total green ~23s; neighbour test_lensing_saddle_rho_guards.py 28 green).
  Gate `_saddle_farfield_analytic_serves(real_delays, w_lo, eta)`.
- T3 SaddleTier1LobeInteriorRefusedTestCase: FOUR-real-image lobe interior
  fixture gamma=1.15 y=(1.0980108865474623,0) -> real_mask.sum()==4,
  eta=0.2473. Fence `len(real)>=4 -> refuse` isolated from eta by re-calling
  gate with synthetic FAVOURABLE_ETA=10.0 + HUGE_W=1e5 (still False);
  TEETH = widest-gap 2-img subset [real[0],real[-1]] at same eta/w -> True;
  forced zero-envelope serve err ~0.20 (>> LEAK_MIN_ERR=1e-2). NOTE: real
  4-img source has eta<floor so eta refuses FIRST; the favourable-eta re-call
  is what actually isolates the count fence.
- T4 SaddleTier1MirrorTieDisciplineTestCase: reuse T2 on-axis tied pair
  (gamma=2.0, rho=0.3 via caustic_geometry direction) at FIXED w=60. (a)
  on-axis dtau=0<=tie_eps, eta=1.307 -> gate False; (b) SAME source +0.1
  along perp=[-dir[1],dir[0]] splits pair mdt=0.110, w*mdt=6.61>=RHO_END=4,
  eta=1.295 -> gate True. Only the tie changes. Geometry needs >=2-pt w grid
  (delays/eta w-independent) — build np.array([60,60.6]), call gate with
  scalar w=60.
- T5 SaddleTier1EtaFloorBoundaryBiteTestCase: T1 cone (gamma=1.2,
  y=(0,3.08), mdt=4.81, w_lo*mdt=38.5>>RHO_END) held fixed, sweep ONLY eta:
  eta=floor-1e-3 -> False, eta=floor+1e-3 -> True; assertNotEqual(below,above)
  is the bite. Two eta values DERIVED from _SADDLE_ETA_FLOOR, not pinned.

## 2026-08-14 (saddle_tier1 eta-gate port)

- SADDLE SERVE-GATE rho->eta PORT (test_lensing_saddle_tier1_accuracy.py,
  23 tests green ~14s): gate signature moved
  `_saddle_farfield_analytic_serves(real_delays, w_lo, rho)` ->
  `(real_delays, w_lo, eta)` where eta = geom.caustic_distance
  (= geometry.nearest_caustic_point(...).distance, DIRECTIONAL) replacing
  retired isotropic `_SADDLE_FARFIELD_RHO_FLOOR`. New constants live in
  cogwheel.lensing.likelihood: `_SADDLE_ETA_FLOOR`=0.5, `_SADDLE_TIE_EPS`=1e-12.
  Gate = (0) deltoid fence len(real)>=4 refuse; (A) eta>=floor; (B) resolvable
  narrowest gap SURVIVING >tie_eps has w_lo*min_dt>=RHO_END(4.0). Oracle =
  ChangRefsdalChannels(w).evaluate(...).exact_total (NOT partition.exact_total —
  the partition has no such attr; the evaluate() return does). Cheap band w<=60.
- TWO SPEC DISCREPANCIES FLAGGED (not pinned): (a) T1 transverse-cone 1e-4
  pointwise bar is NOT achievable — zero-envelope FARFIELD_KERNEL_SUM serve vs
  exact engine is intrinsically ~5e-4; asserted <1e-3. (b) ETA-FLOOR LEAK: the
  0.5 floor admits near-cusp sources served at ~5.5% error (e.g. gamma=1.709
  y=(2.442,-0.506) eta=0.624). Reliable contract domain is rho>=2.0 isotropic +
  eta-gated (p90=5.3e-5, max=2.2e-4). Certify the far-from-caustic domain, flag
  the leak as a spec discrepancy — do NOT pin it.
- T1 fixture gamma=1.2 y=(0,3.08): eta~1.994 in [1,2.5], rho=1.903<2.0 (proves
  old-gate refusal), served=True, maxrel~5.16e-4. T2 fixture gamma=2.0 rho=0.3
  on +x axis: dt=0 structural mirror tie, eta=1.307>=floor so refusal is purely
  Leg B (tie discipline), correctly REFUSES.
