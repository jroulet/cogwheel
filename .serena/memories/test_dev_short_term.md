# Test Dev Short-Term Observations

(empty — last consolidated by Dreamer on 2026-08-18)

## 2026-08-18 (born_certificate WP1/WP2 census-mirror + null-recon shard — VERIFY-ONLY)
- ASSIGNED specs (CENSUS-MIRROR FAITHFULNESS, ENGINE-FREE GUARANTEE ON
  ANALYTIC ROUTES, NULL-RESIDUAL RECONSTRUCTION IDENTITY) were ALREADY
  fully authored by a predecessor mid-turn in test_lensing_born_certificate.py
  as BornCensusMirrorFaithfulnessTestCase (3 tests), BornCensusEngineFreeTestCase
  (1) and BornNullResidualReconstructionTestCase (1). Suite 45 passed 6.7s.
  Assessed sound + KEPT as-is (parsimony: nothing missing, added 0). Neighbors
  test_lensing_born + serve_route_census + born_analytic_reachability +
  born_residual_wiring 160 passed 62s, no regression.
- BACKWARD-COMPAT RESOLVED: the reachability suite my earlier note flagged
  broken by the WP2 _born_reconstruct refactor (_BornAnalyticProbe lacked the
  method) is now GREEN — another agent added _born_reconstruct to its probe.
- SOUNDNESS NOTES for the three new classes (why no twin needed): null-recon
  has intrinsic teeth (bottom-tier oracle=diffractive_amplification, above=
  closed-form ppgo sum both DIFFER from bare carrier, so a missing tiering
  overwrite plateaus >1e-13); requires_comparison=True guards the empty-fixture
  case. Engine-free traps ChangRefsdalChannels.evaluate + _schwinger.f_schwinger
  + _f_schwinger_mpmath (sentinel _EngineDoorTripwire asserted disjoint from the
  live census catch tuples in setUp) and asserts routes land on analytic labels
  (non-vacuous). Mirror delegation proven via mock.Mock(wraps=...) spies on
  born_chart.covers + module _band_split_mask, asserting split taken at
  exp(log_w_grid[0]) (artifact, not literal); route-agreement matrix vs
  _production_born_label across in_box/low_edge/disjoint_high + a refusal-edge row.

## 2026-08-18 (born trained-floor band-split, test_lensing_born_certificate.py)
- COMPLETED a predecessor's partial suite (WP1 Born trained-floor band-split
  + WP2 census mirror). Specs A (BornDisjointEscapeNullSplitTestCase) & B
  (BornTrainedFloorTierRoutingTestCase) were already done+green; only Spec C
  (census revival) was missing. Added BornTrainedFloorCensusRevivalTestCase
  (5 tests) -> 40 passed 6.2s; neighbors test_lensing_born.py +
  test_lensing_serve_route_census.py 95 passed, no regression.
- SPEC C ENGINE-FREE CENSUS PATTERN: drive REAL
  serve_route_census.classify_draw against a synthetic _ProductionModules
  via dataclasses.replace on _load_production_modules(), swapping ONLY
  born_chart + dimensionless_frequency(->FLOOR_DENSE) + ppgo_band_split
  (->w_trust) + ppgo_cell_ceiling(->None) + diffractive_bottom_ceiling
  (->w_low). Everything else (real geometry_partition, shipped
  _band_split_mask, ASTROID_WALL=443.7>>w_hi, _born_trained_floor_route,
  carrier cert) stays production. intercepts 1-5 return before any wave door.
- ROUTE-2-vs-ROUTE-3 TEETH: pass born_carrier_serves=mock.Mock(return_value=
  True); low-edge escape -> route=='born_analytic' AND cert.call_count==0
  (proves Route 2 fired BEFORE the cert); disjoint-HIGH escape (trained
  floor 2.0 > w_hi=0.75, inner _band_split at trained_floor INACTIVE) ->
  route=='born_carrier_only' with cert.call_count==1. cert->False variant
  shows disjoint falls to engine node pass (the pre-WP1 whole-refuse fate
  Route 2 rescues). Predicate teeth: call
  serve_route_census._born_trained_floor_route directly, True low-edge /
  False disjoint.
- FIXTURE ARITHMETIC (FLOOR_DENSE=linspace(0.05,0.75,8), w_low=0.20,
  trained=0.40, w_trust=0.60): host_mask=(0.20,0.60]={0.25,0.35,0.45,0.55},
  engine tier {0.25,0.35}, chart tier {0.45,0.55}. Strict inner sub-band
  requires w_low<trained_floor<w_trust AND both tiers non-empty; a floor
  above w_trust empties chart_mask -> Route 2 False -> falls to Route 3.

## 2026-08-18 (born_certificate degenerate-geometry shard)
- EXTENDED CarrierOmittedTermDegenerateTestCase in
  test_lensing_born_certificate.py with the end-to-end serve-gate pin
  (28 tests, was 24; 5.8s, all engine-free): existing shard pinned only the
  RAW arithmetic (`born_carrier_omitted_term`->inf, inf fails SAFETY*est<=bar);
  the NEW invariant drives the real `likelihood._born_carrier_certificate_
  serves(lens, w_lo, w_hi, images)` and asserts REFUSE for both (a) source
  at origin q2r==0 (gamma=0.30 lens so domain guard passes; refusal is the
  +inf est) and (b) zero-shear gamma==0 (domain guard refuses pre-est).
- KEY: pass WELL-SEPARATED dummy images `[[1,0],[-1,0]]` (sep 2.0 >> the
  0.05 `_SADDLE_FARFIELD_MIN_IMAGE_SEP` backstop) so IF a degenerate query
  reached the backstop it would PASS -> isolates refusal to the degenerate
  guard, not an incidental image-count failure. Added an explicit
  isinf/not-isnan pin (NaN would sneak past `est<=bar` as False on both
  sides, reading like a refusal but undefined) and a TEETH contrast:
  same images + valid positive far-exterior source (gamma=0.30,|y|=80)
  ADMITS, proving the refuse pins aren't vacuous against a refuse-all gate.
- Origin case short-circuits at the est check BEFORE consuming real_images
  (est=inf); zero-shear short-circuits at the `gamma==0` domain check. No
  regression: test_lensing_born.py 53 passed.

## 2026-08-18 (born_certificate serve-routing shard, test_lensing_born_certificate.py)
- EXTENDED existing suite with 3 WP2-lift serve-routing pins (12 tests / 4
  classes), all ENGINE-FREE via a spy-`_born_reconstruct` probe: bind the
  real unbound `LensedRelativeBinningLikelihood._born_residual_analytic`
  onto a `types.SimpleNamespace` via `types.MethodType`; stub
  `_ppgo_band_split`/`_ppgo_cell_ceiling`/`_diffractive_bottom_ceiling` all
  ->None (no-split identity: below_mask all-True, bottom_mask all-False);
  `_StubChart.covers` returns a constant bool, `.evaluate` returns a copy of
  a DISTINCTIVE residual (arange+1j*arange, NOT zeros) so null-identity has
  teeth; spy records residual/masks and returns a sentinel (no heavy tail).
  Patch the MODULE-LEVEL `likelihood._born_carrier_certificate_serves` to
  instrument which branch each query takes.
- THREE PINS: (1) null-identity — in-box covers()==True serves the
  interpolated chart residual byte-identical (cert.call_count==0);
  (2) no-covers()-refusal — EVERY covers()==False query consults the
  certificate exactly once before admitting(zeros)/refusing(None), no
  straight-refusal path survives; (3) saddle resolution fence —
  w_lo*delta_min transition sits EXACTLY at operator.RHO_END(=4.0), below
  refused whole / above serves carrier-only, positive parity NOT fenced.
- FIXTURE NUMERICS (confirmed): positive gamma=0.3,y=80 -> rho=111.55, 2
  images; saddle gamma=1.5,y=150 -> rho=79.06, 2 images, delta_min=22488.29,
  fence_w=1.7787e-4. `real_images`=geom.images is ALREADY real-only (do NOT
  re-mask). `caustic_rho` lives in `cogwheel.lensing.ppgo_map` (NOT
  chang_refsdal.ppgo_map). `_real_images` frequency grid must be strictly
  positive: build band-independent images from linspace(1e-6,...) then probe
  the certificate at w_lo=0.0 separately.
- BACKWARD-COMPAT AUDIT (report-only, NOT my file): the WP2
  `_born_reconstruct` refactor broke sibling
  `test_lensing_born_analytic_reachability.py` — its `_BornAnalyticProbe`
  lacks `_born_reconstruct` (AttributeError at setup) so 10 fail + 10 error
  + anti-vacuity tearDown fires ("0 comparisons ran"). PRE-EXISTING vs my
  work (git shows only my untracked file); another owner must add
  `_born_reconstruct` to their probe. My suite: 24 passed 5.38s;
  test_lensing_born.py no regression.

## 2026-08-18 (born_carrier certificate suite, test_lensing_born_certificate.py)
- NEW SUITE for WP1/WP2 Born carrier-only truncation certificate
  (`_born.born_carrier_omitted_term`), 14 tests / 5 classes, 5.6s. Three
  Architect specs: (1) omitted-term modulus = hypot(a0,0.5*w*b1)/q2r is
  STRICTLY INCREASING in w -> worst case at band CEILING w_hi (guards the
  w_hi-vs-w_lo convention flip; saddle-c3 gates key on w_lo, this one
  keys on w_hi); (2) parity-agnostic invariant b1-a0 == -lam^2*mu_macro
  holds to ~1e-16 across gamma=1 wall incl. macro saddle det_a<0 (proves
  `_born_factors` valid on saddle without the positive-parity policy
  guard); (3) carrier-only serve (`born_lead_carrier` alone) vs exact
  `operator.F_op` <= shipped bar 1e-3 at every admitted point both
  parities, measured worst ~2e-4.
- FRAME PAIRING (no phantom error): `born_lead_carrier` and `F_op` BOTH
  in absolute Fermat-delay frame, both normalized to no-lens
  (F(w->0)=sqrt(mu_macro)) -> directly comparable, NO demodulation. A
  frame mismatch would give O(1) error not 2e-4.
- ADMISSION PREMISE derived live, not pinned: each accuracy fixture
  asserts `_SADDLE_FARFIELD_SAFETY*omitted_term(w_hi) <=
  _SADDLE_FARFIELD_CERT_BAR` (both imported from likelihood) before
  measuring; saddle fixtures also assert resolution fence
  `w_lo*_real_delay_min_separation(source, macro_matrix) >=
  operator.RHO_END` (delta_min ~1e4 for far saddle -> trivially met, so
  the certificate itself is the binding gate). Saddle needs larger |y|
  (150-300) than positive parity (80-100) for the same admission floor.
- COST BOUND: 30 F_op calls (3 pos + 3 saddle) x 5 w-nodes, all w<=0.75
  (exact DD path), shared once via functools.lru_cache(maxsize=1) between
  the accuracy pin and its scatter plot. Self-falsification class proves
  all 3 pins go red (decreasing seq, b1+1e-6 perturbation, served*1.05).
  No regression in sibling test_lensing_born.py (53 passed).
