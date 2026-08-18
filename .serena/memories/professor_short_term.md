# Professor short-term (2026-08-18) — Born floor-split tiering review: PASS

Reviewed `_born_residual_analytic` four-tier floor-split build (likelihood.py
`_engine_envelope_below_split` / `_born_reconstruct`; census mirror in
serve_route_census.py). Ran fast domain tests only; heavy full-sampling
validation operator-deferred (COGWHEEL_TRAIN_TIER).

## Test runs (green, fast)
- test_lensing_born_certificate.py + test_lensing_born_analytic_reachability.py:
  75 passed / 26s.
- test_lensing_serve_route_census.py + test_lensing_saddle_serve_gate.py +
  test_lensing_born_residual_wiring.py: 139 passed / 34s.

## Six spec invariants -> passing test classes (read bodies, not just green)
1. NULL-SPLIT BYTE-IDENTITY -> BornDisjointEscapeNullSplitTestCase +
   NullSplitIdentityTestCase: np.array_equal of residual/below_mask/bottom_mask
   vs box-miss; residual is carrier-only ZERO; engine_envelope/engine_mask None;
   chart.evaluate call_count 0 (both list-count AND explicit wraps-mock). SOUND.
2. FLOOR-SPLIT TIER ROUTING -> BornTrainedFloorTierRoutingTestCase: sentinel
   residual ONLY on [trained_floor,w_trust] chart tier, ZERO elsewhere; engine
   value on [w_low,trained_floor); diffractive F_P bottom [w_lo,w_low); bare
   ppGO above w_trust. Masks derived from shipped `_band_split_mask`, not
   literals; four tiers proven disjoint+covering. SOUND.
3. FLOOR-SPLIT REVIVAL -> BornTrainedFloorCensusRevivalTestCase: low-edge
   escape route=born_analytic (chart evaluate IS called); disjoint-high escape
   NOT born_analytic (=born_carrier_only). Regression Fact-2 whole-refuse cannot
   reappear. SOUND.
4. CENSUS-MIRROR FAITHFULNESS -> BornCensusMirrorFaithfulnessTestCase:
   delegation proven via wraps-spies on PRODUCTION born_chart.covers and
   likelihood._band_split_mask (split taken at exp(log_w_grid[0]), covers probed
   with FLOOR_DENSE[chart_mask]); route-agreement matrix census==production over
   in_box/low_edge/disjoint_high with a genuine mix. SOUND.
5. ENGINE-FREE GUARANTEE -> BornCensusEngineFreeTestCase: tripwire is NOT a
   subclass of any census refusal tuple (asserted against live tuples); doors
   evaluate/f_schwinger/_f_schwinger_mpmath call_count 0; mpmath not freshly in
   sys.modules. Matters physically (the 60<w<=150 mpmath hang/divergence band).
   SOUND.
6. NULL-RESIDUAL RECONSTRUCTION IDENTITY -> BornNullResidualReconstructionTestCase:
   R=0 -> bare carrier <=1e-13 rel on host tier vs `born_carrier_from_partition`;
   diffractive bottom vs `diffractive_amplification`; above-w_trust vs closed-form
   ppGO image-kernel sum. Three INDEPENDENT oracles, none reads the captured
   total. Demod is exact algebraic round-trip -> 1e-13 is the right bar. SOUND.

## Physics ruling
Tiering polarity matches the shipped `_band_split_mask` convention (zero above
split, populate below) I verified last session. The four-tier layering is
physically correct: diffraction at low w, trained residual chart where trained,
engine hosts the untrained-below-trust gap, bare ppGO carrier above the trusted
floor. Tolerances match my own Q6 authority (bit-exact null-split, 1e-13 demod
round-trip, 100% census-mirror agreement). No concerns.

Verdict: PASS.
