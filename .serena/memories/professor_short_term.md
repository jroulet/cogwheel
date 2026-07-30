# Professor short-term (session 2026-07-30, arc-length TubeChart INFERENCE REVIEW)

Reviewed the arc-length-coordinate TubeChart build (the s-vs-theta consult I did
earlier this session, now built). Ran fast domain tests, env
`cogwheel-newlal` python. VERDICT: PASS (heavy engine tier operator-deferred).

## Fast-tier results (all green)
- test_lensing_surrogate.py: 62 passed, 1 skipped (timing smoke). Arc-length specs:
  - ArcLengthMapRoundTripTestCase (both parities): self-inversion |s(theta(s))-s|/s_tot
    MEASURED ~3e-16 (positive astroid) / ~2e-16 (negative saddle) vs certified bound
    1e-6 (ARC_ROUND_TRIP_TOL). 10 orders inside -> calibrated, not perched.
  - ChartSplinesInArcLengthTestCase: served-vs-arc-length contraction diff = 0.0
    (machine precision); served-vs-naive-raw-theta = 0.540 (54%). PROVES interp
    coordinate is arc-length image s, not theta. Exactly the coord change I certified.
  - CoordinateChangeAccuracyTestCase: worst arc rel = 1.979e-04 << 0.05 (F016 COMPLEX
    bar); naive theta contraction = 0.542. Coord change does not move served F beyond
    fit error.
  - TubeChartMapSerializationTestCase + Serialization(MultiChart): npz/pickle round trip
    bit-identical map + served values.
  - IdentityDefaultBackCompatTestCase: served == frozen golden literals; can-go-red
    witness present.
  - ArcLengthSelfFalsificationTestCase: corrupted row breaks 1e-6 bound; non-monotone s
    rejected; map-not-anchored-at-theta_lo rejected; perturbed map moves served value.
    Tests have teeth.
- test_lensing_surrogate_training.py fast tier:
  - ArcLengthNodePlacementGeometryTestCase (7): nodes uniform in arc length, endpoints ==
    arc bounds, independent polyline oracle confirms. PASS.
  - SingleGammaMapAdequacyTestCase (5): band-edge normalized-map deviation < 0.05
    adequacy bar; midpoint reproduces both edges; wide-parity-wall self-falsification.
    Confirms my consult conclusion (A) band-midpoint de-correlates a topology-stable band.

## Operator-deferred (correctly NOT run; my budget forbids)
Engine-backed classes gated behind COGWHEEL_TRAIN_TIER=1 ("minutes per class, driver
runs post-build"): ArcLengthBoundShiftMarginTestCase (knife-edge swing<5% vs incumbent
~20%), ArcLengthNodeEfficiencyTestCase (eps(n4)<0.059 golden literal), ShippedArcLength
TubeGridTestCase. These are the LOAD-BEARING quantitative eps claims -> driver ship gate.
Fast-tier evidence is fully consistent with them.

## Physics judgement
The build does what the consult asked: spline in the arc-length image, monotone
cumulative-trapezoid map (|y'|>=0 guarantees monotone s from theta_lo for BOTH sign
conventions), single band-midpoint gamma map. No defects found in the fast tier. Only
caveat: the eps-improvement / knife-edge numbers themselves are engine-tier deferred.
