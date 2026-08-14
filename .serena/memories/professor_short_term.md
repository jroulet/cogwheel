# Professor short-term (Born-intercept wiring build REVIEW, 2026-08-14)

Reviewed the F077 lifted Born-intercept + band-split build (inference-review mode).
Verdict: PASS. Env python: /home/tejaswi/anaconda3/envs/cogwheel-newlal/bin/python.

## Fast tests run (all green)
- test_lensing_born_residual_wiring.py + born_analytic_reachability.py + born.py:
  118 passed / 42s. Warning "Born-residual chart unavailable (artifact missing) ...
  Regenerate with scripts/train_born_residual.py" is the EXPECTED auto-attach
  fallback-to-None contract firing, not a fault.
- test_lensing_ppgo_bandsplit.py + ppgo_map.py: 101 passed, 4 skipped / 15s. The 4
  skips are COGWHEEL_TRAIN_TIER=1 engine-backed real-chart builds (minutes/class) =
  correctly OPERATOR-DEFERRED post-build tier, NOT silent failures.
- surrogate DefaultSurrogatePathTestCase + LnlikeAccuracyTestCase: 5 passed / 37s.
  Oracle re-point verified: likelihood.py oracle constructs born_residual_chart=None
  (test_lensing_surrogate.py L1651-53, L1942-44) with the engine-pure comment.

## Spec->test mapping (every item a named, anti-vacuity-guarded, passing test)
- SERVE-PATH TRACE/reachability -> reachability suite; live-route pinned by
  MockChartServePath(non-None) vs NoChart(None); recon identity <1e-13
  (test_total_matches_carrier_plus_residual_at_dense). Diagnostic plot
  born_residual_wiring_identity.png regenerated 17:34.
- MAP BAND-SPLIT -> test_w_trust_lands_strictly_inside_band.
- NULL-SPLIT IDENTITY -> test_null_split_map_matches_no_map_byte_exact,
  _w_trust_at_or_above_band, _matches_direct_whole_band_rung, +self-falsification
  _identity_breaks_for_different_chart (np.array_equal).
- BYTE-IDENTITY BATTERY (incl. kappa!=0/beta!=0 fall-through) ->
  test_battery_declines_born_intercept + KappaBetaGuardPrecedenceTestCase.
- CORRUPTED HASH -> test_corrupted_content_hash_refuses (names train_born_residual.py).
- SCHEMA REFUSAL -> test_missing_or_wrong_schema_refuses.
- AUTO-ATTACH FALLBACK -> test_load_failure_refuses_to_none_with_warning +
  test_fallback_serve_equals_explicit_none_serve.
- JSON ROUND-TRIP BOTH CLASSES -> JsonRoundTripBornChartTestCase (RB + Marginalized;
  default drops key & re-auto-loads, explicit None verbatim, in-memory chart ->
  NotImplementedError naming 'source path').
- ORACLE RE-POINT -> LnlikeAccuracyTestCase (explicit None) passes in tolerance.
- ENUMERATED RE-POINTS -> DefaultSurrogatePath, NoChartByteIdentity, bandsplit
  save/restore all green in-suite.

## Physics sanity confirmed
Born gate binding on covers() grid rho>2, kappa=0/beta=0 only; band-split reduces
to un-split Born when w_trust>=w_max (byte-exact), consistent with residual R=F-carrier
decaying in w so bare ppGO is exact above w_trust. kappa!=0 silent-accuracy bug (the
gate's raison d'etre) is pinned refused. Heavy full-posterior/real-chart validation is
operator-deferred (COGWHEEL_TRAIN_TIER=1 tier); verdict rests on fast tests + plots.
