# Professor short-term (this session)

## Session: F041 build inference review (2026-07-29) -- VERDICT PASS

Reviewed the two Test-Developer shards. Ran (cogwheel-newlal python):
`pytest test_lensing_surrogate.py test_lensing_surrogate_training.py -v`
=> 60 passed, 37 skipped (env-gated/heavy), 224s. No TypeError on _find_cusps
=> the stale callers are fixed and RUN green (not merely collect).

Caller fixes verified in situ:
- test_lensing_surrogate.py:1068-69  gamma=gamma, branch=1 added; `shifted`
  wrapper (1947) forwards via **kwargs -> gamma/branch thread through.
- test_lensing_surrogate_training.py:1006-08 gamma=gamma,branch=branch (+
  width_safety/min_halfwidth preserved); 1660-65 frozen+widened both
  gamma=float(gamma),branch=1 with kwargs preserved.
- Anti-vacuity teardown (surrogate.py:685) present; tests PASSED so n_checks>0.
- No removed-constant (_PROBE_ETA/_CLOUD_MARGIN_FRAC/_CUSP_SPEED_REL_FRAC)
  assertion in EITHER shard file (hits exist only in the out-of-scope
  test_lensing_caustic_cusps.py, not touched here).

F041 acceptance (StableGammaBandsF041TestCase) matches my prior ruling exactly:
- A1 dropped==[] AND every band len(arcs)>0 (load-bearing witness; pre-fix the
  (0.01,0.0462) & (0.0462,0.0644) bands built 0 arcs).
- A2 realized as arc EXISTENCE at gamma in {0.02,0.1,0.3,0.9} -- correct, no
  gamma-stable orientation ratio to assert once the magnitude guard is gone.
- A3 image_count==4 (parity constant at kappa=0, (0,1,1,1) An&Evans) +
  inward_sign in (-1,1) + per-index label stability across gamma>=0.1.
- Self-falsification (StableGammaBandsF041SelfFalsificationTestCase) is GENUINE:
  mock strips arcs below gamma<0.05 -> reproduces BOTH a dropped sliver and a
  zero-arc band, and confirms the clean sweep drops nothing. Can go RED.

All value-based (no code-path / removed-constant assertions). Physics sound.
Heavy full-sampling / brute-accuracy validation is operator-deferred (skipped).
