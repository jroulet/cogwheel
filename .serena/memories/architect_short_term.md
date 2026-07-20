# Architect Short-Term Observations

## Build 7b (saddle channel/likelihood/prior integration) — planning notes 2026-07-20

- Engine DONE (6/7a): geometry saddle-capable (macro_matrix/critical_point/
  nearest_caustic_point two-lobe deltoid all verified in-code), _schwinger.py,
  operator parity dispatch, 7a strong-shear fallback. Only two interim guards
  block saddle: top of `channels.ChangRefsdalChannels.evaluate` and
  `LensedWaveformGenerator.__init__` (both `if not 1-kappa>abs(gamma): raise`).
- Professor rulings (carry authority): (Q1) prior = SINGLE uniform gamma
  (0.0, 1.6) identity transform, NO discrete parity label; gamma=1 is a
  measure-zero named refusal (macro_matrix det A=0), zero Jacobian subtlety.
  (Q2) KEEP mass range 3500 + _Y_SCALE=307 unchanged; saddle w<=60 ceiling
  handled by named refusal (accepted inefficiency); w*sqrt(s)<=60 corner
  constraint is LESS restrictive than saddle w<=60 so 307 stays correct.
  (Q3) fold ['u1','u2'] STAYS VALID on deltoid — reflection symmetry of
  Fermat potential is parity-blind, cusp count irrelevant; test = full complex
  F at saddle config + 3 reflections identical 1e-14. (Q4) band-limit refusal
  BEFORE coherent-score is automatic by data-flow (F computed before QMC);
  WP=confirm + spy-test. (Q5) rescued 0.94-nat gap root cause = max|F|
  normalization under-weights deep-cancellation troughs; FIX = _LOO_STOP
  4e-3 -> 1e-3 (keep _LOO_MAX_NODES=48, _ENVELOPE_SCALE_FLOOR=1e-12); gate =
  direct AND ratio vs bruteforce < 0.1 nats on 3-config set; fallback knob =
  seed min-|F| node. (Q6) end-to-end oracle = mpmath 1D-Schwinger-rep,
  AST-guarded import geometry+mpmath only, |F_prod-F_oracle|/|F| < 1e-9 at
  3-5 w in [5,50]; + geometric image-sum cross-check w=40-50 tol 5e-2. (Q7)
  deep-band Morse-phase pins already in engine suite — do NOT duplicate;
  ONE channel-layer add = F009-S flat macro-limit reconstruction |F|=1/
  sqrt(gamma^2-1) 1e-6, flat.
- Simplifier: 4 WPs right granularity, DON'T merge WP1/WP2 (diff files,
  parallel), DON'T add discrete parity label, fold verify is a TEST not code
  (Professor resolved: stays valid), WP4 = 3-turn constant change (don't tangle
  brute-force harness into Coder — Test Dev owns it), watch _LOO_MAX_NODES
  ceiling-vs-stop interaction.
- Existing pinned refusal tests to RECONCILE (Test Dev):
  test_lensing_waveform ConstructionValidationTestCase::
  test_macro_saddle_raises_lens_domain_error + channels saddle-guard test.
- SPEC microlensing rows + in-code "INTERIM/positive-parity-only" docstrings
  need update (SPEC -> Librarian; in-code docstrings -> Coder WPs).

## v1 REVIEW (2026-07-20): WP1/WP2/WP3 + Professor inputs + all 12 domain
  tests APPROVED AS WRITTEN. WP4 REJECTED. Global _LOO_STOP 4e-3->1e-3
  measured 1.44x crown wall time (37.5->54ms probe; ~14ms scaled) THROUGH
  the non-negotiable 10ms gate. RESCOPE WP4: close 0.94-nat gap WITHOUT
  touching certified positive-parity hot-path cost. Chosen mechanism =
  DETERMINISTIC candidate-dependent stop: _LOO_STOP stays 4e-3 for
  |gamma|<0.5 (certified fast region incl. crown), tightens to 1e-3 only
  above (a pure fn of candidate lens params -> fiducial-cache purity holds).
  min-|F| seed retained as documented escalation. WP4 verify MUST have
  BOTH: (a) crown warm lnlike unchanged / FewMsTimingTestCase passes /
  crown envelope node count bit-identical for gamma'<0.5; (b) rescued-node
  gate <0.1 nats direct AND ratio vs brute on 3-config set.
