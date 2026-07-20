# Architect Short-Term Observations

## Build 8a (amplification surrogate) — planning 2026-07-20

- GOAL: additive fast surrogate for Chang-Refsdal F(w) so lensed per-eval
  approaches unlensed. Trained offline vs certified engine, serves only
  in validated domain, exact engine fallback, all named refusals kept.
  Two-tier: in-build = SMALL reduced-domain in-memory surrogate + fast
  gates; full-box training + census = POST-BUILD driver. Sampling stays
  parked (owner ruling A) until this lands.
- SEAM (established by exploration): expensive call is
  likelihood.py `_amplification_coefficients` -> `_evaluate_envelope` ->
  `ChangRefsdalChannels.evaluate` (per-candidate seed + up to 48x in
  fiducial LOO `_envelope_loo_nodes`). Only `_exact_total` (F_op_grid on
  wave nodes) is expensive; geometry (nearest_caustic_point, find_images,
  image_kernel, switch) is cheap. LOO/fiducial-cache/ratio consts live in
  likelihood.py: _LOO_SEED_NODES=8, _LOO_STOP_FAST=4e-3/_STRONG=1e-3,
  _LOO_MAX_NODES=48, fiducial lattice _FID_*_SPACING.
- CHOSEN ARCH: surrogate emulates SACR-C ENVELOPE E(w) (beat-free, smooth
  by construction). At query: cheap geometry-only partition + surrogate
  envelope + `reconstruct_from_envelope` -> channels -> reduce to (k0,k1).
  Needs an ADDITIVE geometry-only method on channels.py (channels.py is
  NOT in the fenced engine list geometry/operator/_schwinger/_hyp1f1/
  _gauge/_dd; `evaluate` stays byte-identical). Single intercept at top of
  `_amplification_coefficients`; default surrogate=None -> exact path,
  crown byte-identical; existing cache/ratio/LOO become the fallback.
- PROFESSOR rulings (carry authority): (Q1) E smooth in (gamma,y1,y2,beta)
  WITHIN a fixed image-count region; decomposition changes topology at
  caustics (2<->4 img), gamma=1 parity boundary, lobe jumps -> build PER
  region/parity, engine fallback near caustics (caustic_distance<0.05
  excluded; |mu|~1/eta). MVP = 2 reduced boxes each in ONE 2-image region
  (pos-parity gamma[0.05,0.45]; saddle gamma[1.1,1.5]); full-box caustic
  tiling deferred post-build. (Q2) beta ELIMINABLE EXACTLY via eigenframe
  rotation exp(-i beta) at 3 code levels -> 4D surrogate (w,gamma,y1_eig,
  y2_eig); envelope invariant under the rotation (delays/tau_c/H_a/switch
  rotation-invariant). Tensor-product CUBIC SPLINE, log-w axis, real/imag
  SEPARATE (never mag/phase - wrap aliasing); ~15 nodes/decade in w, ~6-8
  per param axis. NO quadrant-reflection reduction in MVP (train full box;
  reflection is a 1e-14 TEST). (Q3) gates: (a) envelope eps=max_w|F_sur-
  F_eng|/max_w|F_eng|<1e-3 both parities on HELD-OUT (Sobol+corners+edge
  mids, off training grid, F002-independent fresh engine eval); (b) lnlike
  three-tier: <=0.01 nats crown-family gamma'<0.5 (relax to 0.05 only if
  unreachable from unrelated RB contrib, NEVER past 0.1), <=0.1 nats
  saddle/strong-shear non-rescued (ACCURATE_ATOL), <=max(1.5,0.01*|lnL_bf|)
  rescued/general (RB tol, F016 binning-limited); (c) refusal Boolean +
  F010; (d) timing warm <2.0ms saddle & >=5x, smoke (CI-skippable). (Q4)
  refusal-conservative = axis-aligned containment of certified training box
  + exclusion balls (delta_refuse=grid spacing) around refused pts +
  per-w refusal propagation (ANY refused w at a param pt -> whole pt
  fallback). NO learned mask (false-neg = F005 bug). Store refused-pt set
  in artifact. (Q5) default None correct; enable-by-default deferred
  post-build (needs full-box artifact + >=95% census + PP-plot).
- SIMPLIFIER: keep 2 code WPs core (surrogate module + wiring) — I added a
  3rd focused channels.py geometry-only WP (byte-identity risk warrants
  separation). TRIM per-region partitioning to one-interpolant-per-box.
  TRIM versioned hash to lightweight provenance. WATCH JSONMixin — extend
  `__getstate__` minimally, surrogate passed at construction (pickle-
  preserve; JSON of non-None surrogate deferred). Reuse scipy CubicSpline
  in same log-w/not-a-knot convention as fiducial cache. Training oracle =
  engine directly on a DENSE grid (offline, no LOO), keep surrogate.py
  independent of likelihood.py (avoid circular import).

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
