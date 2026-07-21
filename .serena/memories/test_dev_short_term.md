# Test Dev Short-Term Observations

## 2026-07-21 build8f/g WP1/WP4 test_lensing_airy_fold.py (EXTENDED: ladder wiring)
- Added FOUR shards to the existing Airy/Pearcey suite (do-not-rewrite):
  ServingLadderDeterminismTestCase, CertifiedPathByteIdentityTestCase,
  CornerCensusContractTestCase, LadderByteIdentitySelfFalsificationTestCase.
  FULL FILE FAST: 48 passed/7 skipped/1 xfailed ~18s. New classes gated
  (COGWHEEL_BRUTE_ACCURACY=1): 18 passed/1 xfailed ~26s. Geometry neighbor
  13 passed clean. NO production touched.
- Imports added: ast, importlib.util, subprocess, sys, tempfile, lru_cache,
  expectedFailure, `from cogwheel.lensing import surrogate_census`.
- LADDER ROUTE mirror `_ladder_route` (independent of F_op_grid): w<=60->
  schwinger; w>60 saddle & w*dmin>=RHO_END(4.0)->geometric; else fold arm
  ->cusp arm->refusal. Geometric rung is SADDLE-ONLY (`_saddle_grid`);
  `_positive_parity_grid` has NO geometric rung (verified by reading both).
  `_LADDER_NODES` 5 regimes: schwinger(w40,g0.5), geometric(w100,r1.2,g1.5
  saddle), fold(w500,r0.14,ang1.0,g0.3), cusp(w80,r0.18,ang0.3pi,g0.5),
  refusal(w100,r0.28,g0.3). Served value == arm/geometric value bit-for-bit
  (np.complex128.tobytes): `_uniform_arm_value` returns complex(arm), stored
  verbatim; geometric stores complex(geometric_amplification). Verified.
- BYTE-IDENTITY vs HEAD: `_head_operator()` lru_cached loads HEAD operator.py
  via `git show HEAD:...` -> REAL temp .py file + importlib.spec_from_file_
  location (numba @njit(cache=True) needs a file locator; exec-into-synthetic
  module raises "no locator available"). w<=60 grid [5,20,40,55,60] over 4
  configs (2 pos-parity, 1 saddle, 1 kappa=0.1): max|diff|=0.0 + orders/conv
  array_equal. select_branch identical over 3x3x4 grid. Only-change witness:
  fold node w500 HEAD refused / CUR served. ~14s (HEAD jit compile).
- CROSS-ARM (spec "<=1e-3 where both valid", gated) -> PREMISE REPAIR not
  tolerance repair: arms' cert gates are NOT mutually exclusive. Measured
  double-certify node gamma0.5 r0.14 ang~0.45pi w150 where fold AND cusp both
  certify but envelopes disagree 29% (0.293). So loose "both certify" != genuine
  shared validity; literal 1e-3 over that set is FALSE. Reframed
  test_cross_arm_conflicts_resolved_by_fixed_priority: at every double-certify
  node the ladder serves the FOLD arm bit-for-bit (fold-before-cusp priority)
  -> the spec's PRIMARY clause "no node served by two arms with different
  answers" WITH TEETH; falls back to disjointness assertion if no overlap
  (non-vacuous). Spread saved as diagnostic. At the LADDER FIXTURES themselves
  arms ARE disjoint (fold node: cusp=None; cusp node: fold=None) -> FAST
  test_uniform_arms_disjoint sum(serving)==1 holds.
- CENSUS spec = HONEST CONTRACT (WP1 extended census NOT landed;
  surrogate_census.run report has served_fraction/fallthrough/per_chart_eps/
  lnl_tiers/binning_floor but NO fold/cusp arg CDFs, Wilson intervals, (a)-(d)
  fractions). FAST-testable now: operator.L_MAX==48 pinned; select_branch two-
  condition gate pinned (geometric iff w*dmin>=RHO_END AND L>L_MAX; L==48 ->
  wave); AST purity walk of surrogate_census.py forbids Name.id/Attribute.attr
  in {f_schwinger,F_op,F_op_grid} + ImportFrom _schwinger (currently 0 refs,
  confirmed by grep). @expectedFailure tripwire flips RED when extended API
  lands (hasattr corner_census/wilson_interval/... OR extended tokens in run()
  source). Bump n_checks BEFORE assertion (expectedFailure covers body not
  tearDown). `_census_run_source()` slices run() body via ast lineno/end_lineno.
- SELF-FALSIFICATION (FAST, all confirmed): one-ULP nextafter breaks byte gate;
  corrupt fold arm *(1.0001) via mock.patch.object -> served tracks corruption
  != true captured value; forced double-serve (patch cusp_amplification->
  const) -> sum(serving)==2; patch operator.L_MAX=999 -> select_branch(100,
  0.05,49) flips geometric->wave (pin load-bearing; select_branch is PLAIN
  python reading module globals, patchable); AST purity positive control flags
  synthetic operator.F_op_grid call.
- NEIGHBOR DRIFT (pre-existing per prior builds, report don't touch):
  test_lensing_schwinger RefusalAboveCeilingTestCase + test_lensing_operator
  positive-parity refusal reds are the WP1/WP4 re-baseline owned elsewhere.

## 2026-07-21 build8f WP3/WP4 test_lensing_airy_fold.py (EXTENDED: Pearcey cusp arm)
- Added THREE Pearcey/fall-through shards to the existing Airy uniform-
  arms suite (do-not-rewrite): PearceyPrimitiveCertificationTestCase,
  PearceyCuspScalingTestCase, UniformArmFallThroughTestCase,
  PearceyCuspSelfFalsificationTestCase. FAST 31 passed/6 skipped; gated
  (COGWHEEL_BRUTE_ACCURACY=1) 37 passed. ~12s gated single file.
- Module `_pearcey_cusp.py`: pearcey(x,y) = rotated-contour (pi/8) paired
  N/2N Gauss-Legendre, certified in place vs _CERTIFICATION_TOL=3e-10;
  coarse=panels_central, fine=2*panels_central; returns fine or None.
  cusp_amplification(w,source,gamma,*,beta,kappa,envelope_bar=0.05): builds
  x=delta_par*sqrt(w)/sqrt|C4|, y=delta_perp*w^0.75/|C4|^0.25; REFUSES when
  radius=hypot(x,y) < radius_min=(_UNIFORM_ERROR_CONST/envelope_bar)^(2/3)
  (_UNIFORM_ERROR_CONST=1.0). Docstring ADMITS served amplitude "awaits a
  brute-force cross-check" — my gated engine-match IS that cross-check.
- THREE independent oracles for P (F002): closed form P(0,0)=(Gamma(1/4)/2)
  e^{i pi/8} (3.8e-15); scipy adaptive QUADPACK on single rotated pi/8 line
  (FAST, reldiff 0.0); 40-dps mpmath.quad on same line (gated ~1e-13). All
  use ONE straight line — different contour decomposition + quadrature than
  module's central-segment+two-tails fixed-order rule.
- SCALING SWAP done as PURE-MATH falsification, NOT a brittle nonlinear
  engine fit: semicubical 27y^2=-8x^3 is P's fold caustic (stationary pts
  coalesce, phi''->0, asymptotic diverges); correct 1/2,3/4 exponents make
  common w^{3/2} cancel so a source ON the fold arc STAYS on it at all w
  (residual ~4e-14), swapped 3/4,1/2 leaves w^1 vs w^{9/4} and walks it off
  (residual ~-74). Exponents ALSO fit from captured controls: recording
  wrapper `_capture_cusp_controls` patches _pearcey_cusp.pearcey to log the
  ACTUAL (x,y) the code feeds (never reconstructs them); polyfit log-log
  slopes 0.5/0.75 within 5%.
- FALL-THROUGH probe node MUST be CUSP-served, not fold-served: my first
  pick (gamma0.5 r0.15 ang0.35pi w80) was served by the FOLD arm first
  (grid != cusp; cusp corruption inert!). Correct probe: gamma0.5 r0.18
  ang0.3pi w80 -> fold_amplification None, cusp serves, grid==cusp
  BIT-for-bit (np.complex128.tobytes). Corruptions -> named
  _schwinger.SchwingerCertificationError (w>60): patch _CERTIFICATION_TOL=0
  (cusp never certifies) OR patch pearcey->nan. Threshold flip BOTH knobs:
  _UNIFORM_ERROR_CONST cross = _DEFAULT_ENVELOPE_BAR*radius^1.5 (0.5x serve
  /2x refuse, grid route follows); envelope_bar cross = const/radius^1.5
  (loose serve/tight refuse). Dead-code control: two consts BOTH below
  crossing -> no flip (isolates flip to CROSSING not mutation).
- ENGINE-MATCH premise repair (NOT tolerance repair): spec bar 1e-2 along
  fold arcs. Weakest-shear fixture (gamma0.3 r0.10) at w=50 gives 1.07e-2
  (7% over) from cluster/far-image interference BEAT (error non-monotone in
  w: gamma0.5 0.0051@50 -> 0.0071@60). At CEILING w=60 (deepest reachable
  node in exact Schwinger band, w<=60) ALL 3 fixtures clear (worst 0.007).
  So gate strict 1e-2 at _ENGINE_MATCH_W=W_CEILING_SCHWINGER; witness (non-
  gating) sub-ceiling w=50 < 0.05. F_op_grid at w=60 uses exact f_schwinger
  (60 not >60) so engine=truth.
- Self-falsification (FAST, all red-confirmed): corrupt primitive *(1+1e-6)
  caught by scipy ref (>1e-8) but tainted oracle blind; loosen
  _CERTIFICATION_TOL=1e6 + _panel_count->1 serves wrong value ref exposes;
  wrong origin form Gamma(1/4)/2 (no e^{i pi/8}) rejected; one-ULP nextafter
  breaks byte-identity; same-side threshold move doesn't flip.
- NEIGHBOR DRIFT (report, don't touch; I edited ONLY the test file):
  test_lensing_schwinger.py RefusalAboveCeilingTestCase 3 fail + 2 err —
  asserts F_op RAISES for every (gamma,y,w>60) but WP4 wired uniform arms
  into the ladder so some w>60 nodes now SERVE. Owned by schwinger re-
  baseline. test_lensing_operator.py 23 passed, geometry 13 passed (clean).


## 2026-07-21 build8e WP2 test_lensing_airy_fold.py (NEW suite, Airy fold arm)
- Module `chang_refsdal/_airy_fold.py`: airy_fold_value(w,tau_bar,xi,p,q,
  sigma)=2sqrt(pi)e^{i(w tau_bar+sigma)}[p w^{1/6}Ai(-xi) - i q w^{-1/6}
  Ai'(-xi)] via scipy.special.airy(-xi). fold_amplification serves ONLY the
  high-w corner (error-gate refuses w=200@r0.14, serves w>=500); q=0 leading
  order, sigma=-pi/4; NOT overlapping w<=60 Schwinger engine (no engine
  oracle). 6 classes, 17 tests (2 GATED under COGWHEEL_BRUTE_ACCURACY).
- TWO independent oracles (F002): (1) `_mp_airy_fold` = mpmath.airyai at
  dps40 re-eval of the SAME closed form (different Airy evaluator than
  scipy) -> transcription gate 1e-9; (2) `_geometric_two_image_sum` from
  geometry primitives (find_images/delay/morse/magnification) = exact
  sqrt|mu_+|e^{iw tau_+}+sqrt|mu_-|e^{iw tau_- -i pi/2}.
- KEY currency = MAX-NORMALIZED ENVELOPE (|F|), NOT complex pointwise:
  pointwise resid is dominated by interference nulls + huge w*tau_bar
  carrier (0.01-0.2, non-monotone); envelope error is clean xi^{-3/2}
  (3.85e-4@xi40, ratio 2.83=2^{3/2}/doubling). Far-field amplitudes fed to
  airy_fold_value: p=(s_++s_-)/2 w^{-1/6} xi^{1/4} (SUM->Ai),
  q=(s_--s_+)/2 w^{1/6} xi^{-1/4} (DIFF->Ai'). fold_amplification's SERVED
  q=0 CANNOT hit 1e-3 on asymmetric folds (floors ~asymmetry ~10%) -> I do
  NOT gate that; the 1e-3 far-field cert lives at airy_fold_value level with
  full p_sum/q_diff. Fixture ray angle 1.0, gamma0.3 -> sqrt|mu| ratio ~1.2
  (asymmetric, needed so p/q swap can't hide).
- GOTCHAS: on-caustic |F| peak is NOT at xi=0 — |Ai(-xi)| peaks at first
  fringe xi~1.02 (Ai max at -1.019); finiteness test asserts peak on present
  side 0<=xi<2, not "at caustic". _OUTSIDE_RADIUS 0.35 still SERVES (2-image
  min/saddle still merges) — dropped the "outside refuses" claim; real
  refusals = w<=0, nan/wrong-shape src, envelope_bar<=0, kappa>=1 TypeIII,
  |gamma|==1-kappa parity, near-caustic(r0.285) error gate, low-w(200).
- Self-falsification (FAST, all confirmed red): sign flip via
  mock.patch.object(_airy_fold,'airy', lambda z: scipy_airy(-z)) -> Ai(+xi):
  transcription resid 0.74, present-side maxima 4->0; p<->q swap -> envelope
  0.78; sqrt|mu| amplitude -> |F(0)| tracks divergence (bad grows 2.34x,
  calibrated flat 1.009x, bad_growth==sqrt_mu_growth to 6 places); tainted-
  oracle positive control (oracle calling airy_fold_value blind to flip while
  mpmath catches it). NO production code touched. geometry neighbor suite 13
  passed. Plots: sign_convention_handoff/at_caustic_finite_peak/
  far_field_envelope_convergence.png.

## 2026-07-20 build8d test_lensing_surrogate BYTE-PIN RE-BASELINE (WP1)
- WP1 reroutes SHEARED positive-parity F_op/F_op_grid (gamma'>0) from
  legacy operator-series to exact Schwinger; legacy demoted to
  `operator.legacy_operator_oracle` (=_grid_certified, NOT in __all__) +
  gamma'==0 point-lens exit. Positive-parity F changes ~5e-15 (byte flip).
- CRITICAL: the existing CrownByteIdentityTestCase does NOT catch this —
  `_head_likelihood_class()` reloads only HEAD likelihood.py; both cur+head
  import the SAME working-tree operator.py, so it stays green (correctly
  certifies 8a likelihood wiring is additive-neutral). Only operator.py is
  modified this build. So I ADDED CrownContractFlipWitnessTestCase (8 tests)
  witnessing the flip at the F_op level: NEW Schwinger vs OLD
  legacy_operator_oracle (independent algo, F002) on the legacy-CERTIFIED
  overlap, max-normalized 1e-10. `_flip_witness_metrics` collects overlap
  per-node (legacy per-node loop cheap ~0.26s/60; Schwinger ~58ms/node),
  batches F_op_grid on overlap, gates max(metric_re,metric_im)<1e-10.
- Witness table: A/crown,B/two-image,crown,near-fold ~5e-15; sub-critical
  (gamma=0.35) partial overlap (legacy refuses w>17.6) tightest ~4.8e-11 —
  that's the LEGACY oracle's own 1e-10 cert floor near its cancellation
  edge (Schwinger=exact truth), NOT a Schwinger defect; still clears gate.
- Dispatch seams (Python-level even though targets njit): spy
  `mock.patch.object(schwinger_module,'f_schwinger',...)` +
  `mock.patch.object(operator_module,'_grid_certified',...)`. gamma'>0 crown
  -> f_schwinger=n_w, _grid_certified=0; gamma'=0 pointlens (gamma=0.0) ->
  f_schwinger=0, _grid_certified>0. legacy_operator_oracle holds its own ref
  to real _grid_certified so patching operator._grid_certified can't leak.
  F010 mutation: patch f_schwinger*(1+1e-4) -> witness metric 9.2e-5 RED.
  Refusal: F_op(w=68>W_CEILING_SCHWINGER=60, gamma'>0) raises
  SchwingerCertificationError. Full suite 41 passed 1 skipped ~11min.
- NEIGHBOR DRIFT (report, don't touch): test_lensing_operator.py 6 fail +
  4 err — all OLD positive-parity operator refusal contract
  (test_raises_named_error_above_w_ceiling, _former_silent_nan_config_now_
  refuses, _certified_or_named_refusal_across_band, _cancellation_ratio_
  field_matches_independent) that WP1 rerouted to Schwinger. Operator
  suite's own re-baseline run owns these; my change is test-only.

## 2026-07-20 build8c test_lensing_surrogate (TEST 11/12/13)
- TEST 11 (preservation): 8a suite greened vs landed multi-chart surrogate.
  Root cause of 7 reds = `_refusal_surrogate()` box (0.8,1.2) centres on
  gamma=1.0, and `from_engine` calls UNWRAPPED `_box_region_labels` at box
  centre -> LensDomainError. Fix: box->(0.8,1.3), n_gamma=6 preserves 0.1
  spacing + the gamma=1 refusal column, nudges centre to valid 1.05. No
  tolerance weakened. Only 2 real single-box scrapes needed re-target:
  `_param_spacing`->`charts[0].param_spacing`; F010 mutation now patches
  `surrogate_module._in_exclusion_ball` (the load-bearing guard both
  `envelope`+`in_domain` share via `_farfield_raw_chart`), not `in_domain`.
  Property shims (gamma_grid etc.->charts[0]) already cover the rest.
- TEST 12/13 added: `_multichart_fixture()` (lru_cached) builds a 4-chart
  surrogate (pos_tube/pos_ff/sad_tube/sad_ff) from synthetic smooth tensors
  via TubeChart/FarFieldChart.from_values — NO engine calls, runs instant.
  Overlap band engineered via ff eta_overlap_min=0.02 < tube eta_max=0.05
  (eta in (0.02,0.05] double-matches -> tube priority wins). Negative-theta
  saddle wedge theta_grid=[-0.39,-0.09] exercises `_theta_into_frame` unwrap
  (query theta=2pi-0.19 -> -0.19). Provenance MUST use lists not tuples
  (json round-trip value-equality). refused_points always (n,3) via
  `_normalize_refused` (None->empty(0,3)) so npz allow_pickle=False safe.
  Self-falsification: dataclasses.replace(tube, eta_max=0.025) flips overlap
  selection tube->ff. No-double-serve proved by isolating each chart in a
  1-chart surrogate. Full suite: 33 passed, 1 skipped (timing opt-in), ~5min.
