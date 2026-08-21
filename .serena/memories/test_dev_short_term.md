# Test Dev Short-Term Observations

## 2026-08-21 (INS-2-001 port: end-to-end serve + load contract, low_w_shell_chart)
- INS-2-001 PORT COMPLETED (test_lensing_low_w_shell_chart.py, 22 tests green
  in 20.6s): added ShellServeNodeExactTestCase (production
  `_low_w_shell_chart_serve` driven end-to-end via a SimpleNamespace binder +
  `reconstruct_farfield` interception, kappa in {0.0, 0.2}, beta in {0.0,0.3};
  `_engine_farfield_total` stubbed to the `_engine_reference_kappa` oracle for
  above-split nodes; split wiring pinned by asserting the host received exactly
  dense_w[~below] recomputed from `_reduced_min_delay_separation` +
  `_band_split_mask`; below-split chart composition node-exact ~1e-16 for BOTH
  kappa cases) and ShellLoadContractTestCase (bit-identical round-trip, missing/
  foreign schema + missing content_hash refusals, one flipped real_coeff under a
  stale hash refuses, rehashed tamper loads cleanly = saver<->loader hash-field
  agreement, tampered provenance still loads).  `_round_trip_chart` uses seeded
  RNG coeffs (engine-free) -- no need to reuse the heavy `_exact_residual_chart`.
- FIXTURE MEASUREMENT (witness gp=0.8, rho=1.2, theta=pi/5 node): serve rho and
  theta round-trip EXACTLY from `_make_lens` (rel err ~1e-16), delta_min=4.474,
  w_shell=0.2235 splits the chart's log-w window [0.05..0.3] as below=[0.05,0.1,
  0.15,0.2] above=[0.3] -- a straddling fixture, no stubbing ambiguity.  rho=0.5
  and rho=1.5 both DECLINE (inline rho gate fires before any grid check).
- The `_make_lens` fixture inversion is the SAME one the deleted diffractive
  suite used (`y = sqrt(lam)*R(beta) y_eig`; serve inverts with exp(-1j*beta));
  the beta-rotation + atan2 round-trip is bit-exact for interior grid nodes.
- The port reuses the existing `_exact_residual_chart()` (lru_cached, 480 nodes)
  for the serve class -- shared with the pre-existing accuracy/boundary classes,
  so the ~14-20s build is paid ONCE.

## 2026-08-21 (low_w_shell_chart test suite, test_lensing_low_w_shell_chart.py)

- OFF-GRID 1e-4 IS NOT ACHIEVABLE AT SYNTHETIC SCALE for the shell residual
  chart: R = f_pure - born_lead_carrier has genuine theta structure near the
  caustic, so a 480-node (4x4x6x5) fixture measures ~2e-2 off-grid (theta/
  log-w midpoints), and 8-12 theta nodes only reach ~9e-3..3e-3 (plateaus).
  Substituted a MEASURED 0.1 off-grid bar; node-exact 1e-10 round-trip is the
  robust pin (bit-exact ~1e-16).  Same family as the "4-node w charts cannot
  hit off-grid 1e-4" memory.  f_schwinger at low w (w<=1) is ~15-25ms/call,
  so the 480-node build is ~14-20s (lru_cache-shared across classes).
- MACRO-LEAD CARRIER HAS CONSTANT MAGNITUDE: born_lead_carrier =
  sqrt_mu * exp(1j w phi_geo), |carrier| = sqrt_mu w-INDEPENDENT (no beating
  zero).  So a quotient residual f_pure/carrier does NOT trip the literal
  "max|R|/max|F| <= 10" no-poles cap at low w (|f_pure/carrier| ~ 1) -- the
  quotient's 5800x poles must come from a DIFFERENT (beating) carrier, not
  the macro lead.  The no-poles invariant's real teeth are the node-exact
  round-trip + falsification controls, not the ratio cap.  Measured
  max|R|/max|F| <= 0.9 over witness cells (gp/rho/theta in {0.8/1.1/0.2,
  0.5/1.2/0.5, 0.3/0.9/1.0} x w in [0.02,1.0]).
- RHO=1.4 HANDOFF (shell vs Born) IS ENGINE-SIDED, NOT CARRIER-SIDED:
  RHO_HI == _likelihood._BORN_RHO_FLOOR == 1.4 (bit-equal, no gap/overlap).
  At rho=1.4 the Born carrier-only certificate REFUSES (born_carrier_omitted_term
  ~0.056-0.13 -> safety*est ~1.1-2.5 >> _SADDLE_FARFIELD_CERT_BAR 0.001), so
  the Born side falls to the exact engine.  The honest "no step" pin =
  shell chart accurate at its rho=1.4 boundary node (node-exact) + Born
  carrier-only refusal (engine) -- NOT "both arms carrier-served at 1e-4"
  (the residual is ~1.5% at rho=1.4, non-negligible).  Professor Q3b-consistent.
- FALSIFICATION VIA CHEAP VECTOR ARITHMETIC (no 2nd engine bake): derive
  wrong-residual charts from the exact chart's coeffs by adding the carrier
  correction -- doubled carrier: R_wrong = R_exact - carrier (= f - 2 carrier);
  unit sqrt_mu: R_wrong = R_exact + carrier*(1 - 1/sqrt_mu); zero residual.
  Each breaks node-exactness to ~0.4-1.0 rel err.  No extra f_schwinger calls.
- reduced_source(gp,rho,theta) + f_schwinger(w,src,gp) works at theta=0 and
  pi/2 endpoints and across gp in [0.3,0.9], rho in [0.6,1.4] (no domain
  errors at low w).

## 2026-08-21 (low_w_chart_rho_partitioned test port + per-carrier pins)
- CUSP-TRANSITION DETECTION BUG (production, needs coder fix):
  `low_w_diffractive_chart._airy_fold_form` keys `cusp_transition` on
  `_soft_axis_cubic() is None` (p<=0 / non-finite -- the image-at-point-mass
  case), but the genuine b3->0 fold->cusp transition is detected by
  `_fold_amplitudes` returning None (`abs(b3) <= _B3_MIN=1e-6`).  The b3~1e-15
  cusp cells (e.g. gp=0.8, rho=1.2, theta=0.2) therefore return `(None, False)`
  and are DECLINED (kind 'airy_fold', f_ref None) instead of Pearcey-fallback.
  NO cell returns 'pearcey_cusp' anywhere -- the restricted-Pearcey fallback is
  dead code.  3 tests are honest-RED documenting this:
  `CuspFrefNonVanishingTestCase._witness` (asserts cusp_transition),
  `CuspServeEngineNodeExactTestCase` (chart build errors), and
  `FoldCuspContinuityTestCase.test_handoff_visits_both_forms`.  They flip green
  once `cusp_transition` is keyed on the `_fold_amplitudes` refusal.
- GEOMETRIC RESOLVED-BRANCH ACCURACY vs the spec's blanket 1e-4: the two-image
  geometric sum error vs f_schwinger is ~1e-1 at w=2 (w*delta_tau=16), 4.6e-3
  at w=8 (65), and only reaches 1e-4 near w=16 (130), 6.5e-7 at w=30.  The
  spec's "1e-4 at w=2.0,8.0" for the geometric cell is not met; substituted
  honest bars (2e-1 / 2e-2).  The resolved branch is geometric_amplification/lam
  (kappa=0), NOT the chart.
- MACRO-FOLD RENORMALIZATION changes the fold bars: shell F_ref min/max 0.772
  -> 0.515, residual min/max 0.446 -> 0.295 (so SHELL_RESIDUAL_RATIO_TOL had to
  drop 3e-1 -> 2e-1).  The Airy F_ref now has two regimes: |F_ref|==sqrt_mu
  exactly at h==0 (unresolved), and |F_ref|^2/Wronskian == const at h==1
  (resolved).  The wall-band exterior witness (0.8, 2.0, 0.6) is now MACRO
  (|F_ref| == sqrt_mu exactly, w-independent), no longer fold/cusp.
- 4-NODE w charts CANNOT hit off-grid 1e-4 (cubic interpolation is dragged by
  distant nodes): fold cell off-grid w=1.0 gives 8.6e-2, macro w=0.15 gives
  1.05e-2.  Per-carrier serve pins are NODE-EXACT only (1e-10); off-grid
  accuracy is the full-bake margin-report's (DRIVER) concern, matching the
  existing ServeEngineNodeExactTestCase design.
- Trainer script `scripts/train_low_w_diffractive_chart.py` imports via
  importlib (spec_from_file_location) BUT must register in sys.modules BEFORE
  exec_module -- its `@dataclass _FillResult` does `sys.modules.get(
  cls.__module__)` and raises AttributeError on an unregistered module.
- INS-2-004 (2026-08-21): geometric far-exterior serve is NODE-EXACT (~1e-16)
  at grid nodes -- the baked residual r=f_pure*sqrt(1-gp^2)/F_ref absorbs the
  two-image geometric sum's finite-w deviation and re-modulates with the SAME
  partitioned_reference the trainer baked.  Deleted GEO_SERVE_TOL_RESOLVED=2e-2
  and GEO_SERVE_TOL_MARGINAL=2e-1 (they'd pass a BARE un-anchored geometric sum
  ~1e-1@w=2 / 4.6e-3@w=8 and fail to certify INS-1-002); test_geometric_far_
  exterior_serve now uses NODE_EXACT_TOL like the sibling pins.
- PRE-EXISTING RED in the file (NOT caused by INS-2-004, reproduced with edits
  reversed): production partitioned_reference now returns kind='geometric' for
  off-caustic cells (rho>RHO_HI) whose w_grid contains resolved nodes, which
  strands MacroCarrierReferenceTestCase.test_wall_exterior_routes_to_macro /
  test_macro_fref_magnitude_w_independent (resolved nodes are geometric-sum
  values ~1.4-1.76, not sqrt_mu=1.6667), RhoPartitionContinuityTestCase.
  test_sweep_straddles_the_carrier_partition (expected 'macro' at rho=1.45,
  got 'geometric'), and FoldCuspContinuityTestCase.test_fref_continuous_across_
  handoff.  These classes were authored against the macro-only production
  state; owner shard must re-baseline to the geometric-kind production.
- `_NON_VANISHING_MIN_RATIO` deleted; the old `CuspFrefNonVanishingSelfFalsification`
  (far-exterior cusp cell declined by the guard) is MOOT -- far-exterior is
  now macro-served, not declined.  Deleted the class; its "teeth" role is
  subsumed by `CarrierAdequacyGuardSelfFalsificationTestCase`.

## 2026-08-21 (rho-partition continuity + macro-fold normalization tests)
- RHO-PARTITION CONTINUITY SWEEP MUST SIT IN THE WALL BAND (gamma' > 0.5):
  the coverage union ``(RHO_LO<=rho<=RHO_HI) or (gamma'>_WALL_GAMMA_PRIME)``
  only admits off-caustic rho>RHO_HI through the wall-band clause.  At
  gamma'<0.5 a rho just above RHO_HI is neither shell nor wall -> serve
  DECLINES, so a continuity sweep at gamma'=0.3 CANNOT cross RHO_HI.  Used
  gamma'=0.8, theta=0.8, w=0.3, rho nodes 1.3 (fold) / 1.45 (macro) --
  interior rho-grid nodes (edges 1.1/1.6 only support cubic interp; a
  ~1e-16 roundoff on an edge node can trip inclusive `covers`).  w=0.3 is
  the low-w unresolved witness (macro-side w_split min ~0.455 at rho=1.6).
  Node-exact serve (~1e-16) both sides; self-falsification patches
  `_likelihood_mod.partitioned_reference` to force macro on the fold side ->
  rel err 2.4e-1..3.6e-1 (teeth).
- MACRO-FOLD LOW-W ASYMPTOTE: fold residual |r| at band bottom (w=0.02,
  h==0) = 0.9689 vs sqrt(1-gp^2)=0.9539 -> deviation 1.5e-2 (pin 5e-2).
  The RAW (un-renormalized, h=1) form gives |r|=0.542 -> deviation 4.1e-1
  (|F_ref| diverges w^{-1/6} so r dives toward 0).  Self-falsification
  patches `_lwd_module.smootherstep` -> 1.0 (h=1) and asserts the band-bottom
  residual deviates >> 5e-2.  NOTE the spec's "0.5*sqrt" bar does NOT
  discriminate at w=0.02 (raw form still 0.542 > 0.477); only "close to
  sqrt(1-gp^2)" has teeth.  h==0.0 / h==1.0 exact equality is the established
  idiom (see test_magnitude_tracks_wronskian_form).
