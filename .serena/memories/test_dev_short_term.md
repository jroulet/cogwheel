# Test Dev Short-Term Observations

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
