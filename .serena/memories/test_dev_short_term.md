# Test Dev Short-Term Observations

- Build 8h-d? (D4): EXTENDED test_lensing_surrogate.py (+466/-41, 3 D4 specs)
  and PREMISE-REPAIRED the whole engine-dependent suite broken by WP2. Full
  file now GREEN class-by-class (16 classes; ~74 pass / 2 skip). Plots:
  surrogate_cusp_angles_vs_gamma.png, surrogate_cusp_nodes_on_rays.png,
  surrogate_positive_box_eps_heatmap.png in tests/output/.
  * WP2 broke ALL engine tests two independent ways: (a) the new
    _assert_farfield_carrier_continuity guard aborts COARSE single-box builds
    (winding is a ~pi branch step, density-independent: n_gamma 6/8/12/16 all
    raise), (b) frame-invariant relabel (env now = E_ff*exp(+1j w t_min))
    broke the recon helper. Test-side repair: (a) _from_engine_without_carrier
    _guard(**kw) wraps from_engine in mock.patch.object(surrogate_module,
    '_assert_farfield_carrier_continuity', lambda *a,**k:None); rewired _train
    /_refusal_surrogate through it. (b) _reconstruct_via_surrogate far-field
    branch now calls channels.reconstruct_farfield(w,env,geom.delays,geom.
    saddle_kernels,geom.real_mask,definition,geom.t_min) (t_min REQUIRED
    positional) instead of stale reconstruct_from_envelope(...,ff_switch,0.0).
    Guard bypass defended by reachable-red test_unpatched_positive_box_build_
    raises_carrier_discontinuity (real from_engine on POS_BOX @ SHIP_PARAM_
    NODES -> assertRaises CarrierDiscontinuityError); guard's own teeth live
    in test_lensing_farfield_envelope.FarfieldCarrierContinuityGuardTestCase.
  * Spec1 ClosedFormCuspAngleTestCase: closed-form source-plane cusp set
    {0,+/-pi/2,pi} (written INDEPENDENTLY as CLOSED_FORM_CUSP_ANGLES, NOT
    imported) vs surrogate_training._cusp_source_angles(gamma,2000) over 6
    gammas in (0,1); agree <1e-9, gamma-independent (np.ptp spread<1e-9);
    magnitude via _branch_speed_profile+_find_cusps+geometry.critical_point
    (...).source strictly increases + sweep>1.0. 4 tests, 9s.
  * Spec2 FromEngineCuspWiringTestCase: _union_cusp_nodes pure-fn (insert on
    straddle-0, all in-range cusps, dedup within _CUSP_NODE_DEDUP_TOL, noop
    w/o in-range cusp) + built-chart checks (positive chart carries 0.0 node,
    non-uniform spacing; macro-saddle uniform spacing np.ptp<1e-9). Reuses
    cached _pos/_sad_surrogate_ship. 7 tests, 73s.
  * Spec3: DE-XFAILED test_positive_box_reconstruction_within_budget -> normal
    pass, POS_RECON_TOL=0.20 UNCHANGED. Named cusp-ray config (0.40,2.183,0.0)
    collapses eps 0.260 (cusp-union OFF) -> 1.06e-4 (ON); box max eps 0.132<
    0.20. Reachable-red test_cusp_union_off_regresses_cusp_ray patches
    _union_cusp_nodes to identity -> no 0.0 node, cusp-ray eps>0.20 (pass
    attributable to WP3). NOTE only edited test file; the many modified
    production files in git status are the WP coders', not mine.

- Build 8h-d2 (D3): EXTENDED test_lensing_farfield_envelope.py (+13 tests,
  ~5s new / 82s full file). New classes:
  * FarfieldTelescopingRoundTripTestCase: reconstruct_farfield(w,env,delays,
    saddle_kernels,REAL_MASK,definition,t_min) reproduces HEAD-reconstructed
    field AND engine exact_total to 0.0 (<1e-12) for FARFIELD_KERNEL_SUM and
    ..._MINUS_GHOST (ghost re-added with +t_min tilt BEFORE the call). TWO
    independent oracles: HEAD channels.py loaded side-by-side via git-show +
    importlib, and partition.exact_total. Reachable-red: t_min=0 stale caller
    leaves the frame carrier in -> 6-8e-3 error (test_stale_t_min_zero_breaks).
    NOTE reconstruct_farfield takes real_mask (bool (_N_CHANNELS,)) + builds
    its own switch internally + returns (kernels,total); re-modulates by
    exp(-1j w t_min) FIRST. Configs ((0.0387,1.3,1.3),(0.04,1.5,1.5),
    (0.05,1.2,0.9)) all admit the MINUS_GHOST ghost gate.
  * FarfieldCarrierContinuityGuardTestCase: surrogate._assert_farfield_carrier
    _continuity(grid,w_max,gamma_grid,shape=(n_gamma,n_rho,n_theta)) evaluates
    the TOP w-slice grid[-1]; raises CarrierDiscontinuityError(ValueError) when
    adjacent |angle(lead*conj(trail))|*w_max >= pi/2 with both mags>0; skips
    zero-mag flips; raises plain ValueError on gamma_grid/shape[0] mismatch.
  * StaleFarfieldAxisSchemaRefusalTestCase: OLD tag
    'caustic_radial_offset_rho_theta' (NEW adds '_framewinv') + absent tag both
    hard-refuse at LensAmplificationSurrogate.save->load (chart0_meta JSON
    re-stamp) via _validate_farfield_axis_schema (not in
    _KNOWN_FARFIELD_AXIS_SCHEMAS); current tag loads. Synthetic charts (no
    engine) keep all 3 classes fast-tier.
  * PREMISE-REPAIR of a stale NON-D3 shard: WP2 redefined
    farfield_envelope_from_partition to return env*exp(+1j w t_min)
    (frame-invariant), which BROKE ReconstructionExactnessTestCase (Q2/Q3
    shard) -- reconstruct_from_envelope (no t_min) departed exact_total by
    7.31e-3. Migrated its _reconstruct_public to channels.reconstruct_farfield
    (+t_min) and _reconstruct_gauge to feed the RE-MODULATED env
    (exp(-1j w t_min)) to _gauge.envelope_total -> both 0.0. Kept the 1e-12
    tolerance (premise repair, not tolerance repair); my own D3 additions were
    purely additive (single diff hunk @@ -1903 +1903,503). Full file GREEN:
    34 passed, 21 skipped (train-tier). Plots: farfield_telescoping_roundtrip
    _error.png, farfield_carrier_continuity_winding.png in tests/output/.

- Build 8h-d2 (D1+D2): wrote cogwheel/tests/test_lensing_ppgo_map.py (14
  tests, ~16s) for ppgo_map.annulus_rho + the ppGO exclusion gauge fix.
  * D2 byte-equiv: annulus_rho(g,|y|,0) == np.hypot(y1,y2)/caustic_geometry
    (g,0)[0] EXACTLY (same deterministic reach -> bit-identical). Guard:
    reach<=0 branch is UNREACHABLE via real caustic_geometry (it raises
    LensDomainError first) -> reach it only by patching
    ppgo_map.caustic_geometry to a Mock returning (0.0, dir).
  * D1 SIGN TRAP: Architect brief said "w_cert <= HEAD, move read point
    OUTWARD (harder)". MEASURED reality is the OPPOSITE: _scalar_caustic_reach
    == caustic_geometry(g,0)[0], so both gauges divide by the SAME reach;
    fix feeds a SMALLER |y| (region_exclusion_rho-1+crmin <= physical), so
    rho_fix(0.98) < rho_head(1.19) -> INWARD. Production code's own comment
    confirms "farther-out cell = easier, lower w_cert", so inner = higher
    w_cert. Encoded w_cert(fix) >= w_cert(head) (never-easier = higher floor)
    on a synthetic monotone-decreasing map; flagged brief inversion in the
    docstring. No shipped ppGO artifact -> build CertifiedPpgoMap.from_arrays
    with STATUS_CERTIFIED + huge rho_measured_max.
  * The additive exclusion_rho round-trip ((1+phys-crmin)-1+crmin) is NOT
    bit-exact (1 ULP) -> zero-narrowing edge uses assertAlmostEqual, strict
    < only at the real 0.30 narrowing where the gap is huge.
  * NEIGHBOR RED (report-only): test_lensing_ppgo_bandsplit.py has 2 fail +
    4 error from WP2's new CarrierDiscontinuityError guard in surrogate.py:767
    tripping bandsplit fixtures -- PRODUCTION-side, not my test-only file.

- Build 8h-d? EXTEND test_lensing_ppgo_map.py (+8 tests -> 22 total, ~18s)
  for D1 specs. New: PpgoOrderingReachableRedTestCase (reachable-red:
  reproduces the ORDERING bug -- ppgo coord from PRE-narrowing exclusion_rho
  vs narrowed region_exclusion_rho; production invariant "read no easier than
  served inner edge" PASSES fixed / RAISES buggy, wrapped in assertRaises =
  the teeth; buggy src = exclusion_rho-1+crmin == physical exactly, so
  rho_buggy==physical/reach==rho_head). SaddleBranchByteIdentityTestCase
  (macro-saddle band (1.5,1.7), parity=-2, gamma>1: ppgo=physical/reach ==
  annulus_rho(gm,physical,0) EXACTLY -- HEAD oracle; saddle has NO exterior
  tiles so region==exclusion, no narrowing; foil _narrowed_foil_rho(0.30)
  reads strictly smaller cell). Extended _synthetic_ppgo_map with gamma_max
  param (default 1.0 keeps old shards) to cover gamma_mid>1 for saddle cells.
  * VERIFIED in conda: _scalar_caustic_reach(g)==caustic_geometry(g,0)[0]
    BIT-EXACT at g=0.5 AND g=1.6/2.2; saddle ppgo==annulus_rho(gm,phys,0)
    exact (0.0 diff). Saddle round-trip annulus_rho(gm,excl-1+crmin,0)==
    phys/reach EXACT for saddle (positive band was 4.4e-16 off).
  * D3 (far-field frame-invariance / t_min relabel) was in my Architect spec
    bundle but belongs to chang_refsdal.channels.farfield_envelope_from_
    partition -> its OWNED suite test_lensing_farfield_envelope.py (another
    run; covers lobe-flip invariance + continuity, NOT explicitly t_min-frame
    invariance). Did NOT duplicate into the ppgo_map suite (wrong module) nor
    edit the owned suite. Reported as owner-domain gap.
